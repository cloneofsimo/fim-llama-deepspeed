
import os
import torch
import random
import numpy as np
from transformers import set_seed, AutoTokenizer
import json
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.accelerator import get_accelerator
import torch.nn as nn


def print_rank_0(msg, rank=None):
    if rank is not None and rank <= 0:
        print(msg)
    elif is_rank_0():
        print(msg)


def is_rank_0():
    """Check whether it is rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            return True
        else:
            return False
    else:
        return True


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


class MovingAverage:

    def __init__(self):
        self.count = 0
        self.total = 0
        self.mean = 0

    def update(self, num):
        self.total += num
        self.count += 1
        self.mean = self.total / self.count

        return self.mean


class ExponentialMovingAverage:

    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.ema = None

    def update(self, num):
        prev_ema = num if self.ema is None else self.ema
        self.ema = self.alpha * prev_ema + (1.0 - self.alpha) * num
        return self.ema

    def get(self):
        return self.ema if self.ema is not None else 0.


def get_tokenizer(model_name_or_path, fast_tokenizer=True):
    if "llama" in model_name_or_path:
        from transformers.models.llama import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer)
        if tokenizer.pad_token is None:
            # assert tokenizer.eos_token is not None
            # tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.padding_side = 'right'
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer)
        tokenizer.pad_token = tokenizer.eos_token
        # make sure tokenizer is right pad in our logic
        tokenizer.padding_side = 'right'
    return tokenizer


def load_hf_tokenizer(model_name_or_path,
                      fast_tokenizer=True,
                      add_special_tokens=None):
    if os.path.exists(model_name_or_path):
        # Locally tokenizer loading has some issue, so we need to force download
        model_json = os.path.join(model_name_or_path, "config.json")
        if os.path.exists(model_json):
            model_json_file = json.load(open(model_json))
            model_name = model_json_file.get("_name_or_path",
                                             model_name_or_path)
            tokenizer = get_tokenizer(model_name,
                                      fast_tokenizer=fast_tokenizer)
    else:
        tokenizer = get_tokenizer(model_name_or_path,
                                  fast_tokenizer=fast_tokenizer)

    if add_special_tokens is not None:
        add_special_tokens = [add_special_tokens] if isinstance(add_special_tokens, str) \
            else add_special_tokens
        tokenizer.add_special_tokens(
            {'additional_special_tokens': add_special_tokens})
        
        print("Added special tokens: ", add_special_tokens)
        # print ids
        print("Added special token ids: ", tokenizer.convert_tokens_to_ids(add_special_tokens))
    return tokenizer


def save_hf_format(model, tokenizer, args, sub_folder=""):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(args.output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    for key in list(save_dict.keys()):
        if "lora" in key:
            del save_dict[key]
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)


def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        get_accelerator().manual_seed_all(seed)


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


# This function is a modified version of code available in the from_pretrained API of HuggingFace Transformers
# The code is copied and modified from: https://github.com/huggingface/transformers/blob/5ee9693a1c77c617ebc43ef20194b6d3b674318e/src/transformers/modeling_utils.py#L498
# This function helps load a HF format checkpoint into a DeepSpeed wrapped model that has been sharded using ZeRO Stage 3
def load_state_dict_into_model(model_to_load=None,
                               state_dict=None,
                               start_prefix="",
                               zero_stage=0):

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: nn.Module, state_dict, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # Parameters of module and children will start with prefix. We can exit early if there are none in this
        # state_dict
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            if zero_stage == 3:
                # In sharded models, each shard has only part of the full state_dict, so only gather
                # parameters that are in the current state_dict.
                named_parameters = dict(
                    module.named_parameters(prefix=prefix[:-1], recurse=False))
                params_to_gather = [
                    named_parameters[k] for k in state_dict.keys()
                    if k in named_parameters
                ]
                if len(params_to_gather) > 0:
                    # because zero3 puts placeholders in model params, this context
                    # manager gathers (unpartitions) the params of the current layer, then loads from
                    # the state dict and then re-partitions them again
                    with deepspeed.zero.GatheredParameters(params_to_gather,
                                                           modifier_rank=0):
                        if torch.distributed.get_rank() == 0:
                            module._load_from_state_dict(*args)
            else:
                module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".")

    load(model_to_load, state_dict, prefix=start_prefix)
    # Delete `state_dict` so it could be collected by GC earlier. Note that `state_dict` is a copy of the argument, so
    # it's safe to delete it.
    del state_dict

    return error_msgs


def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    embed_lr=3e-4,
    no_decay_name_list=[
        "bias", "layer_norm.weight", "layernorm.weight", "norm.weight",
        "ln_f.weight"
    ],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n.lower()
                                                    for nd in ['embed_tokens']))
            ],
            "weight_decay":
            weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if "embed_tokens" in n and p.requires_grad
            ],
            "weight_decay":
            weight_decay,
            "lr":
            embed_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n.lower()
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]

    non_empty_groups = []
    for group in optimizer_grouped_parameters:
        if group["params"]:
        
            non_empty_groups.append(group)
            
    def print_all_groups():
        for group in non_empty_groups:
            for p in group["params"]:
                print(p.shape, p.requires_grad)
    
    
            
    return non_empty_groups


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def moving_average(model, model_ema, beta=0.992, device=None, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    with torch.no_grad():
        for param, param_ema in zip(model.parameters(),
                                    model_ema.parameters()):
            # TODO: use prefiltering for efficiency
            params_to_fetch = _z3_params_to_fetch([param, param_ema
                                                   ]) if zero_stage_3 else []
            should_gather_param = len(params_to_fetch) > 0
            with deepspeed.zero.GatheredParameters(
                    params_to_fetch, enabled=should_gather_param):
                data = param.data
                if device is not None:
                    data = data.to(device)
                param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))


def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    os.makedirs(save_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)

    model_to_save = model_ema.module if hasattr(model_ema,
                                                'module') else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():

            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v
                                                                            ]),
                                                       enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict


# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4


def get_train_ds_config(offload,
                        dtype,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512,
                        enable_tensorboard=False,
                        enable_mixed_precision_lora=False,
                        tb_path="",
                        tb_name=""):

    device = "cpu" if offload else "none"
    if dtype == "fp16":
        data_type = "fp16"
        dtype_config = {"enabled": True, "loss_scale_window": 100}
    elif dtype == "bf16":
        data_type = "bfloat16"
        dtype_config = {"enabled": True}
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False
    }
    if enable_mixed_precision_lora:
        zero_opt_dict["zero_quantized_nontrainable_weights"] = True
        if dist.get_world_size() != get_accelerator().device_count():
            zero_opt_dict["zero_hpz_partition_size"] = get_accelerator(
            ).device_count()
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        data_type: dtype_config,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
        "tensorboard": {
            "enabled": enable_tensorboard,
            "output_path": f"{tb_path}/ds_tensorboard_logs/",
            "job_name": f"{tb_name}_tensorboard"
        }
    }


def get_eval_ds_config(offload, dtype, stage=0):
    device = "cpu" if offload else "none"
    if dtype == "fp16":
        data_type = "fp16"
        dtype_config = {
            "enabled": True,
        }
    elif dtype == "bf16":
        data_type = "bfloat16"
        dtype_config = {"enabled": True}
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        },
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        data_type: dtype_config,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }



### MODEL UTILS



def configure_dropout(model_config, dropout):
    if dropout is not None:
        for key in ('dropout', 'attention_dropout', 'hidden_dropout',
                    'activation_dropout'):
            if hasattr(model_config, key):
                print(f"Setting model_config.{key} to {dropout}")
                setattr(model_config, key, dropout)


def causal_lm_model_to_fp32_loss(model):
    """ Convert CausalLM model to calculate loss in fp32 """

    def causal_lm_forward(
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **deprecated_arguments,
    ):
        kwargs = dict() if model.config.model_type == "llama" else dict(
            head_mask=head_mask)
        output = model.__original_forward__(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs)

        return_dict = isinstance(output, dict)
        lm_logits = output.logits if return_dict else output[0]
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].float().contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size),
                shift_labels.view(batch_size * seq_length))

        if not return_dict:
            # re-pack output with fp32 loss
            return ((loss, ) + output) if loss is not None else output

        output.loss = loss
        return output

    model.__original_forward__ = model.forward
    model.forward = causal_lm_forward

import random

import os
import math
import torch
from transformers import (
    AutoConfig,
    AutoModel,
)

from transformers.deepspeed import HfDeepSpeedConfig


def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    rlhf_training=False,
                    dropout=None):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    configure_dropout(model_config, dropout)

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
    if rlhf_training:
        # the weight loading is handled by create critic model
        model = model_class.from_config(model_config)
    else:
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            trust_remote_code=True,
            config=model_config)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8
    
    
    return model


# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch


# This function can be used to print throughput for Step 1 and 2 only
def print_throughput(hf_model, args, e2e_time, rank=0):
    if rank <= 0:
        hf_config = hf_model.config
        num_layers, hidden_size, vocab_size = get_hf_configs(hf_config)

        gpus_per_model = torch.distributed.get_world_size()
        seq_length = args.max_seq_len
        batch_size = args.per_device_train_batch_size
        samples_per_second = batch_size / e2e_time
        checkpoint_activations_factor = 4 if args.gradient_checkpointing else 3
        if args.lora_dim > 0:
            k = args.lora_dim * 2 / hidden_size
            checkpoint_activations_factor -= (1 - k)

        hf_model._num_params = sum([
            p.ds_numel if hasattr(p, "ds_tensor") else p.numel()
            for p in hf_model.parameters()
        ])
        params_in_billions = hf_model._num_params / (1e9)

        # Megatron paper's formula to calculate training flops
        train_flops_per_iteration = calculate_flops(
            checkpoint_activations_factor, batch_size, seq_length, hf_config)

        train_tflops = train_flops_per_iteration / (e2e_time * gpus_per_model *
                                                    (10**12))

        param_string = f"{params_in_billions:.3f} B" if params_in_billions != 0 else "NA"
        print(
            f"Model Parameters: {param_string}, Latency: {e2e_time:.2f}s, TFLOPs: {train_tflops:.2f}, Samples/sec: {samples_per_second:.2f}, Time/seq {e2e_time/batch_size:.2f}s, Batch Size: {batch_size}, Sequence Length: {seq_length}"
        )


# Enhanced version of the function above that provides calculations and printing for Step 3
def print_throughput_step3(actor_model,
                           critic_model,
                           args,
                           e2e_time,
                           gen_exp_time,
                           train_time,
                           rank=0):
    if rank <= 0:
        # Actor model passed here is a HF model.
        actor_hf_config = actor_model.config
        # Critic model passed here is  a DeepSpeed Engine. The module inside is the Reward model (that wraps a HF model).
        critic_hf_config = critic_model.module.config

        actor_num_layers, actor_hidden_size, actor_vocab_size = get_hf_configs(
            actor_hf_config)
        critic_num_layers, critic_hidden_size, critic_vocab_size = get_hf_configs(
            critic_hf_config)

        gpus_per_model = torch.distributed.get_world_size()
        seq_length = args.max_answer_seq_len + args.max_prompt_seq_len
        batch_size = args.per_device_generation_batch_size * args.generation_batches * args.ppo_epochs * gpus_per_model * 1 if args.unsupervised_dataset_name is None else 2
        samples_per_second = batch_size / e2e_time

        actor_checkpoint_activations_factor = 4 if args.actor_gradient_checkpointing else 3
        critic_checkpoint_activations_factor = 4 if args.critic_gradient_checkpointing else 3
        if args.actor_lora_dim > 0:
            k = args.actor_lora_dim * 2 / actor_hidden_size
            actor_checkpoint_activations_factor -= (1 - k)
        if args.critic_lora_dim > 0:
            k = args.critic_lora_dim * 2 / critic_hidden_size
            critic_checkpoint_activations_factor -= (1 - k)

        actor_model._num_params = sum([
            p.ds_numel if hasattr(p, "ds_tensor") else p.numel()
            for p in actor_model.parameters()
        ])
        actor_params_in_billions = actor_model._num_params / (1e9)

        critic_model._num_params = sum([
            p.ds_numel if hasattr(p, "ds_tensor") else p.numel()
            for p in critic_model.parameters()
        ])
        critic_params_in_billions = critic_model._num_params / (1e9)

        # Megatron paper's formula to calculate training flops

        actor_train_flops_per_iteration = calculate_flops(
            actor_checkpoint_activations_factor, batch_size, seq_length,
            actor_hf_config)
        critic_train_flops_per_iteration = calculate_flops(
            critic_checkpoint_activations_factor, batch_size, seq_length,
            critic_hf_config)

        total_train_flops = actor_train_flops_per_iteration + critic_train_flops_per_iteration
        train_tflops = total_train_flops / (train_time * gpus_per_model *
                                            (10**12))

        gen_bs = args.per_device_generation_batch_size * gpus_per_model

        # Modified formula for calculating flops in the forward pass only
        gen_flops_per_iteration = (
            24 * gen_bs * seq_length * actor_num_layers *
            (actor_hidden_size**2)) * (
                1.0 + (seq_length / (6.0 * actor_hidden_size)) +
                (actor_vocab_size /
                 (16.0 * actor_num_layers * actor_hidden_size)))

        gen_tflops = gen_flops_per_iteration / (gen_exp_time * gpus_per_model *
                                                (10**12))

        if actor_hf_config.torch_dtype == torch.float16:
            num_bytes = 2
        elif actor_hf_config.torch_dtype == torch.float32:
            num_bytes = 4
        else:
            num_bytes = -1

        pertok_lat = gen_exp_time / args.max_answer_seq_len
        gen_bw = 1 / pertok_lat * actor_model._num_params * num_bytes / 1e9

        total_flops_per_iteration = total_train_flops + gen_flops_per_iteration * args.generation_batches
        total_tflops = total_flops_per_iteration / (e2e_time * gpus_per_model *
                                                    (10**12))

        print(
            f"End-to-End => Latency: {e2e_time:.2f}s, TFLOPs: {total_tflops:.2f}, Samples/sec: {samples_per_second:.2f}, Time/seq {e2e_time/batch_size:.2f}s, Batch Size: {batch_size}, Total Seq. Length: {seq_length}"
        )
        print(
            f"Generation => Latency: {gen_exp_time:.2f}s, Per-token Latency {pertok_lat*1000:.2f} ms, TFLOPs: {gen_tflops:.2f}, BW: {gen_bw if num_bytes > 0 else num_bytes:.2f} GB/sec, Answer Seq. Length: {args.max_answer_seq_len}"
        )
        print(
            f"Training   => Latency: {train_time:.2f}s, TFLOPs: {train_tflops:.2f}"
        )
        actor_param_string = f"{actor_params_in_billions:.3f} B" if actor_params_in_billions != 0 else "NA"
        critic_param_string = f"{critic_params_in_billions:.3f} B" if critic_params_in_billions != 0 else "NA"
        print(
            f"Actor Model Parameters => {actor_param_string}, Critic Model Parameters => {critic_param_string}"
        )


# Helper function to calculate FLOPs using the Megatron-LM paper's formula
def calculate_flops(checkpoint_activations_factor, batch_size, seq_length,
                    hf_config):
    num_layers, hidden_size, vocab_size = get_hf_configs(hf_config)
    flops_per_iteration = (24 * checkpoint_activations_factor * batch_size *
                           seq_length * num_layers * (hidden_size**2)) * (
                               1.0 + (seq_length / (6.0 * hidden_size)) +
                               (vocab_size /
                                (16.0 * num_layers * hidden_size)))
    return flops_per_iteration


def get_hf_configs(hf_config):
    num_layers = getattr(hf_config, "num_hidden_layers",
                         getattr(hf_config, "n_layer", None))
    hidden_size = getattr(hf_config, "hidden_size",
                          getattr(hf_config, "n_embd", None))
    vocab_size = getattr(hf_config, "vocab_size", None)
    assert all(
        (num_layers, hidden_size, vocab_size)
    ), "Could not determine number of layers, hidden size, and vocab size of the model"

    return num_layers, hidden_size, vocab_size
