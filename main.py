
import argparse

import math
import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed import get_accelerator
from tqdm import tqdm
from data_utils import create_prompt_dataset

from dsutils import (
    print_rank_0,
    to_device,
    save_hf_format,
    set_random_seed,
    get_all_reduce_mean,
    get_optimizer_grouped_parameters,
    save_zero_three_model,
    load_hf_tokenizer,
    get_train_ds_config,
    create_hf_model, causal_lm_model_to_fp32_loss,
    get_hf_configs, calculate_flops
)

import os


def main(
    data_path=[
        "Dahoas/rm-static"
    ],
    data_split="5,1,1",
    model_name_or_path="allenai/tulu-2-dpo-7b",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    max_seq_len=1024,
    learning_rate=2e-5,
    weight_decay=0.1,
    num_train_epochs=1,
    gradient_accumulation_steps=16,
    lr_scheduler_type="cosine",
    num_warmup_steps=50,
    seed=1234,
    gradient_checkpointing=True,
    zero_stage=1,
    output_dir="./test",
    data_output_path="./data_files/",
    offload=True,
    dtype="bf16",
    compute_fp32_loss=False,
    enable_tensorboard=False,
    tensorboard_path="./tensorboard",
    add_eot_token=False,
    print_loss=True,
    replace_chance: float = 0.9,
):
    parser = argparse.ArgumentParser(description='CasualLM')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
        
    
    local_rank = args.local_rank
    print(f"LOCAL RANK : {local_rank}")   
    
    get_accelerator().set_device(args.local_rank)
     
    torch.distributed.init_process_group(backend="nccl")
    
    device = torch.device(get_accelerator().device_name())
    global_rank = torch.distributed.get_rank()
    
    if global_rank == 0:
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(tensorboard_path, exist_ok=True)
        print(f"Output directory: {output_dir}")

    ds_config = get_train_ds_config(
        offload=offload,
        dtype=dtype,
        stage=zero_stage,
        enable_tensorboard=enable_tensorboard,
        tb_path=tensorboard_path,
        tb_name="fim",
    )
    ds_config["train_micro_batch_size_per_gpu"] = per_device_train_batch_size
    ds_config["train_batch_size"] = (
        per_device_train_batch_size
        * torch.distributed.get_world_size()
        * gradient_accumulation_steps
    )

    # Set random seed
    set_random_seed(seed)
    torch.distributed.barrier()

    # Tokenizer
    end_of_conversation_token = ""
    additional_special_tokens = end_of_conversation_token if add_eot_token else None
    tokenizer = load_hf_tokenizer(
        model_name_or_path,
        fast_tokenizer=True,
        add_special_tokens=["<|SUFFIX|>", "<|PREFIX|>", "<|order|>", "<|STARTFIM|>", \
            "<|ENDMIDDLE|>", "<|MIDDLE_0|>", "<|MIDDLE_1|>", "<|MIDDLE_2|>", "<|MIDDLE_3|>", "<|MIDDLE_4|>", "<|MIDDLE_5|>", "<|MIDDLE_6|>"],
        
    )

    # Model
    model = create_hf_model(
        AutoModelForCausalLM, model_name_or_path, tokenizer, ds_config, dropout=None
    )
    
    import random
    
    # set trainable param to only 10% of the model, to save memory
    
    random.seed(0)
    
    for n, p in model.named_parameters():
        if len(p.shape) >= 1 and random.random() < replace_chance and not any(s in n for s in ['lm_head', 'token']):
            p.requires_grad = False
            print(f"Freezing {n} with shape {p.shape}")
            
    
    model.train()

    if compute_fp32_loss:
        print_rank_0(
            f"Using model {model.__class__.__name__} with loss in fp32", global_rank
        )
        causal_lm_model_to_fp32_loss(model)

    # Prepare the data
    train_phase = 1
    train_dataset, eval_dataset = create_prompt_dataset(
        local_rank,
        data_path,
        data_split,
        data_output_path,
        train_phase,
        seed,
        tokenizer,
        max_seq_len,
        rebuild = True
    )

    # DataLoaders
    train_sampler = (
        RandomSampler(train_dataset)
        if local_rank == -1
        else DistributedSampler(train_dataset)
    )
    eval_sampler = (
        SequentialSampler(eval_dataset)
        if local_rank == -1
        else DistributedSampler(eval_dataset)
    )

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        sampler=train_sampler,
        batch_size=per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        sampler=eval_sampler,
        batch_size=per_device_eval_batch_size,
    )

    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in tqdm(enumerate(eval_dataloader)):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        model.train()
        try:
            losses = get_all_reduce_mean(losses)
        except:
            pass
        try:
            perplexity = torch.exp(losses).item()
        except OverflowError:
            perplexity = float("inf")
        return perplexity, losses.item()

    # Optimizer, Scheduler, DeepSpeed Initialization
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, weight_decay)
    AdamOptimizer = DeepSpeedCPUAdam if offload else FusedAdam
    optimizer = AdamOptimizer(
        optimizer_grouped_parameters, lr=learning_rate, betas=(0.9, 0.95)
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )

    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=None,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True,
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    global_step = 0
    
    if global_rank == 0:
        import wandb
        
        project_name = "fimllama"
        
        wandb.init(project=project_name, config={
            "model_name": model_name_or_path,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_train_epochs": num_train_epochs,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "lr_scheduler_type": lr_scheduler_type,
            "num_warmup_steps": num_warmup_steps,
            "seed": seed,
            "gradient_checkpointing": gradient_checkpointing,
            "zero_stage": zero_stage,
            "output_dir": output_dir,
            "data_output_path": data_output_path,
            "offload": offload,
            "dtype": dtype,
            "compute_fp32_loss": compute_fp32_loss,
        })


    for epoch in range(num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            global_rank,
        )
        model.train()
        import time

        for step, batch in enumerate(train_dataloader):
            
            global_step += 1
            
            start = time.time()
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)

            loss = outputs.loss

            model.backward(loss)
            model.step()
            if print_loss:
                print(
                    f"Epoch: {epoch}, Step: {step}, Rank: {torch.distributed.get_rank()}, loss = {loss}"
                )
                # if global_rank == 0:
                #     wandb.log({"TrainingLoss" : get_all_reduce_mean(loss)})
            end = time.time()
            if torch.distributed.get_rank() == 0:
                hf_model = model.module
                e2e_time = end - start
                rank = global_rank

                hf_config = hf_model.config
                num_layers, hidden_size, vocab_size = get_hf_configs(hf_config)

                gpus_per_model = torch.distributed.get_world_size()
                seq_length = max_seq_len
                batch_size = per_device_train_batch_size
                samples_per_second = batch_size / e2e_time
                checkpoint_activations_factor = 4 if gradient_checkpointing else 3

                hf_model._num_params = sum(
                    [
                        p.ds_numel if hasattr(p, "ds_tensor") else p.numel()
                        for p in hf_model.parameters()
                    ]
                )
                params_in_billions = hf_model._num_params / (1e9)

                # Megatron paper's formula to calculate training flops
                train_flops_per_iteration = calculate_flops(
                    checkpoint_activations_factor, batch_size, seq_length, hf_config
                )

                train_tflops = train_flops_per_iteration / (
                    e2e_time * gpus_per_model * (10**12)
                )

                param_string = (
                    f"{params_in_billions:.3f} B" if params_in_billions != 0 else "NA"
                )
                print(
                    f"Model Parameters: {param_string}, Latency: {e2e_time:.2f}s, TFLOPs: {train_tflops:.2f}, Samples/sec: {samples_per_second:.2f}, Time/seq {e2e_time/batch_size:.2f}s, Batch Size: {batch_size}, Sequence Length: {seq_length}"
                )

            # Evaluate perplexity on the validation set.
            
            if global_step % 500 == 10:
                print_rank_0(
                    f"***** Evaluating perplexity, Epoch {epoch+1}/{num_train_epochs} *****",
                    global_rank,
                )
                perplexity, eval_loss = evaluation(model, eval_dataloader)
                print_rank_0(f"ppl: {perplexity}, loss: {eval_loss}", global_rank)
                model.tput_timer.update_epoch_count()
                
                if global_rank == 0:
                    wandb.log({"ppl": perplexity, "loss": eval_loss, "epoch": epoch})

                if output_dir is not None:
                    print_rank_0("saving the final model ...", global_rank)

                    if global_rank == 0:
                        import os

                        # used to save huggingface format, so we can use it for hf.from_pretrained
                        model_to_save = model.module if hasattr(model, "module") else model
                        CONFIG_NAME = "config.json"
                        WEIGHTS_NAME = "pytorch_model.bin"
                        saving_output_dir = os.path.join(output_dir, f"step_{global_step}_final")
                        os.makedirs(saving_output_dir, exist_ok=True)
                        output_model_file = os.path.join(saving_output_dir, WEIGHTS_NAME)
                        output_config_file = os.path.join(saving_output_dir, CONFIG_NAME)
                        save_dict = model_to_save.state_dict()
                        for key in list(save_dict.keys()):
                            if "lora" in key:
                                del save_dict[key]
                        torch.save(save_dict, output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_pretrained(saving_output_dir)

                    if zero_stage == 3:
                        save_zero_three_model(model, global_rank, output_dir, zero_stage=zero_stage)


if __name__ == "__main__":
  
    main()
