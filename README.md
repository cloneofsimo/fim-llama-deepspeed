# FIM-LLAMA and Deepspeed FIM Trainer


https://github.com/cloneofsimo/fim-llama-deepspeed/assets/35953539/e5f4e48c-c46b-4d33-a57a-05ce9568a795


Link to [Huggingface model](https://huggingface.co/cloneofsimo/fim-llama), fine-tuned [Tulu-2-dpo](https://huggingface.co/allenai/tulu-2-dpo-70b) for FIM capability based on this code.

---

Train llama to have Fill-in-the-middle capability, with pure deepspeed.

It is mostly copy-paste from [deepspeed-examples](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/main.py).

## Installation

```bash
deepspeed --num_gpus 8 main.py
```

## How to prompt FIM-LLAMA?

The model is trained with new tokens : 

```python
new_tokens = ["<|SUFFIX|>", "<|PREFIX|>", "<|order|>", "<|STARTFIM|>","<|ENDMIDDLE|>", "<|MIDDLE_0|>", "<|MIDDLE_1|>", "<|MIDDLE_2|>", "<|MIDDLE_3|>", "<|MIDDLE_4|>", "<|MIDDLE_5|>", "<|MIDDLE_6|>"],
```

Here is how you would prompt it:

```python
instruction = f"<|user|> Complete the following text. <|assistant|><|STARTFIM|><|SUFFIX|>{suffix}<|PREFIX|>{prefix}<|MIDDLE|><|MIDDLE_N|>"
```

* $N$ is order of magnitude of the middle you want to fill in.
* Document is in form of prefix + MIDDLE_YOU_WANT + suffix.
