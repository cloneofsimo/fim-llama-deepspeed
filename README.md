# FIM-LLAMA and Deepspeed FIM Trainer


![FIM-LLAMA](contents/intro.mp4)


Link to [Huggingface model](https://huggingface.co/cloneofsimo/fim-llama), fine-tuned Tulu-2-dpo for FIM capability based on this code.

---

Train llama to have Fill-in-the-middle capability, with pure deepspeed.

It is mostly copy-paste from [deepspeed-examples](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/main.py).

## Installation

```bash
deepspeed --num_gpus 8 main.py
```


