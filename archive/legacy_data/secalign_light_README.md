---
model_name: VNU-SecAlign
language:
  - en
tags:
  - security
  - instruction-following
  - finetuning
license: apache-2.0
base_model: meta-llama/Llama-3.1-8B-Instruct
metrics:
  - name: asr
    type: error_rate
    description: Attack success rate / ASR
datasets:
  - Jason-42195/VNU-SecAlign
---

VNU-SecAlign: LoRA adapter and datasets for SecAlign experiments.

This repository contains:
- checkpoints/final_checkpoint: LoRA adapter and tokenizer files.
- data/: datasets used for evaluation and training (moved here).

Usage (load adapter with PEFT):

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(base, device_map='auto')
model = PeftModel.from_pretrained(model, "Jason-42195/VNU-SecAlign")
```

Judge used in evaluation: GPT-4o (deployment `gpt-4o`, temperature=0.0).

