"""CLI entrypoint for DPO / DPO+RPO / DPO+RPO+cDPO training — replaces
external/meta_secalign/secalign_plus_plus.py's torchtune invocation for this project (see
dpo_config.py docstring for why: RPO/cDPO require TRL, not torchtune).

Memory-constrained fallback notes (from archive/legacy_notebooks/{SECAL TRAIN.ipynb, Untitled0.ipynb}
— the only reusable part of those notebooks, everything else there used the wrong
hyperparameters/chat-template): if training OOMs on a single small GPU, in order of preference —
(1) enable 4-bit QLoRA loading, (2) reduce max_seq_len below MAX_LENGTH, (3) reduce
per_device_train_batch_size and raise gradient_accumulation_steps to compensate. None of these
are wired up by default here; add them via DPOConfig/BitsAndBytesConfig if actually needed.

Not run in this pass: needs real preference data from en_preference_gen.py / vi_preference_gen.py
/ attack10_gen.py plus a GPU.
"""

from __future__ import annotations

import argparse

from vi_secalign.training.dpo_config import DPOVariant, build_dpo_config, build_lora_config


def train(
    variant: DPOVariant,
    base_model: str,
    preference_data_path: str,
    output_dir: str,
    lora_target: str = "8b",
    learning_rate: float | None = None,
):
    from datasets import load_dataset  # deferred: heavy dependency
    from transformers import AutoModelForCausalLM, AutoTokenizer  # deferred: heavy dependency
    from trl import DPOTrainer  # deferred: heavy dependency

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)
    dataset = load_dataset("json", data_files=preference_data_path, split="train")

    lora_config = build_lora_config(target=lora_target)
    dpo_config = build_dpo_config(variant, output_dir=output_dir, learning_rate=learning_rate)

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["dpo", "dpo_rpo", "dpo_rpo_cdpo"], required=True)
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--preference_data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lora_target", choices=["8b", "70b"], default="8b")
    parser.add_argument("--learning_rate", type=float, default=None)
    args = parser.parse_args()

    train(
        variant=args.variant,
        base_model=args.base_model,
        preference_data_path=args.preference_data_path,
        output_dir=args.output_dir,
        lora_target=args.lora_target,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
