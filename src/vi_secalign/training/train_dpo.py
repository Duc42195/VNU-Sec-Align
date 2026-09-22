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

Resumability (ckey.vn pod hard-caps rentals at 24h — see .agents/infra_handoff.md, Decision #21):
dpo_config.py sets save_strategy="steps" so HF Trainer already writes full resumable checkpoints
(model+optimizer+scheduler+RNG) to output_dir/checkpoint-<step>/ periodically. Before the pod's 24h
limit, upload output_dir (via tools/hf_upload/*.py) to HF; on the next pod, download it back to the
same output_dir path and pass --resume_from_checkpoint auto (or an explicit checkpoint-<step> path)
to pick up training exactly where it stopped.
"""

from __future__ import annotations

import argparse

from vi_secalign.hf_sync import upload_output
from vi_secalign.training.dpo_config import DPOVariant, build_dpo_config, build_lora_config


def _make_upload_on_save_callback(dest_subdir: str):
    """TrainerCallback that uploads each checkpoint-<step>/ dir to HF right after HF Trainer
    finishes writing it (on_save fires post-write, per transformers' TrainerCallback contract) --
    same 24h-pod-survival reasoning as vi_preference_gen.py's per-chunk upload, see hf_sync.py.
    """
    from transformers import TrainerCallback  # deferred: heavy dependency

    class UploadOnSaveCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            import os

            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            upload_output(checkpoint_dir, dest_subdir)

    return UploadOnSaveCallback()


def train(
    variant: DPOVariant,
    base_model: str,
    preference_data_path: str,
    output_dir: str,
    lora_target: str = "8b",
    learning_rate: float | None = None,
    resume_from_checkpoint: str | None = None,
    upload_checkpoints: bool = True,
    max_length: int | None = None,
):
    import torch  # deferred: heavy dependency
    from datasets import load_dataset  # deferred: heavy dependency
    from transformers import AutoModelForCausalLM, AutoTokenizer  # deferred: heavy dependency
    from transformers.trainer_utils import get_last_checkpoint  # deferred: heavy dependency
    from trl import DPOTrainer  # deferred: heavy dependency

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        # Llama-3.1-Instruct's tokenizer ships with no pad token (only eos/bos) -- DPOTrainer pads
        # batches to the longest sequence and needs a real pad_token_id to do it, matching the
        # standard Llama fallback (pad_token = eos_token; eos itself is masked out of the loss by
        # DPOTrainer's own label handling, so reusing it as pad is safe here).
        tokenizer.pad_token = tokenizer.eos_token
    # dtype=bf16 matches external/meta_secalign/helpers/llama3.1_8B_lora.yaml:94 (torchtune's own
    # anchor config) -- without it, from_pretrained defaults to fp32 (~32GB for an 8B model), which
    # does not fit the same pod GPU that vi_preference_gen.py's vLLM call needed max_model_len
    # capping for (see that script's fix, same 24GB-class card).
    # device_map="auto" makes accelerate write each shard's weights directly to GPU as it's read from
    # disk, instead of fully materializing the ~15GB bf16 model in CPU RAM first and only THEN moving
    # it to GPU (the default with no device_map). Some rental pods pair a big GPU with very little
    # system RAM (e.g. a 32GB-VRAM card on a 16GB-RAM host) -- without this, loading alone could OOM
    # the CPU before training even starts, a failure mode independent of every GPU-VRAM fix so far.
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map="auto")
    # Required for gradient checkpointing to actually save activation memory through a frozen base
    # model + LoRA adapter -- without this, the checkpointed layers' inputs have requires_grad=False
    # (base model is frozen), so torch.utils.checkpoint can't build a backward graph through them and
    # silently keeps full activations anyway (confirmed: real OOM on the pod at batch_size=1, with
    # the exact "None of the inputs have requires_grad=True" warning that flags this known PEFT +
    # gradient-checkpointing gotcha). This hooks the input embeddings' output to force requires_grad.
    model.enable_input_require_grads()
    dataset = load_dataset("json", data_files=preference_data_path, split="train")

    lora_config = build_lora_config(target=lora_target)
    dpo_config = build_dpo_config(
        variant, output_dir=output_dir, learning_rate=learning_rate, max_length=max_length,
    )

    callbacks = [_make_upload_on_save_callback(f"train_dpo/{variant}")] if upload_checkpoints else []
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=callbacks,
    )

    resume = resume_from_checkpoint
    if resume == "auto":
        # None if output_dir has no checkpoint-<step>/ yet (fresh run) -- get_last_checkpoint
        # returns None rather than raising in that case, so this is safe on the very first run too.
        resume = get_last_checkpoint(output_dir)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(output_dir)
    if upload_checkpoints:
        upload_output(output_dir, f"train_dpo/{variant}_final")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["dpo", "dpo_rpo", "dpo_rpo_cdpo"], required=True)
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--preference_data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lora_target", choices=["8b", "70b"], default="8b")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="'auto' to resume from the latest checkpoint-<step>/ under --output_dir if one exists "
        "(safe on a fresh run too -- becomes a no-op), or an explicit checkpoint directory path. "
        "Needed because the rented pod has a 24h max rental (see .agents/infra_handoff.md).",
    )
    parser.add_argument(
        "--no_upload_checkpoints", action="store_false", dest="upload_checkpoints", default=True,
        help="Skip auto-uploading each checkpoint-<step>/ (and the final model) to Hugging Face "
        "(see hf_sync.py). Uploads by default -- needed for the 24h pod rental cap to be survivable.",
    )
    parser.add_argument(
        "--max_length", type=int, default=None,
        help="Override DPOConfig's max_length (default: config.MAX_LENGTH=2048, the cited Meta "
        "anchor value). Lower this on a memory-constrained GPU -- see dpo_config.py's build_dpo_config "
        "docstring for why 2048 can OOM a 24GB-class card even at batch_size=1.",
    )
    args = parser.parse_args()

    train(
        variant=args.variant,
        base_model=args.base_model,
        preference_data_path=args.preference_data_path,
        output_dir=args.output_dir,
        lora_target=args.lora_target,
        learning_rate=args.learning_rate,
        resume_from_checkpoint=args.resume_from_checkpoint,
        upload_checkpoints=args.upload_checkpoints,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
