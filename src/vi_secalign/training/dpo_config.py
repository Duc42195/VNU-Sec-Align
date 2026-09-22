"""LoRA + DPO/RPO/cDPO config factories for VNU-SecAlign v2.

Training uses TRL's DPOTrainer/DPOConfig, not external/meta_secalign's own secalign_plus_plus.py
(which shells out to torchtune's lora_dpo_distributed recipe). This is deliberate: torchtune's
DPOLoss exposes no rpo_alpha/label_smoothing parameters (confirmed: grepping the whole
external/meta_secalign submodule for these terms returns 0 hits) — TRL is the only way to reach
RPO+cDPO, which is a real part of this project's contribution.
"""

from __future__ import annotations

from typing import Literal

from vi_secalign.config import ANCHOR_HYPERPARAMS, LORA_TARGET_MODULES, MAX_LENGTH, MAX_PROMPT_LENGTH

DPOVariant = Literal["dpo", "dpo_rpo", "dpo_rpo_cdpo"]


def build_lora_config(target: Literal["8b", "70b"] = "8b"):
    """Build a peft.LoraConfig matching Meta-SecAlign's published (or equivalent) LoRA setup.

    Citation caveat (verified directly against external/meta_secalign/helpers/llama3.1_8B_lora.yaml):
    `q_proj`/`v_proj` are literal strings from the yaml (line 25, `lora_attn_modules`). `gate_proj`
    /`up_proj`/`down_proj` are NOT literal strings anywhere in Meta's config — the yaml instead
    sets a boolean `apply_lora_to_mlp: True` (line 26), which is torchtune's own shorthand for
    applying LoRA across the Llama-3 MLP block. Listing the three MLP projection names explicitly
    here is this project's equivalent-effect interpretation for PEFT (which has no such boolean
    shorthand), not a direct quote from Meta — cite it that way in the paper.
    """
    from peft import LoraConfig  # deferred: real dependency, keep this module importable without it

    r = ANCHOR_HYPERPARAMS["lora_r"] if target == "8b" else 32  # 70B uses r=32 per helpers/llama3.3_70B_lora.yaml:20
    return LoraConfig(
        r=r,
        lora_alpha=ANCHOR_HYPERPARAMS["lora_alpha"],
        lora_dropout=ANCHOR_HYPERPARAMS["lora_dropout"],
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_dpo_config(
    variant: DPOVariant,
    output_dir: str,
    learning_rate: float | None = None,
    save_steps: int = 200,
    save_total_limit: int = 3,
    max_length: int | None = None,
):
    """Build a trl.DPOConfig for one arm of the DPO vs DPO+RPO vs DPO+RPO+cDPO ablation.

    rpo_alpha and label_smoothing (cDPO) are two distinct, complementary mechanisms, not the same
    thing — do not conflate them when writing this up:

    - `rpo_alpha` scales an additional NLL/SFT loss term computed on the `chosen` field of the
      *same* preference pair used for the DPO loss (i.e. maximize log-likelihood of `chosen` as if
      it were a plain SFT target). This anchors the model against drifting from natural text
      generation quality, which indirectly helps preserve utility (e.g. MMLU) — it does NOT
      compute anything against MMLU or any other benchmark's labels; benchmarks are only used
      post-hoc, at evaluation time, never inside the training loss.
    - `label_smoothing=epsilon` (conservative DPO / cDPO) instead smooths the DPO loss itself:
      L = -(1-eps)*log(sigmoid(h)) - eps*log(sigmoid(-h)), where h is the implicit reward margin
      between chosen and rejected. This assumes a small fraction of preference labels could be
      flipped/noisy — directly relevant here because Phase 1.5/2 preference pairs are
      self-generated (via vLLM sampling) and can contain noisy pairs, unlike a human-curated set.

    variant="dpo" sets both to their neutral/off values (rpo_alpha=None, label_smoothing=0.0),
    i.e. plain DPO, matching both SecAlign and SecAlign++'s actual published method (neither paper
    uses RPO or cDPO — confirmed by reading both directly).

    save_steps/save_total_limit exist because the rented pod (ckey.vn) has a hard 24h max rental —
    a run must be interruptible mid-training without losing all progress. HF Trainer's
    save_strategy="steps" already writes a full resumable checkpoint (model+optimizer+scheduler+
    RNG state) to output_dir/checkpoint-<step>/ on this cadence; train_dpo.py's
    --resume_from_checkpoint flag picks the latest one back up. save_total_limit caps how many are
    kept on disk at once (each LoRA checkpoint is small, but the pod's disk is tight — see
    .agents/infra_handoff.md). See .agents/record.md Decision #21.
    """
    from trl import DPOConfig  # deferred: real dependency

    if variant == "dpo":
        rpo_alpha, label_smoothing = None, 0.0
    elif variant == "dpo_rpo":
        rpo_alpha, label_smoothing = ANCHOR_HYPERPARAMS["rpo_alpha"], 0.0
    elif variant == "dpo_rpo_cdpo":
        rpo_alpha, label_smoothing = ANCHOR_HYPERPARAMS["rpo_alpha"], ANCHOR_HYPERPARAMS["label_smoothing"]
    else:
        raise ValueError(f"Unknown DPO variant {variant!r}, expected one of dpo/dpo_rpo/dpo_rpo_cdpo")

    return DPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate if learning_rate is not None else ANCHOR_HYPERPARAMS["learning_rate"],
        num_train_epochs=ANCHOR_HYPERPARAMS["epochs"],
        max_prompt_length=MAX_PROMPT_LENGTH,
        # Overridable below MAX_LENGTH=2048 (the cited Meta anchor value) for GPU-constrained runs --
        # confirmed on the pod (24GB-class card): even at batch_size=1 + working gradient checkpointing,
        # a batch that happens to contain one of the longer VN preference samples (~99.9th percentile
        # near 1960 tokens, see vi_preference_gen.py's own length-percentile printout) can still OOM
        # with as little as ~1.5GB headroom. This is a real hardware constraint, not a methodology
        # choice -- cite the effective value actually used for a given run, not this default.
        max_length=max_length if max_length is not None else MAX_LENGTH,
        rpo_alpha=rpo_alpha,
        label_smoothing=label_smoothing,
        beta=0.1,  # both SecAlign papers use the DPO-default beta=0.1
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        # HF Trainer's own default (per_device_train_batch_size=8, no grad accum) OOMs a single
        # 24GB-class GPU here -- see ANCHOR_HYPERPARAMS comment (yaml:78,80 effective batch 32).
        per_device_train_batch_size=ANCHOR_HYPERPARAMS["per_device_train_batch_size"],
        gradient_accumulation_steps=ANCHOR_HYPERPARAMS["gradient_accumulation_steps"],
        gradient_checkpointing=True,
        # use_reentrant=False is HF's own recommended setting for PEFT + gradient checkpointing
        # (paired with model.enable_input_require_grads() in train_dpo.py) -- the reentrant
        # (default) variant is the one known to interact badly with frozen-base + LoRA setups.
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
