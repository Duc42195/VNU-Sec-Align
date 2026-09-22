"""Checkpoint registry — one place naming which model/path is used at each phase.

All local paths resolve under checkpoints/ (gitignored — see .gitignore and README). HF ids are
downloaded on demand by whichever loader is used (vLLM/transformers), not pre-fetched here.
"""

from __future__ import annotations

from dataclasses import dataclass

from vi_secalign.config import REPO_ROOT

CHECKPOINTS_ROOT = REPO_ROOT / "checkpoints"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    source: str  # HF hub id, or a local path under CHECKPOINTS_ROOT
    role: str


REGISTRY: dict[str, ModelSpec] = {
    "meta_secalign_8b": ModelSpec(
        name="meta_secalign_8b",
        source="facebook/Meta-SecAlign-8B",
        role="Phase 0 sanity-check baseline; Phase 4 cross-scale (70B spot-check counterpart at 8B); "
        "primary defended-model comparison point throughout.",
    ),
    "meta_secalign_70b": ModelSpec(
        name="meta_secalign_70b",
        source="facebook/Meta-SecAlign-70B",
        role="70B spot-check only (RQ4 scope item) — not used for any training/fine-tuning in this project. "
        "Base model is Llama-3.3-70B-Instruct, NOT Llama-3.1-70B-Instruct (verified 2026-09-16 against "
        "the HF model card and the paper's own text -- a different Llama generation from meta_secalign_8b's "
        "Llama-3.1-8B-Instruct base; any 8B-vs-70B comparison conflates model scale with model generation, "
        "see .agents/record.md Decision #16).",
    ),
    "llama_3_1_8b_instruct": ModelSpec(
        name="llama_3_1_8b_instruct",
        source="meta-llama/Llama-3.1-8B-Instruct",
        role="Undefended baseline; base model for all new fine-tuning (Phase 1.5/2).",
    ),
    "seallm_v3_7b_chat": ModelSpec(
        name="seallm_v3_7b_chat",
        source="SeaLLMs/SeaLLMs-v3-7B-Chat",
        role="Multilingual baseline with no prompt-injection defense -- serves as the Vietnamese-competence "
        "control for RQ1 (introduced Decision #14; actually run as v2.5 with nuanced/confounded results, "
        "Decision #15; upgraded to v3 here, Decision #16 -- v2.5's own instruction-following weakness was "
        "flagged in Decision #15 as a likely confound of the control itself). v3 explicitly lists Vietnamese "
        "among its 12 supported languages and benchmarks strictly better than v2.5 across the board "
        "(instruction-following especially), making it a stronger control. Not yet run as of this upgrade -- "
        "the v2.5 results in Decision #15 remain a valid, separate historical data point, not superseded.",
    ),
    "jason_v1_final_checkpoint": ModelSpec(
        name="jason_v1_final_checkpoint",
        source="Jason-42195/VNU-SecAlign",
        role="V1's published LoRA adapter (subfolder 'checkpoints/final_checkpoint', PEFT r=16/alpha=32/"
        "dropout=0.05/q_proj+v_proj only, base llama_3_1_8b_instruct). Trained on dpo_dataset_clean.json "
        "(PKU-SafeRLHF + VN hate-speech refusal-template data, NOT prompt-injection data) -- go/no-go "
        "test candidate before committing to Phase 1.5/2 retraining, see proposal.md section 3.",
    ),
    "phase1_5_vi_adapter": ModelSpec(
        name="phase1_5_vi_adapter",
        source=str(CHECKPOINTS_ROOT / "phase1_5_vi"),
        role="This project's Vietnamese-augmented LoRA adapter (conditional — only exists if Phase 1.5 ran).",
    ),
    "phase2_final_adapter": ModelSpec(
        name="phase2_final_adapter",
        source=str(CHECKPOINTS_ROOT / "phase2_final"),
        role="This project's final LoRA adapter after Phase 2 (10 novel attack vectors) fine-tuning.",
    ),
}


def get(name: str) -> ModelSpec:
    try:
        return REGISTRY[name]
    except KeyError as e:
        raise KeyError(f"Unknown model {name!r}, expected one of {sorted(REGISTRY)}") from e
