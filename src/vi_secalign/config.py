"""Project-wide constants for VNU-SecAlign v2.

Single source of truth for paths, the 10-vector attack taxonomy, and the benchmark list.
Values here are either copied (not imported) from external/meta_secalign/config.py — to avoid
pulling vllm/openai/google-genai into every module that just needs a constant — or newly defined
for this project. See docs/research_notes/attack_vector_09_10_addendum.md for the taxonomy
rationale and the plan file for full citations.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = REPO_ROOT / "external" / "meta_secalign"

# Copied from external/meta_secalign/config.py:7-8 (not imported — see module docstring).
# "recommended for DPOTrainer when using self-generated labels to cover >99.8% samples" (original comment).
MAX_PROMPT_LENGTH = 384
MAX_LENGTH = 2048

# Meta-SecAlign-8B published/effective anchor hyperparameters (verified against
# external/meta_secalign/helpers/llama3.1_8B_lora.yaml and secalign_plus_plus.py).
# NOTE: the yaml's own default lr is 1e-4 (llama3.1_8B_lora.yaml:68); 1.6e-4 is the value injected
# at runtime via secalign_plus_plus.py's --lr CLI default (line 12) — cite the effective trained
# value, not the yaml literal.
ANCHOR_HYPERPARAMS = {
    "learning_rate": 1.6e-4,
    "lora_r": 64,
    "lora_alpha": 8,
    "lora_dropout": 0.1,
    "epochs": 3,
    "rpo_alpha": 0.5,
    "label_smoothing": 0.1,
    # yaml:78,80 says batch_size=2/grad_accum=16 (effective 32), but DPOTrainer's per-peft-adapter
    # reference-log-prob pass (compute_ref_log_probs, disables the LoRA adapter and does a SECOND
    # forward pass on the same base model) adds on top of the policy forward+backward already in
    # flight -- confirmed by real OOM on the pod even at batch_size=2 (23.50/23.56 GiB used, failed
    # inside that ref pass specifically, not the policy pass). Halved again to batch_size=1 to keep
    # a real margin; grad_accum doubled to 32 to hold the same effective batch of 32.
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 32,
}

# q_proj/v_proj are literal from helpers/llama3.1_8B_lora.yaml:25. gate_proj/up_proj/down_proj are
# this project's equivalent-effect interpretation of apply_lora_to_mlp=True (yaml:26) — NOT a
# literal string from Meta's config. Cite accordingly (see dpo_config.py docstring).
LORA_TARGET_MODULES = ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]

# Benchmarks used for evaluation. GSM8K deliberately dropped (absent from both SecAlign papers
# and from external/meta_secalign entirely — 0 grep hits). CyberSecEval2 is scoped explicitly to
# the prompt-injection subtask, which is all external/meta_secalign wires up (data/CySE_prompt_injections.json,
# commented out by default in run_tests.py:18) — never report it as the full CyberSecEval2 suite.
BENCHMARKS = [
    "alpacafarm",
    "sep",
    "cyberseceval2_pi_subtask",
    "injecagent",
    "mmlu",
    "vi_injecteval",
    "attack10_heldout",
]

# The 4 encoding/obfuscation-class vectors share a measurement-validity problem: raw ASR
# conflates two independent phenomena — (1) whether the model correctly decodes the obfuscated
# payload at all (a capability effect, larger models generally decode more accurately — see e.g.
# Yuan et al. "GPT-4 Is Too Smart To Be Safe: Stealthy Chat with LLMs via Cipher", ICLR 2024, for
# the documented real-world version of this effect), and (2) whether, GIVEN correct decoding, the
# model still respects the trust boundary (the actual SecAlign-relevant question). A model that
# fails to decode looks "safe" for the wrong reason. Reported ASR for these 4 vectors must
# therefore be split into decode_accuracy_rate and decode-conditional ASR, not just a single raw
# ASR number — see evaluation/attack_vectors_eval.py::probe_decode_accuracy /
# evaluate_vector_with_decode_check, and docs/research_notes/attack_vector_09_10_addendum.md.
ENCODING_CLASS_VECTORS = ["ATTACK-02", "ATTACK-06", "ATTACK-09", "ATTACK-10"]

# The 10-vector taxonomy for RQ4. ATTACK-01..08 come from
# docs/research_notes/secalign_mechanism_analysis_and_attack_taxonomy.txt (section 3.2, the only
# 8 that document has a written rationale for). ATTACK-09 replaces the original dataset's "Base64
# Obfuscation" (which duplicated the base64 branch already inside ATTACK-02) with a new,
# orthogonal vector. ATTACK-10 keeps the original dataset's Leetspeak Obfuscation, with a
# rationale added post hoc (it had none in the analysis doc either).
# See docs/research_notes/attack_vector_09_10_addendum.md for the full write-up of 09/10.
ATTACK_VECTORS = {
    "ATTACK-01": {
        "name": "Multi-Turn Gradual Escalation",
        "gap": "GAP-01: no multi-turn injection coverage",
        "source": "docs/research_notes/secalign_mechanism_analysis_and_attack_taxonomy.txt",
    },
    "ATTACK-02": {
        "name": "Encoding Obfuscation Chain",
        "gap": "GAP-02: no encoding-chain attacks (base64/rot13/hex/mixed)",
        "source": "docs/research_notes/secalign_mechanism_analysis_and_attack_taxonomy.txt",
    },
    "ATTACK-03": {
        "name": "Cross-Lingual Code-Switching",
        "gap": "GAP-03: no cross-lingual injection coverage",
        "source": "docs/research_notes/secalign_mechanism_analysis_and_attack_taxonomy.txt",
    },
    "ATTACK-04": {
        "name": "Semantic Injection (No Keyword Match)",
        "gap": "GAP-04: no non-keyword semantic injection coverage",
        "source": "docs/research_notes/secalign_mechanism_analysis_and_attack_taxonomy.txt",
    },
    "ATTACK-05": {
        "name": "Context Overflow / Attention Dilution",
        "gap": "GAP-07: no context-overflow/attention-dilution coverage",
        "source": "docs/research_notes/secalign_mechanism_analysis_and_attack_taxonomy.txt",
    },
    "ATTACK-06": {
        "name": "Unicode Homoglyph Role Confusion",
        "gap": "GAP-06: no homoglyph role-confusion coverage",
        "source": "docs/research_notes/secalign_mechanism_analysis_and_attack_taxonomy.txt",
    },
    "ATTACK-07": {
        "name": "Payload Splitting Across Roles",
        "gap": "GAP-05: no payload-splitting-across-messages coverage",
        "source": "docs/research_notes/secalign_mechanism_analysis_and_attack_taxonomy.txt",
    },
    "ATTACK-08": {
        "name": "Temporal / Conditional Logic Injection",
        "gap": "Exploits model reasoning capabilities against itself (not tied to a numbered GAP in the source doc)",
        "source": "docs/research_notes/secalign_mechanism_analysis_and_attack_taxonomy.txt",
    },
    "ATTACK-09": {
        "name": "Structured Data-Field Injection",
        "gap": (
            "No existing vector (01-08, 10) or public benchmark tests structured/agentic-style "
            "payloads (a JSON key or CSV cell simulating tool-output/API-response content) as "
            "opposed to free-text manipulation of a single `input` field; also directly probes "
            "whether recursive_filter/the input-role boundary hold against structured content."
        ),
        "source": "docs/research_notes/attack_vector_09_10_addendum.md",
        "replaces": "original dataset's ATTACK-09 'Base64 Obfuscation', which duplicated ATTACK-02's base64 branch",
    },
    "ATTACK-10": {
        "name": "Leetspeak Obfuscation",
        "gap": (
            "Other ATTACK-02 encoding variants (base64/rot13/hex) produce non-alphanumeric "
            "strings that heuristics can flag as 'looks suspicious'; leetspeak substitutes "
            "visually-similar characters (a->4, e->3, i->1, o->0, ...) while remaining valid "
            "letters/digits, evading such heuristics while staying readable to the model."
        ),
        "source": "docs/research_notes/attack_vector_09_10_addendum.md",
    },
}
