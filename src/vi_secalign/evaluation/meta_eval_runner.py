"""Thin subprocess wrapper around external/meta_secalign's own evaluation scripts.

Reuses the official ASR-judging logic verbatim (test.py's after_inference_evaluation dispatcher:
SEP/CyberSecEval/AlpacaFarm branches) rather than reimplementing it, unlike the prior project's
notebooks (archive/legacy_notebooks/secalign_SECALIGN_Untitled1.ipynb), which rewrote lm_eval/vLLM
orchestration and ASR scoring from scratch via subprocess calls and a hand-rolled is_safe()
keyword matcher.

All calls run with cwd=EXTERNAL_ROOT because test.py/test_lm_eval.py/etc. use paths relative to
that directory (e.g. 'data/...', 'lm_eval_config').
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from vi_secalign.config import EXTERNAL_ROOT


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=EXTERNAL_ROOT, check=True, capture_output=True, text=True)


def run_alpacafarm(model_name_or_path: str, attack: str, defense: str = "none") -> subprocess.CompletedProcess:
    """AlpacaFarm ASR/utility. See external/meta_secalign/test.py + run_tests.py:17,19."""
    return _run(
        [
            "python",
            "test.py",
            "--model_name_or_path",
            model_name_or_path,
            "--attack",
            attack,
            "--defense",
            defense,
            "--test_data",
            "data/davinci_003_outputs.json",
        ]
    )


def run_sep(model_name_or_path: str, attack: str, defense: str = "none") -> subprocess.CompletedProcess:
    """SEP ASR/utility. See external/meta_secalign/test.py + run_tests.py:19."""
    return _run(
        [
            "python",
            "test.py",
            "--model_name_or_path",
            model_name_or_path,
            "--attack",
            attack,
            "--defense",
            defense,
            "--test_data",
            "data/SEP_dataset_test.json",
        ]
    )


def run_injecagent(model_name_or_path: str) -> subprocess.CompletedProcess:
    """See external/meta_secalign/test_injecagent.py."""
    return _run(["python", "test_injecagent.py", "--model_name_or_path", model_name_or_path])


def run_lm_eval(model_name_or_path: str) -> subprocess.CompletedProcess:
    """MMLU/IFEval/BBH/GPQA/MMLU-Pro via lm-evaluation-harness. See external/meta_secalign/test_lm_eval.py.

    Note: GSM8K is not among the 5 tasks defined here (test_lm_eval.py:17) — deliberately dropped
    from this project's benchmark list, see config.BENCHMARKS.
    """
    return _run(["python", "test_lm_eval.py", "--model_name_or_path", model_name_or_path])


def run_cyberseceval2_pi_subtask(model_name_or_path: str, defense: str = "none") -> subprocess.CompletedProcess:
    """CyberSecEval2 prompt-injection subtask ONLY (see config.BENCHMARKS docstring for scope caveat).

    external/meta_secalign/run_tests.py:18 has the equivalent test.py invocation commented out by
    default; we call test.py directly here instead of editing the vendored run_tests.py.
    Requires data/CySE_prompt_injections.json to already exist (produced by setup.py:37,191-194).
    """
    if not (EXTERNAL_ROOT / "data" / "CySE_prompt_injections.json").exists():
        raise FileNotFoundError(
            "data/CySE_prompt_injections.json not found under EXTERNAL_ROOT — run "
            "external/meta_secalign/setup.py first to download/process it."
        )
    return _run(
        [
            "python",
            "test.py",
            "--model_name_or_path",
            model_name_or_path,
            "--attack",
            "straightforward",
            "--defense",
            defense,
            "--test_data",
            "data/CySE_prompt_injections.json",
        ]
    )


def run_70b_spotcheck(vector_id: str, heldout_path: Path, n_samples: int = 100, model: str = "facebook/Meta-SecAlign-70B"):
    """Stratified spot-check of one attack vector's held-out set against the public 70B checkpoint.

    Interface-only stub for this pass (see plan: "70B spot-check" scope item) — not runnable
    without a rented multi-GPU instance or a hosted inference endpoint for Meta-SecAlign-70B.
    Intentionally samples n_samples (not the full held-out set) to keep the cost of this check
    bounded; see project memory for why a full 70B re-run of everything is out of scope.
    """
    raise NotImplementedError(
        f"70B spot-check for {vector_id} against {model} is a documented future action "
        f"(would sample {n_samples} rows from {heldout_path}), not implemented in this pass — "
        "see README 'Known limitations & deliberate scope decisions'."
    )
