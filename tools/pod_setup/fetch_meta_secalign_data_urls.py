#!/usr/bin/env python3
"""Fetch the 14 small raw-file data dependencies that external/meta_secalign/setup.py downloads
at its start (its `data_urls` list, copied verbatim below), PLUS replicate the one CPU-only
derived-data step that meta_eval_runner.py's run_cyberseceval2_pi_subtask() needs
(data/CySE_prompt_injections.json) -- WITHOUT running the rest of that script.

Why not just run `python3 setup.py` (2026-09-22): past the data_urls loop, the original script
unconditionally calls `snapshot_download()` on 5 full model repos (facebook/Meta-SecAlign-8B/70B
adapters + the ~16GB meta-llama/Llama-3.1-8B-Instruct + the ~140GB meta-llama/Llama-3.3-70B-
Instruct + meta-llama/Meta-Llama-3-8B-Instruct base models), then later asserts
`torch.cuda.device_count() > 0` and runs a real vLLM generation pass to build SEP reference
outputs. None of that belongs in tools/pod_setup/build_env_cache.sh, whose only job is caching
small data files + Python packages -- this project downloads/evaluates models through its own
`src/vi_secalign/models/registry.py` / `meta_bridge.py` path, not by re-running Meta's setup.py
wholesale. This script exists so we get the useful CPU-only parts without the rest.

Two derived-data steps in setup.py were found (2026-09-22, while planning what to run after
pod_init.sh) that this script does NOT replicate, and must be handled separately:
  - TaskTracker_dataset_test.json (setup.py:208-563): pulls SQuAD/hotpot/CodeAlpaca-20k/
    BeaverTails/do-not-answer via extra downloads. NOT needed -- TaskTracker isn't in this
    project's benchmark scope (config.BENCHMARKS: AlpacaFarm/SEP/CyberSecEval2-PI/InjecAgent/
    MMLU/Vi-InjectEval/held-out), so skipped entirely, not just deferred.
  - SEP_dataset_test.json + its reference file (setup.py:565-604): genuinely needs a GPU --
    it loads meta-llama/Meta-Llama-3-8B-Instruct via vLLM to generate reference answers. This
    cannot be precomputed on a laptop; it must run once on the pod itself before T1-T3's
    run_sep() will work. Not handled by this script.

Does NOT modify external/meta_secalign/ (per .agents/CLAUDE.md: don't edit that submodule
directly) -- writes to external/meta_secalign/data/, the same destination setup.py itself uses,
so anything downstream that expects files there still finds them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Verbatim copy of external/meta_secalign/setup.py's `data_urls` list (lines ~29-42) -- keep in
# sync if that list ever changes upstream (check via `git log` on the submodule).
DATA_URLS = [
    "https://huggingface.co/datasets/hamishivi/alpaca-farm-davinci-003-2048-token/resolve/main/davinci_003_outputs.json",
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/refs/heads/main/alpaca_data.json",
    "https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/refs/heads/main/client_configs/openai_configs_example.yaml",
    "https://raw.githubusercontent.com/egozverev/Should-It-Be-Executed-Or-Processed/refs/heads/main/datasets/SEP_dataset.json",
    "https://raw.githubusercontent.com/microsoft/TaskTracker/refs/heads/main/task_tracker/dataset_creation/datasets/saved_injections_test.txt",
    "https://raw.githubusercontent.com/microsoft/TaskTracker/refs/heads/main/task_tracker/dataset_creation/datasets/jailbreaks_pair_attack.txt",
    "https://raw.githubusercontent.com/microsoft/TaskTracker/refs/heads/main/task_tracker/dataset_creation/datasets/llm_adaptive_attacks.txt",
    "https://raw.githubusercontent.com/meta-llama/PurpleLlama/refs/heads/main/CybersecurityBenchmarks/datasets/prompt_injection/prompt_injection.json",
    "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/refs/heads/main/data/attacker_simulated_responses.json",
    "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/refs/heads/main/data/test_cases_dh_base.json",
    "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/refs/heads/main/data/test_cases_dh_enhanced.json",
    "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/refs/heads/main/data/test_cases_ds_base.json",
    "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/refs/heads/main/data/test_cases_ds_enhanced.json",
    "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/refs/heads/main/data/tools.json",
]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "external" / "meta_secalign" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for url in DATA_URLS:
        dest = data_dir / url.split("/")[-1]
        if dest.exists():
            print(f"{dest} already exists.")
            continue
        print(f"wget -P {data_dir} {url}")
        subprocess.run(["wget", "-q", "-P", str(data_dir), url], check=True)

    print(f"Done -- {len(DATA_URLS)} data_urls checked/fetched into {data_dir}")

    _build_cyberseceval2_pi_subtask(data_dir)


def _build_cyberseceval2_pi_subtask(data_dir: Path) -> None:
    """Verbatim port of setup.py:191-203 -- filters prompt_injection.json down to the
    'indirect' injection_type subset that run_cyberseceval2_pi_subtask() (meta_eval_runner.py)
    reads. Pure CPU/JSON reshaping, no model/GPU involved.
    """
    out_path = data_dir / "CySE_prompt_injections.json"
    if out_path.exists():
        print(f"{out_path} already exists.")
        return
    src_path = data_dir / "prompt_injection.json"
    with open(src_path) as f:
        data = json.load(f)
    data_sft_format = [
        {
            "instruction": d["test_case_prompt"],
            "input": d["user_input"],
            "judge_question": d["judge_question"],
        }
        for d in data
        if d["injection_type"] == "indirect"
    ]
    with open(out_path, "w") as f:
        json.dump(data_sft_format, f, indent=2)
    print(f"Built {out_path} ({len(data_sft_format)} indirect-injection samples)")


if __name__ == "__main__":
    sys.exit(main())
