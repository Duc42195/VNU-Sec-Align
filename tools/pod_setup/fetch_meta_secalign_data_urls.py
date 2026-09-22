#!/usr/bin/env python3
"""Fetch just the 14 small raw-file data dependencies that external/meta_secalign/setup.py
downloads at its start (its `data_urls` list, copied verbatim below) -- WITHOUT running the rest
of that script.

Why not just run `python3 setup.py` (2026-09-22): past that data_urls loop, the original script
unconditionally calls `snapshot_download()` on 5 full model repos (facebook/Meta-SecAlign-8B/70B
adapters + the ~16GB meta-llama/Llama-3.1-8B-Instruct + the ~140GB meta-llama/Llama-3.3-70B-
Instruct + meta-llama/Meta-Llama-3-8B-Instruct base models), then later asserts
`torch.cuda.device_count() > 0` and runs a real vLLM generation pass to build SEP reference
outputs. None of that belongs in tools/pod_setup/build_env_cache.sh, whose only job is caching
small data files + Python packages -- this project downloads/evaluates models through its own
`src/vi_secalign/models/registry.py` / `meta_bridge.py` path, not by re-running Meta's setup.py
wholesale. This script exists so we get the useful 14-URL part without the rest.

Does NOT modify external/meta_secalign/ (per .agents/CLAUDE.md: don't edit that submodule
directly) -- writes to external/meta_secalign/data/, the same destination setup.py itself uses,
so anything downstream that expects files there still finds them.
"""

from __future__ import annotations

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


if __name__ == "__main__":
    sys.exit(main())
