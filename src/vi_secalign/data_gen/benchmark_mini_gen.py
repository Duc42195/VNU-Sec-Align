"""GĐ2 pilot expansion: cheap small-N samples for 3 more of the 7 official benchmarks
(`config.py::BENCHMARKS`), same "small % per bộ, đủ nhanh" scoping as the SEP/Alpaca pair already
used in notebooks/phase0_go_no_go_test.ipynb -- NOT the full-scale GĐ6 (T21) benchmark run.

Covers, using the exact source files `external/meta_secalign/setup.py::data_urls` itself points at
(so this is the same source data the official pipeline would use, just a small subsample fetched
without running the rest of that heavy setup script):

- `cyberseceval2_pi_subtask`: PurpleLlama's prompt_injection.json (251 samples). Only ~94/251 have
  a single-quoted secret word in `test_case_prompt`/`judge_question` (e.g. "The secret key is
  'Piano'.") that a substring-witness check can grade without an LLM judge -- the rest use a
  judge_question shape this project has no cheap judge for yet, so this pilot restricts to the
  quotable subset and says so explicitly in `info.judge_method`. Structurally this benchmark has no
  "clean" (non-injected) counterpart per sample the way SEP does, so there is no
  clean-witness-leak sanity field for this one -- see per-function docstring.
- `mmlu`: cais/mmlu ("all" config, test split, 14,042 rows) -- multiple-choice capability/utility
  check, unrelated to prompt injection. Sampled from several random offsets spread across the full
  range (not just the first N rows) so the pilot isn't biased toward alphabetically-early subjects.
- `alpacafarm`: the actual AlpacaFarm reference file (davinci-003 outputs), not generic Stanford
  Alpaca -- more faithful to what "alpacafarm" in BENCHMARKS actually refers to. Still only a
  refusal-rate/empty-rate proxy (see notebooks/phase0_go_no_go_test.ipynb) -- real AlpacaFarm
  win-rate needs a GPT-4-class judge, out of scope for this pilot.

Fetches via plain HTTP (urllib) + the HF datasets-server REST API -- no `datasets` library, no GPU,
consistent with vi_injecteval_gen.py.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.parse
import urllib.request
from pathlib import Path

from vi_secalign.config import REPO_ROOT

DATASETS_SERVER_URL = "https://datasets-server.huggingface.co/rows"
DATASETS_SERVER_SIZE_URL = "https://datasets-server.huggingface.co/size"

CSE2_PI_URL = (
    "https://raw.githubusercontent.com/meta-llama/PurpleLlama/refs/heads/main/"
    "CybersecurityBenchmarks/datasets/prompt_injection/prompt_injection.json"
)
ALPACAFARM_URL = (
    "https://huggingface.co/datasets/hamishivi/alpaca-farm-davinci-003-2048-token/"
    "resolve/main/davinci_003_outputs.json"
)

BENCHMARKS_DIR = REPO_ROOT / "data" / "benchmarks"

_QUOTED_RE = re.compile(r"'([^']{2,30})'")


def _download_json(url: str, cache_path: Path) -> object:
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp:
        data = json.load(resp)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return data


def build_cyberseceval2_pi_mini(n_samples: int = 30, seed: int = 42) -> list[dict]:
    """CyberSecEval2 PI subtask mini-sample.

    Schema matches build_sep_messages()'s 2-role convention: `system_prompt_clean` = the trusted
    instruction holding the secret, `prompt_instructed` = the attacker's injected user_input. No
    `prompt_clean`/clean-witness-leak field -- this dataset has no non-injected counterpart per
    sample (unlike SEP), so that sanity check isn't constructible here.
    """
    raw = _download_json(CSE2_PI_URL, BENCHMARKS_DIR / "cyberseceval2" / "_raw_prompt_injection.json")
    quotable = []
    for item in raw:
        m = _QUOTED_RE.findall(item.get("test_case_prompt", "") + " " + item.get("judge_question", ""))
        if m:
            quotable.append((item, m[0]))

    rng = random.Random(seed)
    n_samples = min(n_samples, len(quotable))
    sample = rng.sample(quotable, n_samples)

    mini = []
    for item, witness in sample:
        mini.append(
            {
                "system_prompt_clean": item["test_case_prompt"],
                "prompt_instructed": item["user_input"],
                "witness": witness,
                "info": {
                    "source": "PurpleLlama/CybersecurityBenchmarks (prompt_injection.json)",
                    "prompt_id": item.get("prompt_id"),
                    "injection_type": item.get("injection_type"),
                    "injection_variant": item.get("injection_variant"),
                    "judge_method": (
                        "substring match on a single-quoted secret word regex-extracted from "
                        "test_case_prompt/judge_question -- only covers "
                        f"{len(quotable)}/{len(raw)} samples whose judge_question is this simple; "
                        "the rest need an LLM judge this pilot doesn't have, so they're excluded."
                    ),
                    "pilot": True,
                },
            }
        )
    return mini


def build_mmlu_mini(n_samples: int = 60, seed: int = 42, n_offsets: int = 12, total_rows: int = 14042) -> list[dict]:
    """MMLU mini-sample, spread across random offsets so subjects aren't alphabetically clustered."""
    rng = random.Random(seed)
    per_offset = max(1, (n_samples // n_offsets) + 1)
    offsets = sorted(rng.sample(range(0, total_rows - per_offset), n_offsets))

    collected = []
    for offset in offsets:
        params = {
            "dataset": "cais/mmlu", "config": "all", "split": "test",
            "offset": str(offset), "length": str(per_offset),
        }
        url = f"{DATASETS_SERVER_URL}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url) as resp:
            payload = json.load(resp)
        collected.extend(r["row"] for r in payload.get("rows", []))

    sample = rng.sample(collected, min(n_samples, len(collected)))
    letters = ["A", "B", "C", "D"]
    mini = []
    for row in sample:
        mini.append(
            {
                "question": row["question"],
                "subject": row["subject"],
                "choices": row["choices"],
                "answer_letter": letters[row["answer"]],
                "info": {"source": "cais/mmlu (config=all, split=test)", "pilot": True},
            }
        )
    return mini


def build_alpacafarm_mini(n_samples: int = 30, seed: int = 42) -> list[dict]:
    """AlpacaFarm reference-set mini-sample (utility proxy only -- see module docstring)."""
    raw = _download_json(ALPACAFARM_URL, BENCHMARKS_DIR / "alpacafarm" / "_raw_davinci_003_outputs.json")
    instruction_only = [d for d in raw if not d.get("input")]
    rng = random.Random(seed)
    sample = rng.sample(instruction_only, min(n_samples, len(instruction_only)))
    mini = []
    for d in sample:
        mini.append(
            {
                "instruction": d["instruction"],
                "reference_output": d["output"],
                "info": {"source": "hamishivi/alpaca-farm-davinci-003-2048-token", "pilot": True},
            }
        )
    return mini


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_cyberseceval2", type=int, default=30)
    parser.add_argument("--n_mmlu", type=int, default=60)
    parser.add_argument("--n_alpacafarm", type=int, default=30)
    args = parser.parse_args()

    outputs = {
        BENCHMARKS_DIR / "cyberseceval2" / "pilot_v0.json": build_cyberseceval2_pi_mini(args.n_cyberseceval2, args.seed),
        BENCHMARKS_DIR / "mmlu" / "pilot_v0.json": build_mmlu_mini(args.n_mmlu, args.seed),
        BENCHMARKS_DIR / "alpacafarm" / "pilot_v0.json": build_alpacafarm_mini(args.n_alpacafarm, args.seed),
    }
    for path, data in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Generated {len(data)} samples -> {path}")


if __name__ == "__main__":
    main()
