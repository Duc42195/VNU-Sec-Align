"""Phase 2 (RQ4) / T15: builds the per-vector clean-instruction pool, then writes the 80/20
train/held-out split via splits.split_vector_pool().

Design note: the 10 attack transforms in attack10_gen.py (ATTACK_TRANSFORMS) are applied LIVE at
eval time (see evaluation/attack_vectors_eval.py::_build_prompts, which calls
`injection_method(deepcopy(d))` on each loaded row) -- not baked into the stored pool. So this
script's job is only to gather >=1000 CLEAN {'instruction','input'} pairs per vector (the "variant
pool" plan.csv T15 asks for is variety of BASE instructions to attack, one draw per vector), not to
pre-apply any transform or generate preference pairs -- that happens later, per vector, only for
vectors T17 confirms the model is actually vulnerable to (see attack10_gen.generate_attack_preference_pairs,
used by T18, not by this script).

Source corpus: external/meta_secalign/data/alpaca_data.json (Stanford Alpaca, already fetched by
fetch_meta_secalign_data_urls.py -- no new download/license question, and it's the same corpus
external/meta_secalign's own generate_preference_dataset reads). Filtered to non-empty 'input'
(an attack needs an input field to inject into; 20,679/52,002 rows qualify -- comfortably enough
for 10 vectors x >=1000, sampled independently per vector so pools may overlap across vectors,
which is fine: the leakage guard in splits.py is about train_80 vs heldout_20 WITHIN one vector,
not about reusing a base instruction across different vectors).

Must NEVER read data/attack_vectors/_legacy_v2_reference/*.json (see attack10_gen.py docstring).

Not run in this pass (no GPU needed for this step, but not executed yet pending T14 completion --
GĐ5 has not started, see plan.csv). Pure CPU/JSON, could actually run on the laptop once T14 is
done -- listed here so it exists ahead of time, per the "as much code ready as possible" request.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from vi_secalign.config import ATTACK_VECTORS, EXTERNAL_ROOT
from vi_secalign.data_gen import splits

DEFAULT_SOURCE = EXTERNAL_ROOT / "data" / "alpaca_data.json"
DEFAULT_N_PER_VECTOR = 1000


def load_clean_pool(source: Path = DEFAULT_SOURCE) -> list[dict]:
    with open(source, encoding="utf-8") as f:
        data = json.load(f)
    return [{"instruction": d["instruction"], "input": d["input"]} for d in data if d.get("input", "").strip()]


def build_vector_pool(vector_id: str, clean_pool: list[dict], n: int) -> list[dict]:
    """Independently-seeded (per vector_id, matching attack10_gen.generate_attack_preference_pairs's
    own `random.Random(vector_id)` convention) sample without replacement within this vector."""
    if vector_id not in ATTACK_VECTORS:
        raise ValueError(f"Unknown attack vector {vector_id!r}, expected one of {sorted(ATTACK_VECTORS)}")
    if n > len(clean_pool):
        raise ValueError(f"Requested n={n} > {len(clean_pool)} available clean samples with non-empty input.")
    rng = random.Random(vector_id)
    return rng.sample(clean_pool, n)


def build_all_pools(
    vector_ids: list[str] | None = None,
    n_per_vector: int = DEFAULT_N_PER_VECTOR,
    source: Path = DEFAULT_SOURCE,
) -> dict[str, tuple[list[dict], list[dict]]]:
    """Builds + splits (80/20) the pool for each requested vector. Returns {vector_id: (train, heldout)}.

    Skips (does not re-sample or re-split) any vector whose heldout_20 file already exists --
    matches splits.split_vector_pool's own write-once guard, but checked here first so a re-run
    across multiple vectors doesn't abort partway through on the first already-done vector.
    """
    vector_ids = vector_ids or sorted(ATTACK_VECTORS)
    clean_pool = load_clean_pool(source)
    print(f"Loaded {len(clean_pool)} clean (non-empty-input) candidates from {source}")

    results = {}
    for vector_id in vector_ids:
        heldout_path = splits.HELDOUT_DIR / f"{vector_id}.jsonl"
        if heldout_path.exists():
            print(f"{vector_id}: {heldout_path} already exists, skipping.")
            continue
        pool = build_vector_pool(vector_id, clean_pool, n_per_vector)
        train, heldout = splits.split_vector_pool(vector_id, pool)
        print(f"{vector_id}: {len(train)} train_80 / {len(heldout)} heldout_20")
        results[vector_id] = (train, heldout)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vector_ids",
        nargs="*",
        default=None,
        choices=sorted(ATTACK_VECTORS),
        help="Subset of vectors to build (default: all 10).",
    )
    parser.add_argument("--n_per_vector", type=int, default=DEFAULT_N_PER_VECTOR)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()

    results = build_all_pools(vector_ids=args.vector_ids, n_per_vector=args.n_per_vector, source=args.source)
    print(f"Done -- built {len(results)} vector pool(s).")


if __name__ == "__main__":
    main()
