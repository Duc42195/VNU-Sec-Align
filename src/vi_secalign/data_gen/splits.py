"""Phase 2 train/held-out split — the leakage guard required by RQ4's methodology.

`data/attack_vectors/splits/heldout_20/` must never be touched by any data-generation step after
its first write. This module enforces that at the code level (not just by convention): writing to
an existing held-out file raises instead of silently overwriting.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from vi_secalign.config import ATTACK_VECTORS, REPO_ROOT

SPLITS_ROOT = REPO_ROOT / "data" / "attack_vectors" / "splits"
TRAIN_DIR = SPLITS_ROOT / "train_80"
HELDOUT_DIR = SPLITS_ROOT / "heldout_20"

TRAIN_FRACTION = 0.8
SPLIT_SEED = 20260830  # fixed for reproducibility


class HeldoutAlreadyExistsError(RuntimeError):
    """Raised when a split step would overwrite an existing held-out file."""


def split_vector_pool(vector_id: str, pool: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split one attack vector's variant pool 80/20, writing to train_80/ and heldout_20/.

    Raises HeldoutAlreadyExistsError if the held-out file for this vector already exists — this
    is the leakage guard: heldout_20 is write-once per vector, never touched again after the
    first split (see project memory / README "Important methodology constraints").
    """
    if vector_id not in ATTACK_VECTORS:
        raise ValueError(f"Unknown attack vector {vector_id!r}, expected one of {sorted(ATTACK_VECTORS)}")

    heldout_path = HELDOUT_DIR / f"{vector_id}.jsonl"
    train_path = TRAIN_DIR / f"{vector_id}.jsonl"
    if heldout_path.exists():
        raise HeldoutAlreadyExistsError(
            f"{heldout_path} already exists — held-out data is write-once per vector and must "
            "never be regenerated or overwritten. If you genuinely need to rebuild the pool from "
            "scratch, delete the file manually first as an explicit, deliberate action."
        )

    shuffled = list(pool)
    random.Random(SPLIT_SEED).shuffle(shuffled)
    cut = int(len(shuffled) * TRAIN_FRACTION)
    train_split, heldout_split = shuffled[:cut], shuffled[cut:]

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    HELDOUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(train_path, train_split)
    _write_jsonl(heldout_path, heldout_split)
    return train_split, heldout_split


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
