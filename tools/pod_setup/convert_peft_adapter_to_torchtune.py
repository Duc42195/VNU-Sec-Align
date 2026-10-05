"""T9b setup: convert facebook/Meta-SecAlign-8B's PEFT-format LoRA adapter into torchtune's native
LoRA key naming + head layout, so it can be loaded directly into a torchtune lora_llama3_1_8b model
and CONTINUED training (not merged into frozen base weights -- see Decision #36/#37 in
.agents/record.md for why torchtune's own checkpointer.adapter_checkpoint can't be used for this:
3 independent gates block it, and even past those, it loads with zero key/permute conversion).

torchtune ships the forward direction only (torchtune-native -> PEFT, for export:
torchtune.models.convert_weights.tune_to_peft_adapter_weights) -- it has no reverse (its own code
comment says so: "we do NOT have a fn convert_weights.peft_to_tune"). This writes that reverse by
inverting the exact same key-mapping + head-permutation logic, then PROVES correctness by
round-tripping the result back through torchtune's own (trusted) forward function and asserting it
reproduces the original PEFT state dict exactly, before writing any output file. A wrong-direction
permute would otherwise run fine and silently corrupt attention projections -- no crash, no NaN,
just a quietly broken model -- so the round-trip is the only check that can't be fooled by a
plausible-looking loss curve. Needs HF_TOKEN (facebook/Meta-SecAlign-8B is gated); no GPU needed,
pure tensor reshuffling on CPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vi_secalign.hf_sync import upload_output  # noqa: E402

ADAPTER_REPO = "facebook/Meta-SecAlign-8B"

# Llama-3.1-8B architecture constants -- this script only ever targets this one model, no need to
# read them from a config file.
DIM = 4096
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = DIM // NUM_HEADS  # 128


def _build_inverse_mapping() -> dict[str, str]:
    """PEFT-key-template -> tune-key-template -- exact inverse of the `full_mapping` built inside
    torchtune's own tune_to_peft_adapter_weights (torchtune/models/convert_weights.py)."""
    from torchtune.models.convert_weights import _FROM_HF, _TO_PEFT_KEYS

    full_mapping: dict[str, str] = {}
    for peft_key, peft_val in _TO_PEFT_KEYS.items():
        for hf_key, hf_val in _FROM_HF.items():
            if hf_val is None:
                continue
            if peft_key == "magnitude":
                adapter_key = hf_val.replace(".weight", f".{peft_key}")
                adapter_val = hf_key.replace(".weight", f".{peft_val}")
            else:
                adapter_key = hf_val.replace(".weight", f".{peft_key}.weight")
                adapter_val = hf_key.replace(".weight", f".{peft_val}.weight")
            full_mapping[adapter_key] = adapter_val
    return {v: k for k, v in full_mapping.items()}  # peft-template -> tune-template


def _unpermute_lora_b(t: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Inverse of torchtune's `_permute_lora_matrix` (inside tune_to_peft_adapter_weights). Uses
    the opposite .view() axis order from that function -- the same axis-swap trick torchtune uses
    between hf_to_tune's and tune_to_hf's base-weight `_permute` (proven self-inverse pair in
    torchtune's own code), substituting `rank` for `dim`."""
    rank = t.shape[-1]
    return t.view(n_heads, 2, HEAD_DIM // 2, rank).transpose(1, 2).reshape(HEAD_DIM * n_heads, rank)


def peft_to_tune_adapter_weights(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    from torchtune.models.convert_weights import get_mapped_key

    inverse_mapping = _build_inverse_mapping()
    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        stripped = key.removeprefix("base_model.model.")
        new_key = get_mapped_key(stripped, inverse_mapping)
        if "q_proj" in stripped and "lora_B" in stripped:
            value = _unpermute_lora_b(value, NUM_HEADS)
        elif "k_proj" in stripped and "lora_B" in stripped:
            value = _unpermute_lora_b(value, NUM_KV_HEADS)
        converted[new_key] = value
    return converted


def _round_trip_check(original: dict[str, torch.Tensor], converted: dict[str, torch.Tensor]) -> None:
    """The real correctness gate. Converts back via torchtune's OWN trusted forward function and
    demands an exact match -- not a review of the reasoning above, a mechanical proof of it."""
    from torchtune.models.convert_weights import tune_to_peft_adapter_weights

    roundtripped = tune_to_peft_adapter_weights(
        converted, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, dim=DIM, head_dim=HEAD_DIM
    )
    assert set(roundtripped) == set(original), (
        f"round-trip KEY mismatch -- missing={set(original) - set(roundtripped)} "
        f"extra={set(roundtripped) - set(original)}"
    )
    for key, orig_val in original.items():
        assert torch.equal(orig_val, roundtripped[key]), (
            f"round-trip VALUE mismatch at {key!r} -- permute or key mapping is wrong, DO NOT use this output"
        )
    print(f"[convert] Round-trip OK -- {len(original)} tensors match exactly.")


def main(output_path: Path) -> None:
    print(f"[convert] Downloading {ADAPTER_REPO} adapter_model.safetensors (needs HF_TOKEN, gated model)...")
    local_path = hf_hub_download(repo_id=ADAPTER_REPO, filename="adapter_model.safetensors")
    original = load_file(local_path)
    print(f"[convert] Loaded {len(original)} tensors from PEFT adapter.")

    converted = peft_to_tune_adapter_weights(original)
    _round_trip_check(original, converted)  # raises if anything is off -- never silently proceeds

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output_path)
    print(f"[convert] Wrote torchtune-native adapter state dict to {output_path}")

    dest = upload_output(output_path, dest_subdir="convert_peft_adapter_to_torchtune")
    if dest:
        print(f"[convert] Uploaded to HF: {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_path", type=Path, default=Path("checkpoints/meta_secalign_8b_adapter_torchtune.pt")
    )
    args = parser.parse_args()
    main(args.output_path)
