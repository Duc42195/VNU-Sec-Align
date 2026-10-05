"""T9b setup step: merge facebook/Meta-SecAlign-8B's LoRA adapter into the Llama-3.1-8B-Instruct
base weights, producing a plain full-weight checkpoint that torchtune's FullModelHFCheckpointer
can load like any other base model.

Why merge instead of loading the adapter via torchtune's `checkpointer.adapter_checkpoint`:
verified directly against torchtune==0.6.0 source (recipes/lora_dpo_single_device.py) -- that
field is only honored when `resume_from_checkpoint=True`, which ALSO requires a full recipe-state
file (optimizer state, epoch counters) that Meta's published adapter does not ship (it's a plain
PEFT adapter_model.safetensors + adapter_config.json, nothing else). Setting resume_from_checkpoint
without that file fails; setting adapter_checkpoint without resume_from_checkpoint silently drops
the adapter weights (recipe passes `lora_weights_state_dict=None` in that branch) -- no error,
just trains a fresh random LoRA as if Meta's adapter was never specified. Merging sidesteps this
entirely: the merged weights become the "base model", and T9b trains a brand-new rank-64 LoRA on
top via the normal (well-exercised) single-device LoRA path, same as every other run so far.

Saved as a SINGLE safetensors shard (max_shard_size forced huge) specifically so the resulting
`checkpointer.checkpoint_files` in the torchtune yaml is always exactly ["model.safetensors"],
regardless of how transformers would have auto-sharded an 8B model -- one less thing that can
silently mismatch between a run and the next.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vi_secalign.hf_sync import upload_output  # noqa: E402

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER = "facebook/Meta-SecAlign-8B"


def merge(output_dir: Path, device_map: str) -> None:
    print(f"[merge] Loading base model {BASE_MODEL} (bf16, device_map={device_map})...")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device_map)

    print(f"[merge] Loading + applying adapter {ADAPTER}...")
    merged = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[merge] Saving merged full-weight model to {output_dir} (single shard)...")
    merged.save_pretrained(output_dir, safe_serialization=True, max_shard_size="900GB")
    AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(output_dir)

    shards = sorted(output_dir.glob("*.safetensors"))
    assert len(shards) == 1, f"expected exactly 1 safetensors shard, got {[s.name for s in shards]}"
    print(f"[merge] Wrote {shards[0].name} ({shards[0].stat().st_size / 1e9:.1f} GB)")

    print("[merge] Fetching original/tokenizer.model (torchtune's llama3_tokenizer needs this exact file)...")
    tok_path = hf_hub_download(repo_id=BASE_MODEL, filename="original/tokenizer.model")
    original_dir = output_dir / "original"
    original_dir.mkdir(exist_ok=True)
    shutil.copy(tok_path, original_dir / "tokenizer.model")
    print(f"[merge] Done. torchtune cache_dir = {output_dir}")

    dest = upload_output(output_dir, dest_subdir="merge_meta_adapter")
    if dest:
        print(f"[merge] Uploaded to HF: {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints/meta_secalign_8b_merged"))
    parser.add_argument(
        "--device_map",
        default="cpu",
        help="'cpu' (default -- safe on a 16GB T4, base+adapter in bf16 is ~16GB, no VRAM headroom "
        "left for merge buffers) or 'auto' (use GPU, fine on a >=24GB card like the T9 pod).",
    )
    args = parser.parse_args()
    merge(args.output_dir, args.device_map)
