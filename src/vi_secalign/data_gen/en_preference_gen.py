"""Phase 1 (RQ1 baseline data) / T9 joint-training input: English preference-pair generation.

2026-09-29: REWRITTEN. Originally a thin pass-through to external/meta_secalign's own
generate_preference_dataset (utils.py:49-127), reused verbatim "not reimplemented" per the
project's replication-fidelity goal. Never run in that form -- when actually preparing to run it
for real N (this session), two real gaps surfaced that the pass-through form cannot fix without
touching the submodule (against project convention -- external/meta_secalign is a read-only
reference, see .agents/CLAUDE.md and training/chat_template.py's own precedent for patching at
this layer instead):

1. No sample cap at all. Meta's function iterates the ENTIRE yahma/alpaca-cleaned corpus (~52K
   rows, non-empty-input rows only) every time -- there is no way to cheaply smoke-test throughput
   the way vi_preference_gen.py's --n_samples already does for VN.
2. `LLM(model=..., tensor_parallel_size=..., trust_remote_code=True)` has no `max_model_len` --
   the exact bug already found and fixed for VN (record.md Decision #23): Llama-3.1's
   auto-detected 131072 context needs ~16GB KV cache, doesn't fit a rented GPU's remaining VRAM
   after model weights.

Given both gaps require changing control flow inside the per-row loop (not just wrapping the
call), this file now mirrors vi_preference_gen.py's approach instead: reimplements the same
generation logic (90/10 straightforward/completion split, randomized injection position,
self-generated chosen/rejected, resumable chunked checkpointing) against the EN corpus
(yahma/alpaca-cleaned by default), reusing only the small shared helpers via meta_bridge
(create_injection_for_completion, jload/jdump, calculate_length_for_preference_dataset) --
exactly the same relationship vi_preference_gen.py already has to Meta's original function. This
was already the established precedent, not a new pattern.

Untrusted `input` is passed through chat_template.sanitize_untrusted_input() before being placed
in a chat-template message, matching vi_preference_gen.py and the project's fix for
recursive_filter never being wired into Meta's real pipeline (see training/chat_template.py) --
Meta's own original en-side function did NOT do this; this is a deliberate, documented deviation
applied uniformly to both EN and VN generation for methodological consistency within this
project's own pipeline.
"""

from __future__ import annotations

import argparse
import os
import time
from copy import deepcopy

import numpy as np

from vi_secalign.config import EXTERNAL_ROOT
from vi_secalign.data_gen import meta_bridge
from vi_secalign.hf_sync import upload_output
from vi_secalign.training.chat_template import build_messages, load_tokenizer, sanitize_untrusted_input

DEFAULT_INSTRUCT_DATASET = "alpaca"


def _load_en_corpus(instruct_dataset: str):
    from datasets import load_dataset  # deferred: heavy dependency

    if instruct_dataset == "alpaca":
        return load_dataset("yahma/alpaca-cleaned")["train"]
    if instruct_dataset == "natural":
        return load_dataset("Muennighoff/natural-instructions", data_dir="train")["train"]
    raise ValueError(f"Unknown instruct_dataset {instruct_dataset!r}, expected 'alpaca' or 'natural'")


def generate_en_preference_dataset(
    preference_data_path: str,
    model_name_or_path: str,
    instruct_dataset: str = DEFAULT_INSTRUCT_DATASET,
    self_generated_response: bool = True,
    randomized_injection_position: bool = True,
    n_samples: int | None = 2000,
    seed: int = 42,
    checkpoint_every: int = 500,
    upload_every_checkpoint: bool = True,
    max_model_len: int = 12288,
):
    """Mirrors vi_preference_gen.py::generate_vi_preference_dataset -- see that docstring for the
    full rationale on n_samples/max_model_len/checkpoint_every/resumability; identical mechanics,
    EN corpus instead of VN.
    """
    import torch  # deferred: heavy dependency
    from vllm import LLM, SamplingParams  # deferred: heavy dependency

    tokenizer = load_tokenizer()
    clean_data = _load_en_corpus(instruct_dataset)
    # injection_data comes from Meta's own fixed pool (data/alpaca_data.json), same file VN uses
    # via meta_bridge -- matches generate_preference_dataset's own behavior (utils.py:65).
    injection_data = meta_bridge.jload(str(EXTERNAL_ROOT / "data" / "alpaca_data.json"))
    ref_inst_resp = {s["instruction"]: s["output"] for s in injection_data}

    preference_data = []
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(clean_data))
    for idx in order:
        if n_samples is not None and len(preference_data) >= n_samples:
            break
        if instruct_dataset == "alpaca":
            current_sample = deepcopy(clean_data[int(idx)])
        else:  # "natural"
            raw = clean_data[int(idx)]
            current_sample = {"instruction": raw["definition"], "input": raw["inputs"], "output": raw["targets"]}
        if not current_sample.get("input"):  # catches missing key, "", and None -- same guard as VN
            continue

        injected_sample = injection_data[rng.integers(len(injection_data))]
        injected_prompt = injected_sample["instruction"] + " " + injected_sample["input"]

        if rng.random() < 0.9:  # 90% straightforward / 10% completion -- matches SecAlign++/VN
            untrusted_input = (
                injected_prompt + " " + current_sample["input"]
                if (rng.random() < 0.5 and randomized_injection_position)
                else current_sample["input"] + " " + injected_prompt
            )
        else:
            fake_response = ref_inst_resp.get(current_sample["instruction"], current_sample["output"])
            untrusted_input = current_sample["input"] + "\n\n" + meta_bridge.create_injection_for_completion(
                fake_response, injected_sample["instruction"], injected_sample["input"]
            )

        untrusted_input = sanitize_untrusted_input(untrusted_input)
        prompt = build_messages(tokenizer, instruction=current_sample["instruction"], untrusted_input=untrusted_input)

        if self_generated_response:
            preference_data.append(
                {
                    "prompt": prompt,
                    "chosen_input": current_sample["instruction"] + "\n\n" + current_sample["input"],
                    "rejected_input": injected_prompt,
                }
            )
        else:
            preference_data.append(
                {
                    "prompt": prompt,
                    "chosen": current_sample["output"] + tokenizer.eos_token,
                    "rejected": injected_sample["output"] + tokenizer.eos_token,
                }
            )

    if self_generated_response:
        completed: list[dict] = []
        if os.path.exists(preference_data_path):
            completed = meta_bridge.jload(preference_data_path)
            if len(completed) > len(preference_data):
                raise ValueError(
                    f"Existing {preference_data_path} has {len(completed)} pairs, more than the "
                    f"{len(preference_data)} this call would produce -- likely different "
                    "n_samples/seed/instruct_dataset args than the run that created it. Refusing "
                    "to guess; pass matching args or a fresh preference_data_path."
                )
            print(f"Resuming: {len(completed)}/{len(preference_data)} pairs already done in {preference_data_path}")

        llm = LLM(
            model=model_name_or_path,
            tensor_parallel_size=torch.cuda.device_count(),
            trust_remote_code=True,
            max_model_len=max_model_len,
        )
        sampling_params = SamplingParams(temperature=0.8, max_tokens=8192, stop=tokenizer.eos_token)

        run_start = time.time()
        n_done_at_start = len(completed)
        remaining = preference_data[len(completed):]
        for chunk_start in range(0, len(remaining), checkpoint_every):
            chunk = remaining[chunk_start : chunk_start + checkpoint_every]
            chunk_t0 = time.time()
            conversations = []
            for sample in chunk:
                conversations.append([{"role": "user", "content": sample["chosen_input"]}])
                conversations.append([{"role": "user", "content": sample["rejected_input"]}])
            outputs = llm.chat(conversations, sampling_params)
            for i, sample in enumerate(chunk):
                sample["chosen"] = outputs[2 * i].outputs[0].text + tokenizer.eos_token
                sample["rejected"] = outputs[2 * i + 1].outputs[0].text + tokenizer.eos_token
            completed.extend(chunk)
            meta_bridge.jdump(completed, preference_data_path)

            chunk_elapsed = time.time() - chunk_t0
            chunk_samples_per_sec = len(chunk) / chunk_elapsed if chunk_elapsed > 0 else float("nan")
            n_done_this_run = len(completed) - n_done_at_start
            overall_elapsed = time.time() - run_start
            overall_samples_per_sec = n_done_this_run / overall_elapsed if overall_elapsed > 0 else float("nan")
            n_left = len(preference_data) - len(completed)
            eta_seconds = n_left / overall_samples_per_sec if overall_samples_per_sec > 0 else float("nan")
            print(
                f"Checkpointed {len(completed)}/{len(preference_data)} -> {preference_data_path} | "
                f"chunk: {len(chunk)} samples in {chunk_elapsed:.1f}s ({chunk_samples_per_sec:.3f} samples/s) | "
                f"run avg: {overall_samples_per_sec:.3f} samples/s | "
                f"ETA remaining {n_left} samples: {eta_seconds / 60:.1f} min"
            )
            if upload_every_checkpoint:
                upload_output(preference_data_path, "en_preference_gen")
        del llm, sampling_params
        preference_data = completed
    else:
        meta_bridge.jdump(preference_data, preference_data_path)
        if upload_every_checkpoint:
            upload_output(preference_data_path, "en_preference_gen")
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=preference_data_path, split="train")
    meta_bridge.calculate_length_for_preference_dataset(dataset, tokenizer)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preference_data_path", required=True)
    parser.add_argument("--model_name_or_path", required=True, help="Model used to self-generate chosen/rejected responses.")
    parser.add_argument("--instruct_dataset", choices=["alpaca", "natural"], default=DEFAULT_INSTRUCT_DATASET)
    parser.add_argument("--no_self_generated_response", action="store_false", dest="self_generated_response", default=True)
    parser.add_argument("--no_randomized_injection_position", action="store_false", dest="randomized_injection_position", default=True)
    parser.add_argument("--n_samples", type=int, default=2000,
                         help="Cap on corpus rows used (None/0 = full ~52K-row alpaca-cleaned corpus, "
                         "NOT recommended without first measuring throughput on a smaller run).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_every", type=int, default=500,
                         help="Flush progress to --preference_data_path every N samples (resume support -- "
                         "the rented pod has a 24h max rental, see .agents/infra_handoff.md).")
    parser.add_argument(
        "--no_upload", action="store_false", dest="upload_every_checkpoint", default=True,
        help="Skip auto-uploading each checkpoint to Hugging Face (see hf_sync.py). Uploads by default.",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=12288,
        help="vLLM max_model_len (prompt+completion budget). Must be set explicitly for "
        "Llama-3.1-family models -- see vi_preference_gen.py's identical flag for the full "
        "rationale (record.md Decision #23).",
    )
    args = parser.parse_args()
    n_samples = args.n_samples if args.n_samples else None

    dataset = generate_en_preference_dataset(
        preference_data_path=args.preference_data_path,
        model_name_or_path=args.model_name_or_path,
        instruct_dataset=args.instruct_dataset,
        self_generated_response=args.self_generated_response,
        randomized_injection_position=args.randomized_injection_position,
        n_samples=n_samples,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        upload_every_checkpoint=args.upload_every_checkpoint,
        max_model_len=args.max_model_len,
    )
    print(f"Generated {len(dataset)} English preference pairs -> {args.preference_data_path}")


if __name__ == "__main__":
    main()
