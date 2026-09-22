"""Phase 1.5 (RQ2, conditional): Vietnamese preference-pair generation.

Only run this if Phase 1's zero-shot eval shows VN_ASR clearly worse than EN_ASR on the same
model (see project memory: Phase 1.5 is conditional, not assumed) — building this data before
that result is in would beg RQ1's question.

Mirrors external/meta_secalign/utils.py::generate_preference_dataset (same 90/10
straightforward/completion ratio, same randomized injection position, same self-generated
chosen/rejected via vLLM) but cannot reuse it directly: that function hard-codes
instruct_dataset in {"alpaca", "natural"} and resolves paths relative to EXTERNAL_ROOT. Reused via
import (not copy-paste): create_injection_for_completion, jload/jdump,
calculate_length_for_preference_dataset (all via meta_bridge).

Corpus decision (see .agents/record.md Decision #18/#19): Go confirmed 2026-09-21 (T6) --
MBZUAI/Bactrian-X, subset "vi" -- CONFIRMED, not a scaffold default anymore:
  - License: CC-BY-NC-4.0 (verified via HF API, 2026-09-21) -- fine for non-commercial thesis use.
  - Quality: spot-checked ~10 samples across the corpus (offsets 0-30000) -- fluent, grammatically
    natural Vietnamese, not stilted MT output. Already independently corroborated by
    vi_injecteval_gen.py's pilot (same corpus, used for eval).
  - Dropped vietgpt/alpaca_vi as fallback: no license tag on HF, 5 total downloads -- not
    trustworthy enough to fall back to, and primary candidate already cleared both checks above.

Every untrusted `input` is passed through chat_template.sanitize_untrusted_input() before being
placed in a chat-template message, per the project's fix for recursive_filter never being wired
into the real train/eval pipeline upstream (see training/chat_template.py).

Needs a GPU (vLLM self-generation) -- not run in this pass, this environment has none. `n_samples`
(new) caps the corpus subsample BEFORE the vLLM generation step -- the full ~67K-row corpus would
mean ~134K generations (chosen+rejected), far beyond the "2 ngày" T8 budget. Default 2000 is a
starting point, not a validated final size -- see docstring on n_samples below: measure real
throughput on the rented pod on a small run first, then decide whether to scale up.

Resumability (ckey.vn pod hard-caps rentals at 24h -- see .agents/infra_handoff.md, Decision #21):
generation runs in chunks of `checkpoint_every` samples (default 500), flushing completed pairs to
`preference_data_path` after every chunk instead of one giant vLLM call at the end. On restart, if
`preference_data_path` already has N completed pairs, the first N (prompt-)samples of the
deterministic seeded order are skipped -- their prompts don't need to be regenerated, only the
still-missing tail is sent to vLLM. This is safe because prompt construction (everything before the
`llm.chat()` call) is pure CPU and fully determined by `seed`/`n_samples`/`corpus_hf_id` -- rerunning
it from scratch each time is cheap and always reproduces the same ordering. Operational flow: before
the pod's 24h limit, upload `preference_data_path` to HF (tools/hf_upload/*.py); on the next pod,
download it back to the same local path before re-running this script with the same arguments.
"""

from __future__ import annotations

import argparse
import os
import time
from copy import deepcopy

import numpy as np

from vi_secalign.config import ANCHOR_HYPERPARAMS
from vi_secalign.data_gen import meta_bridge
from vi_secalign.hf_sync import upload_output
from vi_secalign.training.chat_template import build_messages, load_tokenizer, sanitize_untrusted_input

DEFAULT_CORPUS_HF_ID = "MBZUAI/Bactrian-X"
DEFAULT_CORPUS_SUBSET = "vi"
FALLBACK_CORPUS_HF_ID = "vietgpt/alpaca_vi"


def _load_vi_corpus(hf_id: str, subset: str | None):
    """Load a Vietnamese instruction corpus with 'instruction'/'input'/'output' fields.

    Assumes an Alpaca-shaped schema (matches Bactrian-X and vietgpt/alpaca_vi). If a different
    corpus is substituted, this is the one place that needs adapting.
    """
    from datasets import load_dataset  # deferred: heavy dependency, not needed just to import this module

    if subset:
        return load_dataset(hf_id, subset)["train"]
    return load_dataset(hf_id)["train"]


def generate_vi_preference_dataset(
    preference_data_path: str,
    model_name_or_path: str,
    corpus_hf_id: str = DEFAULT_CORPUS_HF_ID,
    corpus_subset: str | None = DEFAULT_CORPUS_SUBSET,
    self_generated_response: bool = True,
    randomized_injection_position: bool = True,
    n_samples: int | None = 2000,
    seed: int = 42,
    checkpoint_every: int = 500,
    upload_every_checkpoint: bool = True,
):
    """n_samples caps how many rows are drawn from the corpus BEFORE vLLM generation -- each row
    costs 2 generations (chosen+rejected) at up to max_tokens=8192, so this is the main lever on
    wall-clock/$ cost for T8. None = full corpus (~67K rows, ~134K generations -- NOT recommended,
    far beyond the T8 budget; only use None once a smaller run's throughput justifies scaling up).

    checkpoint_every controls how many samples are generated per vLLM `llm.chat()` call before
    flushing to `preference_data_path` -- see module docstring's "Resumability" section. Smaller
    values checkpoint more often (safer against an unexpected pod stop) at a small throughput cost
    (vLLM batches less per call); 500 is a reasonable default, not a validated optimum.
    """
    import torch  # deferred: heavy dependency
    from vllm import LLM, SamplingParams  # deferred: heavy dependency

    tokenizer = load_tokenizer()
    clean_data = _load_vi_corpus(corpus_hf_id, corpus_subset)
    injection_data = list(clean_data)  # same corpus supplies both trusted instructions and injected prompts
    ref_inst_resp = {s["instruction"]: s["output"] for s in injection_data}

    preference_data = []
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(clean_data))
    if n_samples is not None:
        order = order[: n_samples * 2]  # over-sample: the empty-`input` skip below drops some
    for idx in order:
        if n_samples is not None and len(preference_data) >= n_samples:
            break
        current_sample = deepcopy(clean_data[int(idx)])
        if current_sample.get("input", "") == "":
            continue

        injected_sample = injection_data[rng.integers(len(injection_data))]
        injected_prompt = injected_sample["instruction"] + " " + injected_sample["input"]

        if rng.random() < 0.9:  # 90% straightforward / 10% completion — matches SecAlign++
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
        # Resume support: if a prior run already wrote N completed pairs, the first N entries of
        # `preference_data` (deterministic given seed/n_samples/corpus) are the same N samples --
        # skip regenerating them, only send the remaining tail to vLLM. See module docstring.
        completed: list[dict] = []
        if os.path.exists(preference_data_path):
            completed = meta_bridge.jload(preference_data_path)
            if len(completed) > len(preference_data):
                raise ValueError(
                    f"Existing {preference_data_path} has {len(completed)} pairs, more than the "
                    f"{len(preference_data)} this call would produce -- likely different "
                    "n_samples/seed/corpus args than the run that created it. Refusing to guess; "
                    "pass matching args or a fresh preference_data_path."
                )
            print(f"Resuming: {len(completed)}/{len(preference_data)} pairs already done in {preference_data_path}")

        llm = LLM(model=model_name_or_path, tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True)
        sampling_params = SamplingParams(temperature=0.8, max_tokens=8192, stop=tokenizer.eos_token)

        # Throughput/ETA instrumentation (2026-09-22, Decision #21): the real N for T9's EN:VN ratio
        # is deliberately NOT hard-coded yet -- it depends on the samples/sec measured here on the
        # actual rented pod. Printed per-chunk (not just at the end) so a run interrupted by the 24h
        # pod cap still leaves a usable throughput number in whatever log was captured up to that
        # point, not only on a clean full completion.
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
            meta_bridge.jdump(completed, preference_data_path)  # full overwrite -- small N, safest against partial-JSON corruption

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
                # Upload after EVERY checkpoint, not just at the end -- the whole point of
                # checkpoint_every/--resume_from_checkpoint (Decision #21) is surviving the pod's
                # 24h rental cap, which only works if the latest checkpoint is off-pod, not just
                # on local disk that gets wiped when the rental ends.
                upload_output(preference_data_path, "vi_preference_gen")
        del llm, sampling_params
        preference_data = completed
    else:
        meta_bridge.jdump(preference_data, preference_data_path)
        if upload_every_checkpoint:
            upload_output(preference_data_path, "vi_preference_gen")
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=preference_data_path, split="train")
    meta_bridge.calculate_length_for_preference_dataset(dataset, tokenizer)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preference_data_path", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--corpus_hf_id", default=DEFAULT_CORPUS_HF_ID)
    parser.add_argument("--corpus_subset", default=DEFAULT_CORPUS_SUBSET)
    parser.add_argument("--no_self_generated_response", action="store_false", dest="self_generated_response", default=True)
    parser.add_argument("--no_randomized_injection_position", action="store_false", dest="randomized_injection_position", default=True)
    parser.add_argument("--n_samples", type=int, default=2000,
                         help="Cap on corpus rows used (None/0 = full ~67K corpus, NOT recommended for T8's budget).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_every", type=int, default=500,
                         help="Flush progress to --preference_data_path every N samples (resume support -- "
                         "the rented pod has a 24h max rental, see .agents/infra_handoff.md).")
    parser.add_argument(
        "--no_upload",
        action="store_false",
        dest="upload_every_checkpoint",
        default=True,
        help="Skip auto-uploading each checkpoint to Hugging Face (see hf_sync.py). Uploads by "
        "default -- the whole point of checkpointing is surviving the pod's 24h rental cap, which "
        "only works if progress leaves the pod, see .agents/record.md Decision #21.",
    )
    args = parser.parse_args()
    n_samples = args.n_samples if args.n_samples else None

    dataset = generate_vi_preference_dataset(
        preference_data_path=args.preference_data_path,
        model_name_or_path=args.model_name_or_path,
        corpus_hf_id=args.corpus_hf_id,
        corpus_subset=args.corpus_subset,
        self_generated_response=args.self_generated_response,
        randomized_injection_position=args.randomized_injection_position,
        n_samples=n_samples,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        upload_every_checkpoint=args.upload_every_checkpoint,
    )
    print(f"Generated {len(dataset)} Vietnamese preference pairs -> {args.preference_data_path}")


if __name__ == "__main__":
    main()
