"""One-time GPU step: builds data/SEP_dataset_test.json + its reference-output file.

Port of external/meta_secalign/setup.py:565-604 (NOT run via setup.py itself — see
tools/pod_setup/fetch_meta_secalign_data_urls.py's docstring for why: setup.py also unconditionally
downloads 5 full model repos before reaching this point). This is the ONE piece of setup.py's data
pipeline that genuinely needs a GPU (loads meta-llama/Meta-Llama-3-8B-Instruct via vLLM to
self-generate reference answers for the clean, non-injected SEP prompts) and so could not be
precomputed on a laptop in tools/pod_setup/build_env_cache.sh — must run once on the pod itself,
before meta_eval_runner.run_sep() (T1-T3) will work.

Requires (already present in the env-cache tarball, see build_env_cache.sh):
- data/SEP_dataset.json (one of the 14 raw data_urls)
- data/ as a loadable tokenizer dir (data/tokenizer.json, tokenizer_config.json,
  special_tokens_map.json, chat_template.jinja) -- these were produced by an earlier
  (intentionally interrupted, see .agents/record.md) run of setup.py's tokenizer-surgery step
  (setup.py:70-178: loads meta-llama/Llama-3.1-8B-Instruct's tokenizer, since "Llama 3 series
  share the same tokenizer" per that script's own comment, strips the built-in system prompt from
  its chat_template, and saves the result to data/). Reusing that already-saved tokenizer here
  (AutoTokenizer.from_pretrained(EXTERNAL_ROOT/'data')) instead of redoing the template surgery
  keeps this script a pure verbatim port of the generation step, not a reimplementation of the
  template-editing step too.

Output (both required by meta_eval_runner.run_sep(), which reads 'data/SEP_dataset_test.json'):
- data/SEP_dataset_test.json: the clean {instruction,input,injection,witness,output} records.
- data/SEP_dataset_test_Meta-Llama-3-8B-Instruct.json: the reference-labeled version, same records
  with 'instruction' folded to instruction+'\n\n'+input (setup.py:594's exact concatenation).

Not run in this pass (needs vllm + GPU); this is the CLI that will be invoked once a pod is live,
before T1-T3.
"""

from __future__ import annotations

import argparse
import os
import time

from vi_secalign.config import EXTERNAL_ROOT
from vi_secalign.data_gen import meta_bridge
from vi_secalign.hf_sync import upload_output
from vi_secalign.models.registry import get as get_model

REFERENCE_GENERATOR_MODEL = get_model("llama3_8b_instruct_sep_reference").source
OUT_NAME = "SEP_dataset_test.json"
OUT_NAME_REF = f"SEP_dataset_test_{REFERENCE_GENERATOR_MODEL.split('/')[-1]}.json"


def _identity(d_item: dict) -> dict:
    """Equivalent to setup.py's `none` (utils.py) -- no injection, used because this pass
    generates the CLEAN reference answer, not an attacked one."""
    return d_item


def build_sep_reference(data_dir: str = "data") -> tuple[list[dict], list[dict]]:
    """Verbatim port of setup.py:565-604. Must run with cwd == EXTERNAL_ROOT (caller's job, see
    main() below) -- form_llm_input/jload/jdump all resolve paths relative to cwd.
    """
    import torch  # deferred: heavy dependency, matches project convention elsewhere
    import transformers

    out_path = os.path.join(data_dir, OUT_NAME)
    out_path_ref = os.path.join(data_dir, OUT_NAME_REF)
    if os.path.exists(out_path) and os.path.exists(out_path_ref):
        print(f"{out_path} and {out_path_ref} already exist -- nothing to do.")
        return meta_bridge.jload(out_path), meta_bridge.jload(out_path_ref)

    assert torch.cuda.device_count() > 0, (
        "GPU is required to process the SEP dataset and generate reference responses for "
        "evaluation. Please ensure that a compatible GPU is available and properly configured."
    )

    data = meta_bridge.jload(os.path.join(data_dir, "SEP_dataset.json"))
    data_sft_format = []
    for d in data:
        instruction = d["system_prompt_clean"]
        input_ = d["prompt_clean"]
        injection = d["prompt_instructed"].replace(d["prompt_clean"], "")
        if injection.startswith(" ") or injection.startswith("\n"):
            injection = injection[1:]
        data_sft_format.append(
            {"instruction": instruction, "input": input_, "injection": injection, "witness": d["witness"]}
        )

    # Reuse the already-saved, system-prompt-stripped tokenizer from data/ (see module docstring)
    # instead of redoing setup.py's chat_template surgery here.
    tokenizer = transformers.AutoTokenizer.from_pretrained(data_dir)

    model, _ = meta_bridge.load_vllm_model(REFERENCE_GENERATOR_MODEL)
    llm_input = meta_bridge.form_llm_input(
        data_sft_format, _identity, tokenizer.apply_chat_template, instruction_hierarchy=True, defense="none"
    )
    print(llm_input[0])
    time.sleep(5)
    gen_t0 = time.time()
    outputs = meta_bridge.test_model_output_vllm(llm_input, model, tokenizer)
    gen_elapsed = time.time() - gen_t0
    samples_per_sec = len(llm_input) / gen_elapsed if gen_elapsed > 0 else float("nan")
    print(
        f"Generation: {len(llm_input)} samples in {gen_elapsed:.1f}s "
        f"({gen_elapsed / 60:.1f} min, {samples_per_sec:.3f} samples/s)"
    )

    data_reference = []
    for i, d in enumerate(data_sft_format):
        while outputs[i].startswith(" "):
            outputs[i] = outputs[i][1:]
        data_reference.append(
            {
                "instruction": d["instruction"] + "\n\n" + d["input"],
                "input": d["input"],
                "output": outputs[i],
                "witness": d["witness"],
                "injection": d["injection"],
                "instruction_only": d["instruction"],
                "generator": REFERENCE_GENERATOR_MODEL,
            }
        )
        data_sft_format[i]["output"] = outputs[i]

    meta_bridge.jdump(data_reference, out_path_ref)
    meta_bridge.jdump(data_sft_format, out_path)
    return data_sft_format, data_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Relative to EXTERNAL_ROOT (matching setup.py's own path handling) -- default matches "
        "where fetch_meta_secalign_data_urls.py/build_env_cache.sh place everything.",
    )
    parser.add_argument(
        "--no_upload",
        action="store_false",
        dest="upload",
        default=True,
        help="Skip auto-uploading the output to Hugging Face (see hf_sync.py). Uploads by default "
        "since the rented pod has a 24h rental cap -- see .agents/record.md Decision #21.",
    )
    args = parser.parse_args()

    original_cwd = os.getcwd()
    os.chdir(EXTERNAL_ROOT)
    try:
        data_sft_format, data_reference = build_sep_reference(data_dir=args.data_dir)
    finally:
        os.chdir(original_cwd)

    print(f"Built {len(data_sft_format)} SEP test records + {len(data_reference)} reference records.")

    if args.upload:
        upload_output(EXTERNAL_ROOT / args.data_dir / OUT_NAME, "sep_reference_gen")
        upload_output(EXTERNAL_ROOT / args.data_dir / OUT_NAME_REF, "sep_reference_gen")


if __name__ == "__main__":
    main()
