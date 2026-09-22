"""Phase 1 (RQ1 baseline data) / Phase 0 pipeline: English preference-pair generation.

Thin CLI wrapper around external/meta_secalign's own generate_preference_dataset — reused
verbatim via meta_bridge, not reimplemented. Per the project's key pivot, this is NOT needed to
reproduce Meta-SecAlign-8B (that checkpoint is downloaded and used directly) — this script exists
only if a further EN-preference run is ever needed as a control/ablation input.

generate_preference_dataset (external/meta_secalign/utils.py:49-127) resolves 'data/alpaca_data.json'
and a tokenizer from 'data/' using paths relative to the process cwd (utils.py:65,69) — so this
script chdirs to EXTERNAL_ROOT for the duration of the call and restores cwd afterward.

Not run in this pass (needs vllm + GPU); this is the CLI that will be invoked once Phase 1/1.5
work begins.
"""

from __future__ import annotations

import argparse
import os

from vi_secalign.config import EXTERNAL_ROOT
from vi_secalign.data_gen import meta_bridge
from vi_secalign.hf_sync import upload_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preference_data_path",
        required=True,
        help="Output path for the generated preference JSON (relative to EXTERNAL_ROOT, matching "
        "generate_preference_dataset's own path handling).",
    )
    parser.add_argument("--instruct_dataset", choices=["alpaca", "natural"], default="alpaca")
    parser.add_argument("--model_name_or_path", required=True, help="Model used to self-generate chosen/rejected responses.")
    parser.add_argument("--no_self_generated_response", action="store_false", dest="self_generated_response", default=True)
    parser.add_argument("--no_randomized_injection_position", action="store_false", dest="randomized_injection_position", default=True)
    parser.add_argument(
        "--no_upload", action="store_false", dest="upload", default=True,
        help="Skip auto-uploading the output to Hugging Face (see hf_sync.py). Uploads by default.",
    )
    args = parser.parse_args()

    original_cwd = os.getcwd()
    os.chdir(EXTERNAL_ROOT)
    try:
        dataset = meta_bridge.generate_preference_dataset(
            preference_data_path=args.preference_data_path,
            instruct_dataset=args.instruct_dataset,
            self_generated_response=args.self_generated_response,
            randomized_injection_position=args.randomized_injection_position,
            model_name_or_path=args.model_name_or_path,
        )
    finally:
        os.chdir(original_cwd)

    print(f"Generated {len(dataset)} preference pairs -> {args.preference_data_path}")

    if args.upload:
        upload_output(EXTERNAL_ROOT / args.preference_data_path, "en_preference_gen")


if __name__ == "__main__":
    main()
