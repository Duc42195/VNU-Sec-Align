"""Shared chat-template rendering + untrusted-input sanitization for VNU-SecAlign v2.

Every place in this project that builds a `user`/`input`-role prompt must go through
build_messages() here, instead of re-deriving the message structure ad hoc (the exact mistake
found in prior notebooks: chat templates built with a single flattened "user" role, which never
exercises the SecAlign++ trust-separation mechanism at all).

Loads the tokenizer from facebook/Meta-SecAlign-8B, which already ships the official "input"-role
chat template — this avoids depending on external/meta_secalign/data/ (which, as checked out,
has no tokenizer_config.json of its own; external/meta_secalign/utils.py's own
generate_preference_dataset relies on a 'data' directory that must be populated separately by
setup.py).
"""

from __future__ import annotations

from typing import Literal

DEFAULT_TOKENIZER_MODEL = "facebook/Meta-SecAlign-8B"

_tokenizer_cache: dict[str, object] = {}

# Ported verbatim from external/meta_secalign/demo.py:11-15 (recursive_filter). That function is
# defined there and used only in the demo script — grep confirms it is never called from
# utils.py/test.py/secalign_plus_plus.py, i.e. the real train/eval pipeline never sanitizes
# untrusted input against special-token spoofing. This project always applies it (see callers in
# data_gen/vi_preference_gen.py, data_gen/attack10_gen.py, training/train_dpo.py).
_SPECIAL_TOKEN_FILTERS = [
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|begin_of_text|>",
]


def sanitize_untrusted_input(text: str, filters: list[str] | None = None) -> str:
    """Recursively strip Llama-3 special-token strings from untrusted content.

    Recursive by construction (matches demo.py): a nested attempt like
    "<|start_header_id<|start_header_id|>|>" only becomes a real special token after one pass, so
    a single non-recursive strip is not sufficient.
    """
    filters = filters if filters is not None else _SPECIAL_TOKEN_FILTERS
    original = text
    for f in filters:
        text = text.replace(f, "")
    if text != original:
        return sanitize_untrusted_input(text, filters)
    return text


def load_tokenizer(model_name_or_path: str = DEFAULT_TOKENIZER_MODEL):
    if model_name_or_path not in _tokenizer_cache:
        from transformers import AutoTokenizer  # deferred: heavy dependency

        _tokenizer_cache[model_name_or_path] = AutoTokenizer.from_pretrained(model_name_or_path)
    return _tokenizer_cache[model_name_or_path]


def build_messages(
    tokenizer,
    instruction: str,
    untrusted_input: str,
    instruction_hierarchy: bool = True,
    role_order: Literal["normal", "swapped"] = "normal",
    add_generation_prompt: bool = True,
) -> str:
    """Render a trusted instruction + untrusted input through the official input-role template.

    Matches external/meta_secalign/utils.py:90-93 / utils.py:252 exactly when
    instruction_hierarchy=True and role_order="normal" (the only mode used for training data
    generation). The other combinations exist specifically for the two mechanism-validity
    ablations required by the plan — they must NEVER be used when generating training data,
    only at eval time:

    - instruction_hierarchy=False: collapses instruction+input into a single `user` message
      (matches utils.py:253-254's `--no_instruction_hierarchy` behavior). If ASR jumps back
      toward baseline under this setting, the defense genuinely depends on the structural role
      separation.
    - role_order="swapped": places the untrusted `input` message before the trusted `user`
      message (opposite of the order used everywhere in training). If ASR rises sharply here,
      the model learned a positional shortcut rather than content-based judgment.
    """
    if not instruction_hierarchy:
        messages = [{"role": "user", "content": instruction + "\n\n" + untrusted_input}]
    elif role_order == "swapped":
        messages = [
            {"role": "input", "content": untrusted_input},
            {"role": "user", "content": instruction},
        ]
    else:
        messages = [
            {"role": "user", "content": instruction},
            {"role": "input", "content": untrusted_input},
        ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
