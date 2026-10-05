"""T9b setup step: overwrite the installed torchtune package's torchtune/datasets/_preference.py
with Meta SecAlign's patched version (external/meta_secalign/helpers/_preference.py in this repo).

Why a standalone script with the patch content embedded verbatim, instead of just `cp`-ing the
repo file: this needs to run on a remote Colab/pod session via `colab exec -f <this file>`, which
transmits only the content of the script actually passed to `-f` (see notebooks/T10_MANUAL.md
step 4 -- "Transparent Code Execution", no separate upload). A second local file path wouldn't
exist on the remote session, so the patch content has to travel inside this one file.

Stock torchtune's _preference.py expects "chosen"/"rejected" as chat-message lists (via a
message_transform + tokenize_messages). Meta's patch instead reads "prompt"/"chosen"/"rejected" as
plain strings via tokenizer.encode() directly -- confirmed by diffing against the real
torchtune==0.6.0 source. This matches vi_preference_gen.py's output schema exactly (verified: both
fields are plain pre-templated strings, same as Meta's own preference_*.json), so no dataset
conversion script is needed -- just this one file swap.
"""

from __future__ import annotations

import torchtune.datasets

_PATCHED_PREFERENCE_PY = '''\
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Mapping, Optional

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset

from torchtune.data import ChosenRejectedToMessages, CROSS_ENTROPY_IGNORE_IDX
from torchtune.modules.transforms import Transform

from torchtune.modules.transforms.tokenizers import ModelTokenizer


class PreferenceDataset(Dataset):
    """Meta SecAlign patched PreferenceDataset -- see this script's module docstring."""

    def __init__(
        self,
        *,
        source: str,
        message_transform: Transform,
        tokenizer: ModelTokenizer,
        filter_fn: Optional[Callable] = None,
        packed: bool = False,
        **load_dataset_kwargs: dict[str, Any],
    ) -> None:
        if packed:
            raise ValueError(
                "Packed is currently not supported for preference datasets."
            )

        self._tokenizer = tokenizer
        self._message_transform = message_transform
        self._data = load_dataset(source, **load_dataset_kwargs)

        if filter_fn is not None:
            self._data = self._data.filter(filter_fn)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> dict[str, list[int]]:

        prompt_tokenized = self._tokenizer.encode(sample["prompt"])
        prompt_mask = [True] * len(prompt_tokenized)
        chosen_tokenized = self._tokenizer.encode(sample["chosen"])
        chosen_mask = [False] * (len(chosen_tokenized) - 1) + [True]
        rejected_tokenized = self._tokenizer.encode(sample["rejected"])
        rejected_mask = [False] * (len(rejected_tokenized) - 1) + [True]

        chosen_input_ids = prompt_tokenized + chosen_tokenized
        rejected_input_ids = prompt_tokenized + rejected_tokenized
        chosen_masks = prompt_mask + chosen_mask
        rejected_masks = prompt_mask + rejected_mask

        chosen_labels = list(
            np.where(chosen_masks, CROSS_ENTROPY_IGNORE_IDX, chosen_input_ids)
        )

        rejected_labels = list(
            np.where(rejected_masks, CROSS_ENTROPY_IGNORE_IDX, rejected_input_ids)
        )

        assert len(chosen_input_ids) == len(chosen_labels)
        assert len(rejected_input_ids) == len(rejected_labels)

        tokenized_dict = dict(
            chosen_input_ids=chosen_input_ids,
            chosen_labels=chosen_labels,
            rejected_input_ids=rejected_input_ids,
            rejected_labels=rejected_labels,
        )

        return tokenized_dict


def preference_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str,
    column_map: Optional[dict[str, str]] = None,
    train_on_input: bool = False,
    new_system_prompt: Optional[str] = None,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: dict[str, Any],
) -> PreferenceDataset:
    """Meta SecAlign patched preference_dataset builder -- see this script's module docstring."""

    message_transform = ChosenRejectedToMessages(
        train_on_input=train_on_input,
        column_map=column_map,
        new_system_prompt=new_system_prompt,
    )

    return PreferenceDataset(
        source=source,
        message_transform=message_transform,
        tokenizer=tokenizer,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs,
    )
'''

if __name__ == "__main__":
    import os

    target = os.path.join(os.path.dirname(torchtune.datasets.__file__), "_preference.py")
    with open(target, "w") as f:
        f.write(_PATCHED_PREFERENCE_PY)
    print(f"[apply_torchtune_preference_patch] Wrote patched preference_dataset to {target}")
