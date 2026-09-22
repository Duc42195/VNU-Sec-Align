"""Shared helper: auto-upload a generation/training script's output to Hugging Face at the end of
its run, so a pod-generated file survives past the pod's 24h rental cap without a separate manual
step (see .agents/record.md Decision #21 for the 24h constraint; this generalizes the manual
upload command used once for sep_reference_gen.py's output to every data-gen/training script).

Uses the same repo (`Jason-42195/VNU-SecAlign`, repo_type="model") already established by
tools/hf_upload/*.py for v1 checkpoints -- pod outputs land under `pod_outputs/<script>/` there,
kept separate from `checkpoints/`/`data/` (v1's own layout) and from the packages-only
`Jason-42195/vnu-secalign-env-cache` dataset repo (a different repo, different purpose -- see
tools/pod_setup/build_env_cache.sh).

Fails soft, not hard: a missing HF_TOKEN or a network error prints a warning and returns None
instead of raising, so a local/laptop dev run (no token, no intent to upload) still completes
normally -- upload is a convenience for pod runs, not a correctness requirement of the generation
itself.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_REPO_ID = "Jason-42195/VNU-SecAlign"


def upload_output(local_path: str | Path, dest_subdir: str, repo_id: str = DEFAULT_REPO_ID) -> str | None:
    """Uploads a file or directory to `repo_id` under `pod_outputs/<dest_subdir>/`.

    `dest_subdir` should name the producing script (e.g. "sep_reference_gen", "vi_preference_gen")
    so multiple scripts' outputs don't collide. Returns the resulting HF path, or None if the
    upload was skipped/failed (see module docstring -- soft-fail by design).
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        print(f"[hf_sync] HF_TOKEN not set -- skipping upload of {local_path} (set HF_TOKEN to enable).")
        return None

    local_path = Path(local_path)
    if not local_path.exists():
        print(f"[hf_sync] {local_path} does not exist -- skipping upload.")
        return None

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[hf_sync] huggingface_hub not installed -- skipping upload.")
        return None

    api = HfApi()
    dest_root = f"pod_outputs/{dest_subdir}"
    try:
        if local_path.is_dir():
            api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(local_path),
                               path_in_repo=f"{dest_root}/{local_path.name}", token=token)
            dest = f"{dest_root}/{local_path.name}"
        else:
            dest = f"{dest_root}/{local_path.name}"
            api.upload_file(repo_id=repo_id, repo_type="model", path_or_fileobj=str(local_path),
                             path_in_repo=dest, token=token)
    except Exception as e:  # noqa: BLE001 -- soft-fail by design, see module docstring
        print(f"[hf_sync] Upload of {local_path} failed: {e}")
        return None

    print(f"[hf_sync] Uploaded {local_path} -> {repo_id}:{dest}")
    return dest
