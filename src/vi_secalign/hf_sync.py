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
import time
from pathlib import Path

DEFAULT_REPO_ID = "Jason-42195/VNU-SecAlign"

# 2026-09-30 (record.md Decision #33): real T9 training run lost ALL ~18 intermediate checkpoint
# uploads over a ~12h run -- two independent real causes found the hard way:
#   1. HF_HUB_OFFLINE=1 (set to dodge a `--base_model` network-resolve issue, see Decision #31)
#      silently blocks upload_output() too -- huggingface_hub checks the offline flag even for
#      the upload path's metadata validation call (`/api/validate-yaml`), not just downloads.
#      Confirmed real: "Cannot reach https://huggingface.co/api/validate-yaml: offline mode is
#      enabled." Fix here: force the flag off for the duration of the upload call regardless of
#      what the calling process has set, since an upload always needs real network by definition.
#   2. A genuine pod network outage (confirmed via curl to huggingface.co/google.com/github.com,
#      IPv4 and IPv6 both timing out) -- no amount of retrying helps while the outage lasts, but
#      a real transient blip (much more common than a full outage) does recover within seconds to
#      low minutes. Added retry-with-backoff for that case, plus a local failure manifest so a
#      TOTAL outage (like this one) still leaves a precise, greppable list of what still needs
#      uploading once network is back -- instead of having to diff HF's file listing by hand.
_UPLOAD_RETRY_DELAYS_S = (15, 45, 90)  # 3 retries: ~15s, 45s, 90s backoff
_FAILURE_MANIFEST = Path(".hf_upload_failures.log")


def _record_upload_failure(local_path: Path, dest: str, repo_id: str, error: Exception) -> None:
    import datetime

    with open(_FAILURE_MANIFEST, "a") as f:
        f.write(f"{datetime.datetime.now(datetime.UTC).isoformat()}\t{local_path}\t{repo_id}:{dest}\t{error}\n")
    print(f"[hf_sync] Ghi lại vào {_FAILURE_MANIFEST} để retry sau -- KHÔNG mất dấu vết như lần trước.")


def upload_output(local_path: str | Path, dest_subdir: str, repo_id: str = DEFAULT_REPO_ID) -> str | None:
    """Uploads a file or directory to `repo_id` under `pod_outputs/<dest_subdir>/`.

    `dest_subdir` should name the producing script (e.g. "sep_reference_gen", "vi_preference_gen")
    so multiple scripts' outputs don't collide. Returns the resulting HF path, or None if the
    upload was skipped/failed (see module docstring -- soft-fail by design). Retries transient
    network failures with backoff (see Decision #33); a failure that survives all retries is
    appended to `.hf_upload_failures.log` in the cwd for a later manual/scripted re-upload pass.
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
    dest = f"{dest_root}/{local_path.name}"

    # Force-allow network for this call even if the caller set HF_HUB_OFFLINE=1 for an unrelated
    # reason (e.g. avoiding a --base_model resolve-over-network issue) -- an upload always needs
    # real network, offline mode blocking it is never the caller's intent. Restored after, so it
    # doesn't change behavior for any other code in the same process.
    prior_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "0"
    try:
        last_error: Exception | None = None
        for attempt, delay in enumerate((0, *_UPLOAD_RETRY_DELAYS_S)):
            if delay:
                print(f"[hf_sync] Retry {attempt}/{len(_UPLOAD_RETRY_DELAYS_S)} upload {local_path} sau {delay}s...")
                time.sleep(delay)
            try:
                if local_path.is_dir():
                    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(local_path),
                                       path_in_repo=dest, token=token)
                else:
                    api.upload_file(repo_id=repo_id, repo_type="model", path_or_fileobj=str(local_path),
                                     path_in_repo=dest, token=token)
                print(f"[hf_sync] Uploaded {local_path} -> {repo_id}:{dest}")
                return dest
            except Exception as e:  # noqa: BLE001 -- soft-fail by design, see module docstring
                last_error = e
                print(f"[hf_sync] Upload of {local_path} failed (lần {attempt + 1}): {e}")
        _record_upload_failure(local_path, dest, repo_id, last_error)
        return None
    finally:
        if prior_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prior_offline


def download_output(
    dest_subdir: str,
    filename: str,
    local_dir: str | Path = "data/pod_synced",
    repo_id: str = DEFAULT_REPO_ID,
) -> Path | None:
    """Counterpart to upload_output: pulls `pod_outputs/<dest_subdir>/<filename>` from `repo_id`
    down to `<local_dir>/<dest_subdir>/<filename>` (default local_dir matches the
    data/pod_synced/ convention -- see that folder's README).

    Uses the ~/.cache/huggingface/token cached by `huggingface-cli login` if HF_TOKEN is unset
    (unlike upload_output, a read of a model-repo file doesn't strictly need a token unless the
    repo is private, but this project's repo is private -- so in practice one of the two must be
    set). Soft-fails like upload_output, for the same reason (don't break a caller that doesn't
    care about syncing).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[hf_sync] huggingface_hub not installed -- skipping download.")
        return None

    token = os.environ.get("HF_TOKEN")  # None is fine here: falls back to the cached CLI login token
    local_dir = Path(local_dir) / dest_subdir
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        cached_path = hf_hub_download(
            repo_id=repo_id, repo_type="model", token=token,
            filename=f"pod_outputs/{dest_subdir}/{filename}",
        )
    except Exception as e:  # noqa: BLE001 -- soft-fail by design, see module docstring
        print(f"[hf_sync] Download of pod_outputs/{dest_subdir}/{filename} failed: {e}")
        return None

    import shutil

    dest = local_dir / filename
    shutil.copy(cached_path, dest)
    print(f"[hf_sync] Downloaded {repo_id}:pod_outputs/{dest_subdir}/{filename} -> {dest}")
    return dest


def retry_failed_uploads(manifest_path: str | Path = _FAILURE_MANIFEST) -> list[str]:
    """Re-attempts every upload logged in `.hf_upload_failures.log` (see upload_output's retry/
    manifest logic, Decision #33). Run this once network is confirmed back up -- e.g.:
        python3 -c "from vi_secalign.hf_sync import retry_failed_uploads; retry_failed_uploads()"
    Truncates the manifest to only the entries that still fail (so a second run only retries what's
    still actually broken), and returns the list of local paths that succeeded this pass.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        print(f"[hf_sync] {manifest_path} không tồn tại -- không có gì để retry.")
        return []

    lines = manifest_path.read_text().splitlines()
    still_failing: list[str] = []
    succeeded: list[str] = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) < 3:
            continue  # malformed line -- skip rather than crash the whole retry pass
        _, local_path, dest_ref, *_ = parts
        repo_id, dest = dest_ref.split(":", 1)
        dest_subdir = dest.removeprefix("pod_outputs/").rsplit("/", 1)[0]
        result = upload_output(local_path, dest_subdir, repo_id=repo_id)
        if result is None:
            still_failing.append(line)
        else:
            succeeded.append(local_path)

    manifest_path.write_text("\n".join(still_failing) + ("\n" if still_failing else ""))
    print(f"[hf_sync] Retry xong: {len(succeeded)} thành công, {len(still_failing)} vẫn thất bại (còn trong {manifest_path}).")
    return succeeded
