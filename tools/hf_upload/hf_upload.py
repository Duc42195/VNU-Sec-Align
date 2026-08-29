from pathlib import Path
import os
import sys

try:
    from huggingface_hub import HfApi, upload_folder
except Exception as e:
    print("Missing huggingface_hub. Please run: python -m pip install --user huggingface_hub")
    raise

repo_id = "Jason-42195/VNU-SecAlign"
base = Path(__file__).resolve().parent.parent
checkpoint = base / "checkpoints" / "final_checkpoint"
data_dir = base / "data"

items = [
    (checkpoint, "checkpoints/final_checkpoint"),
    (data_dir / "dataset_v2_attacks.json", "data/dataset_v2_attacks.json"),
    (data_dir / "hallucination_preferences.json", "data/hallucination_preferences.json"),
]

token = os.environ.get("HF_TOKEN")
api = HfApi()

for src, dest in items:
    if not src.exists():
        print(f"ERROR: source not found: {src}")
        sys.exit(2)
    try:
        if src.is_dir():
            print(f"Uploading folder {src} -> {repo_id}:{dest}")
            upload_folder(repo_id=repo_id, folder_path=str(src), path_in_repo=dest, token=token)
        else:
            print(f"Uploading file {src} -> {repo_id}:{dest}")
            api.upload_file(path_or_fileobj=str(src), path_in_repo=dest, repo_id=repo_id, token=token)
    except Exception as exc:
        print("Upload failed:", exc)
        raise

print("All uploads attempted.")
