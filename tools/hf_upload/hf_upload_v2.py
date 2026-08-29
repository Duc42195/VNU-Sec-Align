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

def sanitize_readme_in_folder(folder: Path):
    readme = folder / "README.md"
    if not readme.exists():
        return
    try:
        text = readme.read_text(encoding="utf-8")
    except Exception:
        print("Warning: cannot read README.md")
        return
    if not text.startswith("---"):
        return
    parts = text.split("---", 2)
    if len(parts) < 3:
        return
    yaml_block = parts[1]
    cleaned_lines = []
    for line in yaml_block.splitlines():
        if not line.strip().startswith("base_model:"):
            cleaned_lines.append(line)
    new_yaml = "\n".join(cleaned_lines)
    new_text = "---" + new_yaml + "---" + (parts[2] if len(parts) > 2 else "")
    backup = readme.with_suffix('.md.bak')
    try:
        readme.replace(backup)
    except Exception:
        # fallback copy
        backup.write_text(text, encoding="utf-8")
    readme.write_text(new_text, encoding="utf-8")
    print(f"Sanitized README.md in {folder}")

for src, dest in items:
    if not src.exists():
        print(f"ERROR: source not found: {src}")
        sys.exit(2)
    try:
        if src.is_dir():
            sanitize_readme_in_folder(src)
            print(f"Uploading folder {src} -> {repo_id}:{dest}")
            upload_folder(repo_id=repo_id, folder_path=str(src), path_in_repo=dest, token=token)
        else:
            print(f"Uploading file {src} -> {repo_id}:{dest}")
            api.upload_file(path_or_fileobj=str(src), path_in_repo=dest, repo_id=repo_id, token=token)
    except Exception as exc:
        print("Upload failed:", exc)
        raise

print("All uploads attempted.")
