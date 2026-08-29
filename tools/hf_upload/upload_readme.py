from pathlib import Path
import os
from huggingface_hub import HfApi

api = HfApi()
repo = "Jason-42195/VNU-SecAlign"
readme = Path(__file__).resolve().parent.parent / "README.md"
if not readme.exists():
    print("Local README.md not found", readme)
    raise SystemExit(2)

api.upload_file(path_or_fileobj=str(readme), path_in_repo="README.md", repo_id=repo)
print("README uploaded")
