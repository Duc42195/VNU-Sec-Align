from huggingface_hub import HfApi
api = HfApi()
repo = "Jason-42195/VNU-SecAlign"
files = api.list_repo_files(repo_id=repo)
print(f"Total files: {len(files)}")
for f in files[:200]:
    print(f)
