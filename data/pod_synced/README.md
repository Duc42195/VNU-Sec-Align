# data/pod_synced/

Local working copies of files that scripts under `src/vi_secalign/data_gen/` and
`src/vi_secalign/training/` uploaded from the pod via `hf_sync.upload_output()`. Source of truth
is Hugging Face (`Jason-42195/VNU-SecAlign:pod_outputs/<script_name>/...`), not this folder — it is
gitignored (except this README and `.gitkeep`s) and safe to delete/re-sync at any time.

Layout mirrors the HF prefix: `data/pod_synced/<script_name>/<filename>`, e.g.
`data/pod_synced/vi_preference_gen/vn_preference_test200.jsonl`.

To pull a file down:

```python
from vi_secalign.hf_sync import download_output
download_output("vi_preference_gen", "vn_preference_test200.jsonl")
```

or equivalently from the CLI:

```bash
python3 -c "from vi_secalign.hf_sync import download_output; download_output('vi_preference_gen', 'vn_preference_test200.jsonl')"
```

Needs `huggingface_hub` installed and either `HF_TOKEN` set or a cached `huggingface-cli login`
token (`~/.cache/huggingface/token`) — the repo is private.
