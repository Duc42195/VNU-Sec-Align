# T9b — domain-incremental VN-only train qua torchtune (tự chạy)

> Thực thi kế hoạch `/home/j/.claude/plans/radiant-sprouting-firefly.md`. **Đã đổi lại 1 lần nữa**
> (2026-10-05, xem `.agents/record.md` Decision #36/#37): không merge adapter Meta vào base nữa —
> giờ **tiếp tục train ĐÚNG adapter đó**. torchtune không hỗ trợ nạp adapter PEFT ngoài (3 lớp chặn
> + không có hàm convert key-name/permute ngược), nên: (1) tự viết + tự kiểm tra (round-trip) hàm
> convert PEFT→torchtune, (2) patch 1 điểm nhỏ trong 1 bản copy recipe để nạp adapter đã convert
> trực tiếp vào model, né hẳn cơ chế `adapter_checkpoint`/`resume_from_checkpoint` bị chặn của
> torchtune.

## Phase 0 — Convert + tự-kiểm-tra OFFLINE (BẮT BUỘC, chặn trước mọi bước tốn GPU)

Chạy trên máy có `HF_TOKEN` (quyền truy cập `facebook/Meta-SecAlign-8B`, model gated) — không cần
GPU, không cần Colab, chỉ cần `torch torchtune==0.6.0 safetensors huggingface_hub` cài local.

```bash
pip install torch torchtune==0.6.0 safetensors "huggingface_hub[hf_transfer]"
export HF_TOKEN=hf_xxx   # KHÔNG dùng token cũ đã lộ
python tools/pod_setup/convert_peft_adapter_to_torchtune.py \
  --output_path checkpoints/meta_secalign_8b_adapter_torchtune.pt
```

**Phải thấy chính xác dòng**: `[convert] Round-trip OK -- N tensors match exactly.` TRƯỚC KHI đi
tiếp bất kỳ bước nào khác. Nếu script báo `AssertionError` ở round-trip (key hoặc value mismatch) —
**DỪNG**, không tự sửa số rồi thử lại mù quáng — báo lại để xem xét logic convert, đây đúng là
loại lỗi (permute sai chiều) có thể làm hỏng model âm thầm nếu bỏ qua.

Script tự upload file `.pt` kết quả lên HF (`pod_outputs/convert_peft_adapter_to_torchtune/`) —
Phase 1/2 dưới có thể tải lại file này, không cần convert lại mỗi lần (round-trip check không đổi
kết quả giữa các lần chạy, chỉ cần làm 1 lần).

## Phase 1 — Smoke-test RẺ trên Colab T4 (miễn phí) — chỉ làm nếu Phase 0 PASS

### 1. Tạo session

```bash
colab new -s t9b_smoke --gpu T4
colab status -s t9b_smoke   # xác nhận Tesla T4 thật
```

### 2. Cài dependency

```bash
echo 'torch
transformers
accelerate
torchtune==0.6.0
huggingface_hub[hf_transfer]' > /tmp/t9b_requirements.txt
colab install -s t9b_smoke -r /tmp/t9b_requirements.txt
```

### 3. Đăng nhập HF trên session

```bash
echo 'from huggingface_hub import login; login(token="hf_xxx")' | colab exec -s t9b_smoke
```

### 4. Tải base model THUẦN qua `tune download` (KHÔNG dùng pod_init.sh's download — thiếu `original/tokenizer.model`)

```bash
echo '
import os
os.system(
    "tune download meta-llama/Llama-3.1-8B-Instruct "
    "--output-dir /content/llama3.1_8b_instruct "
    "--ignore-patterns original/consolidated.00.pth"
)
' | colab exec -s t9b_smoke
```

`--ignore-patterns original/consolidated.00.pth` chỉ loại bỏ checkpoint gốc 16GB không cần (đã có
bản HF-format safetensors) — **giữ lại** `original/tokenizer.model` (torchtune's llama3_tokenizer
cần đúng file này, khác `pod_init.sh`'s `ignore_patterns=['original/*']` loại bỏ cả thư mục).

### 5. Tải adapter đã convert từ Phase 0 (không convert lại)

```bash
echo '
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="Jason-42195/VNU-SecAlign",
                        filename="pod_outputs/convert_peft_adapter_to_torchtune/meta_secalign_8b_adapter_torchtune.pt")
print("adapter tại:", path)
' | colab exec -s t9b_smoke
```

### 6. Patch dataset loader của torchtune (bắt buộc, không đổi từ bản trước)

```bash
colab exec -s t9b_smoke -f tools/pod_setup/apply_torchtune_preference_patch.py
```

### 7. Cắt subset nhỏ từ data VN thật để smoke-test (KHÔNG phải kết quả nghiên cứu, chỉ test cơ chế)

```bash
echo '
from huggingface_hub import hf_hub_download
import json
path = hf_hub_download(repo_id="Jason-42195/VNU-SecAlign",
                        filename="pod_outputs/vi_preference_gen/vn_preference_n19000.jsonl")
data = json.load(open(path, encoding="utf-8"))
json.dump(data[:40], open("/content/vn_preference_smoke40.json", "w"), ensure_ascii=False)
print(len(data), "-> smoke subset 40 ghi ra /content/vn_preference_smoke40.json")
' | colab exec -s t9b_smoke
```

### 8. Ghi yaml lên session, rồi chạy recipe đã patch — cả 2 qua `colab exec -f` (mỗi lệnh gửi đúng 1 file)

```bash
# (1) Ghi yaml ra /content/t9b.yaml (write_t9b_config.py nhúng sẵn nội dung yaml, cùng pattern
#     apply_torchtune_preference_patch.py đã dùng cho _preference.py)
colab exec -s t9b_smoke -f tools/pod_setup/write_t9b_config.py -- /content/t9b.yaml

# (2) Chạy recipe đã patch (tự chứa toàn bộ logic, chạy trực tiếp bằng python -- không phụ thuộc
#     `tune run`'s dotpath resolution, né rủi ro import module trên máy lạ)
colab exec -s t9b_smoke -f external/meta_secalign/helpers/lora_dpo_single_device_t9b.py -- \
  --config /content/t9b.yaml \
  cache_dir=/content/llama3.1_8b_instruct \
  manual_adapter_checkpoint=<path in từ bước 5> \
  output_dir=/content/t9b_smoke_out \
  dataset.data_files=/content/vn_preference_smoke40.json \
  epochs=1 max_steps_per_epoch=3
```

Không dùng `colab upload` cho yaml (lỗi 500 với file nhỏ — xem `T10_MANUAL.md` mục 4). Nếu
`write_t9b_config.py`'s nội dung nhúng bị lệch so với file yaml thật trong repo (sửa 1 chỗ mà quên
chỗ kia) — luôn coi `external/meta_secalign/helpers/llama3.1_8B_lora_t9b_single_device.yaml` là
nguồn thật, đồng bộ lại `write_t9b_config.py` nếu sửa yaml sau này.

**Tiêu chí PASS**: log in đúng dòng `[T9B PATCH] Loaded manual_adapter_checkpoint ... 0 unexpected`
VÀ vài step chạy xong với loss hợp lệ (không NaN/Inf), không traceback. Nếu `unexpected > 0` —
nghĩa là shape/tên key không khớp cấu trúc LoRA khai báo trong yaml (`lora_attn_modules`/
`apply_lora_to_mlp`) — kiểm tra lại `adapter_config.json` thật của Meta (cần so khớp
`lora_attn_modules`/`lora_rank`/`lora_alpha`/`lora_dropout` đúng số thật, không chỉ tin giả định).

```bash
colab exec -s t9b_smoke -- ls -la /content/t9b_smoke_out
colab stop -s t9b_smoke
colab sessions   # xác nhận không còn session chạy
```

## Phase 2 — Chỉ sau khi Phase 1 PASS: thuê GPU thật, chạy N=19.157 VN thật

### 1. Thuê pod (RTX ≥32GB như pod T9, áp checklist `infra_handoff.md`)

### 2. SSH vào, cài dependency

```bash
pip install torch transformers accelerate "torchtune==0.6.0" "huggingface_hub[hf_transfer]"
huggingface-cli login --token hf_xxx
```

### 3. Tải base model + adapter đã convert (như Phase 1 bước 4-5, chạy trực tiếp qua SSH thay vì `colab exec`)

```bash
tune download meta-llama/Llama-3.1-8B-Instruct \
  --output-dir checkpoints/llama3.1_8b_instruct \
  --ignore-patterns original/consolidated.00.pth
python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download(repo_id='Jason-42195/VNU-SecAlign',
      filename='pod_outputs/convert_peft_adapter_to_torchtune/meta_secalign_8b_adapter_torchtune.pt'))
"
```

### 4. Patch dataset loader

```bash
python tools/pod_setup/apply_torchtune_preference_patch.py
```

### 5. Chạy N thật đầy đủ (không giới hạn step, không subset)

```bash
tune run external/meta_secalign/helpers/lora_dpo_single_device_t9b.py \
  --config external/meta_secalign/helpers/llama3.1_8B_lora_t9b_single_device.yaml \
  cache_dir=checkpoints/llama3.1_8b_instruct \
  manual_adapter_checkpoint=<path tải ở bước 3> \
  output_dir=checkpoints/phase1_5_vi_incremental \
  dataset.data_files=data/pod_synced/vi_preference_gen/vn_preference_n19000.jsonl
```

Nếu `tune run <path>.py` không import được (lỗi dotpath) — fallback:

```bash
python external/meta_secalign/helpers/lora_dpo_single_device_t9b.py --config ... <cùng override>
```

Ước lượng thời gian: ~7h40m (suy ra tuyến tính từ T9: 15h25m × 19157/38314) — **ước lượng trung
tâm, chưa đo thật torchtune**. Nạp tiền dư cho ~12h để an toàn.

### 6. Upload checkpoint lên HF ngay khi xong — KHÔNG lặp lại lỗi mất checkpoint Decision #33/#35

```bash
python -c "
from vi_secalign.hf_sync import upload_output
upload_output('checkpoints/phase1_5_vi_incremental', dest_subdir='train_dpo_t9b')
"
```

Xác nhận log in ra đường dẫn HF thật (không phải `None`) TRƯỚC KHI xoá pod.

### 7. Đánh giá — ĐƠN GIẢN HƠN bản merge trước (không cần suy đoán cấu trúc)

torchtune's `save_checkpoint(..., adapter_only=True)` tự convert output sang **đúng format PEFT
chuẩn** (`tune_to_peft_adapter_weights`/`tune_to_peft_adapter_config`, có sẵn trong torchtune, đã
tin tưởng — xem `_checkpointer.py`) — `output_dir` sau khi train xong là 1 adapter PEFT bình
thường (`adapter_model.safetensors` + `adapter_config.json`). Vì ta nạp thẳng weight Meta vào CÙNG
module LoRA rồi tiếp tục train (không tạo thêm adapter thứ 2), checkpoint cuối **CHÍNH LÀ** adapter
đã học thêm VN, load được trực tiếp qua `peft.PeftModel.from_pretrained(base, output_dir)` như mọi
checkpoint khác trong project — không cần merge lại/load 2 adapter tuần tự. T10b (`plan.csv`) chỉ
cần thêm 1 `MODEL_SPECS` entry trỏ base gốc + adapter này, tái dùng y nguyên cách
`t10_vn_asr_eval.py` đã load `meta_secalign_8b`.
