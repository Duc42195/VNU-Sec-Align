# T10 — chạy VN_ASR eval trên Colab T4 qua `google-colab-cli`

> Tài liệu thao tác — không phải nguồn sự thật kỹ thuật. Script thật là
> `notebooks/t10_vn_asr_eval.py`.

## 0. Cài + xác thực (chỉ 1 lần/máy)

```bash
uv tool install google-colab-cli
export PATH="$HOME/.local/bin:$PATH"
colab sessions   # sẽ in URL OAuth nếu chưa đăng nhập -- mở URL, đăng nhập Google, dán mã code lại
```

Đã xác thực xong trên máy này (2026-09-30) — nếu mở máy/session mới thì lặp lại bước trên.

## 1. Tạo session GPU T4

```bash
colab new -s t10 --gpu T4
colab status -s t10   # xác nhận Tesla T4 thật, không phải CPU-only
```

## 2. Đăng nhập Hugging Face trên session (model gated)

```bash
echo 'from huggingface_hub import login; login(token="hf_xxx")' | colab exec -s t10
```

Thay `hf_xxx` bằng token HF thật (quyền read, đã accept license Llama-3.1). **Không dùng token cũ
đã lộ trong lịch sử chat trước đây** — dùng token khác hoặc token đã rotate.

## 3. Cài dependency

```bash
echo 'transformers
accelerate
peft
bitsandbytes
huggingface_hub[hf_transfer]' > /tmp/t10_requirements.txt
colab install -s t10 -r /tmp/t10_requirements.txt
```

## 4. Upload data benchmark + script (nhỏ, vài trăm KB tổng)

```bash
cd /home/j/Workspace/VNU/Final
colab exec -s t10 <<< "import os; os.makedirs('data/benchmarks/vi_injecteval', exist_ok=True); os.makedirs('data/benchmarks/cyberseceval2', exist_ok=True); os.makedirs('data/benchmarks/mmlu', exist_ok=True); os.makedirs('data/benchmarks/alpacafarm', exist_ok=True)"

colab upload -s t10 data/benchmarks/vi_injecteval/pilot_v0_1.json data/benchmarks/vi_injecteval/pilot_v0_1.json
colab upload -s t10 data/benchmarks/vi_injecteval/pilot_v0_1_en_matched.json data/benchmarks/vi_injecteval/pilot_v0_1_en_matched.json
colab upload -s t10 data/benchmarks/cyberseceval2/pilot_v0.json data/benchmarks/cyberseceval2/pilot_v0.json
colab upload -s t10 data/benchmarks/mmlu/pilot_v0.json data/benchmarks/mmlu/pilot_v0.json
colab upload -s t10 data/benchmarks/alpacafarm/pilot_v0.json data/benchmarks/alpacafarm/pilot_v0.json
```

## 5. Chạy eval (script tự tải model+LoRA từ HF, sinh + chấm điểm 5 bộ benchmark)

```bash
colab exec -s t10 -f notebooks/t10_vn_asr_eval.py
```

Thời gian ước lượng: model 8B ở 4-bit NF4 (~5-6GB VRAM) + 5 bộ benchmark nhỏ (50+50+30+60+30=220
mẫu, batch 16, max 256 token) — khoảng **10-20 phút** trên T4 (chưa tính thời gian tải model lần
đầu qua mạng, có thể thêm 5-15 phút tuỳ băng thông Colab).

## 6. Tải kết quả về

```bash
colab download -s t10 results/phase3_t10_held_out/t10_metrics.json results/phase3_t10_held_out/t10_metrics.json
colab download -s t10 results/phase3_t10_held_out/t10_raw_outputs.json results/phase3_t10_held_out/t10_raw_outputs.json
```

Script tự in bảng so sánh với baseline GĐ2 (`llama_3_1_8b_instruct`: vn_asr=0.54;
`meta_secalign_8b`: vn_asr=0.10) ngay trong log — xem lại bằng:

```bash
colab log -s t10 -n 50
```

## 7. Dọn dẹp (BẮT BUỘC — tránh tốn quota/compute unit)

```bash
colab stop -s t10
colab sessions   # xác nhận không còn session nào chạy
```

## Lưu ý quan trọng

- **T4 dùng fp16, không phải bf16** cho compute dtype của 4-bit quantization — T4 (compute cap 7.5,
  Turing) không có bf16 tensor core native (đã xác nhận thật lúc setup CLI, xem
  `.agents/record.md`). `t10_vn_asr_eval.py` đã đặt đúng `bnb_4bit_compute_dtype=torch.float16`,
  khớp `phase1_rq1_zero_shot.ipynb` đã dùng cho baseline GĐ2 — không đổi phương pháp đo giữa 2 lần,
  nếu không số liệu không còn so sánh được.
- Script chỉ chạy **1 model mới** (`phase1_5_vi_joint`, checkpoint T9) — KHÔNG chạy lại
  `llama_3_1_8b_instruct`/`meta_secalign_8b` (đã có số liệu GĐ2, tốn thời gian T4 vô ích khi chạy
  lại với cùng phương pháp).
- Nếu `colab exec -f notebooks/t10_vn_asr_eval.py` báo thiếu file `data/benchmarks/...` — kiểm tra
  lại bước 4 đã upload đủ 5 file, đúng đường dẫn tương đối (script chạy với cwd là thư mục gốc
  session, không phải `notebooks/`).
