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

## 4. (KHÔNG CẦN NỮA) — bài học thật, 2026-09-30

Bản đầu của manual này bảo `colab upload` 5 file JSON benchmark nhỏ lên VM trước khi chạy — **lỗi
thật**: `colab upload` báo `500 Internal Server Error` cho cả 5 file (nghi do thư mục đích tạo qua
`colab exec <<< "os.makedirs(...)"` không nhận đúng, hoặc bug server-side của chính CLI). Sửa:
`t10_vn_asr_eval.py` giờ **tự tải 5 file pilot từ HF** (đã upload sẵn lên
`Jason-42195/VNU-SecAlign:pod_outputs/benchmarks/...`) ngay khi import — không cần `colab upload`
bất kỳ file data nào nữa, chỉ cần transfer đúng 1 file `.py` ở bước 5 (chính `colab exec -f` đã tự
đọc file local và gửi nội dung qua, không cần upload trước — xem "Transparent Code Execution" trong
README của `google-colab-cli`). Đúng góp ý: chỉ "upload" code (qua `exec -f`, không phải lệnh
`upload`), data tải bằng code (`hf_hub_download`) tại runtime.

## 5. Chạy eval (script tự tải data pilot + model+LoRA từ HF, sinh + chấm điểm 5 bộ benchmark)

```bash
colab exec -s t10 -f notebooks/t10_vn_asr_eval.py
```

Thời gian thật đo được (2026-09-30, lần chạy đầu): **~40 phút** trên T4 (chậm hơn ước lượng ban đầu
10-20 phút — batch generation 256 token/mẫu trên T4 4-bit tốn thời gian hơn dự tính, xem
`.agents/record.md` Decision #35). GPU utilization ~49% khi đang chạy thật — không phải bug/treo,
`colab status -s t10` báo `BUSY` suốt thời gian này là bình thường.

## 6. Kết quả tự upload lên HF — KHÔNG còn phụ thuộc `colab download`

**2026-09-30, SỬA LẠI (bài học nghiêm trọng, xem Decision #35)**: lần chạy đầu, `colab download`
báo lỗi `File or directory not found` (path không khớp cwd thật của session) — và `colab stop`
lỡ chạy ngay sau đó trong cùng 1 loạt lệnh TRƯỚC KHI phát hiện lỗi. Session (đĩa ephemeral) bị xoá
→ **mất hẳn `t10_raw_outputs.json`** (chỉ `t10_metrics.json` cứu được vì đã lỡ in ra màn hình
trước đó qua `colab exec` riêng). Đã sửa `t10_vn_asr_eval.py`: script giờ **tự upload cả 2 file
kết quả lên HF** (`Jason-42195/VNU-SecAlign:pod_outputs/t10_held_out_eval/`) ngay trong `main()`,
KHÔNG cần `colab download` nữa — chỉ cần chờ script chạy xong (status IDLE) rồi `colab stop` an
toàn.

**Nếu vẫn muốn có bản local ngay** (không bắt buộc, dữ liệu đã an toàn trên HF):
```bash
colab download -s t10 results/phase3_t10_held_out/t10_metrics.json results/phase3_t10_held_out/t10_metrics.json
colab download -s t10 results/phase3_t10_held_out/t10_raw_outputs.json results/phase3_t10_held_out/t10_raw_outputs.json
```
**QUY TẮC BẮT BUỘC nếu làm bước này**: xác nhận CẢ HAI lệnh trên in ra `Downloaded ...` (không có
chữ `failed`) TRƯỚC KHI chạy `colab stop` — không bao giờ gộp download + stop vào cùng 1 loạt lệnh
không kiểm tra kết quả giữa chừng.

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
