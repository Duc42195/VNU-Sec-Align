# Infra handoff — phiên thuê GPU đang chạy (ckey.vn)

> File tạm, KHÔNG phải Decision log của record.md — chỉ ghi trạng thái hạ tầng/tiến độ setup
> để mở session Claude Code khác vẫn tiếp tục được ngay. Xoá file này sau khi setup xong hẳn
> và đã bắt đầu chạy T1-T3 formal (lúc đó thông tin ở đây hết giá trị).

## Máy đã thuê

- Nhà cung cấp: **ckey.vn**, listing #105728 (RTX 3090 24GB, 6 core i5-8600T, ~15GB RAM, uptime 96.88%)
- SSH: `ssh -i ~/.ssh/ckey_vast_ai_ed25519 -o IdentitiesOnly=yes -p 1298 root@n1.ckey.vn`
- **Không cần mật khẩu** — khoá SSH đã tạo và authorize xong (public key đã append vào
  `~/.ssh/authorized_keys` trên máy thuê). Private key nằm ở `~/.ssh/ckey_vast_ai_ed25519` trên
  chính máy local này (`/home/j`) — session Claude Code khác chạy trên cùng máy local vẫn dùng
  được ngay, không cần tạo lại khoá.
- Disk: `overlay /` — 73GB tổng, ~50GB trống ngay sau khi trừ OS gốc. **Sát mép** — không cache
  2 model base cùng lúc (chỉ giữ 1 trong 2: Llama-3.1-8B HOẶC SeaLLM tại 1 thời điểm).
- Ngân sách: đã nạp 50.000 VND để test (≈6.5h ở giá ~7.700đ/h) — không phải để chạy train thật,
  chỉ đủ cho setup + T1-T3 formal sanity check.
- **Giới hạn thuê tối đa 24h/lượt** (ckey.vn) — không thể thuê liên tục nhiều ngày trong 1 lượt.
  Mọi việc tốn nhiều giờ (T8 sinh dữ liệu VN, T9/T9b train) PHẢI resumable qua ranh giới lượt thuê:
  dừng trước 24h, lưu tiến độ, upload lên HF, thuê lượt mới, tải về, chạy tiếp. Đã vá code cho việc
  này (xem `.agents/record.md` Decision #21): `vi_preference_gen.py` sinh theo chunk + tự resume từ
  file output cũ; `train_dpo.py`/`dpo_config.py` có `--resume_from_checkpoint`/`save_steps`. Quy
  trình upload/download thủ công dùng `tools/hf_upload/*.py`, chưa tự động hoá thành 1 script.

## Đã làm xong (trên máy thuê)

1. Cài `python3` (3.10.12) + `pip3` + `python3-venv` qua apt (máy gốc không có sẵn — image
   "Ubuntu 22.04" trần, không phải "PyTorch" như dự định chọn, hoặc lựa chọn template không có
   tác dụng — cần lưu ý kiểm tra lại template lúc thuê máy tiếp theo).
2. Tạo venv tại `~/venv` trên máy thuê, đã upgrade pip.
3. Clone repo: `~/repo` — **lưu ý: nhánh trên GitHub tên là `main`, không phải `clean-main`**
   (`clean-main` chỉ là tên nhánh cục bộ trên máy local của user, đã push lên remote `main`).
   Submodule `external/meta_secalign` đã init xong.
4. **(2026-09-22) `sep_reference_gen.py` chạy xong full 9160/9160 mẫu SEP** — log:
   `results/pod_logs/sep_gen.txt`. Đây là bước bắt buộc trước T1-T3's `run_sep()` (Meta không ship
   sẵn reference output, tự sinh bằng `meta-llama/Meta-Llama-3-8B-Instruct` qua vLLM, đúng
   `external/meta_secalign/setup.py:565-604`).
   - **Throughput thật đo được** (dùng để ước lượng N cho T8, xem record.md Decision #21/#22):
     model 8B qua vLLM, KV-cache-bound (~4.97GiB khả dụng sau khi load model → concurrency ~5x),
     **3.23 prompt/s** sinh thuần (9160 prompt / 47 phút 14 giây); cộng ~5 phút load+init model lần
     đầu (phần lớn là tải weight, cache lại thì nhanh hơn nhiều các lần sau) → tổng ~52-53 phút cho
     9160 prompt (1 generation/prompt).
   - Ngoại suy ban đầu cho `vi_preference_gen.py` từ số SEP này (~3.23 prompt/s, khác script/corpus)
     **đã bị thay bằng số đo THẬT trên chính script đó** — xem `record.md` Decision #23: chạy thật
     `--n_samples 200` cho kết quả **2.351 samples/s** (200 mẫu / 85.1s, log
     `results/pod_logs/vi_preference.txt`). Dùng số này (không phải số ngoại suy từ SEP) để ước
     lượng N cuối cho T9: N=12.5K (giữa khoảng 10-15K đã chốt, Decision #21) ≈ 12500/2.351 ≈
     **~1h29m sinh thuần** — dư dả trong giới hạn 24h/lượt của pod.

## Đang chạy / cần kiểm tra lại khi mở session mới

- **(2026-09-22, cập nhật)** `pip install torch` cũ (pid 1605) hoá ra vẫn còn sống khi phiên này
  mở lại — mạng máy thuê tới PyPI/download.pytorch.org rất chậm (~250-290 KB/s đo trực tiếp,
  THẤP HƠN NHIỀU so với spec 28-69 Mbps của listing — băng thông chung tới Cloudflare đo được
  ~1.3MB/s/10Mbps thì ổn, nên nghẽn có vẻ riêng ở route/CDN PyPI, không phải mạng máy thuê hỏng).
  Đã viết `~/remote_setup.sh` trên máy thuê (nội dung: chờ pid 1605 xong → xác nhận torch import
  → cài `transformers/peft/trl/bitsandbytes/accelerate` → kiểm tra disk (dừng nếu <15GB trống,
  KHÔNG tự cài vllm nếu sát disk) → cài `vllm` nếu disk đủ), chạy nền qua
  `nohup bash ~/remote_setup.sh > ~/setup.log 2>&1 & disown` — sống độc lập với phiên SSH, sập
  kết nối local không ảnh hưởng. **Việc đầu tiên khi mở session mới**: kiểm tra log:
  ```bash
  ssh -i ~/.ssh/ckey_vast_ai_ed25519 -o IdentitiesOnly=yes -p 1298 root@n1.ckey.vn "cat ~/setup.log"
  ```
  Tìm dòng cuối: `SETUP_DONE_FULL` (xong hết kể cả vllm), `SETUP_DONE_PARTIAL` (xong tới
  transformers/peft/trl/bitsandbytes nhưng dừng trước vllm vì disk <15GB — cần quyết định thủ
  công: dọn bớt hay cài vllm liều), hoặc `SETUP_FAILED: ...` (dừng giữa chừng, đọc log để biết
  bước nào lỗi).

## Việc còn lại theo đúng thứ tự

1. Đọc `~/setup.log` (lệnh trên) — nếu chưa có dòng `SETUP_DONE_*`/`SETUP_FAILED` nghĩa là vẫn
   đang chạy (mạng chậm, có thể mất 30-60+ phút cho riêng bước torch) — chờ thêm, đừng chạy lại
   `pip install torch` đè lên (lãng phí phần đã tải).
2. Nếu `SETUP_DONE_PARTIAL` (thiếu vllm do disk): quyết định — dọn bớt (vd. xoá cache pip
   `~/venv/.cache` hoặc `pip cache purge`) rồi tự chạy `pip install vllm`, hoặc chấp nhận dùng
   `transformers`+`peft` thuần cho T1-T3 (giống notebook go/no-go, chỉ là "chính thức hoá" qua
   `meta_bridge.py` không nhất thiết cần vllm nếu N nhỏ) — **cần quyết định, không tự chọn**.
3. `huggingface-cli login` hoặc `export HF_TOKEN=...` trên máy thuê — **bắt buộc trước khi tải
   model**, vì `Llama-3.1-8B-Instruct` và `Meta-SecAlign-8B` đều là gated model trên HF.
4. Chạy chính thức **T1-T3** (`plan.csv`, GĐ1) qua `meta_bridge.py`/vLLM thật — không phải
   notebook Colab nhẹ đã dùng cho go/no-go (Decision #11). Sau khi pass, commit theo quy ước
   `.agents/CLAUDE.md` mục 3: `[T3] done: sanity check khớp paper gốc` (hook tự đẩy Status →
   "In progress", KHÔNG tự set Done).
5. Dừng lại sau T1-T3 nếu ngân sách 50k gần hết — đây là điểm dừng hợp lý cho đợt test này.
   GĐ3 (Phase 1.5) cần nạp thêm. **T6 (go/no-go) người dùng đã xác nhận bằng lời "chốt go"
   (2026-09-21) — xem record.md Decision #18/#19 — nhưng Status trong `plan.csv` vẫn "In progress"
   vì chỉ người dùng/hook mới được set Done (CLAUDE.md mục 2); không phải lý do để dừng lại ở GĐ3.**

## Quyết định liên quan (đã ghi ở record.md, không lặp lại chi tiết ở đây)

- Decision #13: hạ tầng thuê GPU theo giờ, không mua Colab Pro.
- Decision #18: RQ1 có tín hiệu sơ bộ, khuyến nghị "Go" cho GĐ3 — chờ user xác nhận T6.
