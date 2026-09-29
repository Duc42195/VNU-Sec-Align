# TODO — T9 (Phase 1.5, joint EN+VN DPO training) — chạy tay từ đây

> File tạm thao tác, không phải nguồn sự thật — xoá sau khi T9 xong hẳn và đã ghi Decision vào
> `.agents/record.md`. Pod hiện tại: `n1.ckey.vn:1211`, RTX 5090 32GB thật, giá 21.818 VND/h.

## Trạng thái hiện tại (lúc viết file này)

- ✅ **VN**: xong hoàn toàn, đúng **19.157/19.157** mẫu → `data/preference/vn_preference_n19000.jsonl`
  (tên file giữ nguyên "n19000" dù thực tế đã top-up lên 19.157 — đừng đổi tên, sẽ mất resume).
  Đã upload HF: `pod_outputs/vi_preference_gen/vn_preference_n19000.jsonl`.
- 🔄 **EN**: đang chạy nền (qua `nohup ... &`, tự chạy tiếp dù bạn ngắt SSH), mục tiêu
  **19.157/19.157** → `data/preference/en_preference_n19157.jsonl`. Chạy `--checkpoint_every 1000`,
  tự upload từng checkpoint lên HF (`pod_outputs/en_preference_gen/`).

## Bước 1 — Kiểm tra EN đã xong chưa

```bash
grep "Checkpointed\|Generated" /root/en_pref_19157.log | tail -5
pgrep -af en_preference_gen   # nếu không ra gì => đã xong (hoặc crash, xem log kỹ nếu vậy)
```

Nếu chưa xong, cứ đợi (chạy nền độc lập, không cần giữ SSH mở) — `tail -f /root/en_pref_19157.log`
để xem trực tiếp. Nếu **crash** (khác "process finished" bình thường), dán log lỗi ra để debug —
đã gặp + fix 2 lỗi hạ tầng thật trong phiên này (xem mục "Lỗi đã biết" cuối file), khả năng cao nếu
lỗi khác thì là lỗi mới.

## Bước 2 — Gộp VN + EN thành 1 file training chung cho T9

`train_dpo.py` chỉ nhận **1** `--preference_data_path` — phải gộp tay trước khi train (T9 = joint
EN+VN, xem `.agents/record.md` Decision #20).

```bash
cd ~/repo && source ~/venv/bin/activate
python3 -c "
import json
vn = json.load(open('data/preference/vn_preference_n19000.jsonl'))
en = json.load(open('data/preference/en_preference_n19157.jsonl'))
print('VN:', len(vn), 'EN:', len(en))
combined = vn + en
json.dump(combined, open('data/preference/t9_joint_en_vn.jsonl', 'w'), indent=2, default=str)
print('Total:', len(combined))
"
```

Kỳ vọng in ra: `VN: 19157 EN: 19157` rồi `Total: 38314`. Nếu số khác — DỪNG, đừng train, báo lại
(có thể 1 trong 2 file chưa sinh xong hẳn).

## Bước 3 — Chạy train_dpo.py cho T9

```bash
export HF_TOKEN=hf_xxx   # token quyền read/write, đã accept license Llama-3/Llama-3.1
export PYTHONPATH=/root/repo/src
export HF_HUB_OFFLINE=1
cd ~/repo
nohup python3 -m vi_secalign.training.train_dpo \
  --variant dpo \
  --base_model /root/models/llama_3_1_8b_instruct \
  --preference_data_path data/preference/t9_joint_en_vn.jsonl \
  --output_dir checkpoints/phase1_5_vi \
  --max_length 2048 \
  > /root/train_t9.log 2>&1 &
disown
sleep 2
pgrep -af train_dpo
```

**2026-09-29, SỬA LẠI (bài học thật)**: bản đầu của file này khuyên dùng HF id
(`meta-llama/Llama-3.1-8B-Instruct`) để tránh bug upload-metadata dưới đây — SAI, gây lỗi chặn hẳn
việc train (`OSError: We couldn't connect to huggingface.co ... couldn't find them in the cached
files`). Lý do: model được tải qua `snapshot_download(..., local_dir=...)`, KHÔNG ghi vào cache
chuẩn `~/.cache/huggingface/hub/` mà `from_pretrained(<HF id>)` tra cứu — nên dùng HF id vẫn phải
gọi mạng thật, và mạng pod lúc đó timeout. Dùng PATH LOCAL (`/root/models/llama_3_1_8b_instruct`)
mới thật sự không cần mạng (đọc thẳng từ đĩa). Đổi lại: chấp nhận phải sửa tay README.md sau khi
train xong (xem mục lỗi `"base_model" with value ...` bên dưới), đơn giản hơn nhiều so với bị chặn
hẳn không train được. `HF_HUB_OFFLINE=1` thêm vào để chặn luôn các lệnh gọi mạng phụ khác của
transformers — không ảnh hưởng bước upload checkpoint (dùng `HfApi` riêng, không bị cờ này chặn).

**Theo dõi**: `tail -f /root/train_t9.log`. Ước lượng thời gian: N=38.314, batch hiệu dụng 32,
3 epoch → `ceil(3×38314/32)` ≈ **3593 step** × 12.31s/step ≈ **12.3h** (đo thật trên chính GPU này ở
N nhỏ hơn, xem `.agents/record.md` Decision #30) + overhead upload checkpoint mỗi `save_steps=200`
(~18 lần, mỗi lần ~2.28GB) ≈ thêm ~1-1.5h. **Kiểm tra ngân sách/thời gian pod trước khi chạy bước
này** — đây là bước tốn nhất, không phải bước rẻ như sinh data.

Checkpoint tự lưu vào `checkpoints/phase1_5_vi/checkpoint-<step>/` mỗi 200 step, tự upload HF
(`pod_outputs/train_dpo/dpo/`) — an toàn nếu pod dừng giữa chừng, dùng
`--resume_from_checkpoint auto` khi chạy lại lệnh trên (thêm vào cuối lệnh) để tiếp tục đúng chỗ.

## Sau khi train xong

1. Kiểm tra log cuối: tìm dòng `train_loss`, `rewards/accuracies` — xác nhận có giảm loss/tăng
   accuracy hợp lý (tham khảo N=200 test: loss 0.417→0.063, accuracies 85.8%→98.6%).
2. Cập nhật `plan.csv` dòng T9: KHÔNG tự set `Status=Done` (theo `.agents/CLAUDE.md` mục 2) — chỉ
   ghi chú kết quả vào cột Ghi chú, để người dùng tự set Done.
3. Ghi Decision mới vào `.agents/record.md` (N thật dùng, throughput/thời gian thật, kết quả
   loss/accuracy) — theo đúng khuôn Context/Decision/Rejected alternatives/Consequences.
4. T10 (đánh giá lại VN_ASR trên held-out, so với GĐ2) là bước tiếp theo, phụ thuộc T9 — chưa làm.

## Lỗi đã biết trong phiên này (nếu gặp lại)

- **`-lcuda: No such file or directory`** khi vLLM/Triton compile — pod chạy trên WSL, thiếu
  symlink `libcuda.so`. Fix: `ln -sf $(find /usr/lib/wsl -iname "libcuda.so.1" | head -1)
  /usr/lib/x86_64-linux-gnu/libcuda.so && ldconfig`. (Chưa gặp trên pod n1.ckey.vn:1211 này.)
- **`ValueError: Loading this dataset requires ... trust_remote_code=True`** hoặc treo vô thời hạn
  khi load Bactrian-X — đã fix trong `vi_preference_gen.py` (commit `19a1e52`), không cần làm gì
  thêm nếu code đã pull bản mới nhất.
- **HF upload checkpoint fail: `"base_model" with value "<path>" is not valid`** — do dùng path
  local làm `--base_model` (Bước 3 trên dùng path local CÓ CHỦ ĐÍCH, để né lỗi mạng nghiêm trọng
  hơn — xem mục "SỬA LẠI" ở Bước 3). Đây là đánh đổi đã chấp nhận, không phải lỗi cần né: khi gặp,
  sửa tay dòng `base_model:` trong README.md của checkpoint thành
  `meta-llama/Llama-3.1-8B-Instruct` rồi gọi lại `vi_secalign.hf_sync.upload_output(...)` tay.
- **`SyntaxError` trong `pod_init.sh` bước tải model** — đã fix (commit `112fabb`), chỉ xảy ra nếu
  chạy bản code cũ hơn commit đó.
