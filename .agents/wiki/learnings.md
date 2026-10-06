# Learnings
Append only. Every error fixed becomes an entry so nobody hits it twice.

Format:
## YYYY-MM-DD — short title
- Symptom:
- Root cause:
- Fix:
- Lesson:

## 2026-10-05 — Colab T4 (14.5GB) không chạy được smoke test DPO Llama-3.1-8B bf16
- Symptom: `OutOfMemoryError` ngay lúc `LoRA` model instantiate (`nn.Linear` lm_head, 1002 MiB alloc, GPU 14.56 GiB full).
- Root cause: base bf16 ~16GB > VRAM T4; smoke manual Phase 1 giả định T4 đủ cho cả model + DPO reference forward.
- Fix: smoke chuyển sang pod RTX ≥32GB (Phase 2) hoặc Colab Pro GPU ≥24GB; không còn đường T4 miễn phí.
- Lesson: smoke test phải tính trước footprint tối thiểu (base + ref + optimizer + activation), không chọn GPU chỉ dựa vào nhãn "miễn phí".

## 2026-10-06 — `torchtune 0.6.0` build model `quantize_base=True` bị kẹt trên Colab torch 2.11
- Symptom: `smoke_t9b_generate.py` chạy >10 phút không in gì, CPU 99%, VRAM 3MiB (GPU không dùng), log trống tới tận dòng đầu tiên.
- Root cause: `lora_llama3_1_8b(quantize_base=True)` của torchtune 0.6.0 gọi `torchao.dtypes.nf4tensor.quantize_tensor_nearest` — với torchao 0.18 + torch 2.11, vòng lượng tử hóa NF4 này chạy trên tensor CPU trong build model (device placement lệch), gần như không bao giờ xong.
- Fix: smoke Phase 1 chuyển sang đường `transformers + peft + bitsandbytes 4-bit NF4` (cùng kỹ thuật T10 eval dùng trên T4) — pass: adapter loaded ok, injection bị chặn, VN benign trả lời bình thường.
- Lesson: khi một bước "quantization build" CPU-bound kéo dài bất thường mà VRAM không tăng, nghi device placement của weights trước khi đổ lỗi cho model/data.

## 2026-10-06 — `colab exec` mặc định timeout 30s → mọi bước nặng đều "Connection was lost"
- Symptom: `tune download`, `pip install`, smoke generate trên Colab đều fail với `RuntimeError: Connection was lost` dù session vẫn READY.
- Root cause: `colab exec` (typer) mặc định `--timeout 30.0` giây; websocket bị cắt khi quá hạn → traceback `TimeoutError: Timeout waiting for reply` xuất hiện khi tune download 16GB hoặc pip install torch.
- Fix: truyền `--timeout` lớn cho mọi `colab exec`/`colab install`, hoặc chạy detached trên VM (`python -u script.py > /content/x.log 2>&1 &` rồi poll file) khi bước có thể mất >20 phút; session T4 miễn phí hay bị prune → mọi thứ trên `/content` mất theo, luôn log ra file + copy code lên HF để tái tạo nhanh.
