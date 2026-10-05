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
