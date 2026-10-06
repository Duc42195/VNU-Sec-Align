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

## 2026-10-06 — Optimizer mặc định của HF Trainer KHÔNG khớp Meta (cosine→linear, no-clip→clip 1.0)
- Symptom: T9 (TRL) có vẻ "không sát Meta" khi viết so sánh optimizer, trong khi dùng chung LoRA config.
- Root cause: mặc định HF Trainer là `AdamW non-fused`, `lr_scheduler_type="linear"`, `max_grad_norm=1.0`; Meta (torchtune yaml `llama3.1_8B_lora.yaml:64-81`) là `AdamW fused=True, wd 0.0, cosine(warmup 0), clip_grad_norm: null`. Paper 2507.02735v3 không nêu optimizer/scheduler/clipping → phải đọc từ yaml trong repo.
- Fix: `DPOConfig(optim="adamw_torch_fused", lr_scheduler_type="cosine", max_grad_norm=0.0)` khớp 100% Meta; loss đã chứng minh khớp tuyệt đối bằng `tools/verify_dpo_loss_equivalence.py`. Chỉ còn lệch tổng step: HF `trainer.py:5682` ceil → 1797 vs torchtune floor → 1794 (0.17%, chấp nhận được).
- Lesson: "framework khác" không đồng nghĩa "loss khác" — TRL nhánh sigmoid DPO là cùng công thức với torchtune `DPOLoss`; phần lệch nằm ở default optimizer/scheduler. Verify bằng số (assert từng phần tử trong float64), đừng chỉ grep.

## 2026-10-07 — torchtune 0.6.0 không pin torchao → torchao 0.18 xoá `dtypes/nf4tensor` làm vỡ cả `import torchtune`
- Symptom: `import torchtune` chết ngay với `ModuleNotFoundError: No module named 'torchao.dtypes.nf4tensor'`, dù đã cài torchao 0.18.0 mới nhất. Trên Colab cùng bản lại treo trong `quantize_tensor_nearest` thay vì raise.
- Root cause: torchtune 0.6.0 import cứng `from torchao.dtypes.nf4tensor import linear_nf4, to_nf4` (`torchtune/modules/low_precision/nf4tensor.py:15`) và `NF4Tensor` (`common_utils.py:19`), nhưng `pyproject.toml` **không pin torchao** — chỉ check "có torchao không". torchao đã xoá module đó ở v0.18 (đã dò: v0.11–v0.13 và v0.15–v0.17 còn, v0.14 không tồn tại, v0.18 không còn).
- Fix: dùng đúng bộ Meta đã pin — `torch==2.8.0` + `torchao==0.11.0` + `torchtune==0.6.0` (`external/meta_secalign/requirements.txt:230,232`). Cài torchao mới không cứu được; phải khớp cả torch vì torchao đóng gói theo version torch.
- Lesson: thư viện không pin dependency không phải lỗi của ta — nhưng đừng kết luận "torchtune không làm được 4-bit". Nó làm được, chỉ là cần bộ version đúng. Và khi một thư viện con báo ImportError ở local nhưng treo ở nơi khác, đó là **hai version khác nhau** — đừng gộp thành một kết luận.
