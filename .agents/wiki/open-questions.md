# Open questions
Add when a question opens. Change its state when it moves. Move to "Closed" with the answer; do not delete.

## Open
- YYYY-MM-DD — question — owner

## Closed
- YYYY-MM-DD — question — answer (link to decision/ADR)

## Open
- 2026-10-07 — Dual-run TRL vs torchtune cùng 2000 mẫu có còn làm để có bằng chứng paper, hay chỉ
  sanity nội bộ (3 kwarg + test code đã đủ cho T9b)? — user. Ràng kỹ thuật nếu làm: N phải đủ dài
  để scheduler kịm phân kỳ (2000 mẫu ≈ 188 step, T9 thật ≈ 3600 step — quá ngắn, cho kết quả rỗng),
  phải so **eval ASR** chứ không chỉ loss (loss chỉ là proxy), phải định bar pass/fail trước
  (vd ΔASR cuối < 3pp) vì bf16 + kernel khác nhau khiến trajectory rời nhau theo thời gian, và phải
  dùng **một** file jsonl đã format chung cho cả hai bên để không so pipeline data thay vì framework.
- 2026-10-07 — 3 kwarg (`optim`/`lr_scheduler_type`/`max_grad_norm`) có sửa luôn trong
  `build_dpo_config()` hay để Optuna gán ở runtime? — user/T9b.
