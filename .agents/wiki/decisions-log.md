# Decisions log
Append only. Newest at the bottom. Never edit or delete an entry: to change a decision, add a new entry that supersedes it.

Format:
## YYYY-MM-DD — short title
- Topic: <slug of the part of the system this decides, e.g. auth-method, dataset-split>
- Decision: what we decided, in one clear statement
- Why: the forces behind it
- Who / task: the person and the plan.csv task id
- Supersedes: <date — title of the entry this replaces> (only when it replaces one)
- Result: <run-id · config · metrics · artifact path> (only for the chosen result of an experiment)

## 2026-10-07 — TRL tương đương torchtune về loss, và tồn tại bộ cấu hình khớp 100%
- Topic: trl-torchtune-parity
- Decision: Chấp nhận TRL thay torchtune làm đường train (giữ record.md #39/#40). Mức bằng chứng
  đạt được: (1) loss DPO + reference log-prob **khớp tuyệt đối** — assert từng phần tử trên
  float64, nguồn `tools/verify_dpo_loss_equivalence.py`; (2) tồn tại bộ cấu hình TRL khớp 100%
  với anchor của Meta, chỉ cần **3 kwarg**: `optim="adamw_torch_fused"`,
  `lr_scheduler_type="cosine"`, `max_grad_norm=0.0`. Với 3 kwarg đó TRL khớp Meta trên **toàn bộ**
  siêu tham số train, không chỉ loss. **Không cần** chạy lại T9, **không cần** dựng môi trường
  torchtune chỉ để so parity.
- Why: chỉ có 3 dòng lệch thật, đã đối chiếu từng dòng giữa
  `external/meta_secalign/helpers/llama3.1_8B_lora.yaml` và
  `src/vi_secalign/training/{train_dpo,dpo_config}.py` + `src/vi_secalign/config.py`. Còn lại đã
  khớp sẵn: LoRA r=64/α=8/dropout=0.1, 3 epochs, effective batch 32 (yaml `2×16` vs T9 `1×32` —
  cùng effective, khác chỉ vì OOM ở reference-logprob pass), lr **1.6e-4 effective** (yaml literal
  1e-4 là default, giá trị train thật do `secalign_plus_plus.py --lr` inject — `config.py:20-24`),
  β=0.1, label_smoothing=0.0 (plain DPO), wd=0.0, betas/eps trùng, bf16, grad-ckpt on, max_len
  2048. Chạy lại T9 (~15h pod) cho delta kỳ vọng < nhiễu đo là lãng phí; dựng torchtune (torch
  2.11 không import được vì torchao 0.18 xoá `dtypes/nf4tensor`) tốn nửa–một ngày chỉ để tái lập
  một so sánh mà 3 dòng kwparg đã đóng.
- Who / task: user + T9b
- Ghi chú giới hạn (đừng viết quá claim trong paper): dù khớp config, **data không khớp** — data
  EN của dự án là regenerate (pool `alpaca-cleaned`), không phải file synthetic_alpaca của Meta.
  Claim được phép là *"TRL thay torchtune được trên data của dự án"*, **không phải** *"tái lập
  được Meta"*. Ngoài ra 2 khác biệt nhỏ đã biết: tokenizer (`llama3_tokenizer` vs HF
  `AutoTokenizer`) cần verify byte-identical trước khi so số; số step HF dùng **ceil** → 1797 vs
  torchtune floor → 1794 (0.17%).
- Liên quan: record.md Decision #40 (T9b dùng Optuna, trial đầu = chính bộ config khớp Meta này;
  `lr_scheduler_type` nên nằm trong search space). 3 kwarg ở trên **chưa** được thêm vào
  `build_dpo_config()` — xem open question về dual-run.
