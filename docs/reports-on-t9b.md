# Báo cáo T9b — nạp model + adapter, và chọn framework

Cập nhật: 2026-10-06. Thay thế nội dung cũ của file này (vốn mô tả 4 phương án xử lý
`checkpointer.adapter_checkpoint` của torchtune). Lý do viết lại: phát hiện ra toàn bộ 4 phương
án đó đều là hệ quả của một lựa chọn framework không còn cần thiết.

## 1. Mục tiêu T9b

Domain-incremental: bắt đầu từ defense EN đã có (`facebook/Meta-SecAlign-8B` = Llama-3.1-8B-Instruct
+ LoRA adapter train trên security preference data EN), train tiếp **chỉ bằng dữ liệu VN**
(`vn_preference_n19000.jsonl`), đo VN_ASR giảm và EN_ASR có bị quên hay không (catastrophic forgetting).

Cần chính xác "cùng một adapter được train tiếp" — không merge, không tạo LoRA mới — đúng nghĩa
domain-incremental của Decision #20.

## 2. Định nạp model + adapter

> Phân biệt rõ: **đường Meta gốc là torchtune** (`tune run lora_dpo_distributed`, xem #2a/#2b
> đối chứng và `secalign_plus_plus.py:93`). TRL là lựa chọn của ta (Decision #5) cho cả T9 lẫn
> T9b — cùng framework với T9 để so sánh được, và đủ `rpo_alpha`/`label_smoothing` như ghi ở
> `docs/reports-on-t9b.md` §3. Dưới đây là hai đường, đường torchtune dựng xong nhưng không dùng
> cho T9b.

### 2a. Đường TRL (`transformers` + `peft` + `trl`) — **chọn**

```python
model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "facebook/Meta-SecAlign-8B", is_trainable=True)  # <- khác T9 1 dòng
model.enable_input_require_grads()
trainer = DPOTrainer(model=model, args=dpo_config, ...)   # bỏ peft_config
```

Ba dòng, tất cả đã có sẵn trong `src/vi_secalign/training/train_dpo.py` (đường này đã chạy thật cho T9).
`peft` đọc `adapter_config.json` của Meta nên rank/alpha/target_modules lấy đúng số thật, không phải khai
báo bằng tay. Checkpoint sau train là adapter PEFT thuần → T10b load bằng
`peft.PeftModel.from_pretrained(base, output_dir)` y hệt mọi checkpoint khác trong project.

### 2b. Đường torchtune (đường cũ, đã dựng xong nhưng không dùng)

```
build lora_llama3_1_8b → FullModelHFCheckpointer.load_checkpoint() (base, qua hf_to_tune)
→ convert file .pt (peft_to_tune_adapter_weights) → patch T9B PATCH: load_state_dict(strict=False)
qua config key `manual_adapter_checkpoint`
```

Ba bước này chỉ tồn tại vì torchtune **không đọc được adapter format PEFT**. Nguyên nhân (đã verify từ
source pinned v0.6.0): `adapter_checkpoint` bị chặn bởi (1) `get_adapter_checkpoint_path()` trả `None`
nếu không `resume_from_checkpoint`; (2) recipe cũng gate riêng `lora_weights_state_dict`; (3) bật
`resume_from_checkpoint` lại đòi file recipe-state mà adapter Meta không có; (4)
`FullModelHFCheckpointer` nạp adapter bằng `safe_torch_load()` **không convert key-name**, còn
`q_proj`'s `lora_B` còn phải permute chiều head do layout RoPE HF/torchtune khác nhau — torchtune chỉ
có hàm xuôi `tune_to_peft_adapter_weights`, code tự ghi chú *"we do NOT have a fn
convert_weights.peft_to_tune"*. Nạp bừa → weights sai key, chạy được, loss hợp lệ, chỉ ra số liệu tệ
(lỗi âm thầm).

## 3. Khác biệt giữa hai framework

| | TRL 0.22.1 | torchtune 0.6.0 |
|---|---|---|
| Nạp adapter PEFT ngoài | `PeftModel.from_pretrained`, native | không hỗ trợ → convert + patch |
| Code phải tự viết | ~3 dòng | convert script + patch recipe + smoke script |
| cDPO (`label_smoothing`) | có | **có** (`DPOLoss(beta, label_smoothing)`) |
| RPO (`rpo_alpha`) | có | không |
| Reference log-prob | adapter-disable forward | adapter-disable forward |
| lr scheduler mặc định | `"linear"` | yaml: cosine |
| Tình trạng upstream | active | "wound down", phải vendor recipe |
| Đã chạy train thật trong project | T9 (xong) | chưa |

**Sửa một hiểu lầm đã ghi trong Decision #5:** lý do chọn TRL được ghi là "torchtune `DPOLoss` không có
`rpo_alpha`/`label_smoothing`". Đọc source thật của `torchtune/rlhf/loss/dpo.py` (tag v0.6.0):
`DPOLoss.__init__(self, beta=0.1, label_smoothing=0.0)` — **có `label_smoothing`**. Lý do grep hồi đó
là grep trong submodule `external/meta_secalign` (chỉ chứa yaml/recipe của Meta), không phải trong thư viện
torchtune, nên kết luận sai. Đúng là torchtune chỉ thiếu `rpo_alpha`; Decision #5 vẫn đúng về kết luận
(chọn TRL) nhưng sai về lý do nửa.

## 4. Xác minh loss & optimizer có khác nhau không

Không đọc source suông — chạy cả hai hàm thật trên cùng batch giả rồi assert:
`tools/verify_dpo_loss_equivalence.py` (CPU, không cần GPU/mạng sau lần tải source đầu).

```
$ python tools/verify_dpo_loss_equivalence.py
[1] tong log-prob: KHOP (max|diff| = 0.000e+00)
[2] loss DPO beta=0.1 label_smoothing=0.0: KHOP (max|diff| = 0.000e+00)
[2] loss DPO beta=0.1 label_smoothing=0.1: KHOP (max|diff| = 0.000e+00)
```

### 4a. Loss — GIỐNG HỆT

| Hạng mục | TRL 0.22.1 | torchtune 0.6.0 | Kết quả |
|---|---|---|---|
| Tổng log-prob | `per_token_logps[:, 1:].sum(-1)` (`dpo_trainer.py:1579`) | `(per_token_log_probs * loss_mask).sum(-1)` (`sequence_processing.py:144`) | giống — cùng SUM, không phải mean |
| Token bị bỏ | `loss_mask` (prompt + padding) | `label_pad_token_id = -100` | giống |
| Công thức | `-(1-ls)·logsigmoid(β·h) - ls·logsigmoid(-β·h)`, `loss_type="sigmoid"` | y hệt dòng 82-85 | giống tuyệt đối |
| `beta` | 0.1 | 0.1 | giống |

Hai điểm từng lo ngại đã được loại trừ bằng số:
- **Sum vs mean**: cả hai đều SUM trên token của response → **thang loss giống nhau**, `lr=1.6e-4` của
  anchor giữ nguyên ý nghĩa khi chuyển framework.
- **Chia `(1 - 2·label_smoothing)`**: chỉ nhánh `loss_type="robust"` của TRL chia
  (`dpo_trainer.py:1071-1073`); nhánh `"sigmoid"` (mặc định) **không chia**. torchtune cũng không chia.

### 4b. Reference log-prob — GIỐNG

Cả hai đều lấy log-prob của chính model đang train với LoRA **bị tắt** (`disable_adapter`), tức base
đóng băng — hàm này không đổi trong suốt train. Khác nhau ở *cách tính*, không phải *giá trị*:
torchtune chạy forward reference mỗi step; TRL dùng `precompute_ref_log_probs=True` chạy một lần rồi
cache (chính là fix memory đã ghi trong `dpo_config.py`). Giá trị thu được bằng nhau.

### 4c. RPO — KHÁC, và đây là khác biệt có ý nghĩa

`rpo_alpha` (thêm `alpha · NLL(chosen)`) **chỉ có ở TRL**. Hệ quả: các nhánh ablation
`dpo_rpo` / `dpo_rpo_cdpo` trong `proposal.md` chỉ chạy được trên TRL. Nhưng arm chính của T9b là **plain
DPO** — và đó cũng đúng phương pháp mà cả hai paper SecAlign/SecAlign++ thực sự dùng (không RPO, không
cDPO, đã đọc trực tiếp). Nên thiếu RPO không chặn T9b; nó chỉ định nghĩa ablation về sau chạy trên
đường TRL, và phải ghi rõ trong bài.

### 4d. Optimizer / scheduler — khác 1 chỗ, sửa được bằng 1 tham số

| | torchtune yaml | TRL/HF `DPOConfig` hiện tại | Xử lý |
|---|---|---|---|
| optimizer | AdamW `fused=True` | AdamW (HF default, không fused) | tương đương về toán học |
| `weight_decay` | 0.0 | 0.0 (HF default) | khớp |
| `learning_rate` | 1.6e-4 | 1.6e-4 (từ `ANCHOR_HYPERPARAMS`) | khớp |
| scheduler | **cosine**, `num_warmup_steps: 0` | **"linear"** (HF default) | **phải sửa:** thêm `lr_scheduler_type="cosine"` |
| `batch_size` / `grad_accum` | 1 / 32 | 1 / 32 | khớp (effective batch 32) |
| `epochs` | 3 | 3 | khớp |

Đây là **sai lệch duy nhất còn lại** giữa hai đường, và nó là cấu hình chứ không phải toán học: sửa bằng
một kwarg trong `build_dpo_config()`.

### 4e. Cắt chuỗi — khác nhẹ, phải ghi khi viết bài

torchtune `max_seq_len=2048` cắt prompt/completion riêng theo dataset impl; TRL `max_length=2048` +
`max_prompt_length=384`, cắt prompt từ trái và completion từ phải. Với prompt VN thường <384 token nên
gần như không khác; chỉ các mẫu dài (theo ghi chú trong `dpo_config.py`, ~99.9th percentile ≈1960 token)
mới bị xử lý khác. Cần nêu effective truncation khi báo cáo, không coi là blocker.

## 5. Quyết định

**Dùng TRL cho T9b.** Căn cứ:
1. Loss và reference log-prob tương đương tuyệt đối (đo bằng số, mục 4a/4b).
2. Optimizer/scheduler lệch đúng một chỗ, sửa bằng `lr_scheduler_type="cosine"`.
3. Nạp adapter Meta gọn 1 dòng — bỏ được `convert` + `patch` + round-trip check.
4. **Không lệch framework với T9**: T9 (joint EN+VN từ đầu) chạy TRL. Nếu T9b chạy torchtune thì so sánh
   T9 vs T9b lệch cả framework lẫn cách pha dữ liệu, không tách được yếu tố nào.
5. RPO/cDPO có sẵn cho ablation, đúng như Decision #5 và `proposal.md`.

**Việc còn phải làm trước khi thuê pod:**
- Thêm `lr_scheduler_type="cosine"` vào `build_dpo_config()`.
- Smoke trên pod: 40 mẫu, `max_steps=3` → xác nhận loss khác 0, không NaN, `rewards/accuracies` hợp lý.
- Ghi rõ trong bài: T9b dùng TRL 0.22.1/peft 0.14.0/transformers 4.57.1, tiếp tục train adapter Meta,
  cosine + warmup 0 + lr 1.6e-4 + effective batch 32 + beta 0.1 + max_length 2048.

**Tư liệu cũ giữ làm fallback:** `tools/pod_setup/convert_peft_adapter_to_torchtune.py`,
`external/meta_secalign/helpers/lora_dpo_single_device_t9b.py`, `tools/pod_setup/merge_meta_adapter.py`,
`tools/pod_setup/smoke_t9b_generate.py` — không xoá (`.agents/record.md` Decision #37 giữ
`merge_meta_adapter.py` làm fallback theo cùng nguyên tắc).

## 6. Trạng thái Phase 0/1 cũ (torchtune) — đã xong, không còn là cổng chặn

- **Phase 0** PASS: `[convert] Round-trip OK -- 320 tensors match exactly.` → cấu trúc adapter thật của
  Meta khớp giả định (`lora_attn_modules=['q_proj','v_proj']`, `rank=64`). File `.pt` đã upload HF.
- **Phase 1** smoke PASS (2026-10-06): adapter đã convert nạp vào base không lỗi (`0 unexpected`),
  prompt injection bị chặn, prompt VN trả lời mạch lạc. Đường torchtune `quantize_base=True` bị kẹt
  trên Colab torch 2.11 (torchao 0.18 đã bỏ `torchao.dtypes.nf4tensor` — đã tái hiện lại khi dựng venv
  ở máy local), nên smoke chạy bằng đường tương đương `transformers + peft + 4-bit`, cùng kỹ thuật T10
  eval dùng trên T4.

Hai kết quả này vẫn có giá trị: chúng xác nhận **adapter Meta đọc được và defense còn nguyên vẹn** —
đúng thứ cần xác nhận trước khi train tiếp, dù đường train là đường nào.