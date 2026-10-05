# Báo cáo vấn đề & các phương án xử lý — T9b (tiếp tục train defense của Meta trên dữ liệu VN)

Ngày lập báo cáo: 2026-10-05. Nguồn gốc quyết định: `.agents/record.md` Decision #36, #37 (#37 thay thế #36).

## 1. Bối cảnh và vấn đề cụ thể

**Mục tiêu T9b:** domain-incremental fine-tuning — bắt đầu từ defense của Meta
(`facebook/Meta-SecAlign-8B`, thực chất là Llama-3.1-8B-Instruct + một LoRA adapter train
trên security preference data EN), train tiếp chỉ bằng dữ liệu preference tiếng Việt
(`vn_preference_n19000.jsonl`), đo VN_ASR giảm và EN_ASR không bị quên (catastrophic forgetting).

**Plan ban đầu** định nạp trực tiếp adapter PEFT của Meta làm điểm khởi đầu LoRA qua field
`checkpointer.adapter_checkpoint` trong yaml torchtune 0.6.0 — không cần merge, không cần
convert. Khi viết code thật (session sau), verify lại trực tiếp bằng source pinned
(`raw.githubusercontent.com/pytorch/torchtune/v0.6.0/...`) thì phát hiện giả định này **sai**.
Cụ thể có 4 lớp chặn:

1. **Recipe gating:** `lora_dpo_single_device.py::setup()` chỉ truyền
   `checkpoint_dict[training.ADAPTER_KEY]` vào `_setup_model(lora_weights_state_dict=...)`
   khi `resume_from_checkpoint=True`. Đặt `False` → adapter bị **bỏ qua âm thầm** (không lỗi,
   không warning), model train với LoRA random init.
2. **`resume_from_checkpoint=True` kéo theo `should_load_recipe_state=True`:** checkpointer đòi
   file `recipe_checkpoint` (optimizer state, epoch...) mà adapter PEFT của Meta không có →
   không chạy được.
3. **`get_adapter_checkpoint_path()` (`orchestration/_utils.py:445`) trả `None` ngay từ đầu**
   nếu `should_load_recipe_state=False`, **bất kể** `adapter_checkpoint` có khai báo trong yaml —
   nên kể cả vá dòng ternary ở recipe cũng không đủ.
4. **Không có hàm convert ngược:** `FullModelHFCheckpointer.load_checkpoint()` nạp
   `adapter_checkpoint` bằng `safe_torch_load()` rồi dùng **nguyên**, không convert key-name
   nào. Adapter Meta ở format PEFT (key `lora_A`/`lora_B`, phân biệt hoa/thường) và còn bị
   permute chiều head cho `q_proj`/`k_proj` do RoPE layout khác nhau giữa HF và torchtune.
   torchtune chỉ có `tune_to_peft_adapter_weights` (export), code tự ghi chú: *"we do NOT have
   a fn convert_weights.peft_to_tune"*. Nạp bừa → weights nằm sai key, attention projection bị
   xáo trộn **âm thầm** (chạy được, loss hợp lý, không crash, chỉ ra số liệu tệ khó giải thích).

Kết luận chung: torchtune 0.6.0 không có đường cấu hình yaml thuần nào để "load adapter PEFT
ngoài làm init, train như mới". `adapter_checkpoint` chỉ dành cho việc resume đúng checkpoint
do chính torchtune sinh ra.

## 2. Các phương án đã tìm hiểu

### Phương án A — Merge adapter Meta vào base, train LoRA mới (Decision #36, bị thay thế)

**Cách hoạt động:**
- `tools/pod_setup/merge_meta_adapter.py`: `peft.PeftModel.from_pretrained(base, "facebook/Meta-SecAlign-8B").merge_and_unload()` → lưu full-weight `.safetensors`.
- Train một LoRA **mới** (rank 64, random init) trên bản merged bằng đường LoRA DPO single-device tiêu chuẩn — không cần `adapter_checkpoint`/`resume_from_checkpoint`.

**Đóng góp / ý nghĩa:**
- EN-delta bị "đóng băng" (bake vào base, không trainable) → không có rủi ro ghi đè trực tiếp, an toàn về EN forgetting; toàn bộ cập nhật VN nằm ở LoRA riêng.
- Nhược: không còn là "continue-train đúng adapter Meta" theo nghĩa đen — điểm phương pháp luận
  của Decision #20 bị lệch (vẫn là domain-incremental hợp lệ, nhưng cần công bố đúng biến thể,
  ghi Limitations, không gọi chung chung "continue-train").

### Phương án B — Tạo file `recipe_checkpoint` giả để lách điều kiện resume (bị loại)

**Cách hoạt động:** tự điền `EPOCHS_KEY=0`, `OPT_KEY={}`... vào file recipe-state để torchtune
chấp nhận `resume_from_checkpoint=True`, rồi khai báo `adapter_checkpoint` trỏ adapter Meta.

**Đóng góp / ý nghĩa:** hình thức giống resume nhưng không đúng — đang đoán cấu trúc file nội
bộ không có trong docs công khai. Rủi ro silent-corrupt cao hơn lợi ích. (Kèm theo đó là lớp
chặn #4 ở trên: dù qua được gating, weights PEFT vẫn vào sai key → hỏng âm thầm.)

### Phương án C — Fork recipe, sửa 1 dòng điều kiện ternary (bị loại một phần, sau loại hẳn)

**Cách hoạt động:** sửa `checkpoint_dict[training.ADAPTER_KEY] if self._resume_from_checkpoint else None` trong `setup()`.

**Đóng góp / ý nghĩa:** từng được xem là đủ, nhưng sau khi phát hiện lớp chặn #3
(`get_adapter_checkpoint_path()` trả `None` không điều kiện) thì vá 1 dòng không giải quyết được;
muốn xong phải vá cả helper + giữ đồng bộ với nhiều phần nội bộ, trong khi torchtune đã "wound
down" — tự vá phải gánh mãi qua mọi version.

### Phương án D — Continue-train ĐÚNG adapter Meta: convert ngược + patch recipe (Decision #37, **đang chọn**)

**Cách hoạt động:**
1. `tools/pod_setup/convert_peft_adapter_to_torchtune.py` viết hàm `peft_to_tune_adapter_weights()` — đảo ngược hàm xuôi `tune_to_peft_adapter_weights` thật của torchtune (import `_FROM_HF`/`_TO_PEFT_KEYS`/`get_mapped_key` từ `torchtune.models.convert_weights`, không chép tay): đảo tên key (`lora_A`/`lora_B` ↔ `lora_a`/`lora_b`) + đảo chiều permute head cho `q_proj`/`k_proj`.
2. **Round-trip self-check bắt buộc:** kết quả convert phải qua ngược lại bằng `tune_to_peft_adapter_weights` thật và `assert torch.equal` từng tensor với state dict PEFT gốc; lệch tuyệt đối → từ chối ghi file. Đã test trên dữ liệu giả: logic convert khớp tuyệt đối, và test âm (cố tình đảo permute sai chiều) → round-trip bắt đúng lỗi. Đây là cổng chặn chính cho rủi ro hỏng âm thầm.
3. `external/meta_secalign/helpers/lora_dpo_single_device_t9b.py`: vendor + patch **đúng 1 điểm** — ngay sau `self._model = self._setup_model(...)`, nạp file convert qua `load_state_dict(strict=False)`, validate bằng `validate_missing_and_unexpected_for_lora` có sẵn. Config key mới `manual_adapter_checkpoint`, **không đụng** `adapter_checkpoint`/`resume_from_checkpoint` của torchtune.
4. Checkpoint cuối tự nằm đúng format PEFT chuẩn (`save_checkpoint(adapter_only=True)` của torchtune tự convert ngược) → T10b load thẳng bằng `peft.PeftModel.from_pretrained`, không cần cấu trúc 2-adapter như lo ngại ở bản cũ.

**Đóng góp / ý nghĩa:**
- Đúng nghĩa đen Decision #20: cùng một adapter, tiếp tục train bằng gradient VN — VN-delta và
  EN-delta cùng nằm trong 1 LoRA, có thể quan sát trôi/dạt EN thật sự (khả năng đo được EN
  forgetting ở mức weight, không chỉ mức hành vi).
- Chi phí trả: phải tự viết + tự chứng minh hàm convert ngược rủi ro cao, đổi lại bằng round-trip
  check có "răng" thật và chạy được trên dữ liệu giả trước khi tốn GPU.
- Giữ `merge_meta_adapter.py` làm fallback nếu round-trip không pass và không kịp sửa.

### Các nhánh phụ đã cân nhắc, chưa làm

- **(a) Mỗi ngôn ngữ 1 LoRA độc lập trên base thuần** (không qua adapter Meta): trả lời câu hỏi
  khác hẳn (VN defense có cần init từ EN defense không), không phải domain-incremental theo
  Decision #20 → ablation rẻ, deferred sang T22.
- **(b) Chạy cả merge-approach lẫn continue-approach để so sánh:** bị từ chối ở ngay bước này —
  tốn thêm 1 vòng GPU rental để so sánh trước khi biết có vấn đề thật không; chỉ quay lại nếu
  kết quả D bất thường.

## 3. Bảng so sánh nhanh

| Phương án | Đúng "continue-train adapter Meta"? | Rủi ro chính | Mức công sức | Kết luận |
|---|---|---|---|---|
| A. Merge + LoRA mới | Không (EN đóng băng) | Lệch claim phương pháp | Thấp | #36 — thay thế bởi #37, giữ làm fallback |
| B. Recipe-state giả | Hình thức | Silent-corrupt, nặng nhất | Thấp | Loại |
| C. Vá 1 dòng recipe | Có, nếu đủ | Chưa đủ (lớp #3), gánh fork mãi | Trung bình | Loại |
| **D. Convert ngược + patch 1 điểm** | **Có** | Permute sai chiều → hỏng âm thầm | **Cao** | **#37 — đang chọn**, gate bằng round-trip check |

## 4. Trạng thái hiện tại

- Code D đã viết xong, round-trip tự verify pass trên dữ liệu giả (không cần HF_TOKEN/GPU).
- Còn thiếu DUY NHẤT: chạy thật `convert_peft_adapter_to_torchtune.py` với `HF_TOKEN` để xác nhận
  adapter thật của Meta khớp giả định (`lora_attn_modules=['q_proj','v_proj']`, `rank=64`) —
  làm theo `manual-t9b.md` Phase 0, chặn trước mọi bước tốn GPU.
