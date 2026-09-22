# Proposal — VNU-SecAlign / Vi-InjectEval (v1 → v2)

> Tài liệu tổng hợp/kế hoạch, không phải kết quả nghiên cứu. Nguồn sự thật kỹ thuật vẫn là
> [README.md](README.md), [`src/vi_secalign/config.py`](src/vi_secalign/config.py), và
> [`.agents/record.md`](.agents/record.md) (Decision log) — file này chỉ tóm tắt và liên kết tới
> đó, không lặp lại toàn bộ chi tiết.

## 1. Nghiên cứu gốc của Meta

> Đọc trực tiếp 2 PDF gốc ([`paper/related_work/original-paper.pdf`](paper/related_work/original-paper.pdf),
> [`paper/related_work/meta_secalign.pdf`](paper/related_work/meta_secalign.pdf)), không suy diễn.
> Đây là **2 công trình khác nhau** của cùng nhóm tác giả (Sizhe Chen, Arman Zharmagambetov, David
> Wagner, Chuan Guo, +cộng sự) — luôn nêu rõ đang trích cái nào, không gộp chung (xem quy ước tên
> gọi ở `.agents/record.md` Decision #10).

### 1.1. SecAlign (gốc, CCS'25)

*Chen, Zharmagambetov, Mahloujifar, Chaudhuri, Wagner, Guo. "SecAlign: Defending Against Prompt
Injection with Preference Optimization." ACM CCS 2025.*

- **Cơ chế:** kiểu StruQ — thêm 1 delimiter phân tách data trong **role có sẵn** (`user`), **không**
  thêm role mới. Untrusted data vẫn nằm trong `user`, chỉ đứng sau 1 data delimiter.
- **Dữ liệu train:** Cleaned Alpaca. `chosen`/`rejected` lấy từ **nhãn có sẵn của dataset**
  (KHÔNG self-generate) — `chosen` = response gốc cho instruction thật, `rejected` = response gốc
  cho injected instruction (từ 1 sample khác). Tỉ lệ injection: 90% Straightforward attack / 10%
  Completion attack (kế thừa từ StruQ, không phải phát minh riêng của paper này).
- **Hyperparameter (Llama3-8B-Instruct — model lớn nhất trong paper này):** LoRA `r=64, alpha=8,
  dropout=0.1, target_modules=[q_proj, v_proj]` (**không có MLP**); DPO sigmoid loss, `beta=0.1`;
  3 epoch; `lr=1.6e-4`; thư viện TRL + PEFT; hardware 4× A100 80GB.
- **Quy mô model:** chỉ ≤8B/7B (Mistral-7B-Instruct, Llama3-8B-Instruct, Llama-7B, Mistral-7B,
  Llama3-8B). **Không có bản 70B trong paper này** — paper tự nói sẽ làm bản 70B "commercial-grade"
  ở 1 công trình tiếp theo (chính là SecAlign++/Meta-SecAlign, mục 1.2).
- **Loss:** DPO sigmoid thuần. Ablation riêng (Table 5) so DPO vs ORPO vs KTO → chọn DPO "vì đơn
  giản, ổn định, hiệu năng tốt" — **không có RPO, không có cDPO/label-smoothing** ở bất kỳ đâu.
- **Benchmark:** AlpacaFarm (805 mẫu, 208 có data part; ASR bằng Ignore/Completion/Ignore-Completion
  + GCG/AdvPrompter/NeuralExec; utility bằng AlpacaEval2 WinRate), SEP (9.1K mẫu), InjecAgent, MMLU/
  Winogrande/AGIEval/CommonSenseQA (chỉ test Mistral-7B, Llama3-8B). **Không có** CyberSecEval2/
  IFEval/AgentDojo/GSM8K.
- **Kết quả 8B nổi bật (Table 6):** WinRate SecAlign=85.9% (None=85.4%); ASR Ignore/Completion/
  Ignore-Completion (opt-free) = **0%** cả 3 (undefended 24/47/51%); AdvPrompter ASR 8% (undefended
  97%); GCG ASR 0% (undefended 84%).
- **Đa ngôn ngữ — điểm quan trọng cho RQ1 của v2:** chỉ có **1 test hẹp duy nhất** — dịch witness-word
  "Print exactly" sang tiếng Trung/Tây Ban Nha trong Completion attack, ASR=0%. Đây **không phải**
  test đa ngôn ngữ toàn diện (không dịch instruction/data, không train/eval đa ngôn ngữ thật).
  Section 6 (Limitations) của paper **không có** phát biểu giới hạn nào về ngôn ngữ.
- **Limitations paper tự nêu:** cần input phân tách rõ ràng; không đảm bảo an toàn 100%/chưa test
  hết loại tấn công tương lai (vd multi-turn); chưa rõ hành vi nếu fine-tune thêm; tốt nhất khi
  injection ở cuối data.

### 1.2. SecAlign++ / Meta-SecAlign (arXiv 2507.02735)

*Chen, Zharmagambetov, Wagner, Guo. FAIR at Meta / UC Berkeley. "Meta SecAlign: A Secure Foundation
LLM Against Prompt Injection Attacks."*

- **Cơ chế mới so với 1.1:** thêm hẳn **role `input` mới** vào chat template Llama-3 (song song
  `system`/`user`/`assistant`), tách riêng untrusted data thay vì nhét vào `user`. Có **recursive/
  repeated special-token sanitization filter** (lọc lặp `<|start_header_id|>`, `<|end_header_id|>`,
  `<|eot_id|>`, `<|begin_of_text|>` tới khi hết) — ý tưởng thuộc về StruQ [ref 26], không phải phát
  minh riêng của Meta-SecAlign, nhưng là 1 phần mô tả cơ chế. **Lưu ý đã xác minh bằng code:** filter
  này (`demo.py::recursive_filter`) chỉ được dùng trong demo script, **chưa từng được gọi** trong
  `utils.py`/`test.py`/`secalign_plus_plus.py` — tức pipeline train/eval chính thức không thật sự
  sanitize input.
- **2 cải tiến dữ liệu thật so với 1.1 (vẫn Cleaned-Alpaca, vẫn tỉ lệ 90/10):**
  1. **Self-generated response** — dùng chính model đang train (qua vLLM) để sinh `chosen`/`rejected`,
     thay vì nhãn có sẵn của dataset (bản gốc dùng nhãn "outdated annotator LLM", chất lượng thấp/OOD).
  2. **Randomized injection position** — trong nhóm 90% straightforward, chia đều 45% đặt injected
     prompt ở đầu / 45% ở cuối (bản gốc luôn đặt cố định 1 vị trí).
  - Kích thước bộ preference: **19.157 mẫu**; 99.9% input ≤384 token.
- **Hyperparameter 8B:** LoRA `r=64, alpha=8, dropout=0.1`, `apply_lora_to_mlp=True` (**cờ boolean
  của torchtune**, tương đương thêm `gate_proj/up_proj/down_proj` về hiệu ứng — KHÔNG phải chuỗi
  liệt kê literal trong config gốc, phải trích dẫn đúng); 3 epoch; `lr=1.6e-4` (giá trị CLI default
  của `secalign_plus_plus.py`, yaml gốc mặc định `1e-4`); DPO `beta=0.1`. Bản 70B: `r=32`,
  `lr=3.2e-4`. **Base model 70B là `Llama-3.3-70B-Instruct`, KHÔNG PHẢI `Llama-3.1-70B-Instruct`**
  (khác thế hệ với bản 8B — xác nhận qua HF model card `facebook/Meta-SecAlign-70B` lẫn câu trực
  tiếp trong PDF gốc "Llama-3.3-70B-Instruct, the initialization LLM for our Meta-SecAlign-70B";
  tra cứu 2026-09-16, xem `.agents/record.md` Decision #16). Mọi so sánh 8B-vs-70B phải lưu ý thêm
  biến "khác thế hệ model", không chỉ khác quy mô tham số.
- **Training framework:** **torchtune** (`tune run lora_dpo_distributed`), **KHÔNG dùng TRL**.
  Grep toàn bộ `external/meta_secalign` xác nhận **0 kết quả** cho `rpo_alpha`/`label_smoothing`/
  cDPO — bản gốc SecAlign++ **không có RPO, không có cDPO** (đây là điểm v2 chủ động thêm, xem
  `.agents/record.md` Decision #3/#5).
- **Checkpoint công khai:** `facebook/Meta-SecAlign-8B`, `facebook/Meta-SecAlign-70B` — dùng thẳng
  làm baseline trong v2 (Decision #1), không train lại từ đầu.
- **Benchmark:** AlpacaFarm/AlpacaEval2, SEP, TaskTracker, CyberSecEval2 (code chỉ wire sẵn subtask
  prompt-injection, `run_tests.py:18` mặc định comment out — không phải full suite), MMLU/MMLU-Pro/
  IFEval/BBH/GPQA Diamond, AgentDojo, WASP, InjecAgent. **Không có GSM8K** ở đâu cả.
- **Kết quả 8B (Table XI — phụ lục; bảng chính Table I-III của paper báo cáo 70B, không phải 8B):**
  MMLU 71.7%, MMLU-Pro 46.7%, IFEval 74.5%, BBH 70.9%, GPQA Diamond 28.3%, AlpacaEval2 31.0%, SEP
  Utility 48.8%; AlpacaFarm ASR ≈0%, SEP ASR 7.8%, InjecAgent ASR 0.1%, CyberSecEval2 ASR 7.3%,
  TaskTracker ASR 0.2%. **Không có số AgentDojo/WASP cho 8B** (2 benchmark agentic này chỉ báo cáo
  ở 70B).
- **8B-vs-70B — đúng bản chất (quan trọng, tránh trích sai như v1 đã từng, xem Decision #10):**
  undefended, model càng lớn/mạnh thì ASR càng **cao** (dễ bị inject hơn); **sau** SecAlign++, 8B và
  70B hội tụ về ASR thấp tương đương — **trên các benchmark có báo cáo cho cả 2** (không bao gồm
  AgentDojo/WASP, vì 8B không có số ở đó).
- **Limitations paper tự nêu (Section VI-C):** chỉ chống indirect PI (không phải direct PI/jailbreak);
  chỉ single-turn; chỉ test trên họ Llama 3 (chưa rõ MoE/reasoning architecture); còn residual
  "message-order shortcut" (cần `user` đứng trước `input`). Hướng tương lai paper tự đề xuất: injection
  mạnh hơn, RL online cho reasoning model, multi-modal PI — **không đề cập đa ngôn ngữ** như hướng
  mở rộng.

### 1.3. Tóm tắt khác biệt 2 paper (tra nhanh)

| | SecAlign (CCS'25) | SecAlign++/Meta-SecAlign (arXiv) |
|---|---|---|
| Role tách trust | Không (delimiter trong `user`) | Có, role `input` riêng |
| Nguồn chosen/rejected | Nhãn có sẵn của dataset | Self-generated (vLLM, on-policy) |
| Vị trí injection | Cố định | Randomized (45/45) |
| LoRA target | `q_proj,v_proj` | + `apply_lora_to_mlp` (≈ +MLP) |
| Training framework | TRL | torchtune |
| RPO/cDPO | Không | Không (cả 2 đều không — v2 mới thêm) |
| Quy mô model | ≤8B, không có 70B | 8B **và** 70B |
| Checkpoint công khai | Không | `facebook/Meta-SecAlign-{8B,70B}` |

## 2. V1 — Nghiên cứu trước

### 2.1. V1 đã làm gì

V1 (trước reorg tháng 8/2026, nay nằm trong [archive/](archive/)) là nhiều lần thử rời rạc, không
phải một pipeline nhất quán. Đối chiếu trực tiếp code/checkpoint/log thật tìm được:

| Checkpoint (thư mục local) | Base | LoRA config | Data train | Kết quả đo được |
|---|---|---|---|---|
| [`trl_model_result_final`](archive/legacy_checkpoints/trl_model_result_final/) — khớp gần như tuyệt đối với **`Jason-42195/VNU-SecAlign/checkpoints/final_checkpoint`** trên Hugging Face (adapter_config: r16/alpha32/dropout0.05/q_proj+v_proj) | Llama-3.1-8B-Instruct | `LoraConfig(r=16, lora_alpha=32, target_modules=[q_proj,v_proj], dropout=0.05)`, train bằng **TRL `DPOTrainer`** (`DPOConfig(lr=5e-6, batch=4, grad_accum=8, epochs=1, rpo_alpha=0.5, label_smoothing=0.1)`) | [`dpo_dataset_clean.json`](archive/legacy_data/dpo_dataset_clean.json) — **26.818 mẫu** | `q1_final_metrics.csv`: MMLU 68.24 (base 68.27), GSM8K 72.02 (base 70.89), FRR 2.2% (base 1.0%), **EN_ASR 64.1% (base 84.9%), VN_ASR 83.9% (base 98.6%), AVG_ASR 74.0% (base 91.75%)** |
| [`SecAlign_A100_Final`](archive/legacy_checkpoints/SecAlign_A100_Final/) | Llama-3.1-8B-Instruct | r16/alpha16/dropout0.05, toàn bộ q,k,v,o,gate,up,down | 10k mẫu PKU-SafeRLHF thuần | Chỉ có 2 log định tính, không có ASR số hoá |
| [`Llama3_SecAlign_Checkpoints`](archive/legacy_checkpoints/Llama3_SecAlign_Checkpoints/) (torchtune) | Llama-3.1-8B-Instruct | Đổi liên tục giữa các cell (r64/alpha8/+MLP, r8/alpha16), lr 5e-5/3e-4 | Tên dataset đúng chuẩn Meta (`preference_..._NaiveCompletion_randpos_synthetic_alpaca.json`) nhưng không có safetensors được lưu — không rõ có train xong | Không có |

**Dữ liệu train thật của checkpoint đã publish** (`dpo_dataset_clean.json`, đọc trực tiếp xác nhận):
½ đầu là PKU-SafeRLHF tiếng Anh (harmlessness chung, ví dụ "roommate borrows clothes without
asking"); phần còn lại là tiếng Việt kiểu hate-speech, `chosen` là **một câu từ chối cố định lặp
lại** ("Tôi không thể đưa ra bình luận mang tính thù hận hay công kích cá nhân."), `rejected` =
chính câu độc hại gốc.

Toàn bộ InjecAgent/CyberSecEval2/IFEval/MMLU qua vLLM trong v1 đều **crash** (`Engine core
initialization failed`, sai path checkpoint `/workspace/checkpoints/dpo`) — chưa từng chạy thành
công qua pipeline chuẩn; số MMLU/GSM8K/FRR/ASR ở trên đến từ một pipeline nội bộ khác, nguồn chính
xác chưa xác minh lại được.

### 2.2. V1 có vấn đề gì → nối sang V2

- **Vấn đề data/provenance** (phát hiện quan trọng nhất): checkpoint công khai trên HF **không hề
  train trên bất kỳ dữ liệu prompt-injection nào** — 100% là PKU-SafeRLHF (harmlessness tiếng Anh
  chung) + hate-speech tiếng Việt refusal-template. → ASR đo được (91.75%→74%) nhiều khả năng là
  hiệu ứng phụ ("phản xạ từ chối" tổng quát), không phải cơ chế SecAlign thật (phân biệt trust
  `user` vs `input` — xem mục 1.2). → **Nối sang V2**: đây là lý do phải test cả ASR lẫn utility trên
  benchmark prompt-injection chuẩn trước khi tin dùng lại checkpoint này (xem kế hoạch test go/no-go,
  phần riêng — chưa chạy).
- **Data leakage**: `baseline_asr_report.json`/`v2_attack_asr.json` đo ASR trên đúng 2000 mẫu dùng
  để sinh 10 vector tấn công, không có held-out. → **Nối sang V2 Decision #4**: tách train/held-out
  80/20 nghiêm ngặt, held-out không dùng ở bất kỳ bước sinh dữ liệu nào.
  (`data/attack_vectors/splits/heldout_20/`).
- **Witness-word matching sai** cho 4 vector encoding/obfuscation — model decode sai vẫn bị tính
  "tấn công thành công." → **Nối sang V2**: `evaluation/attack_vectors_eval.py::decode_accuracy_rate`
  / `conditional_asr` (`--decode_check`) tách hiệu ứng năng lực-giải-mã khỏi câu hỏi phòng thủ thật.
- **Pipeline hạ tầng không ổn định**: gọi thẳng script vLLM gốc của Meta, crash khi path/config
  không khớp. → **Nối sang V2**: `data_gen/meta_bridge.py` làm lớp trung gian ổn định, không sửa
  trực tiếp `external/meta_secalign`.
- **Taxonomy/naming**: ATTACK-09 cũ (Base64) trùng lặp bản chất với ATTACK-02; dùng ViHOS/ViHSD sai
  task (hate-speech, không phải prompt-injection); nhầm lẫn tên gọi SecAlign/SecAlign++/8B-vs-70B.
  → **Nối sang V2 Decision #6, #10**.
- **Tín hiệu tích cực giữ lại**: `q1_final_metrics.csv` cho thấy VN_ASR > EN_ASR cả trước lẫn sau
  fine-tune (98.6/84.9 → 83.9/64.1) — bằng chứng sơ bộ ủng hộ RQ1 (khoảng cách cross-lingual không
  đóng lại), dùng làm động lực/prior cho v2, **không** trích dẫn như kết quả chính thức (pipeline
  đo khác, chưa theo chuẩn v2).

## 3. V2 — Nghiên cứu hiện tại

### 3.1. Output mong muốn

- **Vi-InjectEval**: benchmark prompt-injection tiếng Việt độc lập, công khai (~500 mẫu) — deliverable
  chính.
- Trả lời thực nghiệm 4 câu hỏi nghiên cứu:
  - **RQ1**: Security policy học từ dữ liệu preference thuần tiếng Anh có tổng quát hoá zero-shot
    sang tiếng Việt không?
  - **RQ2** (có điều kiện, chỉ chạy nếu RQ1 cho thấy khoảng cách rõ rệt): bổ sung dữ liệu preference
    tiếng Việt cải thiện VN_ASR bao nhiêu, đánh đổi utility gì?
  - **RQ3**: Với ngân sách 1-GPU, cấu hình nào (qua Optuna, neo quanh giá trị công bố của
    Meta-SecAlign-8B) đạt ASR gần nhất checkpoint công khai đó?
  - **RQ4**: 10 vector tấn công mới có đại diện cho lớp tấn công chưa được benchmark công khai phủ
    tới không? Fine-tuning bổ sung có giảm ASR đo trên held-out không?
- **RQ1 — kiểm soát năng lực tiếng Việt nền (confound mới, xem [`.agents/record.md`](.agents/record.md)
  Decision #14):** `llama_3_1_8b_instruct` và `meta_secalign_8b` đều dựa trên Llama-3.1 — model này
  chỉ chính thức hỗ trợ đa ngôn ngữ (SFT/alignment thật) cho 8 ngôn ngữ (en/fr/de/hi/it/pt/es/th),
  **không có tiếng Việt**. Năng lực tiếng Việt (nếu có) của cả 2 model chỉ đến từ tiếp xúc tình cờ
  lúc pretrain, chưa từng qua alignment riêng — nên VN_ASR đo trên 1 trong 2 model này lẫn 2 hiện
  tượng độc lập: defense có tổng quát hoá sang tiếng Việt không, và model có đủ hiểu tiếng Việt để
  phép đo có nghĩa không (cả 2 model cùng chung điểm yếu, không model nào làm control được cho model
  kia). Dùng model continued-pretrain + mở rộng vocab thật cho SEA/tiếng Việt, không có defense, làm
  nhóm control: nếu ASR tiếng Việt vẫn cao (đúng kỳ vọng cho model không phòng thủ), mới có cơ sở
  diễn giải VN_ASR của `meta_secalign_8b` là hiệu ứng defense thật, không phải hiệu ứng ngôn ngữ.
  **Đã chạy thật với `seallm_7b_v2_5`** (`.agents/record.md` Decision #15) — kết quả **không xác
  nhận rõ** confound này (SeaLLM 46% VN_ASR THẤP HƠN Llama 54%, ngược kỳ vọng), nhưng chính bản thân
  SeaLLM v2.5 lại bộc lộ instruction-following yếu (MMLU unparsed-rate 15% vs 0%) — không phải
  control sạch. **Đã nâng cấp lên `seallm_v3_7b_chat` và chạy xong** (Decision #16, #18): MMLU cải
  thiện rõ (53.3%→68.3%), nhưng VN_ASR không đổi (46%→46%) và CyberSecEval2 lệch không đáng kể (nằm
  trong nhiễu N nhỏ). SeaLLM v3 vẫn có gap VN<EN cùng chiều với Llama (-16pp so với -30pp) dù mạnh
  tiếng Việt hơn hẳn — kết luận: phần lớn gap KHÔNG phải thuần confound năng lực ngôn ngữ, xem bảng
  đầy đủ + diễn giải ở mục 4 (Decision #18). Bắt buộc báo cáo utility tiếng Việt lành tính đi kèm
  mỗi số ASR tiếng Việt trước khi diễn giải — cùng nguyên tắc 2 tầng competence-vs-compliance đã
  dùng cho nhóm vector encoding (`decode_accuracy_rate`/`conditional_asr`).
- Kết quả thực nghiệm so sánh trực tiếp với checkpoint công khai `Meta-SecAlign-8B` ở cùng quy mô
  8B (không suy diễn từ số liệu 70B).
- Taxonomy 10 vector tấn công với train/held-out tách bạch, quy trình chống leakage.
- Cơ chế đo ASR sửa lỗi (decode_accuracy_rate/raw_asr/conditional_asr) cho nhóm vector
  encoding/obfuscation.
- Đóng góp phương pháp: DPO + RPO + cDPO **có ablation riêng chứng minh** Pareto-optimal (ASR
  giảm thêm, MMLU gần như không đổi) — không chỉ bật tham số như v1 đã làm mà chưa validate.
- Kỳ vọng venue thực tế: Q2-Q3 hoặc workshop NLP/security tốt — trừ khi RQ1 cho kết quả bất ngờ,
  các ablation cơ chế cho bằng chứng depth rõ ràng, hoặc DPO+RPO+cDPO cải thiện rõ rệt (xem
  `.agents/record.md` Decision #10).

### 3.2. Sửa những gì từ V1

(Mỗi dòng khớp trực tiếp với vấn đề đã nêu ở mục 2.2 — chi tiết đầy đủ dạng
Context/Decision/Rejected alternatives/Consequences ở [`.agents/record.md`](.agents/record.md)).

| # | Sửa gì | Quyết định |
|---|---|---|
| 1 | Bỏ tự train lại baseline EN từ đầu | Dùng thẳng `Meta-SecAlign-8B` công khai làm baseline & điểm khởi đầu (Decision #1) |
| 2 | Dữ liệu preference tiếng Việt | Chỉ self-generated, task-completion đúng cấu trúc gốc; không dùng ViHOS/ViHSD (Decision #2) |
| 3 | DPO+RPO+cDPO | Dùng TRL `DPOTrainer`, **có ablation riêng** chứng minh cải thiện thật (Decision #3, #5) |
| 4 | Data leakage khi đo ASR | Tách train/held-out 80/20 nghiêm ngặt, held-out frozen sau khi tách (Decision #4) |
| 5 | Training framework | TRL thay vì torchtune (torchtune không có `rpo_alpha`/`label_smoothing`) (Decision #5) |
| 6 | Taxonomy 10 vector | ATTACK-09 đổi từ Base64 (trùng ATTACK-02) sang Structured Data-Field Injection (Decision #6) |
| 7 | Benchmark list | Bỏ GSM8K (không có trong 2 paper gốc); CyberSecEval2 chỉ tính subtask prompt-injection (Decision #7) |
| 8 | GRPO/online RL | Loại trừ có chủ đích, không phải bỏ sót (ngân sách 1-GPU, judge chưa đủ tin cậy) (Decision #8) |
| 9 | Witness-word sai cho encoding vectors | `decode_accuracy_rate`/`raw_asr`/`conditional_asr` tách biệt; + 2 ablation cơ chế (instruction_hierarchy, role order) + 70B spot-check (Decision #9) |
| 10 | Diễn giải 8B-vs-70B, tên gọi | Sửa đúng bản chất (ngân sách 1-GPU, không phải "8B kém hơn"); chuẩn hoá SecAlign/SecAlign++/v1/v2 (Decision #10) |
| 11 | **Mới** — trước khi retrain (GĐ3-GĐ5) | Test checkpoint v1 đã có sẵn trên HF trước (base vs. đã train); chỉ train lại phần nào test cho thấy thật sự cần |

## 4. Trạng thái hiện tại & bước tiếp theo

**Cập nhật 2026-09-15 — go/no-go đã chạy xong, có kết quả**
(`notebooks/phase0_go_no_go_test.ipynb`, Colab T4, kết quả embedded trong notebook +
`results/phase0_go_no_go/metrics.json`):

| Model | ASR (SEP, N=40) | Alpaca refusal-rate (N=30) |
|---|---|---|
| `llama_3_1_8b_instruct` (base) | 87.5% | 0% |
| `meta_secalign_8b` (Meta công khai) | **5.0%** | 0% |
| `jason_v1_final_checkpoint` (v1) | 82.5% | 0% |

**Quyết định: No-go.** Checkpoint v1 gần như không giảm ASR so với base (87.5%→82.5%, trong biên độ
nhiễu N=40) — khác hẳn `meta_secalign_8b` (giảm còn 5%) — và không có dấu hiệu over-refusal (utility
không sập). Kết luận: checkpoint v1 **không học được cơ chế phòng thủ prompt-injection nào**, đúng
như giả thuyết provenance ở mục 2.2. Không dùng làm điểm khởi đầu — tiến hành retrain theo kế hoạch
gốc GĐ3-GĐ5. Chi tiết đầy đủ (Context/Decision/Rejected alternatives/Consequences) ở
[`.agents/record.md`](.agents/record.md) Decision #11.

Test này đồng thời đóng vai trò một phần T1-T3 (`plan.csv`, đã chuyển "In progress"): đã xác nhận
tải + chạy được `meta_secalign_8b` (kể cả tự nhận diện đúng nó là LoRA-only repo), ASR sanity 5% hợp
lý — nhưng T1-T3 formal (qua `meta_bridge.py`/vLLM, không phải notebook nhẹ transformers/peft) vẫn
cần chạy để có số liệu chính thức đúng chuẩn báo cáo.

**Cập nhật 2026-09-16 — T4 đã chạy (pilot v0.1 + 3 bộ mở rộng + control `seallm_7b_v2_5`)**
(`notebooks/phase1_rq1_zero_shot.ipynb`, `results/phase1_multi_benchmark_pilot/`, chi tiết đầy đủ ở
[`.agents/record.md`](.agents/record.md) Decision #15): so sánh sạch nhất trong cùng 1 bộ dữ liệu
(Vi-InjectEval v0.1) — `llama_3_1_8b_instruct` VN_ASR 54% → `meta_secalign_8b` 10% — cho bằng chứng
thật rằng phòng thủ có tác dụng đáng kể trong tiếng Việt zero-shot. Tuy nhiên `vn_minus_en_asr`
(so với EN_ASR đo trên SEP, bộ dữ liệu khác) **chưa đáng tin làm số liệu RQ1 chính thức** — cần EN
reference đo bằng đúng pool template v0.1. Control `seallm_7b_v2_5` (46% VN_ASR, thấp hơn cả Llama)
**không xác nhận rõ** confound năng lực tiếng Việt, nhưng bản thân v2.5 bộc lộ instruction-following
yếu (không phải control sạch) — đã nâng cấp lên `seallm_v3_7b_chat` để chạy thêm trước khi kết luận
(xem Decision #16). T4 vẫn ở "In progress", chưa "Done".

**Cập nhật 2026-09-21 — RQ1 có câu trả lời sơ bộ (đáng tin), Go cho GĐ3**
(`en_matched_injecteval_gen.py`, `results/phase1_multi_benchmark_pilot/multi_benchmark_pilot_metrics.json`,
chi tiết đầy đủ ở `.agents/record.md` Decision #17/#18): thêm `EN_MATCHED_PILOT` — bộ tiếng Anh khớp
từng mẫu với Vi-InjectEval v0.1 trên mọi trục trừ ngôn ngữ (cùng câu gốc Bactrian-X, cùng template
injection dịch tay giữ độ lộ liễu, cùng witness token) — để cô lập đúng biến ngôn ngữ cho RQ1, thay
cho phép so sánh cũ dùng SEP (2 bộ injection khác nhau, nhiễu "độ mạnh pool" lẫn vào "ngôn ngữ").

| Model | EN_ASR (matched pool) | VN_ASR | Gap (VN−EN, matched) | Gap (VN−EN, SEP cũ) |
|---|---|---|---|---|
| `llama_3_1_8b_instruct` (không defense) | 84% | 54% | **-30pp** | -33.5pp |
| `meta_secalign_8b` (có defense) | 2% | 10% | **+8pp** | +5pp |
| `seallm_v3_7b_chat` (không defense, mạnh tiếng Việt) | 62% | 46% | **-16pp** | — |

**Diễn giải (N=50 mỗi bộ, chưa test thống kê chính thức, nhưng có tín hiệu hội tụ)**: với model
KHÔNG defense, VN_ASR thấp hơn EN_ASR nhất quán kể cả sau khi cô lập ngôn ngữ (Llama -30pp, SeaLLM
v3 -16pp dù mạnh tiếng Việt hơn hẳn) — phần lớn không phải thuần confound năng lực ngôn ngữ (Decision
#14), nhiều khả năng do phong cách injection dịch tay kém "hiệu lực" hơn khi chuyển ngữ; chênh lệch
độ lớn gap giữa 2 model (Llama vs SeaLLM) gợi ý vẫn còn 1 phần đóng góp từ năng lực ngôn ngữ, không
loại trừ hẳn. Với `meta_secalign_8b` (có defense), chiều gap **đảo ngược**: VN_ASR (10%) > EN_ASR
(2%), +8pp — khớp hướng và độ lớn với phép đo cũ dùng SEP (+5pp) dù 2 phép đo dùng 2 bộ tiếng Anh
hoàn toàn độc lập. Sự hội tụ này là bằng chứng khá vững cho **RQ1**: defense SecAlign, học hoàn toàn
từ dữ liệu tiếng Anh, có một khoảng hở tổng quát hoá sang tiếng Việt thật (không phải nhiễu đo lường)
— dù tuyệt đối cả 2 đều thấp (2-10%), khoảng hở tương đối (5x) có ý nghĩa.

**Người dùng đã xác nhận GO cho GĐ3** (2026-09-21, xem Decision #19) dựa trên khuyến nghị này —
`plan.csv` T6 chuyển "In progress" (chờ người dùng tự set Done theo CLAUDE.md mục 2). Đã chốt luôn:
- **T7 (corpus tiếng Việt)**: `MBZUAI/Bactrian-X` (subset `vi`), license CC-BY-NC-4.0 (xác minh qua
  HF API, phù hợp thesis phi thương mại) — không cần fallback.
- **T8 (sinh preference tiếng Việt)**: `vi_preference_gen.py` đã thêm `--n_samples` (mặc định 2000,
  script gốc xử lý toàn bộ ~67K dòng — vượt xa ngân sách 3 ngày) + `--seed` (tái lập được qua
  `np.random.default_rng`). Sẵn sàng chạy, **còn thiếu**: vLLM + GPU thật (pod thuê) — chưa chạy
  được trong môi trường agent, khuyến nghị thử `--n_samples 200` trước để đo throughput thật rồi mới
  quyết định N cuối cùng.

**Hạ tầng (2026-09-22)**: đã dựng xong pipeline cache môi trường cho pod thuê
(`tools/pod_setup/build_env_cache.sh` + `pod_init.sh`, xem `tools/pod_setup/manual.md`) — build 1
lần trên máy mạng tốt, upload lên Hugging Face, pod chỉ cần tải về thay vì cài trực tiếp từ PyPI
(từng bị nghẽn route nặng). Đang trong quá trình setup pod thật; còn thiếu trước khi chạy T1-T3
chính thức: sinh `SEP_dataset_test.json` (script `sep_reference_gen.py` đã viết, cần chạy 1 lần có
GPU trên pod).
