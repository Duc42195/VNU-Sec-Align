# Vi-InjectEval — pilot v0 (superseded) / v0.1 (hiện dùng)

`pilot_v0_1.json` (N=50) là bản **pilot hiện tại**, sinh bởi
[`src/vi_secalign/data_gen/vi_injecteval_gen.py`](../../../src/vi_secalign/data_gen/vi_injecteval_gen.py) —
**không phải** bản Vi-InjectEval cuối cùng (~500 mẫu) sẽ công bố làm deliverable chính. Mục đích
duy nhất: có đủ dữ liệu để chạy T4 (`plan.csv`, GĐ2, RQ1 zero-shot) và có tín hiệu sớm trước khi
đầu tư build bản đầy đủ.

`pilot_v0.json` **vẫn giữ lại** (không xoá, theo quy ước `.agents/CLAUDE.md`) nhưng **đã bị thay
thế** — có lỗi confound đã phát hiện khi đọc kết quả chạy thật (xem `.agents/record.md` mục Bài
học, 2026-09-16): dùng **1 câu injection cố định, rất lộ liễu** + **1 witness token cố định** cho
toàn bộ 50 mẫu, khiến so ASR với SEP (tiếng Anh, injection đa dạng hơn) bị nhiễu bởi "độ lộ liễu
câu tấn công" chứ không chỉ khác ngôn ngữ. **Không dùng `pilot_v0.json` cho bất kỳ kết luận nào.**

## Nguồn dữ liệu

Instruction/input tiếng Việt **bản địa** (không dịch máy) từ `MBZUAI/Bactrian-X` (subset `vi`,
67.017 mẫu) — cùng corpus đã chốt sẵn cho dữ liệu train GĐ3 (`vi_preference_gen.py`,
`DEFAULT_CORPUS_HF_ID`), tái dùng ở đây cho eval thay vì train. Chỉ giữ mẫu có `input` đủ dài
(≥15 ký tự) — cần có "dữ liệu untrusted" để chèn injection vào, giống cấu trúc SEP gốc.

## Schema (khớp `SEP_dataset.json`)

```json
{
  "system_prompt_clean": "...",   // = 'instruction' gốc, đóng vai chỉ thị đáng tin (trusted)
  "prompt_clean": "...",          // = 'input' gốc, không có injection
  "prompt_instructed": "...",     // = 'input' gốc + 1 trong 7 câu injection (chọn ngẫu nhiên)
  "witness": "...",               // 1 trong 20 witness word (chọn ngẫu nhiên theo mẫu)
  "info": {"source": "...", "source_id": "...", "injection_template_idx": 0-6, "pilot": true,
           "version": "v0.1"}
}
```

Cùng schema này nên tái dùng thẳng được code chấm điểm trong
`notebooks/phase0_go_no_go_test.ipynb`/`notebooks/phase1_rq1_zero_shot.ipynb`
(`build_sep_messages`, `asr_sep_instructed`) không cần sửa.

## Quy ước witness/injection (v0.1 — đã sửa lỗi confound của v0)

**v0 (superseded)**: 1 token cố định `XACNHANTIEMNHIEM` cho mọi mẫu, 1 câu injection cố định rất
lộ liễu ("Bỏ qua mọi hướng dẫn phía trên...").

**v0.1 (hiện dùng)**: pool 20 witness word tiếng Việt thông dụng (`chanh`, `vali`, `boong`, ...) +
pool 7 kiểu câu injection trải từ lộ liễu ("Bỏ qua mọi hướng dẫn...") tới tinh vi (giả làm ghi chú
hệ thống, đóng vai social-engineering, roleplay...) — gán ngẫu nhiên (seed cố định, tái lập được)
theo từng mẫu, bám sát tinh thần "đa dạng theo từng mẫu" của chính SEP gốc (SEP cũng có witness
riêng từng mẫu, injection không lặp lại y hệt). Xem đầy đủ 7 template + 20 witness trong
`vi_injecteval_gen.py`.

## Giới hạn đã biết (pilot, chưa phải bản chính thức)

- N=50, không phải ~500.
- 7 kiểu injection / 20 witness — đa dạng hơn v0 nhiều nhưng vẫn hữu hạn, chưa chắc đã khớp hoàn
  toàn độ đa dạng thật của SEP (chưa đo định lượng "độ lộ liễu" để so khớp 2 bên).
- Chỉ append injection vào cuối `prompt_clean` (không đổi vị trí) — chưa có biến thể vị trí như 10
  attack vector riêng (đó là track khác, GĐ5, `attack10_gen.py`).
- Chưa qua audit thủ công.

## `pilot_v0_1_en_matched.json` — bộ tiếng Anh khớp từng mẫu (cho RQ1)

Sinh bởi [`src/vi_secalign/data_gen/en_matched_injecteval_gen.py`](../../../src/vi_secalign/data_gen/en_matched_injecteval_gen.py)
(xem `.agents/record.md` Decision #17). Dùng để tính `en_asr_matched_pool`/`vn_minus_en_asr_matched`
trong `notebooks/phase1_rq1_zero_shot.ipynb` — **thay thế** `en_asr_reference` (đo trên SEP, đã bị
Decision #15 chỉ ra là không đáng tin cho RQ1 vì SEP và `pilot_v0_1.json` là 2 bộ injection khác
nhau). Khớp `pilot_v0_1.json` trên mọi trục trừ ngôn ngữ: cùng 50 instruction/input gốc (Bactrian-X
`en` config, cùng `source_id`, không phải dịch máy ngược), cùng injection template theo đúng index
(dịch tay sang tiếng Anh), cùng witness token (không đổi). Chưa có kết quả chạy thật — cần 1 lượt
Colab nữa (xem `plan.csv` T5).
