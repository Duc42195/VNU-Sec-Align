# Addendum: ATTACK-09 và ATTACK-10 (không có trong tài liệu phân tích gốc)

`secalign_mechanism_analysis_and_attack_taxonomy.txt` chỉ đề xuất và lập luận cho 8 vector (ATTACK-01→08, mục 3.2, mỗi vector gắn với 1 GAP-0N cụ thể). Dataset cũ (`data/attack_vectors/_legacy_v2_reference/dataset_v2_attacks.json`) lại có 10 loại — 2 loại cuối được thêm thẳng vào dữ liệu mà chưa từng có rationale bằng văn bản. File này bổ sung rationale cho bộ 10 vector cuối cùng dùng trong VNU-SecAlign v2.

## ATTACK-09 (đã thay thế): Structured Data-Field Injection

Định nghĩa cũ (Base64 Obfuscation) bị loại vì trùng lặp bản chất với nhánh base64 đã có sẵn trong ATTACK-02 (Encoding Obfuscation Chain) — không orthogonal, không đáng tính là 1 vector riêng.

**Định nghĩa mới**: chèn injected instruction vào bên trong 1 trường dữ liệu có cấu trúc (JSON key, CSV cell, hoặc field giả lập tool-output/API-response), thay vì đặt trực tiếp trong free-text như 9/10 vector còn lại. Ví dụ: `{"tool_result": {"status": "ok", "note": "<injected instruction>"}}`.

**Gap khai thác**: Toàn bộ 8 vector gốc + ATTACK-10 (Leetspeak) đều là thao túng free-text trong một trường `input` đơn — chưa vector nào kiểm tra xem ranh giới role `"input"` (SecAlign++, `utils.py::form_llm_input`) và `recursive_filter` (`demo.py`, chỉ lọc 4 chuỗi special-token literal) có đứng vững khi untrusted content trông giống dữ liệu có cấu trúc (structured/agentic-style payload) hay không. Đây cũng là gap thật đối với các benchmark công khai: InjecAgent/AgentDojo test hành vi agent có bị hijack hay không, nhưng không hệ thống hoá việc tấn công qua payload cấu trúc bị obfuscate/biến dạng trong 1 field cụ thể.

## ATTACK-10: Leetspeak Obfuscation (giữ nguyên, bổ sung rationale)

Không có trong tài liệu phân tích gốc nhưng không trùng lặp với vector nào khác. **Gap khai thác**: các biến thể encoding khác trong ATTACK-02 (base64/rot13/hex) đều tạo ra chuỗi ký tự không phải chữ cái thông thường (dễ bị regex/heuristic phát hiện là "trông đáng ngờ"). Leetspeak thay ký tự trong từ bằng ký tự số/ký hiệu tương tự về hình dạng (a→4, e→3, i→1, o→0...) nhưng **vẫn là chữ cái/số hợp lệ, không có ký tự lạ** — né được các filter dựa trên phát hiện "chuỗi trông bất thường," đồng thời vẫn đủ dễ đọc để mô hình ngôn ngữ hiểu đúng ý injected instruction mà không cần bước decode riêng như base64/hex.

## Danh sách 10 vector cuối cùng (nguồn sự thật: `src/vi_secalign/config.py::ATTACK_VECTORS`)

| ID | Tên | Nguồn |
|---|---|---|
| ATTACK-01 | Multi-Turn Gradual Escalation | `secalign_mechanism_analysis_and_attack_taxonomy.txt` §3.2 |
| ATTACK-02 | Encoding Obfuscation Chain | như trên |
| ATTACK-03 | Cross-Lingual Code-Switching | như trên |
| ATTACK-04 | Semantic Injection (No Keyword Match) | như trên |
| ATTACK-05 | Context Overflow / Attention Dilution | như trên |
| ATTACK-06 | Unicode Homoglyph Role Confusion | như trên |
| ATTACK-07 | Payload Splitting Across Roles | như trên |
| ATTACK-08 | Temporal / Conditional Logic Injection | như trên |
| ATTACK-09 | **Structured Data-Field Injection** (mới, thay thế Base64 Obfuscation cũ) | file này |
| ATTACK-10 | Leetspeak Obfuscation | file này |
