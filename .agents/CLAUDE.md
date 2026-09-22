# CLAUDE.md — Luật hiện hành cho project Thesis (Vi-InjectEval)

## 0. Việc đầu tiên của MỌI phiên làm việc (không phải chỉ lần đầu)

1. Kiểm tra `.agents/` có đủ 3 file: `CLAUDE.md`, `record.md`, `action-history.md`.
   Kiểm tra root project có file plan (`plan.csv`) và có phải git repo không.
   - Thiếu file nào → **tạo từ template chuẩn** (không tự bịa cấu trúc mới), và
     **in ra màn hình rõ ràng những gì vừa tạo** trước khi làm tiếp.
   - Có đủ → đọc **toàn bộ** `record.md` trước khi chạm vào bất kỳ file nào khác.
2. Không giả định "project này chắc đã setup rồi" chỉ vì thấy 1-2 file quen thuộc.

## 1. Thư mục / file "frozen" — hỏi trước khi đụng, không tự quyết

> Tên thư mục dưới đây đã đối chiếu với workspace thật (2026-09-06) — không phải
> tên giả định ban đầu nữa.

- `data/attack_vectors/splits/heldout_20/` — 20% held-out của 10 vector tấn công (GĐ5).
  Một khi đã tách, **không được đọc/ghi/dùng để sinh thêm dữ liệu train dưới bất kỳ
  hình thức nào**. Guard đã cài ở code (`src/vi_secalign/data_gen/splits.py` —
  ghi đè lần 2 lên file trong thư mục này sẽ raise lỗi, không âm thầm cho qua).
  Đây là lỗi đã từng xảy ra ở nghiên cứu trước (xem `record.md` mục Bài học).
- `checkpoints/<tên_phase>/` (vd. `checkpoints/phase2_final/`, xem
  `src/vi_secalign/models/registry.py`) — checkpoint đã được dùng để tính số liệu
  đã ghi vào bản thảo bài báo. Lưu ý: quy ước thật là tên thư mục phẳng
  `checkpoints/<phase>/`, KHÔNG phải cấu trúc lồng `checkpoints/<phase>/final/`
  như bản CLAUDE.md cũ giả định.
- `results/phase0_sanity_check/`, `results/phase1_zero_shot/`,
  `results/phase1_5_vi_defense/`, `results/optuna_trials/`, `results/phase2_heldout/`,
  `results/ablations/` — một khi 1 file kết quả đã được trích dẫn trong `draft_v*.md`,
  không tự ý chạy lại đè lên. Muốn chạy lại → dừng, hỏi. (Bản cũ ghi
  `benchmark_results/`/`ablation_results/` ở cấp root — không tồn tại trong repo
  thật, đã sửa lại đúng theo `results/` hiện có.)
- Cần đổi/xóa bất kỳ mục nào ở trên → **dừng lại, hỏi tôi**, không tự quyết rồi báo sau.

## 2. Status trong plan.csv — không tự đánh dấu Done

- Agent **không bao giờ** tự set `Status = Done`.
- Agent chỉ được đẩy tối đa tới `In progress` (qua commit, xem mục 3).
- `Done` chỉ được set bởi: (a) tôi tay, hoặc (b) hook tự động khi `DoD (check)`
  của dòng đó pass (xem `plan.csv`, cột `DoD (check)`).
- Nếu nghĩ 1 task đã xong nhưng hook chưa flip Done: đó là tín hiệu để tự hỏi lại
  DoD có thật sự thỏa chưa, không phải để tự sửa cột Status.

## 3. Quy ước commit (bắt buộc)

```
[TaskID] <wip|done|blocked>: <mô tả ngắn>
```
Ví dụ: `[T1] wip: đang tải checkpoint Meta-SecAlign-8B`

Không cần tiền tố tên project — mỗi project đã là 1 git repo riêng.
`TaskID` phải khớp đúng 1 dòng trong `plan.csv`.

## 4. Cập nhật record.md — trách nhiệm của agent, không phải hook

Hook chỉ ghi `action-history.md` (log máy). Agent phải **chủ động** ghi vào
`record.md` khi:
- Đưa ra một quyết định phương pháp/kỹ thuật có phương án bị loại (thêm entry
  theo đúng khuôn Context/Decision/Rejected alternatives/Consequences).
- Rút ra một bài học từ lỗi (thêm vào mục Bài học).
- Có câu hỏi nghiên cứu mới phát sinh, hoặc trả lời được 1 câu đã treo (cập nhật
  mục Câu hỏi treo).

Không xóa entry cũ trong `record.md`. Đổi ý → thêm entry mới, ghi rõ
"thay thế entry #N".

## 5. Dữ liệu nhạy cảm

Project này làm việc với payload prompt-injection / dữ liệu tấn công. Không dán
các payload này vào công cụ AI bên ngoài không rõ chính sách lưu trữ dữ liệu
(không có ZDR) — rủi ro rò rỉ hướng nghiên cứu chưa công bố.
