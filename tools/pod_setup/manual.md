# Hướng dẫn setup pod GPU thuê (tools/pod_setup/)

> Tài liệu thao tác — không phải nguồn sự thật kỹ thuật (nguồn sự thật là chính các file `.sh`/`.py`
> trong thư mục này). Cập nhật file này khi quy trình đổi; nếu lệch với code thật, tin code.

## Tổng quan kiến trúc

```
laptop (mạng tốt)              Hugging Face (kho trung chuyển)         pod thuê (mỗi lần thuê mới)
────────────────────           ──────────────────────────────          ─────────────────────────────
build_env_cache.sh    ──upload──→  Jason-42195/vnu-secalign-env-cache  ──download──→  pod_init.sh
(chạy 1 lần, hoặc khi                 1 file tar.gz (~5.1GB)                (chạy trên từng pod)
 requirements.txt đổi)             (thư viện Python đã giải nén sẵn
                                    + data nhỏ của meta_secalign)
```

**Vì sao tách 2 bước** thay vì để pod tự `pip install` trực tiếp: pod thuê thường có route mạng
riêng tới PyPI/GitHub bị nghẽn nặng (đã gặp thực tế: 250-500KB/s, hay đứt kết nối) dù băng thông
chung của pod bình thường. Build 1 lần trên máy mạng tốt, đóng gói sẵn, pod chỉ cần tải 1 file từ
HF (route khác, ổn định hơn).

## File trong thư mục này

| File | Chạy ở đâu | Việc |
|---|---|---|
| `build_env_cache.sh` | Laptop (mạng tốt) | Cài + đóng gói toàn bộ thư viện Python + data nhỏ, upload lên HF |
| `pod_init.sh` | Mỗi pod mới thuê | Clone code, tải cache từ HF, dựng venv, sẵn sàng chạy |
| `fetch_meta_secalign_data_urls.py` | Được gọi tự động bởi 2 script trên | Tải 14 file data nhỏ của Meta + tự sinh `CySE_prompt_injections.json` |

## Quy trình đầy đủ, theo đúng thứ tự

### 1. Trên laptop (chỉ cần làm lại khi `requirements.txt` của Meta đổi, hoặc đổi version torchtune)

```bash
cd /home/j/Workspace/VNU/Final
bash tools/pod_setup/build_env_cache.sh
```

Yêu cầu trước khi chạy: đã cài `uv`, đã có token HF ghi quyền vào repo
`Jason-42195/vnu-secalign-env-cache` (kiểm tra bằng `python3 -c "from huggingface_hub import whoami; print(whoami())"`).

Mất khoảng vài chục phút tới 1-2 tiếng tuỳ tốc độ cài `torch`/`vllm` thật vào venv tạm. Bước upload
cuối cùng (~5.1GB) tuỳ tốc độ mạng — xem mục Troubleshooting nếu chậm bất thường.

### 2. Trên mỗi pod mới thuê

SSH vào pod, rồi:

```bash
export HF_TOKEN=hf_xxx   # BẮT BUỘC trước khi chạy — token quyền read, đã accept license Llama-3/Llama-3.1
curl -sL https://raw.githubusercontent.com/Duc42195/VNU-Sec-Align/main/tools/pod_setup/pod_init.sh | bash
```

**`HF_TOKEN` phải export TRƯỚC** — repo cache trên HF là private, script tải cache đó ngay từ bước
[3/5], không phải ở bước cuối. Thiếu token sẽ báo lỗi `401 RepositoryNotFoundError` và dừng ngay.

Script làm 5 bước tự động:
1. Cài `git`/`uv` nếu thiếu.
2. Clone/pull code từ GitHub (nhánh `main`).
3. Tạo venv Python 3.13, login HF, tải + giải nén cache 5.1GB, copy vào site-packages, cài lại
   numpy/cupy/ray/vllm tươi (glibc-sensitive, xem Troubleshooting), kiểm tra import.
4. Copy 14 file data nhỏ + tự sinh `CySE_prompt_injections.json`.
5. In sẵn (không tự chạy) 4 lệnh `snapshot_download` mẫu để tải model khi cần.

Kết thúc bằng dòng `SETUP_DONE`.

### 3. Tải model (làm tay, chọn cái cần)

```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Llama-3.1-8B-Instruct', local_dir='~/models/llama_3_1_8b_instruct')"
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/Meta-SecAlign-8B', local_dir='~/models/meta_secalign_8b')"
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Meta-Llama-3-8B-Instruct', local_dir='~/models/llama3_8b_instruct')"
```

Model thứ 3 (`Meta-Llama-3-8B-Instruct`) là model "trọng tài" nhỏ dùng riêng cho bước 4 dưới đây
(sinh câu trả lời tham chiếu SEP) — không phải model đang được đánh giá.

**Lưu ý disk**: kiểm tra dung lượng pod thật bằng `df -h /` — đừng giả định theo con số ghi trong
`pod_init.sh` cũ (từng sai một lần, xem Troubleshooting). Model 8B ở fp16 ~16GB/bản; 4-bit ~5-6GB.

### 4. Sinh dữ liệu tham chiếu SEP (cần GPU, chạy 1 lần/pod)

```bash
cd ~/repo
source ~/venv/bin/activate
PYTHONPATH=src python3 src/vi_secalign/data_gen/sep_reference_gen.py
```

Nạp `Meta-Llama-3-8B-Instruct` qua vLLM, tự sinh câu trả lời tham chiếu cho từng mẫu SEP sạch
(không injection) — không cache trước được vì cần GPU thật. Output:
`external/meta_secalign/data/SEP_dataset_test.json` +
`external/meta_secalign/data/SEP_dataset_test_Meta-Llama-3-8B-Instruct.json`.

### 5. Chạy T1-T3 chính thức (sanity check, đối chiếu paper gốc)

```python
from vi_secalign.evaluation import meta_eval_runner
meta_eval_runner.run_sep("~/models/llama_3_1_8b_instruct", attack="none", defense="none")
meta_eval_runner.run_sep("~/models/meta_secalign_8b", attack="none", defense="none")
# tương tự run_alpacafarm / run_cyberseceval2_pi_subtask / run_injecagent / run_lm_eval
```

## Troubleshooting (sự cố thật đã gặp)

**`401 RepositoryNotFoundError` khi tải cache** — chưa `export HF_TOKEN` trước khi chạy
`pod_init.sh`, hoặc token không có quyền đọc repo `Jason-42195/vnu-secalign-env-cache`. Đã sửa thứ
tự script (login trước khi tải), nhưng vẫn phải tự export token trước.

**`pip install torch` treo hàng chục phút, gần như không tải được gì** — route mạng riêng tới
PyPI/pytorch.org bị nghẽn dù băng thông chung bình thường (đo bằng `curl` tới Cloudflare speed test
để so sánh). Đã chuyển toàn bộ sang `uv` (tải song song, resume tốt hơn).

**Tốc độ upload lên HF đo được cực thấp (~500 B/s) qua `/proc/<pid>/io`** — có thể là nghẽn thật
(đổi network/interface để test, ví dụ hotspot điện thoại khác ISP), nhưng **cũng có thể do đo sai
phương pháp**: nếu dùng `curl` để test tốc độ, luôn kiểm tra `http_code` và `size_download` trước
khi tin số B/s — một response lỗi nhỏ (401/redirect chưa theo `-L`) tải xong trong <1s cũng ra số
B/s rất thấp, dễ nhầm là "mạng chậm" trong khi thực ra mạng bình thường, chỉ là request bị từ chối
nhanh. Cách đo đúng: `curl -L -o /dev/null -s -w "%{speed_download} B/s, %{http_code}"` trên 1 file
public đủ lớn (>10MB) để loại nhiễu.

**Máy có nhiều interface mạng (ethernet + wifi) cùng lúc** — Linux có thể tự chọn route không như
kỳ vọng (ví dụ IPv6 Router Advertisement "pref high" của wifi thắng ethernet dù ethernet có metric
IPv4 thấp hơn). Dùng `curl --interface <tên-nic>` để test từng đường riêng biệt mà không ảnh hưởng
kết nối khác đang chạy, trước khi quyết định tắt hẳn 1 interface.

**`ModuleNotFoundError: torchao.dtypes.nf4tensor`** — tự chọn version mới nhất cho từng gói
(`torch`/`torchao`/`torchtune`) độc lập nhau gây xung đột API. Luôn cài đúng theo
`external/meta_secalign/requirements.txt` (đã pin sẵn tổ hợp version Meta tự kiểm chứng) +
`torchtune==0.6.0` riêng theo README Meta, không tự chọn "mới nhất".

**`setup.py` gốc của Meta tải nhầm 5 model đầy đủ (gồm ~140GB 70B)** — nó không chỉ tải 14 file
data nhỏ, còn tự `snapshot_download()` cả model + chạy vLLM inference. KHÔNG chạy `python3 setup.py`
trực tiếp — dùng `fetch_meta_secalign_data_urls.py` (chỉ 14 URL + xử lý CPU thuần) thay thế.

**Dung lượng đĩa pod không như ghi trong comment cũ** — pod ckey.vn trước đây có 73GB thật (không
phải 100GB như 1 bản comment cũ từng giả định). Luôn `df -h /` để lấy số thật trên chính pod đang
thuê, đừng tin số cũ ghi sẵn trong script — pod mới có thể khác hẳn.

**`ImportError: ... you should not try to import numpy from its source directory` /
`GLIBC_2.38 not found`** — `numpy==1.26.4` (pin trong `requirements.txt`) không có wheel dựng sẵn
cho Python 3.13 trên PyPI, nên lúc build cache trên laptop, `uv` phải tự build nó từ source ngay
trên máy đó — kết quả gắn chặt vào glibc của máy build (không portable sang pod glibc cũ hơn, vd
Ubuntu 22.04/glibc 2.35). `pod_init.sh` đã tự động
`uv pip install -r ~/repo/external/meta_secalign/requirements.txt --reinstall-package numpy` ngay
sau khi copy cache để sửa — nếu vẫn gặp lỗi này (ví dụ chạy tay từng phần script), chạy lại đúng
lệnh đó trong venv đã activate.

**KHÔNG dùng `uv pip install --reinstall numpy==... cupy==... ray==... vllm==...` liệt kê version
trực tiếp như vậy** (bài học thật, 2026-09-22) — thiếu `-r requirements.txt` làm `uv` tự resolve
lại TOÀN BỘ ~148 gói theo "mới nhất tương thích", phá vỡ tổ hợp Meta đã pin (hậu quả thật gặp:
`transformers` 4.57.1→5.17.0, `huggingface-hub` 0.36.0→1.32.0, `protobuf` 5→7 — nhảy nhiều major
version cùng lúc). Luôn dùng `-r requirements.txt --reinstall-package <tên-gói>` để ràng buộc mọi
gói khác giữ đúng version pin, chỉ gói được nêu tên mới bị build/tải lại.

Dấu hiệu khác để nhận diện lỗi glibc trước khi kết luận nhầm: prompt không có tiền tố `(venv)` —
nghĩa là quên `source ~/venv/bin/activate`, dễ nhầm với lỗi glibc vì thông báo lỗi bề ngoài giống
nhau (cả hai đều là "import numpy thất bại"), nhưng nguyên nhân khác nhau — luôn kiểm tra
`which python3`/`python3 --version` (phải ra `~/venv/bin/python3`, `Python 3.13.x`) trước khi kết
luận là lỗi glibc.

**`hf: command not found`** — không có console-script CLI (`huggingface-cli`, `hf`) qua cách cài
này (`uv pip install --target` không tạo shim). Mọi thao tác HF đều qua Python API
(`huggingface_hub.login`/`snapshot_download`/`hf_hub_download`), không gọi lệnh CLI.

## Ghi chú khác

- `.agents/infra_handoff.md` — trạng thái pod cụ thể đang thuê (SSH, disk, ngân sách) — file tạm,
  chỉ đúng cho lần thuê đó, xoá sau khi setup xong hẳn.
- Repo GitHub public (`Duc42195/VNU-Sec-Align`) nên `curl | bash` dùng raw URL hoạt động không cần auth.
- Repo cache trên HF (`Jason-42195/vnu-secalign-env-cache`) là **private** — luôn cần `HF_TOKEN`.
