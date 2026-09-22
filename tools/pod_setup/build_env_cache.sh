#!/usr/bin/env bash
# tools/pod_setup/build_env_cache.sh
#
# Chạy 1 LẦN trên máy có mạng tốt (laptop, hoặc 1 pod mạng nhanh) — KHÔNG chạy trên pod
# mạng chậm. Mục đích: đóng gói (1) toàn bộ thư viện Python cần cho project dưới dạng cache
# đã tải sẵn, và (2) các file dữ liệu nhỏ mà external/meta_secalign/setup.py phải tải rải
# rác từ ~14 URL GitHub raw khác nhau — gộp lại thành 1 file duy nhất, up lên Hugging Face.
# Pod sau chỉ cần tải 1 file này về, không phải đụng PyPI/GitHub raw (nơi đã đo được nghẽn
# mạng nặng từ pod thuê ở Nga — xem .agents/infra_handoff.md).
#
# 2026-09-22: đổi từ pip sang `uv` sau khi `pip install torch` treo cứng 41 phút, chỉ tải
# được 13MB, kẹt ở 1 file 72MB (đứt kết nối liên tục, route riêng tới PyPI có vẻ nghẽn dù
# băng thông chung bình thường). `uv` tải song song nhiều luồng + resume tốt hơn hẳn.
#
# 2026-09-22 (lần 2): đổi từ tự chọn version "mới nhất" cho từng gói sang dùng THẲNG
# `external/meta_secalign/requirements.txt` (lockfile 265 dòng, đã pin đúng version Meta tự
# test) + `torchtune==0.6.0` cài riêng theo đúng README của Meta (mục Environment Setup) --
# lý do: tự chọn "mới nhất" cho torch/torchao/torchtune độc lập nhau làm 3 bản không tương
# thích (torchtune 0.6.0-nhánh-mới nhất cần torchao.dtypes.nf4tensor, module này đã bị đổi/xoá
# ở torchao bản mới nhất tính tới hôm nay) -- 2 lần chạy đầu đều fail ở bước `import torchtune`
# trong setup.py vì lý do này. requirements.txt của Meta là tổ hợp version đã CHỨNG MINH chạy
# được cùng nhau, không phải phỏng đoán.
#
# Lưu ý quan trọng: `uv` KHÔNG có lệnh tương đương `pip download` (chỉ có install/sync/compile).
# Thay vì tải .whl thô rồi cài lại trên pod (`pip install --no-index --find-links`), script này
# dùng `uv pip install --target <dir>` để tải THẲNG package ở dạng đã giải nén, khớp Python
# 3.13/Linux của pod (README Meta dùng `uv venv --python 3.13` -- không dùng system Python của
# pod, `uv` tự tải interpreter 3.13 riêng, nhỏ, không phụ thuộc pod có sẵn bản nào). Nhược điểm:
# --target không tạo console-script (vd. lệnh `huggingface-cli`) đúng path cho môi trường đích
# -- pod_init.sh vì vậy gọi qua Python API thay vì gọi thẳng lệnh CLI, xem file đó.
#
# Chỉ cần chạy lại khi: requirements.txt của Meta đổi, đổi version torchtune, hoặc setup.py của
# Meta đổi URL nguồn dữ liệu — KHÔNG cần chạy lại mỗi phiên làm việc.
#
# 2026-09-22 (lần 3): phát hiện numpy==1.26.4 (pin trong requirements.txt) không có wheel dựng sẵn
# cho Python 3.13 trên PyPI -- bước [2b/4] dưới đây phải tự build nó từ sdist NGAY TRÊN MÁY CHẠY
# SCRIPT NÀY, ra .so gắn chặt glibc của máy đó (không portable sang pod glibc cũ hơn -- lỗi thật
# gặp phải: "GLIBC_2.38 not found" khi import trên pod Ubuntu 22.04/glibc 2.35). KHÔNG đổi version
# numpy ở đây để né lỗi (numpy là dependency chung, đổi version có thể phá vỡ tổ hợp version Meta
# đã pin) -- thay vào đó, pod_init.sh tự `uv pip install --reinstall numpy==1.26.4` NGAY TRÊN POD
# sau khi copy cache, để build lại đúng theo glibc thật của từng pod. cupy-cuda12x/ray/vllm cũng
# có tag .dist-info khác thường ("linux_x86_64" không phải "manylinux*") nhưng xác minh là wheel
# PyPI thật (không phải build tại chỗ) -- reinstall thêm cho các gói này trong pod_init.sh chỉ là
# phòng hờ, không phải fix bắt buộc như numpy.
#
# Yêu cầu trước khi chạy: `uv` đã cài (curl -LsSf https://astral.sh/uv/install.sh | sh), và đã
# `huggingface-cli login`/có token HF_TOKEN với quyền write vào repo đích.

set -euo pipefail

# ==== SỬA 2 DÒNG NÀY TRƯỚC KHI CHẠY ====
# Token cache cục bộ (~/.cache/huggingface/token) chỉ có scope repo.write cho entity
# Jason-42195, không có quyền ghi vào Duc42195 (kiểm tra qua /api/whoami-v2 ngày 2026-09-22).
HF_REPO_ID="Jason-42195/vnu-secalign-env-cache"
HF_PRIVATE=true                                 # false nếu muốn repo public
# ========================================

REPO_ROOT="$(git rev-parse --show-toplevel)"
REQUIREMENTS_TXT="$REPO_ROOT/external/meta_secalign/requirements.txt"
TORCHTUNE_PIN="torchtune==0.6.0"
TORCHTUNE_INDEX="https://download.pytorch.org/whl/cu126"   # đúng README Meta -- không phải PyPI thường
POD_PYTHON_VERSION="3.13"   # khớp README Meta ("uv venv metasecalign --python 3.13")

# KHÔNG dùng mktemp -d mặc định (/tmp) -- trên máy này /tmp là tmpfs (RAM) chỉ 16GB, đã từng
# gây lỗi "Disk quota exceeded" giữa chừng khi torch+vllm tổng cộng vượt quá đó (2026-09-22).
# Dùng thư mục tạm trên ổ đĩa thật.
mkdir -p "$REPO_ROOT/.tmp_env_cache_build"
WORKDIR="$(mktemp -d "$REPO_ROOT/.tmp_env_cache_build/build.XXXXXX")"
POD_PKGS_DIR="$WORKDIR/pod_site_packages"   # package đã giải nén sẵn, khớp Python 3.13/Linux của pod
DATA_DIR="$WORKDIR/meta_secalign_data"
mkdir -p "$POD_PKGS_DIR" "$DATA_DIR"

echo "=== [1/4] Tạo venv tạm để chạy setup.py (cần cài THẬT, không chỉ tải package) ==="
uv venv "$WORKDIR/build_venv" --python "$POD_PYTHON_VERSION"
source "$WORKDIR/build_venv/bin/activate"

echo "=== [2/4] Cài thật vào venv, đúng thứ tự README Meta (requirements.txt trước, torchtune sau) ==="
uv pip install -r "$REQUIREMENTS_TXT"
uv pip install "$TORCHTUNE_PIN" --index-url "$TORCHTUNE_INDEX"
# bitsandbytes: KHÔNG có trong requirements.txt của Meta (họ dùng torchtune full-precision, dự
# án này dùng QLoRA qua TRL -- xem record.md Decision #5). Thêm riêng, không pin theo Meta vì họ
# không dùng gói này.
uv pip install bitsandbytes

echo "=== [2b/4] Tải riêng bản khớp Python $POD_PYTHON_VERSION/Linux cho pod (giải nén thẳng, không qua .whl thô) ==="
# KHÔNG ép --only-binary=:all: nữa (2026-09-22, lần 3): antlr4-python3-runtime==4.9.3 (kéo theo
# bởi requirements.txt của Meta) chỉ có sdist, không có wheel -- ép only-binary làm cả lệnh
# fail. Bỏ ràng buộc, cho phép build từ sdist khi cần (đa số case như thế này là pure-python,
# build không cần compiler khớp target).
uv pip install --target "$POD_PKGS_DIR" \
  --python-version "$POD_PYTHON_VERSION" --python-platform linux \
  -r "$REQUIREMENTS_TXT"
uv pip install --target "$POD_PKGS_DIR" \
  --python-version "$POD_PYTHON_VERSION" --python-platform linux \
  --index-url "$TORCHTUNE_INDEX" \
  "$TORCHTUNE_PIN"
uv pip install --target "$POD_PKGS_DIR" \
  --python-version "$POD_PYTHON_VERSION" --python-platform linux \
  bitsandbytes

echo "=== [3/4] Tải 14 data_urls nhỏ của Meta (SEP/CyberSecEval2/InjecAgent/...) ==="
# KHÔNG chạy `python3 setup.py` gốc nữa (2026-09-22, đã thử 3 lần, phát hiện dần từng lớp vấn
# đề): script đó, sau đoạn tải 14 URL nhỏ ta cần, còn tự snapshot_download() 5 model ĐẦY ĐỦ
# (gồm cả ~140GB meta-llama/Llama-3.3-70B-Instruct) và sau đó assert torch.cuda.device_count()>0
# để tự chạy vLLM sinh dữ liệu tham chiếu SEP -- hoàn toàn ngoài phạm vi script cache-package
# này (project tải/eval model qua registry.py/meta_bridge.py riêng, không qua setup.py). Dùng
# tools/pod_setup/fetch_meta_secalign_data_urls.py -- copy verbatim đúng 14 URL đó, không đụng
# gì khác, không sửa external/meta_secalign/ (đúng luật CLAUDE.md "không sửa trực tiếp").
python3 "$REPO_ROOT/tools/pod_setup/fetch_meta_secalign_data_urls.py"
cp -r "$REPO_ROOT/external/meta_secalign/data/." "$DATA_DIR/"

deactivate

echo "=== [4/4] Đóng gói + upload lên HF ($HF_REPO_ID) ==="
TARBALL="$WORKDIR/vnu_secalign_env_cache.tar.gz"
tar -czf "$TARBALL" -C "$WORKDIR" pod_site_packages meta_secalign_data

PRIVATE_FLAG=""
if [ "$HF_PRIVATE" = true ]; then PRIVATE_FLAG="--private"; fi

source "$WORKDIR/build_venv/bin/activate"   # huggingface-cli sống trong venv này, cài ở bước 2
huggingface-cli repo create "$HF_REPO_ID" --repo-type dataset $PRIVATE_FLAG -y || true
huggingface-cli upload "$HF_REPO_ID" "$TARBALL" vnu_secalign_env_cache.tar.gz --repo-type dataset
deactivate

echo "XONG. File tại HF: https://huggingface.co/datasets/$HF_REPO_ID/blob/main/vnu_secalign_env_cache.tar.gz"
echo "Dọn thư mục tạm: rm -rf $WORKDIR"
