#!/usr/bin/env bash
# tools/pod_setup/build_env_cache.sh
#
# Chạy 1 LẦN trên máy có mạng tốt (laptop, hoặc 1 pod mạng nhanh) — KHÔNG chạy trên pod
# mạng chậm. Mục đích: đóng gói (1) toàn bộ thư viện Python cần cho project dưới dạng wheel
# đã build sẵn, và (2) các file dữ liệu nhỏ mà external/meta_secalign/setup.py phải tải rải
# rác từ ~14 URL GitHub raw khác nhau — gộp lại thành 1 file duy nhất, up lên Hugging Face.
# Pod sau chỉ cần tải 1 file này về, không phải đụng PyPI/GitHub raw (nơi đã đo được nghẽn
# mạng nặng từ pod thuê ở Nga — xem .agents/infra_handoff.md).
#
# Chỉ cần chạy lại khi: đổi danh sách thư viện, đổi version, hoặc setup.py của Meta đổi URL
# nguồn dữ liệu — KHÔNG cần chạy lại mỗi phiên làm việc.
#
# Yêu cầu trước khi chạy: `huggingface-cli login` (cần quyền write vào repo đích).

set -euo pipefail

# ==== SỬA 2 DÒNG NÀY TRƯỚC KHI CHẠY ====
# Đổi sang Jason-42195: token cache cục bộ (~/.cache/huggingface/token) chỉ có scope
# repo.write cho entity Jason-42195, không có quyền ghi vào Duc42195 (kiểm tra qua
# /api/whoami-v2 ngày 2026-09-22) -- Duc42195 sẽ lỗi 403 lúc repo create/upload.
HF_REPO_ID="Jason-42195/vnu-secalign-env-cache"
HF_PRIVATE=true                                 # false nếu muốn repo public
# ========================================

REPO_ROOT="$(git rev-parse --show-toplevel)"
WORKDIR="$(mktemp -d)"
WHEELS_DIR="$WORKDIR/wheels"
DATA_DIR="$WORKDIR/meta_secalign_data"
mkdir -p "$WHEELS_DIR" "$DATA_DIR"

echo "=== [1/4] Tạo venv tạm để chạy setup.py (cần cài THẬT, không chỉ tải wheel) ==="
python3 -m venv "$WORKDIR/build_venv"
source "$WORKDIR/build_venv/bin/activate"
pip install -q --upgrade pip

# Danh sách thư viện cố định 1 chỗ — sửa ở đây nếu project cần thêm/bớt gói.
PACKAGES=(torch transformers peft trl bitsandbytes accelerate datasets huggingface_hub \
          vllm torchtune alpaca_eval)

echo "=== [2/4] Cài thật (để chạy setup.py) + tải wheel cho pod (Linux x86_64, Python 3.10) ==="
pip install -q "${PACKAGES[@]}" wget

# Tải wheel đúng target của pod (KHÔNG phải target của máy đang chạy script này).
pip download --dest "$WHEELS_DIR" \
  --python-version 3.10 --implementation cp --abi cp310 \
  --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 \
  --only-binary=:all: \
  "${PACKAGES[@]}"
# wget (PyPI package) chỉ có sdist, không có wheel -> tải riêng, không ép --only-binary.
pip download --dest "$WHEELS_DIR" wget

echo "=== [3/4] Chạy setup.py gốc của Meta để gom data rải rác (SEP/CyberSecEval2/InjecAgent/...) ==="
pushd "$REPO_ROOT/external/meta_secalign" >/dev/null
python3 setup.py
popd >/dev/null
cp -r "$REPO_ROOT/external/meta_secalign/data/." "$DATA_DIR/"

deactivate

echo "=== [4/4] Đóng gói + upload lên HF ($HF_REPO_ID) ==="
TARBALL="$WORKDIR/vnu_secalign_env_cache.tar.gz"
tar -czf "$TARBALL" -C "$WORKDIR" wheels meta_secalign_data

PRIVATE_FLAG=""
if [ "$HF_PRIVATE" = true ]; then PRIVATE_FLAG="--private"; fi

huggingface-cli repo create "$HF_REPO_ID" --repo-type dataset $PRIVATE_FLAG -y || true
huggingface-cli upload "$HF_REPO_ID" "$TARBALL" vnu_secalign_env_cache.tar.gz --repo-type dataset

echo "XONG. File tại HF: https://huggingface.co/datasets/$HF_REPO_ID/blob/main/vnu_secalign_env_cache.tar.gz"
echo "Dọn thư mục tạm: rm -rf $WORKDIR"
