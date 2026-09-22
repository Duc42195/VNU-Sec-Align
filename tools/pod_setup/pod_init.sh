#!/usr/bin/env bash
# tools/pod_setup/pod_init.sh
#
# Chạy trên MỖI pod mới thuê (sau khi build_env_cache.sh đã chạy ít nhất 1 lần và có file
# trên HF). Không cần internet tốt tới PyPI/GitHub — chỉ cần tới Hugging Face.
#
# Cách dùng: dán toàn bộ nội dung file này vào terminal SSH của pod, hoặc:
#   curl -sL <raw-url-của-file-này-trên-github> | bash
# (chỉ dùng được cách curl sau khi repo đã push lên GitHub — xem ghi chú cuối file).

set -euo pipefail

# ==== SỬA THEO ĐÚNG GIÁ TRỊ BUILD_ENV_CACHE.SH ĐÃ DÙNG ====
# Đổi sang Jason-42195 (khớp build_env_cache.sh, 2026-09-22 -- token ghi được cục bộ chỉ có
# quyền cho entity này, xem ghi chú trong build_env_cache.sh).
HF_REPO_ID="Jason-42195/vnu-secalign-env-cache"
REPO_URL="https://github.com/Duc42195/VNU-Sec-Align.git"
REPO_BRANCH="main"   # KHÔNG phải "clean-main" -- đó chỉ là tên nhánh cục bộ trên laptop
# ===========================================================

echo "=== [1/6] Kiểm tra công cụ cơ bản (pod có thể là Ubuntu trần, không có gì cả) ==="
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git)
command -v python3 >/dev/null || (apt-get update -qq && apt-get install -y -qq python3 python3-pip python3-venv)
python3 --version
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
df -h / | tail -1

echo "=== [2/6] Clone/pull repo (nhánh $REPO_BRANCH) ==="
if [ -d ~/repo/.git ]; then
  cd ~/repo && git pull --quiet
else
  git clone --quiet "$REPO_URL" ~/repo
  cd ~/repo
fi
git checkout --quiet "$REPO_BRANCH"
git submodule update --init --quiet

echo "=== [3/6] Tạo venv + cài từ cache HF (không đụng PyPI) ==="
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install -q --upgrade pip huggingface_hub

CACHE_DIR=~/env_cache
mkdir -p "$CACHE_DIR"
huggingface-cli download "$HF_REPO_ID" vnu_secalign_env_cache.tar.gz \
  --repo-type dataset --local-dir "$CACHE_DIR"
tar -xzf "$CACHE_DIR/vnu_secalign_env_cache.tar.gz" -C "$CACHE_DIR"

df -h / | tail -1  # kiểm tra disk TRƯỚC khi cài (dừng thủ công nếu quá sát mép)
pip install --no-index --find-links="$CACHE_DIR/wheels" \
  torch transformers peft trl bitsandbytes accelerate datasets huggingface_hub \
  vllm torchtune alpaca_eval wget
df -h / | tail -1  # và SAU khi cài, để biết còn bao nhiêu cho model/data

echo "=== [4/6] Chép data nhỏ (SEP/CyberSecEval2/InjecAgent/...) vào đúng chỗ setup.py cũ trỏ tới ==="
cp -r "$CACHE_DIR/meta_secalign_data/." ~/repo/external/meta_secalign/data/

echo "=== [5/6] Đăng nhập HF để tải model gated (Llama-3.1-8B-Instruct, Meta-SecAlign-8B) ==="
if [ -z "${HF_TOKEN:-}" ]; then
  echo "CHƯA có HF_TOKEN trong biến môi trường."
  echo "Chạy: huggingface-cli login   (dán token có quyền read, đã accept license Llama-3.1)"
  echo "Sau đó tự chạy tiếp bước 6 (tải model) theo nhu cầu, không tự động ở đây."
else
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
fi

echo "=== [6/6] (Tuỳ chọn) Tải sẵn 4 model — bỏ comment dòng nào cần, disk 100GB đủ giữ cả 2 base model ==="
cat << 'EOF'
# Chạy tay khi cần (không tự động, vì không phải lúc nào cũng cần cả 4 ngay):
#   huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ~/models/llama_3_1_8b_instruct
#   huggingface-cli download facebook/Meta-SecAlign-8B --local-dir ~/models/meta_secalign_8b
#   huggingface-cli download SeaLLMs/SeaLLMs-v3-7B-Chat --local-dir ~/models/seallm_v3_7b_chat
#   huggingface-cli download Jason-42195/VNU-SecAlign --local-dir ~/models/jason_v1
EOF

echo "SETUP_DONE — kiểm tra 'python3 -c \"import torch; print(torch.cuda.is_available())\"' trước khi chạy T1-T3."
