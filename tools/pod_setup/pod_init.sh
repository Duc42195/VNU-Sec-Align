#!/usr/bin/env bash
# tools/pod_setup/pod_init.sh
#
# Chạy trên MỖI pod mới thuê (sau khi build_env_cache.sh đã chạy ít nhất 1 lần và có file
# trên HF). Không cần internet tốt tới PyPI/GitHub — chỉ cần tới Hugging Face (+ astral.sh cho
# uv, file nhỏ).
#
# Cách dùng: dán toàn bộ nội dung file này vào terminal SSH của pod, hoặc:
#   curl -sL <raw-url-của-file-này-trên-github> | bash
# (chỉ dùng được cách curl sau khi repo đã push lên GitHub — xem ghi chú cuối file).
#
# 2026-09-22: đổi sang uv + copy trực tiếp (không qua `pip install --no-index --find-links`
# nữa) -- xem build_env_cache.sh để hiểu vì sao (cache HF giờ chứa package đã giải nén sẵn
# khớp Python 3.13/Linux, không phải .whl thô). Hệ quả: KHÔNG có console-script (vd. lệnh
# `huggingface-cli`) cho các gói cài qua đường này -- toàn bộ thao tác HF trong file này dùng
# thẳng Python API (`huggingface_hub.login`/`snapshot_download`), không gọi CLI.
#
# 2026-09-22 (lần 2): venv pod giờ dùng Python 3.13 qua `uv venv --python 3.13` (uv tự tải
# interpreter riêng, KHÔNG dùng system python3 của pod nữa, dù pod có sẵn 3.10) -- khớp đúng
# README Meta ("uv venv metasecalign --python 3.13") và khớp bản package đã đóng gói trong cache
# HF (build_env_cache.sh cũng dùng 3.13). Dùng lẫn 2 version Python giữa lúc build cache và lúc
# copy vào venv thật sẽ ra site-packages sai ABI, import lỗi khó hiểu -- đã từng thấy hệ quả
# tương tự khi trộn version torchao/torchtune không khớp (xem build_env_cache.sh).

set -euo pipefail

# ==== SỬA THEO ĐÚNG GIÁ TRỊ BUILD_ENV_CACHE.SH ĐÃ DÙNG ====
HF_REPO_ID="Jason-42195/vnu-secalign-env-cache"
REPO_URL="https://github.com/Duc42195/VNU-Sec-Align.git"
REPO_BRANCH="main"   # KHÔNG phải "clean-main" -- đó chỉ là tên nhánh cục bộ trên laptop
# ===========================================================

# 2026-09-28: model(s) tự tải ở bước [5/5] -- danh sách key của registry.py, phân cách bằng dấu
# phẩy (KHÔNG phải HF id thô, để registry.py (src/vi_secalign/models/registry.py) luôn là nguồn
# sự thật duy nhất cho HF id -- sửa version/tên model chỉ cần sửa 1 chỗ, không phải sửa cả script
# này). Mặc định chỉ tải đúng 1 model thật sự cần ngay (base cho vi_preference_gen.py/train_dpo.py,
# xem .agents/infra_handoff.md "Việc còn lại" bước 2) -- KHÔNG mặc định tải thêm meta_secalign_8b
# (cần cho baseline T1-T3/so sánh T10 nhưng không chặn smoke-test) vì disk pod mới chưa biết chắc
# (pod cũ chỉ có ~50GB trống, 2 model 8B fp16 ~32GB đã sát mép -- xem manual.md Troubleshooting).
# Override khi cần, ví dụ:
#   MODELS="llama_3_1_8b_instruct,meta_secalign_8b" bash pod_init.sh
#   MODELS="" bash pod_init.sh   # bỏ qua tải model, tự tải tay sau (xem lệnh in ra cuối script)
MODELS="${MODELS:-llama_3_1_8b_instruct}"
export MODELS   # bước [5/5] đọc lại qua os.environ trong subprocess python3, phải export
# ===========================================================

echo "=== [1/5] Biến môi trường + công cụ cơ bản (pod có thể là Ubuntu trần, không có gì cả) ==="
# 2026-09-28: kiểm tra HF_TOKEN NGAY ĐẦU script, trước cả git clone/tạo venv -- cả cache riêng tư
# ở bước [3/5] lẫn model ở bước [5/5] đều cần nó. Thứ tự cũ (kiểm tra ở giữa bước 3, SAU khi đã
# clone+tạo venv) lỡ lãng phí đúng phần việc đó nếu thiếu token -- fail sớm ở đây tiết kiệm thời
# gian pod thật (pod tính tiền theo giờ, xem .agents/CLAUDE.md/feedback_pod_priority).
if [ -z "${HF_TOKEN:-}" ]; then
  echo "LỖI: chưa có HF_TOKEN trong biến môi trường -- cache thư viện (private) và model gated đều cần nó."
  echo "Chạy: export HF_TOKEN=hf_xxx   (token có quyền read, đã accept license Llama-3/Llama-3.1)"
  echo "Rồi chạy lại toàn bộ script."
  exit 1
fi
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git)
command -v python3 >/dev/null || (apt-get update -qq && apt-get install -y -qq python3 python3-pip python3-venv)
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"   # uv installer thường đặt vào đây, cần có trong PATH ngay
python3 --version
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
df -h / | tail -1

echo "=== [2/5] Clone/pull repo (nhánh $REPO_BRANCH) ==="
if [ -d ~/repo/.git ]; then
  cd ~/repo && git pull --quiet
else
  git clone --quiet "$REPO_URL" ~/repo
  cd ~/repo
fi
git checkout --quiet "$REPO_BRANCH"
git submodule update --init --quiet

echo "=== [3/5] Tạo venv (Python 3.13, khớp build_env_cache.sh) + tải cache từ HF ==="
uv venv ~/venv --python 3.13
source ~/venv/bin/activate
uv pip install huggingface_hub   # nhỏ, tải thẳng từ PyPI cũng nhanh -- chỉ cần để tải tarball cache

# repo cache ($HF_REPO_ID) là PRIVATE (build_env_cache.sh: HF_PRIVATE=true) -- HF_TOKEN đã kiểm tra
# tồn tại ở bước [1/5] (fail sớm trước khi tốn thời gian clone/venv nếu thiếu), dùng lại ở đây.
python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

CACHE_DIR=~/env_cache
mkdir -p "$CACHE_DIR"
python3 -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id='$HF_REPO_ID', filename='vnu_secalign_env_cache.tar.gz',
                     repo_type='dataset', local_dir='$CACHE_DIR', token='$HF_TOKEN')
print('downloaded:', p)
"
tar -xzf "$CACHE_DIR/vnu_secalign_env_cache.tar.gz" -C "$CACHE_DIR"

df -h / | tail -1  # kiểm tra disk TRƯỚC khi copy (dừng thủ công nếu quá sát mép)
SITE_PACKAGES="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "site-packages đích: $SITE_PACKAGES"
cp -r "$CACHE_DIR/pod_site_packages/." "$SITE_PACKAGES/"
df -h / | tail -1  # và SAU khi copy, để biết còn bao nhiêu cho model/data

# 2026-09-22: numpy==1.26.4 KHÔNG có wheel dựng sẵn cho Python 3.13 trên PyPI -- khi
# build_env_cache.sh chạy `uv pip install --target ...`, uv phải tự build numpy từ sdist NGAY
# TRÊN LAPTOP, ra 1 file .so gắn chặt glibc của laptop (2.43 trên máy build lần này). Copy sang
# pod glibc cũ hơn (vd Ubuntu 22.04, glibc 2.35) sẽ lỗi "GLIBC_2.38 not found" ngay khi import
# torch (torch import numpy nội bộ). Fix: cài lại numpy NGAY TRÊN POD (không qua cache) để nó tự
# build/tải đúng theo glibc thật của máy này. cupy-cuda12x/ray/vllm cũng từng bị nghi tương tự khi
# rà `Tag:` trong .dist-info (không phải "manylinux*") nhưng xác minh lại là wheel PyPI thật (nhà
# phát hành tự đóng gói vậy, không phải build tại chỗ) -- reinstall thêm cho chắc, không hại gì
# nếu bản cache đã đúng sẵn (uv sẽ chỉ redownload).
#
# QUAN TRỌNG (2026-09-22, phát hiện khi test thật trên pod): KHÔNG dùng
# `uv pip install --reinstall numpy==... cupy==... ray==... vllm==...` trực tiếp như bản đầu của
# fix này -- không truyền -r requirements.txt nghĩa là uv tự resolve lại TOÀN BỘ dependency graph
# theo "mới nhất tương thích" cho mọi gói KHÔNG được nêu tên trong lệnh, phá vỡ tổ hợp version
# Meta đã pin (hậu quả thật gặp phải: transformers 4.57.1 -> 5.17.0, huggingface-hub 0.36.0 ->
# 1.32.0, protobuf 5->7, starlette 0->1, openai 2->3 -- toàn bộ 148 gói bị resolve lại, không chỉ
# 4 gói cần sửa). Đúng cách: dùng `-r requirements.txt` làm ràng buộc đầy đủ, chỉ ép build lại
# riêng numpy bằng --reinstall-package (uv giữ nguyên version mọi gói khác theo lockfile).
echo "=== Cài lại numpy khớp glibc thật của pod (giữ nguyên mọi version khác theo requirements.txt) ==="
uv pip install -r ~/repo/external/meta_secalign/requirements.txt --reinstall-package numpy

echo "--- kiểm tra import (không phải chỉ copy xong là chắc chắn chạy được) ---"
python3 -c "
import torch, transformers, peft, trl, bitsandbytes, accelerate, vllm, torchtune, torchao
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('vllm', vllm.__version__)
"

echo "=== [4/5] Chép data nhỏ (SEP/CyberSecEval2/InjecAgent/...) vào đúng chỗ setup.py cũ trỏ tới ==="
cp -r "$CACHE_DIR/meta_secalign_data/." ~/repo/external/meta_secalign/data/
# Sinh data/CySE_prompt_injections.json từ prompt_injection.json đã có sẵn trong cache (CPU
# thuần, không cần mạng) -- cần cho run_cyberseceval2_pi_subtask() ở meta_eval_runner.py
# (2026-09-22: phát hiện thiếu bước này khi rà lại những gì setup.py gốc làm ngoài phần tải
# model/data thô -- xem docstring tools/pod_setup/fetch_meta_secalign_data_urls.py).
python3 ~/repo/tools/pod_setup/fetch_meta_secalign_data_urls.py

echo "=== [5/5] Tải model theo \$MODELS (mặc định: $MODELS) ==="
# Đã login HF ở bước [3/5] -- không cần login lại ở đây.
# Lưu ý disk thật (xem .agents/infra_handoff.md): pod cũ chỉ có ~50GB trống, KHÔNG phải 100GB --
# mỗi model 8B fp16 ~16GB/bản. Luôn kiểm tra df -h trước khi thêm model vào $MODELS. KHÔNG bao giờ
# thêm 70B (~140GB fp16) vào đây -- để dành pod khác lớn hơn, GĐ6.
#
# 2026-09-28: đổi từ chỉ IN lệnh (yêu cầu chạy tay) sang TỰ tải theo $MODELS -- key tra cứu qua
# chính src/vi_secalign/models/registry.py (import trực tiếp, không hardcode HF id lặp lại ở đây)
# để registry.py luôn là nguồn sự thật duy nhất. local_dir quy ước "~/models/<key>", khớp đúng
# đường dẫn manual.md/record.md đã dùng để tham chiếu model đã tải trên pod.
if [ -z "$MODELS" ]; then
  echo "MODELS rỗng -- bỏ qua tải model. Tải tay sau, ví dụ:"
  echo "  python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('facebook/Meta-SecAlign-8B', local_dir='$HOME/models/meta_secalign_8b')\""
else
  echo "Đang tải model ($MODELS) -- có thể mất 20-30 phút/model tuỳ tốc độ mạng thật của pod."
  echo "CHẬM KHÔNG CÓ NGHĨA LÀ TREO -- KHÔNG Ctrl+C giữa chừng (đã có checkpoint/resume của"
  echo "huggingface_hub, Ctrl+C giữa chừng chỉ làm mất tiến độ đã tải, không phải lỗi thật)."
  # hf-xet==1.2.0 đã pin trong requirements.txt (nằm sẵn trong cache bước [3/5]) -- huggingface_hub
  # tự dùng nó để tăng tốc cho repo hỗ trợ Xet, KHÔNG cần biến môi trường gì thêm. KHÔNG set
  # HF_HUB_ENABLE_HF_TRANSFER=1 ở đây -- đó là biến của gói `hf_transfer` (khác hf-xet), gói đó
  # KHÔNG có trong requirements.txt; set biến này mà thiếu gói sẽ làm huggingface_hub raise
  # ValueError ngay lập tức, không phải chỉ bỏ qua.
  df -h / | tail -1   # trước khi tải -- so sánh với sau để biết model chiếm bao nhiêu
  PYTHONPATH="$HOME/repo/src" python3 -c "
import os
from huggingface_hub import snapshot_download
from vi_secalign.models.registry import REGISTRY

keys = [k.strip() for k in os.environ['MODELS'].split(',') if k.strip()]
home = os.path.expanduser('~')
for key in keys:
    spec = REGISTRY.get(key)
    if spec is None:
        raise SystemExit(f'MODELS: khong tim thay key {key!r} trong registry.py. Cac key hop le: {sorted(REGISTRY)}')
    if '/' not in spec.source or spec.source.startswith(home):
        # local path (adapter chua train xong, vd phase2_final_adapter) -- khong phai HF id, bo qua
        print(f'[skip] {key}: source={spec.source!r} khong phai HF hub id, khong tai tu dong')
        continue
    # dọn lock mồ côi trước mỗi lần tải -- an toàn (không có tiến trình khác giữ lock thật ở đây)
    os.system('find ~/.cache/huggingface -iname \"*.lock\" -delete 2>/dev/null')
    local_dir = os.path.join(home, 'models', key)
    print(f'[downloading] {key} <- {spec.source} -> {local_dir}')
    # 2026-09-28: xác nhận thật trên pod (n2.ckey.vn:2500) -- thiếu ignore_patterns khiến
    # snapshot_download tải luôn thư mục original/ (vd. Llama-3.1-8B-Instruct co
    # original/consolidated.00.pth ~16GB, ban torchtune/native format trung lap hoan toan voi cac
    # .safetensors) -- transformers/vllm chi doc .safetensors, khong bao gio dung original/*, nen
    # day la lang phi thuan tuy: gan gap doi thoi gian tai (~40 phut thay vi ~20) va gan gap doi
    # dung luong dia cho MOI model. loai bo bang ignore_patterns.
    snapshot_download(spec.source, local_dir=local_dir, ignore_patterns=["original/*"])
    print(f'[done] {key}')
"
  df -h / | tail -1   # sau khi tải
fi

echo "SETUP_DONE — kiểm tra 'python3 -c \"import torch; print(torch.cuda.is_available())\"' trước khi chạy T1-T3."
