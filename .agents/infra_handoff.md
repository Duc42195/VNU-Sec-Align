# Infra handoff — trạng thái hạ tầng hiện tại

> File tạm, KHÔNG phải Decision log của record.md — chỉ ghi trạng thái hạ tầng/tiến độ setup
> để mở session Claude Code khác vẫn tiếp tục được ngay. Xoá file này sau khi setup xong hẳn
> và đã bắt đầu chạy N thật cho T9 (lúc đó thông tin ở đây hết giá trị).

## Pod cũ (ckey.vn #105728, RTX 3090 24GB) — ĐÃ XOÁ (2026-09-22)

Người dùng đã xoá pod này sau khi hoàn tất go/no-go pipeline test. SSH/thông tin máy cũ **không
còn hiệu lực**, đừng dùng lại. Không mất dữ liệu gì — mọi thứ quan trọng đã đẩy lên git (code) và
HF (data/checkpoint), xem mục "Trạng thái đã đạt được" bên dưới.

## Pod #2 (n2.ckey.vn:2961, "RTX 5090") — THẤT BẠI MỤC TIÊU, đang chờ xoá (2026-09-28)

Thuê để né vấn đề VRAM của 3090. Thực tế `nvidia-smi` cho thấy đây là **"RTX 5090 Laptop GPU",
chỉ 24463 MiB (~24GB) VRAM** — không phải bản desktop 32GB như kỳ vọng khi quyết định thuê. Kết
quả thật đo được trên pod này (xem `.agents/record.md` Decision #29 để có chi tiết đầy đủ):

1. **Không giải quyết được vấn đề gốc**: `train_dpo.py --max_length 2048` OOM thật ở step 2/21
   (`Tried to allocate 748MiB, 23.42GiB total, 581MiB free`) — **giống hệt** lỗi từng gặp trên
   3090. Compute capability 12.0 (Blackwell) không giúp gì cho vấn đề VRAM.
2. **Sinh data còn chậm hơn 3090**: `vi_preference_gen.py --n_samples 200` đạt 1.825 samples/s,
   thấp hơn 3090's 2.351 samples/s — khả năng do kernel CUDA/vLLM 0.11.0 chưa tối ưu tốt cho
   sm_120 (kiến trúc rất mới).
3. **Mạng tải file lớn chậm** (~11MB/s đo bằng `curl` cho model shard từ HF CDN) — không phải bug,
   nhưng khiến lần đầu tưởng nhầm là "treo" (xem bài học dưới).
4. **Bug thật tìm được — lock file mồ côi**: sau khi `kill -9` một tiến trình đang tải model giữa
   chừng, `~/.cache/huggingface/hub/.locks/**/*.lock` không được giải phóng sạch (nghi do overlay
   filesystem của container) → lần gọi `from_pretrained`/`snapshot_download` sau đó **treo thật vô
   thời hạn** cho tới khi xoá tay các file `.lock`. Rủi ro thật cho bất kỳ pod nào nếu 1 lần chạy
   bị ngắt giữa chừng ở bước tải rồi chạy lại — `pod_init.sh` nên tự dọn lock trước mỗi lần tải
   (`find ~/.cache/huggingface -iname "*.lock" -delete` trước mỗi `snapshot_download`).
5. **Bug thật khác — wandb crash khi chạy nền**: `nohup ... &` không có tty, `report_to` mặc định
   của `DPOConfig` kích hoạt wandb auto-init, wandb ném `UsageError: api_key not configured
   (no-tty)`, crash training ngay sau khi precompute ref log probs đã chạy xong (lãng phí ~1.5 phút
   pod + xoá sạch câu trả lời thật cho câu hỏi OOM). **Đã fix trong code**
   (`dpo_config.py`: `report_to="none"`), không cần nhớ set `WANDB_DISABLED` tay nữa.

**Dữ liệu vẫn giữ được**: `vn_preference_test200_5090.jsonl` (200 mẫu, sinh xong, đã upload HF
`pod_outputs/vi_preference_gen/vn_preference_test200_5090.jsonl`) — dùng được cho bất kỳ pod nào
sau này, không cần sinh lại.

**Tiêu chí sửa cho lần thuê tiếp theo** (bổ sung, không thay thế mục dưới):
- **Phải xác nhận rõ "desktop" không phải "laptop"** trong tên GPU của listing trước khi thuê —
  bài học thật lần này: tên listing ghi "5090" không đảm bảo VRAM 32GB, laptop SKU cùng dòng chỉ
  có 24GB. Hỏi host xác nhận VRAM cụ thể bằng số (GB) trước khi trả tiền, không suy ra từ tên GPU.
- Vẫn kiểm tra `nvidia-smi` ngay khi SSH vào (đã làm đúng lần này) — nhưng lần sau nên coi đây là
  điều kiện DỪNG NGAY nếu VRAM < 32GB, không thử "biết đâu kiến trúc mới đỡ hơn" nữa (đã thử, không
  đỡ).

## Pod #3 (n2.ckey.vn:2500, RTX 5090 DESKTOP thật 32GB) — THÀNH CÔNG, đang dùng (2026-09-28)

Xác nhận `nvidia-smi`: `NVIDIA GeForce RTX 5090, 32607 MiB, compute_cap 12.0` — đúng bản desktop,
đúng tiêu chí đã đặt ra. Kết quả đầy đủ ở `.agents/record.md` Decision #30, tóm tắt:

1. **`train_dpo.py --max_length 2048` chạy xong KHÔNG OOM** — vượt xa điểm OOM cũ (step 2/21) trên
   cả 3090 lẫn "5090 laptop". `train_runtime=258.6s` (so với 744.87s trên 3090 ở `max_length` THẤP
   HƠN 1536) — nhanh hơn nhiều dù khối lượng việc/step nặng hơn. loss 0.417→0.063,
   rewards/accuracies 85.8%→98.6%.
2. **`vi_preference_gen.py` throughput 4.534 samples/s** — gần gấp đôi 3090 (2.351) và 5090-laptop
   (1.825). Đây là hiệu năng thật của Blackwell khi mọi lớp hạ tầng (libcuda, VRAM) đều đúng.
3. **2 bug hạ tầng mới tìm được, đã fix**:
   - **`-lcuda` linker error khi Triton JIT compile** (`torch._inductor.exc.InductorError:
     ... -lcuda: No such file or directory`) — pod này chạy trên WSL (Windows Subsystem for Linux,
     thấy qua path `/usr/lib/wsl/drivers/...`), chỉ có `libcuda.so.1` không có symlink `libcuda.so`
     mà `-lcuda` cần ở link-time. Fix: `ln -sf /usr/lib/wsl/drivers/<driver>/libcuda.so.1
     /usr/lib/x86_64-linux-gnu/libcuda.so && ldconfig`. Đã verify: chạy lại `vi_preference_gen.py`
     ngay sau khi tạo symlink, pass ngay lần đầu.
   - **HF upload checkpoint thất bại nếu `--base_model` là local path**: auto-gen README.md của
     PEFT/Trainer đặt `base_model: <path local>` vào YAML frontmatter, HF Hub từ chối vì không phải
     model id hợp lệ (`"base_model" with value "..." is not valid`). Fix tạm: sửa tay dòng
     `base_model:` trong README.md thành HF id đúng (`meta-llama/Llama-3.1-8B-Instruct`) trước khi
     upload lại. **Chưa fix trong code** — nếu dùng local model path để tránh tải lại (khuyến nghị,
     xem dưới), nhớ việc này hoặc luôn truyền `--base_model` bằng HF id (vLLM/transformers tự dùng
     cache local nếu đã có, không tải lại mạng, nên dùng HF id vẫn nhanh).
4. **`snapshot_download` thiếu `ignore_patterns=["original/*"]`** (đã fix trong `pod_init.sh`, xem
   Decision #29) khiến lần tải `llama_3_1_8b_instruct` đầu tiên trên pod này mất **40 phút thay vì
   ~20** (tải dư ~16GB thư mục `original/`). Fix đã có sẵn cho lần tải model tiếp theo.

**Checkpoint + data đã upload HF**: `pod_outputs/train_dpo/dpo_vn200_5090desktop/` (checkpoint LoRA
thật, max_length=2048, không sai lệch phương pháp luận nào) và
`pod_outputs/vi_preference_gen/vn_preference_test200_5090desktop.jsonl`.

**Kết luận**: pod này đủ điều kiện dùng cho N thật (T9). Không cần thuê thêm pod nào khác trừ khi
disk/thời gian không đủ giữa chừng.

## Pod tiếp theo — cần thuê RTX ≥32GB VRAM THẬT (đã thử 1 lần "5090" không đạt, xem trên)

**Lý do đổi từ 3090 sang GPU ≥32GB VRAM** (xem `record.md` Decision #25, #26): 3090 (24GB) không
đủ dư để chạy đúng `max_length=2048` (giá trị anchor trích dẫn theo config gốc Meta,
`external/meta_secalign/helpers/llama3.1_8B_lora.yaml:94` vùng lân cận) — mọi lần thử trên 3090 đều
cần hạ xuống `--max_length 1536` hoặc thêm QLoRA mới chạy được, tức phải chấp nhận thêm 1 sai lệch
phương pháp luận. GPU ≥32GB (baseline model+LoRA+optimizer chiếm cố định ~19GB, không đổi theo GPU)
cho dư ra ≥13GB activation, đủ chạy `max_length=2048` ở `batch_size=1` mà không cần đánh đổi thêm.

**Tiêu chí chọn máy** (đã rút ra từ nhiều lần khảo giá thất bại tối 2026-09-22, xem record.md
Decision #26 để có bảng đầy đủ các listing đã xem xét):
- **Compute Capability ≥ 8.0 (Ampere trở lên)** — bắt buộc để có bf16 tensor core thật. Card đời
  Turing/Volta/Pascal (V100, Quadro RTX 8000/6000, Titan RTX, P100/P40) dù VRAM to vẫn KHÔNG đạt,
  vì code dùng `dtype=torch.bfloat16` xuyên suốt (khớp config gốc Meta) — chạy bf16 không tensor
  core chậm hơn cả 3090, phủ nhận lợi ích VRAM lớn.
- **RAM hệ thống**: không còn là điều kiện chặn cứng nữa — đã thêm `device_map="auto"` vào
  `train_dpo.py` (commit `c0cb54d`), model nạp thẳng vào VRAM khi đọc từng shard, không cần đủ RAM
  hệ thống giữ bản đầy đủ 15GB model. Vẫn nên ưu tiên máy RAM khá hơn nếu giá tương đương.
  Nếu chỉ dùng để train (không chạy `vi_preference_gen.py`/`sep_reference_gen.py` qua vLLM trên
  cùng máy), CPU yếu không phải vấn đề lớn (dataset nhỏ, không nặng preprocessing).
- **Trước khi thuê dài hạn cho N thật**: LUÔN chạy thử lệnh smoke-test 200 mẫu trước (xem dưới) để
  đo `giây/step` thật, áp vào công thức chi phí (record.md Decision #26) — đừng tin số ước lượng
  suông, tối nay đã sai ít nhất 1 lần (ước lượng ban đầu "3090 đủ" hoá ra sai).

## Trạng thái đã đạt được (không cần làm lại trên pod mới)

Tất cả các mục dưới đây đã **verify bằng chạy thật** trên pod 3090 cũ, code đã push lên git
(`origin/main`), dữ liệu/checkpoint đã upload HF (`Jason-42195/VNU-SecAlign`) — pod mới chỉ cần
`git clone`/`git pull` + `pip install -r requirements.txt` (hoặc dùng env cache đã đóng gói, xem
`tools/pod_setup/build_env_cache.sh`) + tải lại data/checkpoint từ HF nếu cần, KHÔNG cần sửa code
gì thêm cho các bước sau:

1. **T1-T3 prerequisite — `sep_reference_gen.py`**: chạy xong full 9160/9160 mẫu SEP
   (`results/pod_logs/sep_gen.txt`), verify khớp `setup.py:565-604` (record.md Decision #22).
   Output: `pod_outputs/sep_reference_gen/` trên HF.
2. **T8 smoke test — `vi_preference_gen.py --n_samples 200`**: sau khi sửa 3 lỗi thật (input=None,
   oversample factor, `max_model_len` OOM — Decision #23), chạy thành công **2.351 samples/s**
   (200 mẫu / 85.1s, `results/pod_logs/vi_preference.txt`). Dùng số này để ước lượng N cuối cho T9:
   N=12.500 (giữa khoảng 10-15K đã chốt sơ bộ, Decision #21) ≈ 12500/2.351 ≈ **~1h29m sinh thuần**.
3. **T9 pipeline smoke test — `train_dpo.py`**: sau 6 lần vá lỗi liên tiếp trên pod thật (dtype
   fp32→bf16, thiếu `pad_token`, batch_size mặc định quá lớn, gradient checkpointing không hoạt
   động với PEFT/frozen base, `max_length` cần hạ cho 24GB, và cuối cùng `precompute_ref_log_probs`
   để bỏ hẳn forward pass thứ 2 chồng bộ nhớ — xem record.md Decision #25), chạy thành công thật:
   loss 0.439→0.066, `rewards/accuracies` 83.4%→98.6%, `train_loss` TB=0.2427,
   `train_runtime`=744.87s cho N=200×3 epoch. Checkpoint đã upload:
   `pod_outputs/train_dpo/dpo_final/test_dpo_vn200/`. **Lưu ý: run này dùng `--max_length 1536`
   (sai lệch so với 2048 gốc) — không dùng lại config này cho N thật, xem mục "Pod tiếp theo" ở trên.**

## Việc còn lại theo đúng thứ tự (trên pod mới)

1. Thuê máy đạt tiêu chí ở trên (Ampere+, VRAM≥32GB), setup lại từ đầu (clone, venv/env cache,
   `huggingface-cli login`/`HF_TOKEN` — bắt buộc, model gated).
2. Chạy lại đúng lệnh smoke-test 200 mẫu (`vi_preference_gen.py --n_samples 200` rồi `train_dpo.py`
   với `--max_length 2048`, KHÔNG dùng 1536 nữa) để đo `giây/step` thật trên máy mới trước khi cam
   kết N lớn — xem lệnh đầy đủ trong `record.md` Decision #25/#26.
3. Chốt N cuối cùng cho EN:VN (Decision #21 vẫn treo) dựa trên throughput thật vừa đo.
4. Chạy `vi_preference_gen.py` với N thật (VN) — đã có sẵn resumability (`--checkpoint_every`,
   tự resume từ output cũ) nếu cần chạy qua nhiều lượt thuê.
5. Chạy `en_preference_gen.py` với N thật (EN) — **chưa từng chạy lần nào**, nhiều khả năng cũng
   dính đúng lỗi `max_model_len` chưa set trong `external/meta_secalign/utils.py` (đã biết, chưa
   vá — xem Decision #23) vì dùng chung `load_vllm_model`. Vá khi gặp, đừng giả định đã ổn.
6. Train N thật cho T9 (joint EN+VN) và/hoặc T9b (domain-incremental trên `meta_secalign_8b`,
   Decision #20) — dùng `--max_length 2048` đúng chuẩn, không cần `--max_length` override nữa nếu
   máy đủ VRAM.
7. T10/T10b: đánh giá VN_ASR (và EN_ASR cho T10b) trên **held-out**, so với baseline — đây mới là
   bằng chứng thật trả lời RQ2, không phải loss curve của smoke test.

## Quyết định liên quan (đã ghi ở record.md, không lặp lại chi tiết ở đây)

- Decision #13: hạ tầng thuê GPU theo giờ, không mua Colab Pro.
- Decision #18/#19: RQ1 có tín hiệu sơ bộ thật, người dùng đã xác nhận "Go" cho GĐ3.
- Decision #20/#21: nhánh domain-incremental song song; N cuối cùng (EN:VN) vẫn treo, cần đo
  throughput thật trước khi chốt.
- Decision #22/#23/#25: các lần chạy thật + lỗi/fix trên pod 3090 (chi tiết đầy đủ, không lặp ở đây).
- Decision #24: đối chiếu tường minh hyperparameter Meta (4×A100) vs 1 GPU đơn — cái gì giữ nguyên
  được, cái gì buộc phải đổi.
- Decision #26: quyết định thuê GPU ≥32GB Ampere+ thay 3090, đã khảo giá V100/RTX 8000/RTX 5090,
  công thức so sánh chi phí.
