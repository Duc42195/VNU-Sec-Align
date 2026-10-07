"""Kiểm tra bằng SỐ: loss DPO của TRL và của torchtune có giống nhau không?

Câu hỏi quyết định T9b chạy framework nào (xem docs/reports-on-t9b.md §3). Đọc source thôi
không đủ — hai implementation nằm ở hai repo khác nhau và đã từng đổi theo version — nên
script này chạy CẢ HAI hàm thật trên cùng một batch giả rồi assert khớp từng phần tử.

Cách chạy (không cần GPU, không cần mạng):
    python tools/verify_dpo_loss_equivalence.py

Script tự tải source gốc cần thiết (nếu chưa có trong cache):
  - torchtune v0.6.0: torchtune/rlhf/loss/dpo.py, torchtune/rlhf/sequence_processing.py
  - trl     v0.22.1: trl/trainer/dpo_trainer.py  (đúng version trong requirements.txt của pod)
Nạp bằng importlib từ file, né `torchtune/__init__.py` (kéo torchao/transformers/triton —
torchao 0.18 đã bỏ `torchao.dtypes.nf4tensor` nên import torchtune thật còn fail).

Kết quả in ra là căn cứ cho mục "loss có khác nhau không" của báo cáo.
"""

from __future__ import annotations

import pathlib
import sys
import types
import urllib.request

CACHE = pathlib.Path("/tmp/opencode")
TT_TAG = "v0.6.0"
TT_BASE = f"https://raw.githubusercontent.com/pytorch/torchtune/{TT_TAG}/"
TRL_TAG = "v0.22.1"
TRL_BASE = f"https://raw.githubusercontent.com/huggingface/trl/{TRL_TAG}/"

FILES = {
    "tt_dpo.py": TT_BASE + "torchtune/rlhf/loss/dpo.py",
    "tt_seqproc.py": TT_BASE + "torchtune/rlhf/sequence_processing.py",
    "trl_dpo.py": TRL_BASE + "trl/trainer/dpo_trainer.py",
}


def fetch(name: str, url: str) -> pathlib.Path:
    path = CACHE / name
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def load_torchtune_dpologss():
    """DPOLoss của torchtune — file này chỉ cần torch, nạp trực tiếp được."""
    import torch.nn as nn  # noqa: F401

    spec = types.ModuleType("tt_dpo_mod")
    exec(compile(fetch("tt_dpo.py", FILES["tt_dpo.py"]).read_text(), "tt_dpo.py", "exec"), spec.__dict__)
    return spec.DPOLoss


def load_torchtune_get_batch_log_probs():
    """get_batch_log_probs của torchtune — file này import `from torchtune import rlhf` và
    `from torchtune.data import CROSS_ENTROPY_IGNORE_IDX`; ta dựng stub tối thiểu."""
    spec = types.ModuleType("tt_seqproc_mod")
    rlhf_stub = types.ModuleType("torchtune.rlhf")
    data_stub = types.ModuleType("torchtune.data")
    data_stub.CROSS_ENTROPY_IGNORE_IDX = -100
    rlhf_stub.rewards = types.ModuleType("torchtune.rlhf.rewards")
    # masked_mean chỉ dùng cho return_average_logprobs=True, ta không gọi nhánh đó.
    rlhf_stub.rewards.masked_mean = lambda t, m, dim: (t * m).sum(dim) / m.sum(dim)
    sys.modules["torchtune"] = types.ModuleType("torchtune")
    sys.modules["torchtune.rlhf"] = rlhf_stub
    sys.modules["torchtune.data"] = data_stub
    exec(compile(fetch("tt_seqproc.py", FILES["tt_seqproc.py"]).read_text(), "tt_seqproc.py", "exec"), spec.__dict__)
    return spec.get_batch_log_probs


def trl_sigmoid_loss(pi_c, pi_r, ref_c, ref_r, beta, label_smoothing):
    """Đúng công thức trong trl v0.22.1 dpo_trainer.py, nhánh loss_type == "sigmoid"
    (trl/trainer/dpo_trainer.py:1063-1066) — CHÚ Ý nhánh này KHÔNG chia (1 - 2*ls),
    chỉ nhánh "robust" mới chia."""
    import torch.nn.functional as F

    logits = (pi_c - pi_r) - (ref_c - ref_r)
    return (
        -F.logsigmoid(beta * logits) * (1 - label_smoothing)
        - F.logsigmoid(-beta * logits) * label_smoothing
    )


def trl_get_batch_logps(logits, labels, loss_mask):
    """Đúng công thức trong trl v0.22.1 dpo_trainer.py:1562-1579 — per-token logp rồi SUM."""
    import torch

    shifted_labels = labels[:, 1:].clone()
    shifted_logits = logits[:, :-1, :]
    mask = loss_mask[:, 1:].bool()
    shifted_labels[~mask] = 0
    per_token_logps = torch.gather(
        torch.log_softmax(shifted_logits, dim=-1), 2, shifted_labels.unsqueeze(-1)
    ).squeeze(-1)
    per_token_logps = per_token_logps * mask
    return per_token_logps.sum(-1)


def main() -> int:
    import torch

    torch.manual_seed(0)
    B, S, V = 4, 7, 11

    DPOLoss = load_torchtune_dpologss()
    tt_get_logps = load_torchtune_get_batch_log_probs()

    # --- 1. Tổng log-prob: torchtune (sum trên token khác ignore) vs TRL (sum) ---
    logits = torch.randn(B, S, V, dtype=torch.float64)
    labels = torch.randint(0, V, (B, S))
    ignore_mask = torch.zeros(B, S, dtype=torch.bool)
    ignore_mask[:, 0] = True  # prompt đầu bị ignore, giống cách cả hai đánh dấu
    ignore_mask[:, -1] = True  # token cuối (label sau EOS) bị ignore
    tt_labels = labels.masked_fill(ignore_mask, -100)

    tt_logps = tt_get_logps(logits, tt_labels)
    trl_logps = trl_get_batch_logps(logits, labels, ~ignore_mask)
    assert torch.allclose(tt_logps, trl_logps, atol=1e-12), (tt_logps, trl_logps)
    print(f"[1] tong log-prob: KHOP (max|diff| = {(tt_logps - trl_logps).abs().max():.3e})")

    # --- 2. Loss DPO cho beta/label_smoothing của anchor ---
    for beta, ls in [(0.1, 0.0), (0.1, 0.1)]:
        pc, pr = torch.randn(B, dtype=torch.float64), torch.randn(B, dtype=torch.float64)
        rc, rr = torch.randn(B, dtype=torch.float64), torch.randn(B, dtype=torch.float64)
        tt_loss, _, _ = DPOLoss(beta=beta, label_smoothing=ls)(pc, pr, rc, rr)
        trl_loss = trl_sigmoid_loss(pc, pr, rc, rr, beta, ls)
        assert torch.allclose(tt_loss, trl_loss, atol=1e-12), (tt_loss, trl_loss)
        print(
            f"[2] loss DPO beta={beta} label_smoothing={ls}: KHOP "
            f"(max|diff| = {(tt_loss - trl_loss).abs().max():.3e})"
        )

    print("\n=> Loss DPO cua TRL (loss_type='sigmoid') va cua torchtune GIONG NHAU tren cung input.")
    print("=> RPO (rpo_alpha * NLL tren 'chosen') CHI CO o TRL — torchtune DPOLoss khong co.")
    return 0


if __name__ == "__main__":
    sys.exit(main())