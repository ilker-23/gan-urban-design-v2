"""Genel yardımcı işlevler: seed, config yükleme, görsel kaydetme, lr scheduler."""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import LambdaLR


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


_ENV_RE = re.compile(r"\$\{oc\.env:\s*([^,}]+)(?:,\s*([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    """`${oc.env:VAR, default}` syntax'ını ortam değişkenlerine genişletir."""
    if isinstance(value, str):
        def repl(m: re.Match[str]) -> str:
            var = m.group(1).strip()
            default = (m.group(2) or "").strip()
            return os.environ.get(var, default)
        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _expand_env(cfg)


def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    """[-1, 1] tensörünü uint8 HxWxC görüntüye çevirir."""
    t = t.detach().cpu().clamp(-1, 1)
    t = (t + 1.0) / 2.0
    t = (t * 255.0).round().to(torch.uint8)
    if t.dim() == 4:
        t = t[0]
    return t.permute(1, 2, 0).numpy()


def make_lambda_scheduler(optimizer: torch.optim.Optimizer, total_epochs: int, decay_start: int) -> LambdaLR:
    """Pix2Pix'in lineer LR decay scheduler'ı."""
    def lr_lambda(epoch: int) -> float:
        if epoch < decay_start:
            return 1.0
        denom = max(1, total_epochs - decay_start)
        return max(0.0, 1.0 - (epoch - decay_start) / denom)
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
