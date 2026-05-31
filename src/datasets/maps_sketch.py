"""Pix2Pix tipi (A, B) eşli görüntü dataset'i.

Beklenen dizin yapısı:
  <root>/<split>/A/<id>.png  (girdi: sketch veya satellite)
  <root>/<split>/B/<id>.png  (hedef: renkli map)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class PairedImageDataset(Dataset):
    """Eşli görsel dataset'i. [-1, 1] aralığında PyTorch tensörleri döndürür."""

    IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 256,
        augment: dict | None = None,
    ) -> None:
        self.root = Path(root) / split
        self.dir_a = self.root / "A"
        self.dir_b = self.root / "B"
        if not self.dir_a.exists() or not self.dir_b.exists():
            raise FileNotFoundError(
                f"{self.dir_a} veya {self.dir_b} bulunamadı. "
                f"Önce 'python scripts/prepare_sketches.py' çalıştırın."
            )

        self.files = sorted(
            p.name for p in self.dir_a.iterdir()
            if p.suffix.lower() in self.IMG_EXT and (self.dir_b / p.name).exists()
        )
        if not self.files:
            raise RuntimeError(f"{self.dir_a} içinde eşli görsel bulunamadı.")

        self.image_size = image_size
        self.augment = augment or {}
        self.is_train = split == "train"

    def __len__(self) -> int:
        return len(self.files)

    def _load(self, p: Path) -> np.ndarray:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Okunamadı: {p}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        img = img.astype(np.float32) / 127.5 - 1.0   # [-1, 1]
        return torch.from_numpy(img.transpose(2, 0, 1)).contiguous()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        name = self.files[idx]
        a = self._load(self.dir_a / name)
        b = self._load(self.dir_b / name)

        # Aynı boyuta getir
        a = cv2.resize(a, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        if self.is_train:
            if self.augment.get("h_flip", False) and np.random.rand() < 0.5:
                a = a[:, ::-1, :].copy()
                b = b[:, ::-1, :].copy()
            if self.augment.get("rotate90", False):
                k = np.random.randint(0, 4)
                if k:
                    a = np.rot90(a, k).copy()
                    b = np.rot90(b, k).copy()
            if self.augment.get("color_jitter_target_only", False) and np.random.rand() < 0.5:
                # Sadece hedef tarafta hafif renk dalgalanması
                shift = np.random.uniform(-15, 15, size=3).astype(np.float32)
                b = np.clip(b.astype(np.float32) + shift, 0, 255).astype(np.uint8)

        return {"A": self._to_tensor(a), "B": self._to_tensor(b), "name": name}
