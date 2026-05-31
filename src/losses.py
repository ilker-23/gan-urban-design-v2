"""Pix2Pix için kayıp fonksiyonları."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    """LSGAN / Vanilla / WGAN-GP varyantlı çekişmeli kayıp."""

    def __init__(self, mode: str = "lsgan", target_real: float = 1.0, target_fake: float = 0.0) -> None:
        super().__init__()
        self.mode = mode
        self.register_buffer("real_label", torch.tensor(target_real))
        self.register_buffer("fake_label", torch.tensor(target_fake))
        if mode == "lsgan":
            self.loss = nn.MSELoss()
        elif mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif mode == "wgangp":
            self.loss = None
        else:
            raise ValueError(f"Bilinmeyen GAN modu: {mode}")

    def _target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        t = self.real_label if target_is_real else self.fake_label
        return t.expand_as(prediction).to(prediction.device)

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        if self.mode == "wgangp":
            return -prediction.mean() if target_is_real else prediction.mean()
        target = self._target_tensor(prediction, target_is_real)
        return self.loss(prediction, target)


class VGGPerceptualLoss(nn.Module):
    """Opsiyonel VGG-tabanlı algısal kayıp. Pix2PixHD'de kullanılır."""

    def __init__(self, layers: tuple[int, ...] = (3, 8, 15, 22), weights: tuple[float, ...] = (1/32, 1/16, 1/8, 1/4)) -> None:
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg
        self.layers = layers
        self.weights = weights
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Pix2Pix çıkışları [-1, 1] aralığında — VGG için ImageNet normalize'a çeviriyoruz.
        x = (x + 1.0) / 2.0
        return (x - self.mean) / self.std

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        fake_n, real_n = self._normalize(fake), self._normalize(real)
        loss = torch.zeros((), device=fake.device)
        x_f, x_r = fake_n, real_n
        idx = 0
        for i, layer in enumerate(self.vgg):
            x_f = layer(x_f)
            x_r = layer(x_r)
            if i == self.layers[idx]:
                loss = loss + self.weights[idx] * F.l1_loss(x_f, x_r)
                idx += 1
                if idx >= len(self.layers):
                    break
        return loss
