"""Pix2Pix tam model wrapper'ı: G + D + kayıplar + optimizer.

Bir epoch sürecinde dataset üzerinden döngü, `step()` çağrılarıyla yürütülür.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.optim import Adam

from ..losses import GANLoss, VGGPerceptualLoss
from .networks import PatchDiscriminator, UNetGenerator, init_weights


class Pix2PixModel(nn.Module):
    def __init__(self, cfg: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.cfg = cfg

        m = cfg["model"]
        loss_cfg = cfg["loss"]
        opt_cfg = cfg["optim"]

        # Generator: A -> B
        self.G = UNetGenerator(
            in_channels=m["in_channels"],
            out_channels=m["out_channels"],
            num_downs=8,
            ngf=m["gen_features"],
            norm=m["norm"],
            use_dropout=m["use_dropout"],
        ).to(device)

        # PatchGAN discriminator (A,B) çiftini görür
        self.D = PatchDiscriminator(
            in_channels=m["in_channels"] + m["out_channels"],
            ndf=m["disc_features"],
            n_layers=3,
            norm=m["norm"],
        ).to(device)

        init_weights(self.G)
        init_weights(self.D)

        # Kayıplar
        self.criterion_gan = GANLoss(mode=loss_cfg["gan"]).to(device)
        self.criterion_l1 = nn.L1Loss()
        self.lambda_l1 = float(loss_cfg["lambda_l1"])
        self.lambda_perc = float(loss_cfg.get("lambda_perceptual", 0.0))
        self.criterion_perc: VGGPerceptualLoss | None = (
            VGGPerceptualLoss().to(device) if self.lambda_perc > 0 else None
        )

        # Optimizatörler
        self.opt_G = Adam(self.G.parameters(), lr=opt_cfg["lr"],
                          betas=(opt_cfg["beta1"], opt_cfg["beta2"]))
        self.opt_D = Adam(self.D.parameters(), lr=opt_cfg["lr"],
                          betas=(opt_cfg["beta1"], opt_cfg["beta2"]))

        # Eğitim sırasında doldurulur
        self.real_A: torch.Tensor | None = None
        self.real_B: torch.Tensor | None = None
        self.fake_B: torch.Tensor | None = None

    # ---------------------- Eğitim adımı ----------------------

    def set_input(self, batch: dict[str, torch.Tensor]) -> None:
        self.real_A = batch["A"].to(self.device, non_blocking=True)
        self.real_B = batch["B"].to(self.device, non_blocking=True)

    def forward(self) -> torch.Tensor:
        self.fake_B = self.G(self.real_A)
        return self.fake_B

    def _backward_D(self) -> dict[str, float]:
        # Fake çift
        fake_AB = torch.cat([self.real_A, self.fake_B.detach()], dim=1)
        pred_fake = self.D(fake_AB)
        loss_D_fake = self.criterion_gan(pred_fake, target_is_real=False)
        # Real çift
        real_AB = torch.cat([self.real_A, self.real_B], dim=1)
        pred_real = self.D(real_AB)
        loss_D_real = self.criterion_gan(pred_real, target_is_real=True)

        loss_D = 0.5 * (loss_D_fake + loss_D_real)
        loss_D.backward()
        return {"D": loss_D.item(), "D_fake": loss_D_fake.item(), "D_real": loss_D_real.item()}

    def _backward_G(self) -> dict[str, float]:
        fake_AB = torch.cat([self.real_A, self.fake_B], dim=1)
        pred_fake = self.D(fake_AB)
        loss_G_gan = self.criterion_gan(pred_fake, target_is_real=True)
        loss_G_l1 = self.criterion_l1(self.fake_B, self.real_B) * self.lambda_l1

        loss_G = loss_G_gan + loss_G_l1
        log: dict[str, float] = {"G_gan": loss_G_gan.item(), "G_l1": loss_G_l1.item()}

        if self.criterion_perc is not None:
            loss_perc = self.criterion_perc(self.fake_B, self.real_B) * self.lambda_perc
            loss_G = loss_G + loss_perc
            log["G_perc"] = loss_perc.item()

        loss_G.backward()
        log["G"] = loss_G.item()
        return log

    def step(self) -> dict[str, float]:
        self.forward()
        # D güncellemesi
        for p in self.D.parameters():
            p.requires_grad_(True)
        self.opt_D.zero_grad(set_to_none=True)
        log_d = self._backward_D()
        self.opt_D.step()
        # G güncellemesi
        for p in self.D.parameters():
            p.requires_grad_(False)
        self.opt_G.zero_grad(set_to_none=True)
        log_g = self._backward_G()
        self.opt_G.step()
        return {**log_d, **log_g}

    # ---------------------- Inference ----------------------

    @torch.no_grad()
    def inference(self, A: torch.Tensor) -> torch.Tensor:
        self.G.eval()
        out = self.G(A.to(self.device))
        self.G.train()
        return out

    # ---------------------- Kaydet / yükle ----------------------

    def save(self, path: str) -> None:
        torch.save({
            "G": self.G.state_dict(),
            "D": self.D.state_dict(),
            "opt_G": self.opt_G.state_dict(),
            "opt_D": self.opt_D.state_dict(),
            "cfg": self.cfg,
        }, path)

    def load(self, path: str, *, weights_only: bool = False) -> None:
        ckpt: Any = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["G"])
        self.D.load_state_dict(ckpt["D"])
        if not weights_only:
            self.opt_G.load_state_dict(ckpt["opt_G"])
            self.opt_D.load_state_dict(ckpt["opt_D"])
