"""Pix2Pix için U-Net generator ve PatchGAN discriminator.

Referans: Isola vd. (2017) "Image-to-Image Translation with Conditional Adversarial
Networks", CVPR. Junyanz/pytorch-CycleGAN-and-pix2pix implementasyonu temel alındı;
minimal ve okunabilir hale getirildi.
"""

from __future__ import annotations

import functools

import torch
import torch.nn as nn


def get_norm_layer(norm: str = "instance") -> functools.partial:
    if norm == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    raise ValueError(f"Bilinmeyen norm: {norm}")


def init_weights(net: nn.Module, gain: float = 0.02) -> None:
    """Pix2Pix paper'da kullanılan ağırlık başlatma (N(0, 0.02))."""
    def init_fn(m: nn.Module) -> None:
        cls_name = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in cls_name or "Linear" in cls_name):
            nn.init.normal_(m.weight.data, 0.0, gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif "BatchNorm2d" in cls_name:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_fn)


# ---------------------------------------------------------------------------
# U-Net Generator
# ---------------------------------------------------------------------------

class UNetSkipBlock(nn.Module):
    """U-Net'in tekrarlanan blok birimi (encoder + decoder + skip)."""

    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        input_nc: int | None = None,
        submodule: nn.Module | None = None,
        outermost: bool = False,
        innermost: bool = False,
        norm_layer=nn.InstanceNorm2d,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, inplace=True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(inplace=True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                up.append(nn.Dropout(0.5))
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], dim=1)


class UNetGenerator(nn.Module):
    """Pix2Pix U-Net generator (256x256 girdi için 8 katmanlı U-Net)."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_downs: int = 8,           # 256 için 8, 128 için 7
        ngf: int = 64,
        norm: str = "instance",
        use_dropout: bool = True,
    ) -> None:
        super().__init__()
        norm_layer = get_norm_layer(norm)

        # En içteki blok (innermost)
        unet_block = UNetSkipBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                   norm_layer=norm_layer, innermost=True)
        # 3 adet aynı boyutlu ara blok
        for _ in range(num_downs - 5):
            unet_block = UNetSkipBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                       norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UNetSkipBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UNetSkipBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UNetSkipBlock(ngf,     ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UNetSkipBlock(out_channels, ngf, input_nc=in_channels, submodule=unet_block,
                                   outermost=True, norm_layer=norm_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# PatchGAN Discriminator
# ---------------------------------------------------------------------------

class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator (Pix2Pix orijinal)."""

    def __init__(self, in_channels: int = 6, ndf: int = 64, n_layers: int = 3, norm: str = "instance") -> None:
        super().__init__()
        norm_layer = get_norm_layer(norm)
        use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult, nf_mult_prev = 1, 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
