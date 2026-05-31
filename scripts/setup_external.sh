#!/usr/bin/env bash
# Pix2PixHD ve SPADE karşılaştırma deneyleri için harici, doğrulanmış implementasyonları çek.
# Repo'ya commit etmiyoruz (gitignore'da external/), Colab'da çalışma zamanında klonlanır.

set -euo pipefail
EXT="${EXT:-external}"
mkdir -p "$EXT"
cd "$EXT"

# 1) Junyanz — Pix2Pix + CycleGAN
if [ ! -d pytorch-CycleGAN-and-pix2pix ]; then
  git clone --depth=1 https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
fi

# 2) NVlabs SPADE (GauGAN)
if [ ! -d SPADE ]; then
  git clone --depth=1 https://github.com/NVlabs/SPADE.git
fi

# 3) NVIDIA Pix2PixHD
if [ ! -d pix2pixHD ]; then
  git clone --depth=1 https://github.com/NVIDIA/pix2pixHD.git
fi

echo "Harici repolar $EXT/ altında hazır."
