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

cd ..   # repo köküne dön (EXT'in bir üstü)

# Pix2PixHD'yi Python 3.12 / PyTorch 2.x ile uyumlu hale getir (fractions.gcd vb.)
if [ -f "$EXT/pix2pixHD/train.py" ]; then
  echo ">> pix2pixHD Python 3.12 uyumluluk yaması uygulanıyor..."
  python scripts/patch_pix2pixhd.py --repo "$EXT/pix2pixHD" || \
    echo "   (uyarı: patch script çalıştırılamadı; manuel kontrol gerekebilir)"
fi

echo "Harici repolar $EXT/ altında hazır."
