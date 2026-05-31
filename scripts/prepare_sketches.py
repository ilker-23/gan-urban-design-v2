"""Maps dataset'inden sentetik kroki (sketch) çiftleri üretir.

Maps dataset'i her görseli 1200x600 olarak depolar:
  sol 600x600  = satellite aerial
  sağ 600x600  = Google Maps render (renkli plan)

Bu script:
  1) Her görseli sol/sağ olarak böler.
  2) Sağ taraftan (renkli map) Canny + XDoG ile sentetik B&W kroki üretir.
  3) Hem (sat, map) hem (sketch, map) çiftlerini ayrı klasörlere yazar.

Çıktı:
  data/processed/maps_paired/{train,val}/{A=sat, B=map}/<id>.png
  data/processed/maps_sketch/{train,val}/{A=sketch, B=map}/<id>.png

Kullanım:
  python scripts/prepare_sketches.py --input data/maps --output data/processed
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def split_pair(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """1200x600 yan yana yapıştırılmış görseli ortadan iki 600x600 görsele böler."""
    h, w = img.shape[:2]
    half = w // 2
    return img[:, :half], img[:, half:]


def make_sketch(color_img: np.ndarray, method: str = "canny_xdog") -> np.ndarray:
    """Renkli plan görüntüsünden B&W kroki sentezler.

    method:
      - 'canny'      : Sadece Canny edge.
      - 'xdog'       : Sadece XDoG (yumuşak çizgisel).
      - 'canny_xdog' : İkisinin birleşimi (önerilen).
    """
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    if method == "canny":
        edges = cv2.Canny(gray, 50, 150)
        sketch = 255 - edges  # beyaz arka plan, siyah çizgi
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    # XDoG (eXtended Difference of Gaussians)
    sigma = 0.8
    k = 1.6
    g1 = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    g2 = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma * k)
    dog = g1 - 0.98 * g2
    # eşik fonksiyonu
    eps = 0.1
    phi = 200
    xdog = np.where(dog >= eps, 1.0, 1.0 + np.tanh(phi * (dog - eps)))
    xdog = np.clip(xdog * 255.0, 0, 255).astype(np.uint8)

    if method == "xdog":
        return cv2.cvtColor(xdog, cv2.COLOR_GRAY2BGR)

    # Birleşim: XDoG + Canny edge'leri ekle
    canny = cv2.Canny(gray, 50, 150)
    combined = np.minimum(xdog, 255 - canny)
    return cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)


def process_split(
    src_split: Path,
    paired_out: Path,
    sketch_out: Path,
    method: str,
) -> int:
    """Tek bir split'i (train/val) işler. İşlenen görsel sayısını döndürür."""
    for sub in ("A", "B"):
        (paired_out / sub).mkdir(parents=True, exist_ok=True)
        (sketch_out / sub).mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src_split.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        print(f"  UYARI: {src_split} içinde görsel yok, atlanıyor.")
        return 0

    for f in tqdm(files, desc=f"  {src_split.name}"):
        img = cv2.imread(str(f))
        if img is None:
            continue
        sat, mp = split_pair(img)
        sketch = make_sketch(mp, method=method)

        stem = f.stem
        cv2.imwrite(str(paired_out / "A" / f"{stem}.png"), sat)
        cv2.imwrite(str(paired_out / "B" / f"{stem}.png"), mp)
        cv2.imwrite(str(sketch_out / "A" / f"{stem}.png"), sketch)
        cv2.imwrite(str(sketch_out / "B" / f"{stem}.png"), mp)
    return len(files)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/maps"),
                    help="Maps dataset kök dizini (içinde train/ ve val/ olmalı).")
    ap.add_argument("--output", type=Path, default=Path("data/processed"),
                    help="Çıktı kök dizini.")
    ap.add_argument("--method", choices=["canny", "xdog", "canny_xdog"],
                    default="canny_xdog",
                    help="Sketch üretim yöntemi.")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Girdi dizini yok: {args.input}\nÖnce 'bash scripts/download_data.sh maps' çalıştırın.")

    total = 0
    for split in ("train", "val"):
        src = args.input / split
        if not src.exists():
            print(f"  UYARI: {src} yok, atlanıyor.")
            continue
        paired_out = args.output / "maps_paired" / split
        sketch_out = args.output / "maps_sketch" / split
        print(f">> İşleniyor: {split}")
        total += process_split(src, paired_out, sketch_out, method=args.method)

    print(f"Bitti. Toplam {total} görsel işlendi.")
    print(f"Paired (sat<->map):   {args.output / 'maps_paired'}")
    print(f"Sketch (kroki<->map): {args.output / 'maps_sketch'}")


if __name__ == "__main__":
    main()
