"""External GAN repolarının (junyanz / pix2pixHD / SPADE) beklediği veri
klasör yapısına uyum sağlamak için sembolik linkler kurar.

Kaynak yapı (bizim üretim sonrası):
  data/processed/maps_sketch/{train,val}/{A,B}/<id>.png

Hedef yapılar:
  data/external/junyanz/{trainA,trainB,testA,testB}/<id>.png       (CycleGAN için)
  data/external/junyanz_aligned/{train,test}/<id>.png              (Pix2Pix aligned için, AB yan yana)
  data/external/pix2pixhd/{train_A,train_B,test_A,test_B}/<id>.png (Pix2PixHD)

Symlink kullanır — disk yer kazanır. macOS/Linux'ta çalışır; Colab ortamı uyumludur.

Kullanım:
  python scripts/prepare_external_data.py
  python scripts/prepare_external_data.py --src data/processed/maps_paired   # E2 (sat→map) için
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def symlink_files(src_dir: Path, dst_dir: Path) -> int:
    """src_dir içindeki tüm görüntüleri dst_dir altına symlink olarak bağlar."""
    ensure_dir(dst_dir)
    count = 0
    for f in src_dir.iterdir():
        if f.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        link = dst_dir / f.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(f.resolve())
        count += 1
    return count


def make_aligned_pairs(a_dir: Path, b_dir: Path, out_dir: Path) -> int:
    """junyanz/aligned formatı: tek dosya, sol=A sağ=B (yan yana)."""
    ensure_dir(out_dir)
    files = sorted(p.name for p in a_dir.iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                   and (b_dir / p.name).exists())
    for name in tqdm(files, desc=f"  aligned -> {out_dir.name}"):
        a = cv2.imread(str(a_dir / name))
        b = cv2.imread(str(b_dir / name))
        if a is None or b is None:
            continue
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        ab = np.concatenate([a, b], axis=1)
        cv2.imwrite(str(out_dir / name), ab)
    return len(files)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/maps_sketch",
                    help="Kaynak dataset (içinde train/{A,B} ve val/{A,B} olmalı).")
    ap.add_argument("--out", default="data/external",
                    help="External repolar için çıktı kökü.")
    ap.add_argument("--format", choices=["all", "junyanz_unaligned", "junyanz_aligned", "pix2pixhd"],
                    default="all")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not (src / "train" / "A").exists():
        raise SystemExit(f"Kaynak yok: {src}/train/A — önce prepare_sketches.py çalıştırın.")

    splits = [("train", "train", "train"), ("val", "test", "test")]   # bizim val ↔ junyanz/pix2pixHD test

    # ---- 1) junyanz (CycleGAN) unaligned: trainA/B, testA/B ----
    if args.format in ("all", "junyanz_unaligned"):
        print(">> junyanz unaligned (CycleGAN) formatı:")
        base = out / "junyanz_cyc"
        for ours, junyanz_split, _ in splits:
            for ab in ("A", "B"):
                src_dir = src / ours / ab
                dst_dir = base / f"{junyanz_split}{ab}"
                n = symlink_files(src_dir, dst_dir)
                print(f"   {dst_dir}: {n} link")

    # ---- 2) junyanz aligned (Pix2Pix): AB yan yana ----
    if args.format in ("all", "junyanz_aligned"):
        print(">> junyanz aligned (Pix2Pix) formatı:")
        base = out / "junyanz_pix"
        for ours, _, aligned_split in splits:
            a_dir = src / ours / "A"
            b_dir = src / ours / "B"
            n = make_aligned_pairs(a_dir, b_dir, base / aligned_split)
            print(f"   {base / aligned_split}: {n} AB-birleşik")

    # ---- 3) Pix2PixHD: train_A/B, test_A/B ----
    if args.format in ("all", "pix2pixhd"):
        print(">> Pix2PixHD formatı:")
        base = out / "pix2pixhd"
        for ours, _, hd_split in splits:
            for ab in ("A", "B"):
                src_dir = src / ours / ab
                dst_dir = base / f"{hd_split}_{ab}"
                n = symlink_files(src_dir, dst_dir)
                print(f"   {dst_dir}: {n} link")

    print("Bitti.")


if __name__ == "__main__":
    main()
