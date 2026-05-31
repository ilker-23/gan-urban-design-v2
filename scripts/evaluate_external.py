"""External modelin (junyanz/pix2pixHD) test çıktı klasörünü değerlendir:
FID, SSIM, PSNR, LPIPS, L1 — bizim src.evaluate ile aynı protokol.

junyanz çıktı yapısı (test.py sonrası):
  results/<name>/test_latest/images/
    <id>_real_A.png    (girdi)
    <id>_real_B.png    (gerçek hedef)
    <id>_fake_B.png    (üretim)

pix2pixHD çıktı yapısı:
  results/<name>/test_latest/images/
    <id>_input_label.png
    <id>_synthesized_image.png
    <id>_real_image.png

Kullanım:
  python scripts/evaluate_external.py \
      --results-dir external/pytorch-CycleGAN-and-pix2pix/results/sketch2map_cyc/test_latest/images \
      --pattern junyanz \
      --tag cyclegan_v1

  python scripts/evaluate_external.py \
      --results-dir external/pix2pixHD/results/sketch2map_hd/test_latest/images \
      --pattern pix2pixhd \
      --tag pix2pixhd_v1
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def collect_pairs(results_dir: Path, pattern: str) -> list[tuple[Path, Path]]:
    """(real_B, fake_B) çiftleri toplar."""
    files = sorted(results_dir.iterdir())
    pairs: list[tuple[Path, Path]] = []

    if pattern == "junyanz":
        # <id>_real_B.png  ve  <id>_fake_B.png
        reals = {re.sub(r"_real_B\.\w+$", "", f.name): f for f in files if "_real_B." in f.name}
        fakes = {re.sub(r"_fake_B\.\w+$", "", f.name): f for f in files if "_fake_B." in f.name}
        for k in reals:
            if k in fakes:
                pairs.append((reals[k], fakes[k]))

    elif pattern == "pix2pixhd":
        reals = {re.sub(r"_real_image\.\w+$", "", f.name): f for f in files if "_real_image." in f.name}
        fakes = {re.sub(r"_synthesized_image\.\w+$", "", f.name): f for f in files if "_synthesized_image." in f.name}
        for k in reals:
            if k in fakes:
                pairs.append((reals[k], fakes[k]))
    else:
        raise ValueError(f"Bilinmeyen pattern: {pattern}")
    return pairs


def to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    img = img.astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).contiguous()
    return t.to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--pattern", required=True, choices=["junyanz", "pix2pixhd"])
    ap.add_argument("--tag", required=True, help="Çıktı klasörü için etiket")
    ap.add_argument("--out", default="results/external_eval", type=Path)
    args = ap.parse_args()

    if not args.results_dir.exists():
        raise SystemExit(f"Çıktı dizini yok: {args.results_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = collect_pairs(args.results_dir, args.pattern)
    if not pairs:
        raise SystemExit("Çift bulunamadı; pattern doğru mu, klasörde test sonuçları var mı?")
    print(f"Çiftler: {len(pairs)}")

    out_dir = args.out / args.tag
    real_dump = out_dir / "real"; fake_dump = out_dir / "fake"
    real_dump.mkdir(parents=True, exist_ok=True)
    fake_dump.mkdir(parents=True, exist_ok=True)

    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    import lpips as lpips_lib
    lpips_net = lpips_lib.LPIPS(net="alex").to(device).eval()

    per: list[dict] = []
    for i, (rp, fp) in enumerate(tqdm(pairs, desc="Metrikler")):
        real = cv2.imread(str(rp)); fake = cv2.imread(str(fp))
        if real is None or fake is None:
            continue
        if real.shape != fake.shape:
            fake = cv2.resize(fake, (real.shape[1], real.shape[0]))

        # ortak dump (FID için lazım)
        cv2.imwrite(str(real_dump / f"{i:05d}.png"), real)
        cv2.imwrite(str(fake_dump / f"{i:05d}.png"), fake)

        real_rgb = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)
        fake_rgb = cv2.cvtColor(fake, cv2.COLOR_BGR2RGB)

        ssim = structural_similarity(real_rgb, fake_rgb, channel_axis=-1, data_range=255)
        psnr = peak_signal_noise_ratio(real_rgb, fake_rgb, data_range=255)
        l1 = float(np.mean(np.abs(real_rgb.astype(np.float32) - fake_rgb.astype(np.float32))))

        with torch.no_grad():
            d = lpips_net(to_tensor(fake_rgb, device), to_tensor(real_rgb, device))
        per.append({"name": rp.stem, "ssim": float(ssim), "psnr": float(psnr), "l1": l1, "lpips": float(d.item())})

    summary = {
        "ssim_mean":  float(np.mean([d["ssim"]  for d in per])),
        "psnr_mean":  float(np.mean([d["psnr"]  for d in per])),
        "l1_mean":    float(np.mean([d["l1"]    for d in per])),
        "lpips_mean": float(np.mean([d["lpips"] for d in per])),
    }

    print("FID hesaplanıyor (Inception)...")
    from pytorch_fid.fid_score import calculate_fid_given_paths
    summary["fid"] = float(calculate_fid_given_paths(
        [str(real_dump), str(fake_dump)],
        batch_size=32, device=str(device), dims=2048, num_workers=2,
    ))

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_image": per}, f, indent=2, ensure_ascii=False)

    print("\n=== ÖZET ({}): ===".format(args.tag))
    for k, v in summary.items():
        print(f"  {k:>12s} = {v:.4f}")


if __name__ == "__main__":
    main()
