"""Eğitilmiş bir Pix2Pix modelini val seti üzerinde değerlendirir.

Hesaplanan metrikler: FID, SSIM, PSNR, LPIPS, L1.

Kullanım:
  python -m src.evaluate \
      --config configs/pix2pix_default.yaml \
      --checkpoint results/pix2pix_default/checkpoints/pix2pix_epoch_0200.pth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import PairedImageDataset
from .models import Pix2PixModel
from .utils import ensure_dir, load_config, set_seed, tensor_to_image


def compute_pixel_metrics(real: np.ndarray, fake: np.ndarray) -> dict[str, float]:
    """SSIM, PSNR, L1 hesabı (uint8 görüntüler üzerinde)."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    ssim = structural_similarity(real, fake, channel_axis=-1, data_range=255)
    psnr = peak_signal_noise_ratio(real, fake, data_range=255)
    l1 = float(np.mean(np.abs(real.astype(np.float32) - fake.astype(np.float32))))
    return {"ssim": float(ssim), "psnr": float(psnr), "l1": l1}


def compute_lpips(real_t: torch.Tensor, fake_t: torch.Tensor, lpips_net) -> float:
    """LPIPS hesabı ([-1, 1] aralığında tensörler bekler)."""
    with torch.no_grad():
        d = lpips_net(fake_t, real_t)
    return float(d.item())


def compute_fid(real_dir: Path, fake_dir: Path) -> float:
    from pytorch_fid.fid_score import calculate_fid_given_paths
    return float(calculate_fid_given_paths(
        [str(real_dir), str(fake_dir)],
        batch_size=32, device="cuda" if torch.cuda.is_available() else "cpu",
        dims=2048, num_workers=2,
    ))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--out", type=str, default=None, help="Çıktı dizini (varsayılan: run_dir/eval).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(cfg["output"]["run_dir"])
    out_dir = ensure_dir(Path(args.out) if args.out else (run_dir / "eval"))
    real_dump = ensure_dir(out_dir / "real")
    fake_dump = ensure_dir(out_dir / "fake")

    val_ds = PairedImageDataset(
        root=cfg["data"]["root"], split="val",
        image_size=cfg["data"]["image_size"], augment={},
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Val örnek: {len(val_ds)}")

    model = Pix2PixModel(cfg, device=device)
    model.load(args.checkpoint, weights_only=True)
    model.eval()

    import lpips as lpips_lib
    lpips_net = lpips_lib.LPIPS(net="alex").to(device).eval()

    per_image: list[dict] = []
    for batch in tqdm(val_loader, desc="Değerlendirme"):
        fake = model.inference(batch["A"])

        real_img = tensor_to_image(batch["B"][0])
        fake_img = tensor_to_image(fake[0])
        name = Path(batch["name"][0]).stem

        cv2.imwrite(str(real_dump / f"{name}.png"), cv2.cvtColor(real_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(fake_dump / f"{name}.png"), cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR))

        px = compute_pixel_metrics(real_img, fake_img)
        lp = compute_lpips(batch["B"].to(device), fake, lpips_net)
        per_image.append({"name": name, **px, "lpips": lp})

    summary = {
        "ssim_mean":  float(np.mean([d["ssim"]  for d in per_image])),
        "psnr_mean":  float(np.mean([d["psnr"]  for d in per_image])),
        "l1_mean":    float(np.mean([d["l1"]    for d in per_image])),
        "lpips_mean": float(np.mean([d["lpips"] for d in per_image])),
    }

    print("FID hesaplanıyor (Inception aktivasyonları)...")
    summary["fid"] = compute_fid(real_dump, fake_dump)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_image": per_image}, f, indent=2, ensure_ascii=False)

    print("\n=== ÖZET ===")
    for k, v in summary.items():
        print(f"  {k:>12s} = {v:.4f}")
    print(f"\nDetay: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
