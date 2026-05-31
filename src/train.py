"""Pix2Pix eğitim girdisi.

Kullanım:
  python -m src.train --config configs/pix2pix_default.yaml
  python -m src.train --config configs/pix2pix_sat2map.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .datasets import PairedImageDataset
from .models import Pix2PixModel
from .utils import ensure_dir, load_config, make_lambda_scheduler, set_seed, tensor_to_image


def build_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    train_ds = PairedImageDataset(
        root=data_cfg["root"], split="train",
        image_size=data_cfg["image_size"], augment=data_cfg.get("augment", {}),
    )
    val_ds = PairedImageDataset(
        root=data_cfg["root"], split="val",
        image_size=data_cfg["image_size"], augment={},
    )
    train_loader = DataLoader(
        train_ds, batch_size=data_cfg["batch_size"], shuffle=True,
        num_workers=data_cfg["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True,
    )
    return train_loader, val_loader


def save_samples(model: Pix2PixModel, val_loader: DataLoader, out_dir: Path, epoch: int, n: int = 6) -> None:
    out = out_dir / f"epoch_{epoch:04d}"
    ensure_dir(out)
    it = iter(val_loader)
    for i in range(n):
        try:
            batch = next(it)
        except StopIteration:
            break
        fake = model.inference(batch["A"])
        a = tensor_to_image(batch["A"][0])
        b = tensor_to_image(batch["B"][0])
        f = tensor_to_image(fake[0])
        # [A | B (gerçek) | fake] yan yana
        grid = cv2.hconcat([
            cv2.cvtColor(a, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(b, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(f, cv2.COLOR_RGB2BGR),
        ])
        cv2.imwrite(str(out / f"sample_{i:02d}.png"), grid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--resume", type=str, default=None, help="Devam edilecek checkpoint dosyası.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])

    run_dir = ensure_dir(cfg["output"]["run_dir"])
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    sample_dir = ensure_dir(run_dir / "samples")
    log_dir = ensure_dir(run_dir / "tb")
    with open(run_dir / "config_resolved.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader = build_loaders(cfg)
    print(f"Train örnek: {len(train_loader.dataset)} | Val örnek: {len(val_loader.dataset)}")

    model = Pix2PixModel(cfg, device=device)
    if args.resume:
        print(f"Devam ediliyor: {args.resume}")
        model.load(args.resume)

    sched_G = make_lambda_scheduler(model.opt_G, cfg["train"]["epochs"], cfg["train"]["lr_decay_start_epoch"])
    sched_D = make_lambda_scheduler(model.opt_D, cfg["train"]["epochs"], cfg["train"]["lr_decay_start_epoch"])

    writer = SummaryWriter(log_dir=str(log_dir))
    global_step = 0
    epochs = cfg["train"]["epochs"]
    save_every = cfg["train"]["save_every_epochs"]
    log_every = cfg["train"]["log_every_steps"]

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{epochs}")
        running: dict[str, float] = {}
        for batch in pbar:
            model.set_input(batch)
            logs = model.step()
            for k, v in logs.items():
                running[k] = 0.9 * running.get(k, v) + 0.1 * v
            global_step += 1
            if global_step % log_every == 0:
                pbar.set_postfix({k: f"{v:.3f}" for k, v in running.items() if k in {"G", "D"}})
                for k, v in logs.items():
                    writer.add_scalar(f"loss/{k}", v, global_step)

        sched_G.step()
        sched_D.step()
        writer.add_scalar("lr/G", model.opt_G.param_groups[0]["lr"], epoch)
        dt = time.time() - t0
        print(f"  -> bitti, {dt:.1f}s | LR_G={model.opt_G.param_groups[0]['lr']:.2e}")

        # Görsel örnek üret
        save_samples(model, val_loader, sample_dir, epoch=epoch, n=6)

        # Checkpoint
        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = ckpt_dir / f"pix2pix_epoch_{epoch:04d}.pth"
            model.save(str(ckpt_path))
            print(f"  -> checkpoint kaydedildi: {ckpt_path}")

    writer.close()
    print("Eğitim tamamlandı.")


if __name__ == "__main__":
    main()
