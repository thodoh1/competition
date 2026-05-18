"""
train.py – Training script for InfraredSRNet.

Usage examples
--------------
# Stage 1 (2×): train on 320→640
python src/train.py \
    --lr_dir  data/训练数据集/input_320 \
    --hr_dir  data/训练数据集/target_640 \
    --scale   2 \
    --output  weights/stage1 \
    --epochs  200

# Stage 2 (4×): fine-tune from Stage 1 checkpoint
python src/train.py \
    --lr_dir   data/训练数据集/input_160 \
    --hr_dir   data/训练数据集/target_640 \
    --scale    4 \
    --output   weights/stage2 \
    --epochs   200 \
    --pretrain weights/stage1/best_model.pth
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model   import InfraredSRNet
from dataset import make_train_val_split
from losses  import CombinedLoss
from metrics import Evaluator


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="InfraredSRNet Training")
    p.add_argument("--lr_dir",    required=True,  help="Path to LR training images")
    p.add_argument("--hr_dir",    required=True,  help="Path to HR ground-truth images")
    p.add_argument("--scale",     type=int, default=2, choices=[2, 4])
    p.add_argument("--output",    default="weights", help="Directory to save checkpoints")
    p.add_argument("--epochs",    type=int, default=200)
    p.add_argument("--batch",     type=int, default=8)
    p.add_argument("--patch",     type=int, default=128,
                   help="HR patch size for training crops")
    p.add_argument("--nf",        type=int, default=64,  help="Feature channels")
    p.add_argument("--nb",        type=int, default=8,   help="Number of RRDB blocks")
    p.add_argument("--lr",        type=float, default=2e-4, help="Initial learning rate")
    p.add_argument("--val_every", type=int, default=5,   help="Validate every N epochs")
    p.add_argument("--workers",   type=int, default=4)
    p.add_argument("--pretrain",  default=None,
                   help="Path to pretrained weights (FP16 .pth)")
    p.add_argument("--no_perc",   action="store_true",
                   help="Disable perceptual loss (faster, lower LPIPS score)")
    p.add_argument("--amp",       action="store_true",
                   help="Use AMP (automatic mixed precision) during training")
    return p.parse_args()


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    torch.save({
        "epoch":     epoch,
        "metrics":   metrics,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    # Handle FP16 → FP32 conversion if needed
    state = {k: v.float() if v.dtype == torch.float16 else v
             for k, v in ckpt["model"].items()}
    model.load_state_dict(state, strict=False)
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("epoch", 0), ckpt.get("metrics", {})


# ──────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] device={device}  scale={args.scale}×  epochs={args.epochs}")

    # Output directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ────────────────────────────────────
    train_ds, val_ds = make_train_val_split(
        args.lr_dir, args.hr_dir,
        scale=args.scale,
        patch_size=args.patch,
        val_ratio=0.1,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=2,
    )

    # ── Model ───────────────────────────────────
    model = InfraredSRNet(scale=args.scale, nf=args.nf, nb=args.nb).to(device)
    print(f"[Train] Model params: {model.count_parameters()/1e6:.2f}M")

    if args.pretrain:
        epoch_start, _ = load_checkpoint(args.pretrain, model, device=device)
        print(f"[Train] Loaded pretrain from {args.pretrain} (epoch {epoch_start})")
    else:
        epoch_start = 0

    # ── Loss, Optimizer, Scheduler ──────────────
    criterion = CombinedLoss(
        w_pixel=1.0, w_perc=0.1, w_edge=0.05,
        use_perceptual=not args.no_perc,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    # Advance scheduler to match pretrain epoch
    for _ in range(epoch_start):
        scheduler.step()

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device == "cuda")
    evaluator = Evaluator(device=device)

    best_psnr = 0.0
    log_path  = out_dir / "training_log.csv"

    with open(log_path, "w") as f:
        f.write("epoch,train_loss,psnr,ssim,lpips,edge,lr\n")

    # ── Epoch loop ──────────────────────────────
    for epoch in range(epoch_start + 1, args.epochs + 1):
        model.train()
        t0         = time.time()
        total_loss = 0.0

        for lr_imgs, hr_imgs in train_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=args.amp and device == "cuda"):
                sr   = model(lr_imgs)
                loss, _ = criterion(sr, hr_imgs)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        lr_cur   = optimizer.param_groups[0]["lr"]

        # ── Validation ──────────────────────────
        val_metrics = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "edge": 0.0}

        if epoch % args.val_every == 0:
            model.eval()
            with torch.no_grad():
                for lr_imgs, hr_imgs in val_loader:
                    lr_imgs = lr_imgs.to(device)
                    hr_imgs = hr_imgs.to(device)
                    sr      = model(lr_imgs)
                    m       = evaluator.evaluate_batch(sr, hr_imgs)
                    for k in val_metrics:
                        val_metrics[k] += m[k]

            n = len(val_loader)
            val_metrics = {k: v / n for k, v in val_metrics.items()}

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"loss={avg_loss:.4f} | "
                f"PSNR={val_metrics['psnr']:.2f} | "
                f"SSIM={val_metrics['ssim']:.4f} | "
                f"LPIPS={val_metrics['lpips']:.4f} | "
                f"Edge={val_metrics['edge']:.4f} | "
                f"LR={lr_cur:.2e} | "
                f"t={elapsed:.1f}s"
            )

            # Save best
            if val_metrics["psnr"] > best_psnr:
                best_psnr = val_metrics["psnr"]
                best_path = out_dir / "best_model.pth"
                save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, best_path)
                # Also save FP16 version for submission
                model.save_fp16(str(out_dir / "best_model_fp16.pth"))
                print(f"  ✓ New best PSNR={best_psnr:.2f} → saved to {best_path}")

        else:
            elapsed = time.time() - t0
            print(f"Epoch {epoch:4d}/{args.epochs} | loss={avg_loss:.4f} | "
                  f"LR={lr_cur:.2e} | t={elapsed:.1f}s")

        # Periodic checkpoint
        if epoch % 50 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                out_dir / f"epoch_{epoch:04d}.pth"
            )

        # CSV log
        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_loss:.6f},"
                    f"{val_metrics['psnr']:.4f},{val_metrics['ssim']:.4f},"
                    f"{val_metrics['lpips']:.4f},{val_metrics['edge']:.4f},"
                    f"{lr_cur:.2e}\n")

    # ── Final save ──────────────────────────────
    final_path = out_dir / "final_model.pth"
    save_checkpoint(model, optimizer, scheduler, args.epochs, val_metrics, final_path)
    model.save_fp16(str(out_dir / "final_model_fp16.pth"))
    print(f"\n[Train] Done. Best PSNR: {best_psnr:.2f} dB")
    print(f"[Train] Weights saved to {out_dir}/")


if __name__ == "__main__":
    main()
