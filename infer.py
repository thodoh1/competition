"""
infer.py – Inference entry point (required for Stage 2 submission).

Competition command-line interface:
    python src/infer.py \
        --input_dir  <path-to-LR-images> \
        --output_dir <path-to-save-SR-results> \
        --weights    <path-to-model.pth>

Additional options:
    --scale     2 or 4 (auto-detected from weights filename if omitted)
    --nf        feature channels (default 64)
    --nb        number of RRDB blocks (default 8)
    --tile      tile size for VRAM-limited inference (0 = full image, default 0)
    --device    cuda / cpu (auto-detected if omitted)
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# ── Allow running from repo root or src/ directory
sys.path.insert(0, str(Path(__file__).parent))
from model import InfraredSRNet


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def load_lr_image(path: str) -> torch.Tensor:
    """Load grayscale PNG → (1, 1, H, W) float32 tensor in [0, 1]."""
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def save_sr_image(tensor: torch.Tensor, path: str):
    """(1, 1, H, W) float32 tensor → 8-bit grayscale PNG."""
    arr = tensor.squeeze().cpu().float().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def infer_full(model: torch.nn.Module, lr: torch.Tensor, device: str) -> torch.Tensor:
    """Run full-image inference."""
    with torch.no_grad():
        return model(lr.to(device))


def infer_tiled(
    model: torch.nn.Module,
    lr: torch.Tensor,
    device: str,
    tile: int = 128,
    overlap: int = 16,
    scale: int = 2,
) -> torch.Tensor:
    """
    Tile-based inference for large images or limited VRAM.
    Tiles overlap by `overlap` pixels (LR space) to avoid seam artifacts.
    """
    _, _, H, W = lr.shape
    out_H, out_W = H * scale, W * scale
    output = torch.zeros(1, 1, out_H, out_W)
    weight = torch.zeros(1, 1, out_H, out_W)

    step = tile - overlap
    for y in range(0, H, step):
        for x in range(0, W, step):
            y2 = min(y + tile, H)
            x2 = min(x + tile, W)
            y1 = max(0, y2 - tile)
            x1 = max(0, x2 - tile)

            patch = lr[:, :, y1:y2, x1:x2]
            with torch.no_grad():
                sr_patch = model(patch.to(device)).cpu()

            # Target location in SR space
            sy1, sx1 = y1 * scale, x1 * scale
            sy2, sx2 = y2 * scale, x2 * scale

            output[:, :, sy1:sy2, sx1:sx2] += sr_patch
            weight[:, :, sy1:sy2, sx1:sx2] += 1.0

    return output / weight.clamp(min=1.0)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="InfraredSRNet Inference")
    p.add_argument("--input_dir",  required=True,  help="Directory of LR input images (.png)")
    p.add_argument("--output_dir", required=True,  help="Directory to write SR output images")
    p.add_argument("--weights",    required=True,  help="Path to model weights (.pth)")
    p.add_argument("--scale",      type=int, default=None, choices=[2, 4],
                   help="Upscale factor (auto-detected from weight filename if not set)")
    p.add_argument("--nf",         type=int, default=64)
    p.add_argument("--nb",         type=int, default=8)
    p.add_argument("--tile",       type=int, default=0,
                   help="Tile size in LR pixels (0 = no tiling, full image)")
    p.add_argument("--overlap",    type=int, default=16,
                   help="Tile overlap in LR pixels (used only when --tile > 0)")
    p.add_argument("--device",     default=None,
                   help="'cuda' or 'cpu' (auto-detected if not set)")
    return p.parse_args()


def detect_scale_from_filename(weights_path: str) -> int:
    """Heuristic: if 'stage2' or 'x4' appears in filename, use scale=4."""
    name = Path(weights_path).stem.lower()
    if "stage2" in name or "x4" in name or "4x" in name or "scale4" in name:
        return 4
    return 2


def main():
    args = parse_args()

    # ── Device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[Infer] device={device}")

    # ── Scale detection
    scale = args.scale or detect_scale_from_filename(args.weights)
    print(f"[Infer] scale={scale}×")

    # ── Load model
    model = InfraredSRNet.load_fp16(args.weights, scale=scale, nf=args.nf, nb=args.nb,
                                    device=device)
    model.to(device).eval()
    n_params = model.count_parameters()
    print(f"[Infer] Model loaded | params={n_params/1e6:.2f}M")

    # ── Input / output directories
    in_dir  = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(in_dir.glob("*.png"))
    if not img_paths:
        print(f"[Infer] WARNING: no .png files found in {in_dir}")
        return

    print(f"[Infer] Processing {len(img_paths)} images → {out_dir}")

    # ── Inference loop
    for i, img_path in enumerate(img_paths):
        lr = load_lr_image(str(img_path))

        if args.tile > 0:
            sr = infer_tiled(model, lr, device,
                             tile=args.tile, overlap=args.overlap, scale=scale)
        else:
            sr = infer_full(model, lr, device)

        out_path = out_dir / img_path.name
        save_sr_image(sr, str(out_path))

        if (i + 1) % 20 == 0 or (i + 1) == len(img_paths):
            print(f"  [{i+1}/{len(img_paths)}] {img_path.name} → {out_path.name}")

    print(f"[Infer] Done. {len(img_paths)} images saved to {out_dir}")


if __name__ == "__main__":
    main()
