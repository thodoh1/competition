"""
evaluate.py – Compute all competition metrics on a results folder.

Usage:
    python src/evaluate.py \
        --pred_dir  results/preliminary \
        --gt_dir    data/初赛测试集/target_640 \
        --stage     1

Output: per-image CSV + summary table with estimated competition score.
"""

import argparse
import csv
from pathlib import Path

import torch
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent))
from metrics import Evaluator


def load_gray_tensor(path: str) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--gt_dir",   required=True)
    p.add_argument("--stage",    type=int, default=1, choices=[1, 2])
    p.add_argument("--out_csv",  default=None, help="Optional CSV output path")
    p.add_argument("--device",   default="cpu")
    return p.parse_args()


def main():
    args      = parse_args()
    pred_dir  = Path(args.pred_dir)
    gt_dir    = Path(args.gt_dir)
    evaluator = Evaluator(device=args.device)

    pred_files = sorted(pred_dir.glob("*.png"))
    matched    = [(f, gt_dir / f.name) for f in pred_files if (gt_dir / f.name).exists()]

    if not matched:
        print(f"No matching files between {pred_dir} and {gt_dir}")
        return

    print(f"Evaluating {len(matched)} image pairs …")

    rows        = []
    all_metrics = []

    for pred_path, gt_path in matched:
        pred = load_gray_tensor(str(pred_path))
        gt   = load_gray_tensor(str(gt_path))
        m    = evaluator.evaluate_batch(pred, gt)
        all_metrics.append(m)
        rows.append({"file": pred_path.name, **m})

    # Compute means
    mean_m = {k: sum(r[k] for r in rows) / len(rows)
              for k in ("psnr", "ssim", "lpips", "edge")}

    # Estimated competition score
    score = evaluator.stage1_score(mean_m, all_metrics)

    # Print summary
    print("\n─── Evaluation Summary ─────────────────────────")
    print(f"  Images evaluated : {len(matched)}")
    print(f"  PSNR             : {mean_m['psnr']:.4f} dB")
    print(f"  SSIM             : {mean_m['ssim']:.4f}")
    print(f"  LPIPS            : {mean_m['lpips']:.4f}  (lower is better)")
    print(f"  Edge Score       : {mean_m['edge']:.4f}")
    print(f"  ── Estimated Stage {args.stage} Score: {score:.2f} / 100")
    print("────────────────────────────────────────────────\n")

    # Save CSV
    out_csv = args.out_csv or str(pred_dir / "eval_results.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "psnr", "ssim", "lpips", "edge"])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({"file": "MEAN",
                         "psnr": f"{mean_m['psnr']:.4f}",
                         "ssim": f"{mean_m['ssim']:.4f}",
                         "lpips": f"{mean_m['lpips']:.4f}",
                         "edge": f"{mean_m['edge']:.4f}"})

    print(f"Per-image results saved → {out_csv}")


if __name__ == "__main__":
    main()
