# InfraredSRNet – AI ISP Competition Solution

Lightweight RRDB-based super-resolution for uncooled long-wave infrared images.  
Covers **Stage 1** (2× SR, 320×256 → 640×512) and **Stage 2** (4× SR, 160×128 → 640×512).

---

## Directory structure

```
ir_sr/
├── README.md
├── requirements.txt
├── weights/              ← trained model checkpoints
│   ├── best_model_fp16.pth
│   └── ...
├── src/
│   ├── model.py          ← InfraredSRNet architecture
│   ├── dataset.py        ← LR/HR paired dataset loader
│   ├── losses.py         ← Combined loss (pixel + perceptual + edge)
│   ├── metrics.py        ← PSNR, SSIM, LPIPS, Edge metrics
│   ├── train.py          ← Training script
│   ├── infer.py          ← Inference entry point (Stage 2 requirement)
│   └── evaluate.py       ← Batch evaluation + competition score
├── preliminary/          ← Stage 1 output (640×512 PNGs)
├── final_validation/     ← Stage 2 validation output
└── final_test/           ← Stage 2 no-GT test output
```

---

## Environment setup

```bash
pip install -r requirements.txt
# Optional (better LPIPS metric):
pip install lpips
```

Tested on: Python 3.10, PyTorch 2.1, CUDA 12.1.

---

## Model architecture

**InfraredSRNet** is a lightweight RRDB (Residual in Residual Dense Block) network:

| Component       | Details                                      |
|-----------------|----------------------------------------------|
| Head            | Conv 1→64                                    |
| Body            | 8 × RRDB (each = 3 RDB with 5 dense layers) |
| Upsampling      | PixelShuffle ×2 (or ×2 + ×2 for 4×)         |
| Tail            | Conv 64→32→1                                 |
| Parameters      | ~2.5 M                                       |
| Model size (FP16) | ~5 MB                                      |

**Loss function:** Charbonnier (pixel) + VGG perceptual + Sobel edge loss.

---

## Training

### Stage 1 – 2× super-resolution
```bash
python src/train.py \
    --lr_dir  data/训练数据集/input_320 \
    --hr_dir  data/训练数据集/target_640 \
    --scale   2 \
    --output  weights/stage1 \
    --epochs  200 \
    --batch   8 \
    --patch   128
```

### Stage 2 – 4× super-resolution (fine-tune from Stage 1)
```bash
python src/train.py \
    --lr_dir   data/训练数据集/input_160 \
    --hr_dir   data/训练数据集/target_640 \
    --scale    4 \
    --output   weights/stage2 \
    --epochs   200 \
    --pretrain weights/stage1/best_model.pth
```

**Tips:**
- Add `--amp` for faster training with automatic mixed precision on CUDA.
- Use `--no_perc` to skip perceptual loss if VGG download is unavailable.
- Increase `--nb 12` and `--nf 64` for higher quality (larger model).

---

## Inference

### Stage 1 (2×)
```bash
python src/infer.py \
    --input_dir  data/初赛测试集/input_320 \
    --output_dir preliminary/ \
    --weights    weights/stage1/best_model_fp16.pth \
    --scale      2
```

### Stage 2 – validation set (4×)
```bash
python src/infer.py \
    --input_dir  data/训练数据集/input_160 \
    --output_dir final_validation/ \
    --weights    weights/stage2/best_model_fp16.pth \
    --scale      4
```

### Stage 2 – test set without GT (4×)
```bash
python src/infer.py \
    --input_dir  data/决赛测试集/无监督 \
    --output_dir final_test/ \
    --weights    weights/stage2/best_model_fp16.pth \
    --scale      4
```

For GPU-memory-limited machines, add `--tile 128 --overlap 16`.

---

## Evaluation

```bash
python src/evaluate.py \
    --pred_dir preliminary/ \
    --gt_dir   data/初赛测试集/target_640 \
    --stage    1
```

Outputs per-image PSNR/SSIM/LPIPS/Edge and an estimated competition score.

---

## Model precision (competition requirement)

All weights are saved in **FP16** (half precision) using `model.save_fp16()`.  
FP32 and FP64 are explicitly prohibited by the competition rules.

To verify:
```python
import torch
state = torch.load("weights/stage2/best_model_fp16.pth")
for k, v in state.items():
    assert v.dtype == torch.float16, f"{k}: {v.dtype}"
print("All parameters are FP16 ✓")
```

---

## Submission checklist

### Stage 1
- [x] `preliminary/` – 640×512 PNGs matching input filenames
- [x] `README.md`

### Stage 2
- [x] `preliminary/` – Stage 1 results
- [x] `final_validation/` – 4× validation results
- [x] `final_test/` – 4× no-GT test results
- [x] `weights/model.pth` – FP16 model weights
- [x] `src/infer.py` – inference entry point
- [x] `src/model.py`, `src/*.py` – all code
- [x] `README.md`, `requirements.txt`
