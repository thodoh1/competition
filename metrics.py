"""
metrics.py – Evaluation metrics for the competition.

Metrics implemented
-------------------
1. PSNR  (Peak Signal-to-Noise Ratio)          weight 30 / 28
2. SSIM  (Structural Similarity Index)          weight 30 / 22
3. LPIPS (Learned Perceptual Image Patch Sim.)  weight 20 / 10
4. Edge  (Sobel edge-map correlation)           weight 20 / 10

All functions operate on (B, 1, H, W) float32 tensors in [0, 1].
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict


# ──────────────────────────────────────────────
# 1. PSNR
# ──────────────────────────────────────────────

def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Compute mean PSNR over a batch.
    Inputs: (B, 1, H, W) float32 in [0, 1].
    """
    with torch.no_grad():
        mse = F.mse_loss(pred, target, reduction="none")
        mse = mse.view(mse.shape[0], -1).mean(dim=1)          # (B,)
        psnr = 10 * torch.log10(max_val ** 2 / (mse + 1e-8))  # (B,)
    return psnr.mean().item()


# ──────────────────────────────────────────────
# 2. SSIM
# ──────────────────────────────────────────────

def _ssim_kernel(kernel_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Create a 2-D Gaussian kernel for SSIM."""
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g.outer(g)
    return kernel.view(1, 1, kernel_size, kernel_size)


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    kernel_size: int = 11,
    sigma: float = 1.5,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> float:
    """Compute mean SSIM over a batch (grayscale)."""
    kernel = _ssim_kernel(kernel_size, sigma).to(pred.device)
    pad    = kernel_size // 2

    with torch.no_grad():
        mu1    = F.conv2d(pred,   kernel, padding=pad)
        mu2    = F.conv2d(target, kernel, padding=pad)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2= mu1 * mu2

        sigma1_sq = F.conv2d(pred   * pred,   kernel, padding=pad) - mu1_sq
        sigma2_sq = F.conv2d(target * target, kernel, padding=pad) - mu2_sq
        sigma12   = F.conv2d(pred   * target, kernel, padding=pad) - mu1_mu2

        num   = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        den   = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = num / (den + 1e-8)

    return ssim_map.mean().item()


# ──────────────────────────────────────────────
# 3. LPIPS (using built-in torchvision VGG)
# ──────────────────────────────────────────────

class LPIPSMetric:
    """
    Lightweight LPIPS approximation using VGG-16 features.
    (Full LPIPS requires the lpips package; this is a compatible fallback.)

    Try to import the official 'lpips' library first; fall back to VGG-based.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        try:
            import lpips
            self._fn = lpips.LPIPS(net="vgg").to(device)
            self._fn.eval()
            self._use_official = True
            print("[LPIPS] Using official lpips library (vgg)")
        except ImportError:
            self._use_official = False
            self._vgg_feat = self._build_vgg_feat(device)
            print("[LPIPS] lpips not installed – using VGG-feature approximation")

    @staticmethod
    def _build_vgg_feat(device):
        import torchvision.models as models
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        # Use up to relu3_3
        feat = torch.nn.Sequential(*list(vgg.features.children())[:18])
        feat.to(device).eval()
        for p in feat.parameters():
            p.requires_grad = False
        return feat

    def _gray_to_3ch(self, x: torch.Tensor) -> torch.Tensor:
        """Grayscale [0,1] → [-1,1] 3-channel for LPIPS."""
        x3 = x.repeat(1, 3, 1, 1)
        return x3 * 2.0 - 1.0

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred   = pred.to(self.device)
        target = target.to(self.device)

        if self._use_official:
            d = self._fn(self._gray_to_3ch(pred), self._gray_to_3ch(target))
            return d.mean().item()
        else:
            # Normalise to ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1,3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1,3,1,1)
            p3   = (pred.repeat(1,3,1,1)   - mean) / std
            t3   = (target.repeat(1,3,1,1) - mean) / std
            fp   = self._vgg_feat(p3)
            ft   = self._vgg_feat(t3)
            # Normalise feature maps
            fp_n = F.normalize(fp.view(fp.shape[0], -1), dim=1)
            ft_n = F.normalize(ft.view(ft.shape[0], -1), dim=1)
            d    = 1.0 - (fp_n * ft_n).sum(dim=1)   # cosine distance
            return d.mean().item()


# ──────────────────────────────────────────────
# 4. Edge Preservation Score
# ──────────────────────────────────────────────

def compute_edge_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Edge preservation score based on Sobel gradient maps.
    Returns the mean Pearson correlation between pred and target edge maps.
    Range: [-1, 1]; higher is better.
    """
    kx = torch.tensor([[-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.]], device=pred.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.],
                        [ 0.,  0.,  0.],
                        [ 1.,  2.,  1.]], device=pred.device).view(1, 1, 3, 3)

    with torch.no_grad():
        def edge_map(img):
            gx = F.conv2d(img, kx, padding=1)
            gy = F.conv2d(img, ky, padding=1)
            return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

        ep = edge_map(pred).view(pred.shape[0], -1)
        et = edge_map(target).view(target.shape[0], -1)

        # Pearson correlation per image, then average
        ep_c = ep - ep.mean(dim=1, keepdim=True)
        et_c = et - et.mean(dim=1, keepdim=True)
        corr = (ep_c * et_c).sum(dim=1) / (
            ep_c.norm(dim=1) * et_c.norm(dim=1) + 1e-8
        )
    return corr.mean().item()


# ──────────────────────────────────────────────
# Unified evaluator
# ──────────────────────────────────────────────

class Evaluator:
    """
    Compute all four competition metrics on batches or single images.

    Usage:
        ev = Evaluator(device="cuda")
        metrics = ev.evaluate_batch(pred_tensor, target_tensor)
        print(metrics)
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.lpips  = LPIPSMetric(device)

    def evaluate_batch(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, float]:
        """
        Args:
            pred, target: (B, 1, H, W) float32 in [0, 1]
        Returns:
            dict with keys: psnr, ssim, lpips, edge
        """
        pred   = pred.to(self.device).float()
        target = target.to(self.device).float()
        return {
            "psnr": compute_psnr(pred, target),
            "ssim": compute_ssim(pred, target),
            "lpips": self.lpips(pred, target),
            "edge": compute_edge_score(pred, target),
        }

    def stage1_score(self, metrics: Dict[str, float],
                     all_metrics_list: list = None) -> float:
        """
        Compute the Stage 1 competition score (0-100).
        If all_metrics_list is provided, uses proper min-max normalisation.
        Otherwise uses simple clipping normalisation.

        Score = s_PSNR*30 + s_SSIM*30 + (1-s_LPIPS)*20 + s_Edge*20
        """
        if all_metrics_list:
            return self._normalised_score(metrics, all_metrics_list, stage=1)

        # Simple approximation (useful during training monitoring)
        s_psnr = min(metrics["psnr"] / 50.0, 1.0)          # 50 dB ≈ excellent
        s_ssim = max(min(metrics["ssim"], 1.0), 0.0)
        s_lpips= max(min(metrics["lpips"], 1.0), 0.0)       # lower is better
        s_edge = max(min((metrics["edge"] + 1) / 2.0, 1.0), 0.0)  # [-1,1]→[0,1]
        return s_psnr * 30 + s_ssim * 30 + (1 - s_lpips) * 20 + s_edge * 20

    @staticmethod
    def _normalised_score(m: dict, all_m: list, stage: int) -> float:
        """Min-max normalise and compute weighted score."""
        def _norm(key, lower_better=False):
            vals = [x[key] for x in all_m]
            mn, mx = min(vals), max(vals)
            if mx == mn:
                return 1.0
            s = (m[key] - mn) / (mx - mn)
            return (1 - s) if lower_better else s

        if stage == 1:
            return (
                _norm("psnr")  * 30 +
                _norm("ssim")  * 30 +
                _norm("lpips", lower_better=True) * 20 +
                _norm("edge")  * 20
            )
        else:   # stage 2 (params/size handled externally)
            return (
                _norm("psnr")  * 28 +
                _norm("ssim")  * 22 +
                _norm("lpips", lower_better=True) * 10 +
                _norm("edge")  * 10
            )
