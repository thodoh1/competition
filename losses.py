"""
losses.py – Combined loss for infrared super-resolution.

Components
----------
1. L1Loss           – pixel-level fidelity (stable, good for PSNR/SSIM)
2. PerceptualLoss   – VGG-16 feature-space loss (improves LPIPS)
3. EdgeLoss         – Sobel-based edge-preservation loss (scores edge metric)
4. CombinedLoss     – weighted sum of all three
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# 1. Pixel loss
# ──────────────────────────────────────────────

class L1Loss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


class CharbonnierLoss(nn.Module):
    """Charbonnier (robust L1) – smoother gradients near zero."""
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps ** 2))


# ──────────────────────────────────────────────
# 2. Perceptual loss (VGG-16, grayscale adapted)
# ──────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """
    VGG-16 feature-space loss for grayscale images.
    Repeats the single channel to 3 channels before passing to VGG.
    Uses relu2_2 features by default (good balance of low/high-level).
    """

    def __init__(self, layer_idx: int = 9):
        """
        Args:
            layer_idx: VGG feature layer index (default 9 = relu2_2).
                       Other useful values: 4 (relu1_2), 18 (relu3_3)
        """
        super().__init__()
        import torchvision.models as models
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features.children())[:layer_idx + 1])
        for p in self.features.parameters():
            p.requires_grad = False
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Grayscale [0,1] → ImageNet-normalised 3-channel."""
        x3 = x.repeat(1, 3, 1, 1)
        return (x3 - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_feat   = self.features(self._preprocess(pred))
        target_feat = self.features(self._preprocess(target))
        return F.l1_loss(pred_feat, target_feat)


# ──────────────────────────────────────────────
# 3. Edge-preservation loss (Sobel)
# ──────────────────────────────────────────────

class EdgeLoss(nn.Module):
    """
    Encourages the model to reconstruct sharp edges.
    Applies Sobel filters to both prediction and target,
    then computes L1 distance on the gradient magnitude maps.
    """

    def __init__(self):
        super().__init__()
        # Sobel kernels (horizontal and vertical)
        kx = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]]).view(1, 1, 3, 3)
        ky = torch.tensor([[-1., -2., -1.],
                            [ 0.,  0.,  0.],
                            [ 1.,  2.,  1.]]).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def _gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(
            self._gradient_magnitude(pred),
            self._gradient_magnitude(target),
        )


# ──────────────────────────────────────────────
# 4. Combined loss
# ──────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Weighted combination:
        L = w_pixel * L_pixel + w_perc * L_perc + w_edge * L_edge

    Default weights are tuned to match competition scoring emphasis:
        PSNR+SSIM (60%), LPIPS (20%), Edge (20%)
    """

    def __init__(
        self,
        w_pixel: float = 1.0,
        w_perc:  float = 0.1,
        w_edge:  float = 0.05,
        use_perceptual: bool = True,
    ):
        super().__init__()
        self.w_pixel = w_pixel
        self.w_perc  = w_perc
        self.w_edge  = w_edge
        self.use_perceptual = use_perceptual

        self.pixel_loss = CharbonnierLoss()
        self.edge_loss  = EdgeLoss()
        if use_perceptual:
            self.perc_loss = PerceptualLoss(layer_idx=9)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        l_pixel = self.pixel_loss(pred, target)
        l_edge  = self.edge_loss(pred, target)

        loss = self.w_pixel * l_pixel + self.w_edge * l_edge
        log  = {"pixel": l_pixel.item(), "edge": l_edge.item()}

        if self.use_perceptual:
            l_perc  = self.perc_loss(pred, target)
            loss   += self.w_perc * l_perc
            log["perc"] = l_perc.item()

        log["total"] = loss.item()
        return loss, log
