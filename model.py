"""
InfraredSRNet – Lightweight RRDB super-resolution model
Supports 2× (Stage 1: 320×256 → 640×512)
     and 4× (Stage 2: 160×128 → 640×512)

Key design choices:
  • RRDB backbone (Residual in Residual Dense Blocks) – proven quality
  • 32 base channels, 8 RRDB blocks – ~2.5 M params, <10 MB in FP16
  • PixelShuffle upsampling (clean, no checkerboard)
  • Grayscale-in / Grayscale-out (1-channel)
  • Stored in FP16 to comply with competition precision rules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────

class DenseLayer(nn.Module):
    """Single dense connection layer: conv + leaky-relu."""
    def __init__(self, in_ch: int, growth: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, growth, 3, padding=1, bias=True)
        self.act  = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block (RDB) with 5 dense layers.
    Growth channel = 32; base channel = nf.
    """
    def __init__(self, nf: int = 64, gc: int = 32, res_scale: float = 0.2):
        super().__init__()
        self.res_scale = res_scale
        self.d1 = DenseLayer(nf,        gc)
        self.d2 = DenseLayer(nf + gc,   gc)
        self.d3 = DenseLayer(nf + 2*gc, gc)
        self.d4 = DenseLayer(nf + 3*gc, gc)
        self.d5 = nn.Conv2d(nf + 4*gc, nf, 3, padding=1, bias=True)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(torch.cat([x,  x1], 1))
        x3 = self.d3(torch.cat([x,  x1, x2], 1))
        x4 = self.d4(torch.cat([x,  x1, x2, x3], 1))
        x5 = self.d5(torch.cat([x,  x1, x2, x3, x4], 1))
        return x5 * self.res_scale + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block = 3 stacked RDBs."""
    def __init__(self, nf: int = 64, gc: int = 32, res_scale: float = 0.2):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc, res_scale)
        self.rdb2 = ResidualDenseBlock(nf, gc, res_scale)
        self.rdb3 = ResidualDenseBlock(nf, gc, res_scale)
        self.res_scale = res_scale

    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * self.res_scale + x


# ──────────────────────────────────────────────
# Main network
# ──────────────────────────────────────────────

class InfraredSRNet(nn.Module):
    """
    Lightweight RRDB-based infrared super-resolution network.

    Args:
        scale  : upscale factor – 2 or 4
        nf     : number of feature channels (default 64)
        nb     : number of RRDB blocks      (default 8)
        gc     : growth channels in RDB     (default 32)
    """
    def __init__(self, scale: int = 2, nf: int = 64, nb: int = 8, gc: int = 32):
        super().__init__()
        assert scale in (2, 4), "scale must be 2 or 4"
        self.scale = scale

        # Head: 1-ch grayscale → nf feature maps
        self.head = nn.Conv2d(1, nf, 3, padding=1, bias=True)

        # RRDB body
        body = [RRDB(nf, gc) for _ in range(nb)]
        self.body = nn.Sequential(*body)
        self.body_conv = nn.Conv2d(nf, nf, 3, padding=1, bias=True)

        # Upsampling tail
        # For 2×: one PixelShuffle(2)
        # For 4×: two PixelShuffle(2) stages
        self.up1 = nn.Sequential(
            nn.Conv2d(nf, nf * 4, 3, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        if scale == 4:
            self.up2 = nn.Sequential(
                nn.Conv2d(nf, nf * 4, 3, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.tail = nn.Sequential(
            nn.Conv2d(nf, nf // 2, 3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf // 2, 1, 3, padding=1, bias=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W)  – normalised [0, 1]
        Returns:
            sr: (B, 1, H*scale, W*scale) – clipped [0, 1]
        """
        feat = self.head(x)
        body_out = self.body_conv(self.body(feat))
        feat = feat + body_out          # global residual

        feat = self.up1(feat)
        if self.scale == 4:
            feat = self.up2(feat)

        out = self.tail(feat)
        return torch.clamp(out, 0.0, 1.0)

    # ── convenience helpers ──────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save_fp16(self, path: str):
        """Save weights in FP16 (competition requirement)."""
        self.half()
        torch.save(self.state_dict(), path)
        self.float()
        print(f"Saved FP16 weights → {path}")

    @classmethod
    def load_fp16(cls, path: str, scale: int = 2, nf: int = 64, nb: int = 8,
                  device: str = "cpu") -> "InfraredSRNet":
        """Load an FP16 checkpoint, return model in float32 for inference."""
        model = cls(scale=scale, nf=nf, nb=nb)
        state = torch.load(path, map_location=device)
        # Convert FP16 tensors back to FP32 for inference
        state_fp32 = {k: v.float() if v.dtype == torch.float16 else v
                      for k, v in state.items()}
        model.load_state_dict(state_fp32)
        model.eval()
        return model


# ──────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────
if __name__ == "__main__":
    for scale in (2, 4):
        H, W = (256, 320) if scale == 2 else (128, 160)
        model = InfraredSRNet(scale=scale, nf=64, nb=8)
        x     = torch.randn(1, 1, H, W)
        y     = model(x)
        params = model.count_parameters()
        print(f"Scale {scale}×  | input {H}×{W} → output {y.shape[-2]}×{y.shape[-1]} "
              f"| params {params/1e6:.2f}M")
