"""Equations (1), (4)-(6) and the probe's intervention distribution."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def regional_means(x, block):
    # ceil_mode supports partial boundary blocks without averaging padded zeros.
    return F.avg_pool2d(x, block, block, ceil_mode=True, count_include_pad=False)


def expand_regions(gates, block, height, width):
    return gates.repeat_interleave(block, -2).repeat_interleave(block, -1)[..., :height, :width]


class RegionalRouter(nn.Module):
    def __init__(self, channels=(64, 128, 128), projection=16, hidden=24, tau=1.0):
        super().__init__()
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.channels, self.tau = tuple(channels), tau
        self.projections = nn.ModuleList(nn.Linear(c, projection) for c in channels)
        self.predictor = nn.Sequential(nn.Linear(3 * projection, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.normal_(self.predictor[-1].weight, std=0.001)
        nn.init.zeros_(self.predictor[-1].bias)

    def forward(self, summaries):
        parts = summaries.split(self.channels, dim=-1)
        projected = [layer(part) for layer, part in zip(self.projections, parts)]
        return self.predictor(torch.cat(projected, -1)).squeeze(-1)

    def gains(self, summaries):
        return torch.exp(self(summaries).clamp(max=0) / self.tau)


def sample_masks(batch, rows, cols, *, device, dtype):
    """Per-image modes: .30 on, .05 off, .35 binary, .30 continuous.

    Engineering convention: unselected blocks stay on; selected binary blocks
    are removed, selected continuous blocks receive U(.1, 1).
    """
    mode = torch.rand(batch, 1, 1, 1, device=device)
    shape = (batch, 1, rows, cols)
    selected = torch.rand(shape, device=device) < 0.20
    gains = torch.ones(shape, device=device, dtype=dtype)
    gains = torch.where((mode >= .30) & (mode < .35), 0., gains)
    gains = torch.where((mode >= .35) & (mode < .70) & selected, 0., gains)
    continuous = .1 + .9 * torch.rand(shape, device=device, dtype=dtype)
    return torch.where((mode >= .70) & selected, continuous, gains)


class ShallowGateConcat(nn.Module):
    """S is aligned by area averaging; only g*S enters the concatenation."""
    def __init__(self, block=8, mode="exposure"):
        super().__init__()
        self.block, self.mode = block, mode
        self.router = RegionalRouter()
        self.override = None

    def align(self, inputs):
        shallow, topdown, intermediate = inputs
        shallow = F.adaptive_avg_pool2d(shallow, topdown.shape[-2:])
        if intermediate.shape[-2:] != topdown.shape[-2:]:
            raise ValueError("T and I must share the P3 grid")
        return shallow, topdown, intermediate

    def summaries(self, aligned):
        return torch.cat([regional_means(x, self.block) for x in aligned], 1).permute(0, 2, 3, 1)

    def forward(self, inputs):
        aligned = self.align(inputs)
        s, t, i = aligned
        b, _, h, w = s.shape
        rows, cols = math.ceil(h / self.block), math.ceil(w / self.block)
        if self.override is not None:
            g = self.override.to(device=s.device, dtype=s.dtype)
            if g.shape != (b, 1, rows, cols):
                raise ValueError(f"Expected mask {(b, 1, rows, cols)}, got {tuple(g.shape)}")
        elif self.mode == "router":
            g = self.router.gains(self.summaries(aligned)).unsqueeze(1)
        elif self.mode == "off":
            g = s.new_zeros((b, 1, rows, cols))
        elif self.mode == "exposure" and self.training:
            g = sample_masks(b, rows, cols, device=s.device, dtype=s.dtype)
        elif self.mode in ("on", "exposure"):
            g = s.new_ones((b, 1, rows, cols))
        else:
            raise ValueError(f"Unknown gating mode: {self.mode}")
        return torch.cat((expand_regions(g, self.block, h, w) * s, t, i), 1)
