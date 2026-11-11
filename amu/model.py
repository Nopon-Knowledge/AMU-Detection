"""Pinned YOLO11n adapter. See docs/IMPLEMENTATION.md for reconstruction choices."""
import torch
import ultralytics
from ultralytics.nn.tasks import DetectionModel, yaml_model_load
from ultralytics.nn.modules import C3k2
from ultralytics.cfg import get_cfg
from ultralytics.utils.torch_utils import initialize_weights
from .routing import ShallowGateConcat


class UtilityDetector(DetectionModel):
    def __init__(self, nc=8, block=8, verbose=False):
        if ultralytics.__version__ != "8.3.40":
            raise RuntimeError("This adapter requires ultralytics==8.3.40; do not silently upgrade")
        config = yaml_model_load("yolo11n.yaml")
        super().__init__(config, ch=3, nc=nc, verbose=False)
        self.args = get_cfg()
        self.nc = nc
        self.names = {i: str(i) for i in range(nc)}
        self.block = block
        gate = ShallowGateConcat(block=block)
        gate.i, gate.f, gate.type = 15, [2, 14, 4], "ShallowGateConcat"
        gate.np = sum(p.numel() for p in gate.parameters())
        self.model[15] = gate
        # YOLO11n P3 C3k2: one repeat, 64 output channels, expansion .5.
        fusion = C3k2(320, 64, n=1, c3k=False, e=0.5)
        initialize_weights(fusion)  # match upstream BN eps/momentum and activation settings
        fusion.i, fusion.f, fusion.type = 16, -1, "C3k2"
        fusion.np = sum(p.numel() for p in fusion.parameters())
        self.model[16] = fusion
        self.save = sorted(set(self.save + [2, 14]))
        for p in self.gate.router.parameters():
            p.requires_grad_(False)
        if verbose:
            self.info(verbose=True)

    @property
    def gate(self):
        return self.model[15]

    def freeze_probe(self):
        self.eval()
        self.requires_grad_(False)
        self.gate.mode = "on"
        self.gate.override = None
        return self

    def prefix(self, image):
        """Cache all intervention-independent tensors up to the P3 junction."""
        cache, x = [], image
        for module in self.model[:15]:
            if module.f != -1:
                x = cache[module.f] if isinstance(module.f, int) else [x if j == -1 else cache[j] for j in module.f]
            x = module(x)
            cache.append(x)  # keep prefix tensors so tail references remain valid
        return cache

    def tail(self, prefix, mask=None):
        """Run P3/P4/P5 heads with unchanged backbone and reference features."""
        cache, x = list(prefix), prefix[-1]
        old = self.gate.override
        self.gate.override = mask
        try:
            for module in self.model[15:]:
                if module.f != -1:
                    x = cache[module.f] if isinstance(module.f, int) else [x if j == -1 else cache[j] for j in module.f]
                x = module(x)
                cache.append(x)
            return x
        finally:
            self.gate.override = old


def raw_predictions(output):
    return output[1] if isinstance(output, tuple) else output


def save_detector(model, path, metadata=None):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save({"format": "amu-v1", "nc": model.model[-1].nc, "block": model.block,
                "state_dict": state, "names": model.names, "gate_mode": model.gate.mode,
                "metadata": metadata or {}}, path)


def load_detector(path, device="cpu"):
    # Plain tensor/config checkpoint. Do not load arbitrary third-party pickle files.
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("format") != "amu-v1":
        raise ValueError("Expected amu-v1 checkpoint produced by this package")
    model = UtilityDetector(nc=data["nc"], block=data["block"])
    model.load_state_dict(data["state_dict"], strict=True)
    model.names = data["names"]
    model.gate.mode = data["gate_mode"]
    return model.to(device).eval(), data["metadata"]
