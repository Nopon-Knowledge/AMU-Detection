"""Evaluation and prediction use the pinned upstream preprocessing/NMS."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from ultralytics.models.yolo.detect import DetectionValidator, DetectionPredictor
from .model import load_detector


def evaluate(checkpoint, data, output, device="cpu", imgsz=640, batch=1, mode=None):
    model, metadata = load_detector(checkpoint)
    if mode:
        model.gate.mode = mode
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    validator = DetectionValidator(save_dir=output, args={
        "data": str(Path(data).resolve()), "device": device, "imgsz": imgsz,
        "batch": batch, "workers": 0, "conf": .001, "iou": .7, "max_det": 300,
        "half": False, "augment": False, "rect": False, "plots": False,
        "save_json": False, "split": "val", "task": "detect", "verbose": False})
    metrics = validator(model=model)
    payload = {"metrics": {k: float(v) for k, v in metrics.items()}, "metadata": metadata,
               "note": "AP values in metrics are fractions (multiply by 100 for paper percentages).",
               "gate_mode": model.gate.mode}
    ap = np.asarray(validator.metrics.box.all_ap)
    if ap.size:
        payload["AP75"] = float(ap[:, 5].mean())
    else:
        payload["AP75"] = 0.0
    (output / "metrics.json").write_text(json.dumps(payload, indent=2))
    return payload


def predict(checkpoint, source, output, device="cpu", imgsz=640, conf=.25):
    model, _ = load_detector(checkpoint)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    predictor = DetectionPredictor(overrides={"device": device, "imgsz": imgsz,
        "conf": conf, "iou": .7, "max_det": 300, "half": False, "save": True,
        "save_txt": True, "save_conf": True, "project": str(output.parent),
        "name": output.name, "exist_ok": False, "verbose": False})
    for _ in predictor(source=source, model=model, stream=True):
        pass
    return str(predictor.save_dir)


@torch.no_grad()
def benchmark(checkpoint, device="cpu", imgsz=640, warmup=200, iterations=1000, rounds=5):
    model, _ = load_detector(checkpoint, device)
    x = torch.zeros(1, 3, imgsz, imgsz, device=device)
    use_cuda = torch.device(device).type == "cuda"
    def synchronize():
        if use_cuda:
            torch.cuda.synchronize()
        elif torch.device(device).type == "mps":
            torch.mps.synchronize()
    for _ in range(warmup):
        model(x)
    synchronize()
    records = []
    for _ in range(rounds):
        samples = []
        for _ in range(iterations):
            if use_cuda:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                model(x)
                end.record()
                end.synchronize()
                elapsed = start.elapsed_time(end)
            else:
                synchronize()
                start = time.perf_counter()
                model(x)
                synchronize()
                elapsed = (time.perf_counter() - start) * 1000
            samples.append(elapsed)
        records.append({"p50_ms": float(np.percentile(samples, 50)),
                        "p95_ms": float(np.percentile(samples, 95))})
    return {"device": device, "imgsz": imgsz, "batch": 1, "precision": "FP32",
            "warmup": warmup, "iterations_per_round": iterations, "rounds": records,
            "parameters": sum(p.numel() for p in model.parameters()),
            "scope": "network only, no decode/preprocessing/NMS/transfer; unfused model"}
