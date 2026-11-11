"""Tiny synthetic images exercise interfaces; these are NOT research data."""
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
import yaml
from amu.training import train_probe, generate_labels, train_router, train_detection_router
from amu.evaluation import evaluate, predict, benchmark
from amu.model import load_detector


def make_dataset(root):
    rng = np.random.default_rng(10)
    for split, count in (("train", 4), ("val", 2)):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(count):
            image = rng.integers(0, 255, (112, 160, 3), dtype=np.uint8)
            cv2.rectangle(image, (48, 25), (112, 87), (255, 200, 10), -1)
            cv2.imwrite(str(root / "images" / split / f"{i}.jpg"), image)
            (root / "labels" / split / f"{i}.txt").write_text(f"{i % 2} 0.5 0.5 0.4 0.5\n")
    config = root / "data.yaml"
    config.write_text(yaml.safe_dump({"path": str(root), "train": "images/train", "val": "images/val",
                                     "names": {0: "synthetic_a", 1: "synthetic_b"}}))
    return config


def test_train_label_router_evaluate_predict(tmp_path):
    data = make_dataset(tmp_path / "data")
    checkpoint = train_probe(data, tmp_path / "probe", "cpu", epochs=1, batch=2, imgsz=128, workers=0)
    assert Path(checkpoint).is_file()
    manifest = generate_labels(checkpoint, data, tmp_path / "labels", imgsz=128, regions=2, draws=1)
    assert manifest["count"] == 8
    probe, _ = load_detector(checkpoint)
    assert all(v.dtype == torch.float32 for v in probe.state_dict().values() if v.is_floating_point())
    for target, shuffled, name in (("symmetric", False, "m9"), ("reassigned", False, "m8"), ("symmetric", True, "m10")):
        trained = train_router(checkpoint, tmp_path / "labels", tmp_path / name,
                               epochs=1, batch_size=4, target=target, shuffled=shuffled)
        restored, _ = load_detector(trained)
        for key, value in probe.state_dict().items():
            if "router" not in key:
                assert torch.equal(value, restored.state_dict()[key]), key
    m9 = tmp_path / "m9/router_final.pt"
    scores = evaluate(m9, data, tmp_path / "eval", imgsz=128)
    assert "metrics/mAP50-95(B)" in scores["metrics"]
    predictions = predict(m9, str(tmp_path / "data/images/val"), tmp_path / "pred", imgsz=128)
    assert len(list(Path(predictions).glob("*.jpg"))) == 2
    timing = benchmark(m9, imgsz=128, warmup=1, iterations=2, rounds=1)
    assert timing["rounds"][0]["p50_ms"] > 0
    m7 = train_detection_router(checkpoint, data, tmp_path / "m7", epochs=1, batch_size=2, imgsz=128)
    assert Path(m7).exists()


def test_no_label_leakage(tmp_path):
    from amu.model import UtilityDetector, save_detector
    data = make_dataset(tmp_path / "data")
    payload = yaml.safe_load(data.read_text())
    payload["val"] = payload["train"]
    data.write_text(yaml.safe_dump(payload))
    checkpoint = tmp_path / "probe.pt"
    save_detector(UtilityDetector(nc=2), checkpoint)
    with pytest.raises(ValueError, match="leakage"):
        generate_labels(checkpoint, data, tmp_path / "labels", imgsz=128)
