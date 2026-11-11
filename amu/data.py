"""Deterministic square letterbox and YOLO labels for utility generation."""
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics.data.utils import check_det_dataset, img2label_paths


def dataset_info(path):
    # Never silently download a substitute dataset.
    return check_det_dataset(str(Path(path).resolve()), autodownload=False)


def split_images(info, split):
    roots = info[split] if isinstance(info[split], list) else [info[split]]
    images = []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for root in roots:
        p = Path(root)
        if p.is_dir():
            images.extend(x.resolve() for x in p.rglob("*") if x.suffix.lower() in extensions)
        elif p.suffix.lower() == ".txt":
            for line in p.read_text().splitlines():
                if line.strip():
                    q = Path(line.strip())
                    images.append((p.parent / q).resolve() if not q.is_absolute() else q.resolve())
        else:
            raise ValueError(f"Expected image directory or manifest: {p}")
    images = sorted(set(images))
    if not images:
        raise ValueError(f"No images in split {split}")
    return images


def load_example(path, imgsz, nc, device="cpu"):
    if imgsz % 32:
        raise ValueError("imgsz must be a multiple of 32")
    path = str(path)
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape[:2]
    label = Path(img2label_paths([path])[0])
    if not label.exists():
        raise FileNotFoundError(f"Missing label {label}; use an empty .txt for a verified negative image")
    rows = [list(map(float, line.split())) for line in label.read_text().splitlines() if line.strip()]
    labels = np.asarray(rows, dtype=np.float32).reshape(-1, 5)
    if labels.size and (not np.isfinite(labels).all() or (labels[:, 1:] < 0).any() or
                        (labels[:, 1:] > 1).any() or (labels[:, 3:] <= 0).any() or
                        (labels[:, 0] < 0).any() or (labels[:, 0] >= nc).any() or
                        (labels[:, 0] % 1 != 0).any()):
        raise ValueError(f"Invalid YOLO labels in {label}")
    ratio = min(imgsz / height, imgsz / width)
    nw, nh = round(width * ratio), round(height * ratio)
    left, top = round((imgsz - nw) / 2 - .1), round((imgsz - nh) / 2 - .1)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(image, top, imgsz - nh - top, left, imgsz - nw - left,
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))
    if labels.size:
        labels[:, [1, 3]] *= width * ratio / imgsz
        labels[:, [2, 4]] *= height * ratio / imgsz
        labels[:, 1] += left / imgsz
        labels[:, 2] += top / imgsz
    tensor = torch.from_numpy(np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1))).float().div_(255)
    return {"img": tensor.unsqueeze(0).to(device),
            "batch_idx": torch.zeros(len(labels), dtype=torch.long, device=device),
            "cls": torch.from_numpy(labels[:, :1]).to(device),
            "bboxes": torch.from_numpy(labels[:, 1:]).to(device)}
