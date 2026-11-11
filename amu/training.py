"""Full probe training via Ultralytics, followed by frozen-detector fitting."""
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import init_seeds
from .model import UtilityDetector, load_detector, save_detector
from .data import dataset_info, split_images, load_example
from .utility import FixedBundleLoss


def seed_all(seed):
    init_seeds(seed, deterministic=True)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ProbeTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        if weights is not None:
            raise ValueError("Paper protocol starts from random initialization")
        return UtilityDetector(nc=self.data["nc"], block=self.amu_block, verbose=verbose)


def train_probe(data, output, device, seed=17, epochs=300, batch=32, imgsz=640, workers=4, block=8):
    if "," in str(device):
        raise ValueError("This adapter currently supports single-device training only")
    dataset_info(data)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new run directory: {output}")
    trainer = ProbeTrainer(overrides={
        "model": "yolo11n.yaml", "data": str(Path(data).resolve()), "device": device,
        "project": str(output.parent), "name": output.name, "exist_ok": False,
        "epochs": epochs, "batch": batch, "imgsz": imgsz, "workers": workers,
        "seed": seed, "deterministic": True, "pretrained": False,
        "optimizer": "SGD", "lr0": .01, "lrf": .01, "momentum": .937,
        "weight_decay": .0005, "warmup_epochs": 3, "cos_lr": True,
        "amp": False, "patience": 0, "close_mosaic": 0, "mosaic": 1., "mixup": 0.,
        "degrees": 0., "translate": .1, "scale": .5, "shear": 0., "perspective": 0.,
        "flipud": 0., "fliplr": .5, "hsv_h": .015, "hsv_s": .7, "hsv_v": .4,
        "box": 7.5, "cls": .5, "dfl": 1.5, "save": True, "plots": False,
        "val": True, "conf": .001, "iou": .7, "max_det": 300, "freeze": [15],
    })
    trainer.amu_block = block

    def export_final(t):
        if t.epoch == t.epochs - 1:
            model = t.ema.ema
            model.gate.mode = "on"
            save_detector(model, Path(t.save_dir) / "probe_final.pt", {
                "seed": seed, "epochs": epochs, "imgsz": imgsz, "data": str(Path(data).resolve()),
                "selection": "last_epoch_ema", "ultralytics": "8.3.40"})

    # Before final_eval strips/converts the upstream checkpoint to half precision.
    trainer.add_callback("on_fit_epoch_end", export_final)
    trainer.train()
    return str(Path(trainer.save_dir) / "probe_final.pt")


@torch.no_grad()
def generate_labels(checkpoint, data, output, device="cpu", seed=17, imgsz=640,
                    draws=2, regions=4, shard_size=4096, limit=None):
    seed_all(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model, meta = load_detector(checkpoint, device)
    model.freeze_probe()
    info = dataset_info(data)
    if info["nc"] != model.model[-1].nc:
        raise ValueError("Dataset class count differs from checkpoint")
    images = split_images(info, "train")
    overlap = set(images) & set(split_images(info, "val"))
    if overlap:
        raise ValueError(f"Train/validation leakage: {len(overlap)} shared image paths")
    if limit is not None:
        images = images[:limit]
    criterion = FixedBundleLoss(info["nc"])
    rng = torch.Generator().manual_seed(seed)
    pending, shards, count = [], [], 0
    image_ids = [str(p) for p in images]
    source = {"format": "amu-labels-v1", "checkpoint_sha256": file_hash(checkpoint),
              "data_sha256": file_hash(data), "seed": seed, "imgsz": imgsz,
              "draws": draws, "regions_per_draw": regions, "block": model.block,
              "images": image_ids, "limit": limit, "context": "all_other_regions_on"}

    def flush():
        if not pending:
            return
        shard = {k: torch.stack([x[k] for x in pending]) for k in pending[0]}
        name = f"shard_{len(shards):05d}.pt"
        torch.save(shard, output / name)
        shards.append({"file": name, "count": len(pending), "sha256": file_hash(output / name)})
        pending.clear()

    for image_index, path in enumerate(images):
        batch = load_example(path, imgsz, info["nc"], device)
        prefix = model.prefix(batch["img"])
        summaries = model.gate.summaries(model.gate.align([prefix[2], prefix[14], prefix[4]]))
        rows, cols = summaries.shape[1:3]
        if regions > rows * cols:
            raise ValueError("regions_per_draw exceeds available regions")
        on = criterion.decode(model.tail(prefix))
        a_on = criterion.assign(on, batch)
        for draw in range(draws):
            chosen = torch.randperm(rows * cols, generator=rng)[:regions].tolist()
            for index in chosen:
                row, col = divmod(index, cols)
                mask = batch["img"].new_ones((1, 1, rows, cols))
                mask[0, 0, row, col] = 0
                off = criterion.decode(model.tail(prefix, mask))
                values, _, a_off = criterion.contrast(on, off, batch, a_on)
                coverage_on = a_on.gt_ids[a_on.foreground].unique().numel()
                coverage_off = a_off.gt_ids[a_off.foreground].unique().numel()
                record = {"summaries": summaries[0, row, col].cpu().clone(),
                          "image_index": torch.tensor(image_index), "region": torch.tensor([row, col]),
                          "draw": torch.tensor(draw), "gt_count": torch.tensor(len(batch["cls"])),
                          "covered_on": torch.tensor(coverage_on), "covered_off": torch.tensor(coverage_off)}
                record.update({k: v[0].cpu() for k, v in values.items()})
                pending.append(record)
                count += 1
                if len(pending) >= shard_size:
                    flush()
        if (image_index + 1) % 50 == 0:
            print(f"labels: {image_index + 1}/{len(images)} images; {count} pairs", flush=True)
    flush()
    source.update({"shards": shards, "count": count, "complete": True})
    (output / "manifest.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    return source


def train_router(checkpoint, labels, output, device="cpu", seed=17, epochs=30, batch_size=32,
                 target="symmetric", shuffled=False):
    seed_all(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    labels = Path(labels)
    manifest = json.loads((labels / "manifest.json").read_text())
    if not manifest.get("complete") or manifest["checkpoint_sha256"] != file_hash(checkpoint):
        raise ValueError("Incomplete labels or labels generated by a different probe")
    shards = []
    for shard in manifest["shards"]:
        path = labels / shard["file"]
        if file_hash(path) != shard["sha256"]:
            raise ValueError(f"Shard checksum mismatch: {path}")
        shards.append(torch.load(path, map_location="cpu", weights_only=True))
    x = torch.cat([s["summaries"] for s in shards])
    y = torch.cat([s[target] for s in shards])
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Non-finite summaries or utility labels")
    # Conventional median: midpoint of the two central values for even counts.
    scale = y.abs().quantile(.5)
    if scale <= 0:
        raise ValueError("Zero median absolute utility: the regression pilot is uninformative")
    z = y / scale
    if shuffled:
        z = z[torch.randperm(len(z), generator=torch.Generator().manual_seed(seed))]
    model, meta = load_detector(checkpoint, device)
    model.freeze_probe()
    router = model.gate.router
    # Same initialization for M8/M9/M10 under the same seed.
    from .routing import RegionalRouter
    seed_all(seed)
    router.load_state_dict(RegionalRouter().state_dict())
    router.requires_grad_(True).train()
    optimizer = torch.optim.AdamW(router.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=.00001)
    loader = DataLoader(TensorDataset(x, z), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    history = []
    for epoch in range(epochs):
        loss_sum = 0.
        lr = optimizer.param_groups[0]["lr"]
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.smooth_l1_loss(router(features.to(device)), targets.to(device), beta=1.)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 5.)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(features)
        scheduler.step()
        history.append({"epoch": epoch + 1, "loss": loss_sum / len(x), "lr": lr})
        print(f"router epoch {epoch + 1}/{epochs}: loss={loss_sum / len(x):.6g}", flush=True)
    model.eval()
    model.gate.mode = "router"
    metadata = {**meta, "router_seed": seed, "router_epochs": epochs, "utility_scale": float(scale),
                "target": target, "shuffled": shuffled, "probe_sha256": file_hash(checkpoint),
                "labels_manifest_sha256": file_hash(labels / "manifest.json"), "selection": "last_epoch_router"}
    save_detector(model, output / "router_final.pt", metadata)
    (output / "training.json").write_text(json.dumps({"metadata": metadata, "history": history}, indent=2))
    return str(output / "router_final.pt")


def train_detection_router(checkpoint, data, output, device="cpu", seed=17, epochs=30, batch_size=32, imgsz=640):
    """M7 control: only router gradients; fixed BN, dynamic TAL, deterministic images.

    Per-image losses are averaged before one optimizer step per image minibatch.
    """
    from .routing import RegionalRouter
    seed_all(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model, meta = load_detector(checkpoint, device)
    model.freeze_probe()
    seed_all(seed)
    model.gate.router.load_state_dict(RegionalRouter().state_dict())
    model.gate.router.requires_grad_(True)
    model.gate.mode = "router"
    info = dataset_info(data)
    images = split_images(info, "train")
    if info["nc"] != model.model[-1].nc:
        raise ValueError("Dataset class count differs from checkpoint")
    criterion = FixedBundleLoss(info["nc"])
    optimizer = torch.optim.AdamW(model.gate.router.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=.00001)
    history = []
    for epoch in range(epochs):
        order = torch.randperm(len(images)).tolist()
        epoch_loss = 0.
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            for index in indices:
                batch = load_example(images[index], imgsz, info["nc"], device)
                with torch.no_grad():
                    prefix = model.prefix(batch["img"])
                pred = criterion.decode(model.tail(prefix))
                target = criterion.assign(pred, batch)
                loss = criterion.evaluate(pred, target).sum() / len(indices)
                loss.backward()
                epoch_loss += float(loss.detach()) * len(indices)
            torch.nn.utils.clip_grad_norm_(model.gate.router.parameters(), 5.)
            optimizer.step()
        scheduler.step()
        history.append({"epoch": epoch + 1, "loss": epoch_loss / len(images)})
        print(f"M7 epoch {epoch + 1}/{epochs}: {history[-1]['loss']:.6g}", flush=True)
    save_detector(model, output / "router_final.pt", {**meta, "target": "detection_loss", "seed": seed})
    (output / "training.json").write_text(json.dumps(history, indent=2))
    return str(output / "router_final.pt")
