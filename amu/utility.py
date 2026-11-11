"""Complete detached target bundles and per-image, assignment-matched losses."""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors, dist2bbox
from ultralytics.utils.ops import xywh2xyxy
from .model import raw_predictions


@dataclass(frozen=True)
class Predictions:
    distribution: torch.Tensor
    logits: torch.Tensor
    boxes: torch.Tensor  # feature-grid coordinates, each anchor's own stride
    anchors: torch.Tensor
    strides: torch.Tensor
    image_hw: tuple


@dataclass(frozen=True)
class TargetBundle:
    foreground: torch.Tensor
    gt_ids: torch.Tensor  # -1 for background; per-image GT indexing
    scores: torch.Tensor
    boxes: torch.Tensor  # feature-grid coordinates
    weights: torch.Tensor
    normalizer: torch.Tensor  # one denominator per image, NEVER batch-wide
    anchors: torch.Tensor
    strides: torch.Tensor


class FixedBundleLoss:
    def __init__(self, nc, strides=(8, 16, 32), reg_max=16):
        self.nc, self.strides, self.reg_max = nc, tuple(strides), reg_max
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=nc, alpha=.5, beta=6.)
        self.box_loss = BboxLoss(reg_max)

    def decode(self, output):
        feats = raw_predictions(output)
        batch = feats[0].shape[0]
        joined = torch.cat([x.reshape(batch, 4 * self.reg_max + self.nc, -1) for x in feats], 2)
        distribution, logits = joined.split((4 * self.reg_max, self.nc), 1)
        distribution, logits = distribution.permute(0, 2, 1), logits.permute(0, 2, 1)
        anchors, strides = make_anchors(feats, self.strides, .5)
        bins = torch.arange(self.reg_max, device=distribution.device, dtype=distribution.dtype)
        distances = distribution.reshape(batch, -1, 4, self.reg_max).softmax(-1) @ bins
        boxes = dist2bbox(distances, anchors, xywh=False)
        h, w = feats[0].shape[-2:]
        return Predictions(distribution, logits, boxes, anchors, strides, (h * self.strides[0], w * self.strides[0]))

    @torch.no_grad()
    def assign(self, pred, batch):
        device, dtype = pred.logits.device, pred.logits.dtype
        h, w = pred.image_hw
        scale = pred.logits.new_tensor([w, h, w, h])
        foreground, ids, scores, boxes = [], [], [], []
        for b in range(pred.logits.shape[0]):
            select = batch["batch_idx"].view(-1).to(device) == b
            gt_cls = batch["cls"].to(device=device, dtype=dtype)[select].reshape(1, -1, 1)
            gt_boxes = xywh2xyxy(batch["bboxes"].to(device=device, dtype=dtype)[select] * scale).unsqueeze(0)
            valid = torch.ones_like(gt_cls, dtype=torch.bool)
            _, tb, ts, fg, gi = self.assigner(pred.logits[b:b+1].detach().sigmoid(),
                pred.boxes[b:b+1].detach() * pred.strides, pred.anchors * pred.strides,
                gt_cls, gt_boxes, valid)
            foreground.append(fg.bool())
            ids.append(torch.where(fg.bool(), gi.long(), -1))
            scores.append(ts)
            boxes.append(tb / pred.strides)
        fg, gi, ts, tb = [torch.cat(x, 0).detach().clone() for x in (foreground, ids, scores, boxes)]
        return TargetBundle(fg, gi, ts, tb, ts.sum(-1).detach().clone(),
                            ts.sum((1, 2)).clamp_min(1).detach().clone(),
                            pred.anchors.detach().clone(), pred.strides.detach().clone())

    def evaluate(self, pred, target):
        if not torch.equal(pred.anchors, target.anchors) or not torch.equal(pred.strides, target.strides):
            raise ValueError("The reference bundle and prediction must use identical anchors/strides")
        if pred.logits.shape != target.scores.shape:
            raise ValueError("Target and prediction shape mismatch")
        results = []
        for b in range(pred.logits.shape[0]):
            sl = slice(b, b + 1)
            denominator = target.normalizer[b]
            cls = F.binary_cross_entropy_with_logits(pred.logits[sl], target.scores[sl], reduction="sum") / denominator
            box = pred.boxes[sl].sum() * 0
            dfl = pred.distribution[sl].sum() * 0
            if target.foreground[sl].any():
                # BboxLoss derives identical weights from the frozen soft scores.
                box, dfl = self.box_loss(pred.distribution[sl], pred.boxes[sl], target.anchors,
                    target.boxes[sl], target.scores[sl], denominator, target.foreground[sl])
            results.append(torch.stack((7.5 * box, .5 * cls, 1.5 * dfl)))
        return torch.stack(results)  # [B, box/cls/dfl], no extra batch multiplier

    @torch.no_grad()
    def contrast(self, on, off, batch, on_bundle=None):
        a_on = on_bundle if on_bundle is not None else self.assign(on, batch)
        a_off = self.assign(off, batch)
        ll_on = self.evaluate(on, a_on).sum(-1)
        ll_off_on = self.evaluate(off, a_on).sum(-1)
        ll_on_off = self.evaluate(on, a_off).sum(-1)
        ll_off = self.evaluate(off, a_off).sum(-1)
        u_on, u_off = ll_off_on - ll_on, ll_off - ll_on_off
        return {"symmetric": (u_on + u_off) / 2, "on": u_on, "off": u_off,
                "reassigned": ll_off - ll_on}, a_on, a_off
