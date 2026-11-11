from dataclasses import replace
import torch
from ultralytics.utils.loss import v8DetectionLoss
from amu.model import UtilityDetector, save_detector, load_detector
from amu.routing import ShallowGateConcat, RegionalRouter, regional_means, expand_regions, sample_masks
from amu.utility import FixedBundleLoss


def example(batch=1):
    return {"batch_idx": torch.arange(batch), "cls": torch.zeros(batch, 1),
            "bboxes": torch.tensor([[.5, .5, .4, .4]]).repeat(batch, 1)}


def test_zero_gate_has_no_shallow_bypass():
    gate = ShallowGateConcat(block=8, mode="off").eval()
    s, t, i = torch.randn(1, 64, 32, 32), torch.randn(1, 128, 16, 16), torch.randn(1, 128, 16, 16)
    assert torch.equal(gate([s, t, i]), gate([s * 1e4, t, i]))
    gate.mode = "on"
    assert not torch.equal(gate([s, t, i]), gate([s * 1e4, t, i]))


def test_partial_region_pool_and_broadcast():
    x = torch.ones(1, 2, 17, 19)
    means = regional_means(x, 8)
    assert means.shape[-2:] == (3, 3)
    assert torch.equal(means, torch.ones_like(means))
    assert torch.equal(expand_regions(means, 8, 17, 19), x)


def test_gate_mapping_and_mask_support():
    router = RegionalRouter()
    for p in router.parameters():
        p.data.zero_()
    x = torch.zeros(2, 320)
    assert torch.equal(router.gains(x), torch.ones(2))
    router.predictor[-1].bias.data.fill_(-2)
    torch.testing.assert_close(router.gains(x), torch.full((2,), torch.exp(torch.tensor(-2.))))
    torch.manual_seed(1)
    masks = sample_masks(2000, 3, 3, device="cpu", dtype=torch.float32)
    assert masks.min() == 0 and masks.max() == 1
    assert ((masks > .1) & (masks < 1)).any()
    assert .25 < float((masks == 1).all(dim=(1, 2, 3)).float().mean()) < .50


def test_cached_tail_and_serialization(tmp_path):
    torch.manual_seed(3)
    model = UtilityDetector(nc=2).freeze_probe()
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        direct = model(x)
        cached = model.tail(model.prefix(x))
    torch.testing.assert_close(direct[0], cached[0])
    path = tmp_path / "model.pt"
    save_detector(model, path)
    restored, _ = load_detector(path)
    with torch.no_grad():
        torch.testing.assert_close(model(x)[0], restored(x)[0])


def test_fixed_bundle_matches_upstream_single_image_loss():
    torch.manual_seed(4)
    model = UtilityDetector(nc=2).freeze_probe()
    batch = example()
    with torch.no_grad():
        output = model(torch.rand(1, 3, 128, 128))
    criterion = FixedBundleLoss(2)
    pred = criterion.decode(output)
    bundle = criterion.assign(pred, batch)
    assert bundle.foreground.any()
    assert not bundle.scores.requires_grad
    actual = criterion.evaluate(pred, bundle).sum()
    expected, _ = v8DetectionLoss(model)(output, batch)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_per_image_labels_do_not_depend_on_batch_companions():
    torch.manual_seed(5)
    model = UtilityDetector(nc=2).freeze_probe()
    with torch.no_grad():
        features = model(torch.rand(2, 3, 128, 128))[1]
    criterion = FixedBundleLoss(2)
    off_features = [x + .05 * torch.randn_like(x) for x in features]
    on, off = criterion.decode(features), criterion.decode(off_features)
    values, _, _ = criterion.contrast(on, off, example(2))
    alone, _, _ = criterion.contrast(criterion.decode([x[:1] for x in features]),
                                    criterion.decode([x[:1] for x in off_features]), example())
    for key in values:
        torch.testing.assert_close(values[key][0], alone[key][0], rtol=1e-5, atol=1e-5)


def test_symmetric_identity_swap_and_frozen_targets():
    torch.manual_seed(6)
    model = UtilityDetector(nc=2).freeze_probe()
    with torch.no_grad():
        features = model(torch.rand(1, 3, 128, 128))[1]
    criterion = FixedBundleLoss(2)
    on = criterion.decode(features)
    off = criterion.decode([x + .03 * torch.randn_like(x) for x in features])
    values, bundle, _ = criterion.contrast(on, off, example())
    zero, _, _ = criterion.contrast(on, on, example())
    reverse, _, _ = criterion.contrast(off, on, example())
    assert zero["symmetric"].item() == 0
    torch.testing.assert_close(values["symmetric"], -reverse["symmetric"])
    saved = bundle.scores.clone()
    criterion.assign = lambda *a: (_ for _ in ()).throw(AssertionError("Unexpected reassignment"))
    criterion.evaluate(off, bundle)
    assert torch.equal(saved, bundle.scores)


def test_empty_gt_and_router_only_gradient():
    model = UtilityDetector(nc=2).freeze_probe()
    model.gate.mode = "router"
    model.gate.router.requires_grad_(True)
    # Avoid saturation so this test checks propagation through the frozen detector.
    model.gate.router.predictor[-1].bias.data.fill_(-.2)
    x = torch.rand(1, 3, 128, 128)
    criterion = FixedBundleLoss(2)
    pred = criterion.decode(model(x))
    batch = {"batch_idx": torch.empty(0), "cls": torch.empty(0, 1), "bboxes": torch.empty(0, 4)}
    target = criterion.assign(pred, batch)
    assert not target.foreground.any()
    assert target.normalizer.item() == 1
    loss = criterion.evaluate(pred, target).sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.gate.router.parameters())
    assert all(p.grad is None for name, p in model.named_parameters() if "router" not in name)
