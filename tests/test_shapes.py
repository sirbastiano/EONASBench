"""
Unit tests for shape invariants and backbone outputs.
"""
import torch
import pytest
from models.build import build_model

@pytest.mark.parametrize("config_path", [
    "configs/variant_a.yaml",
    "configs/variant_b.yaml"
])
def test_model_shapes(config_path):
    model = build_model(config_path)
    model.eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    seg, cls = out['seg'], out['cls']
    # Check segmentation output shape
    assert seg.shape[0] == 2
    assert seg.shape[1] == model.upernet.seg_head.out_channels
    assert seg.shape[2] == 64 and seg.shape[3] == 64  # H/4, W/4 for 256x256 input
    # Check classifier output shape
    assert cls.shape == (2, model.gap_cls.out_features)
    # Check backbone feature shapes
    feats, _ = model.backbone(x)
    assert len(feats) == 3
    C1, C2, C3 = model.backbone.out_channels
    assert feats[0].shape[1] == C1
    assert feats[1].shape[1] == C2
    assert feats[2].shape[1] == C3
    assert feats[0].shape[2:] == (128, 128)
    assert feats[1].shape[2:] == (64, 64)
    assert feats[2].shape[2:] == (32, 32)
