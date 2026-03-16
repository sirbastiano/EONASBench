"""
Unit tests for shape invariants and backbone outputs.
"""
import torch
import pytest
from models.build import build_model

@pytest.mark.parametrize("config_path", [
    "configs/variant_a.yaml",
    "configs/variant_b.yaml",
    "configs/search_macro_reduced.yaml",
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
    # Check classifier output shape
    assert cls.shape == (2, model.gap_cls.out_features)
    # Check backbone feature shapes
    feats, _ = model.backbone(x)
    assert len(feats) == len(model.backbone.out_channels)
    assert seg.shape[2:] == feats[0].shape[2:]

    for idx, feat in enumerate(feats):
        assert feat.shape[1] == model.backbone.out_channels[idx]

    for prev, curr in zip(feats, feats[1:]):
        assert curr.shape[2] == prev.shape[2] // 2
        assert curr.shape[3] == prev.shape[3] // 2


def test_reduced_profile_head_width():
    model = build_model("configs/search_macro_reduced.yaml")
    assert model.upernet.lateral_dim == 64
    assert model.backbone.out_channels == [32, 64, 128]
