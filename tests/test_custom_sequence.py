"""
Test model assembly from a custom sequence of layer indices.
"""
import torch
from models.build import build_model
from layers.registry import LAYERS

# Map indices to layer IDs in registry order
LAYER_IDS = list(LAYERS.keys())

def build_config_from_indices(indices, num_classes=10, stem_out=64, drop_path_rate=0.1):
    """
    Build a config dict from a sequence of layer indices.
    Args:
        indices (list[int]): Indices into LAYER_IDS.
        num_classes (int): Number of classes.
        stem_out (int): Stem output channels.
        drop_path_rate (float): DropPath schedule max.
    Returns:
        dict: Model config.
    """
    assert all(0 <= i < len(LAYER_IDS) for i in indices), 'Invalid layer index.'
    stages = [
        {"cells": 3, "layer": LAYER_IDS[i]} for i in indices
    ]
    return {
        "model": {
            "num_classes": num_classes,
            "stem_out": stem_out,
            "stages": stages,
            "downsample": {"type": "maxpool_conv1x1"},
            "head": {"type": "upernet+gap"},
            "drop_path_rate": drop_path_rate,
            "norm": "layernorm",
            "amp_dtype": "bf16",
            "init": "trunc_normal_0.02"
        }
    }

def test_custom_sequence():
    # Example: [0, 1, 2] means use first 3 layer types in registry order
    indices = [0, 1, 2]
    config = build_config_from_indices(indices, num_classes=5, stem_out=32)
    model = build_model(config)
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    seg, cls = out['seg'], out['cls']
    assert seg.shape[0] == 2
    assert seg.shape[1] == 5
    assert cls.shape == (2, 5)
    print('Custom sequence test passed.')

if __name__ == "__main__":
    test_custom_sequence()
