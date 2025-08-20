"""
CLI to test model assembly from a sequence of layer indices.
"""
import argparse
import torch
from models.build import build_model
from layers import convnext_v1, convnext_se, convnext_dil, vit_encoder, vit_rpe, vit_window
from layers.registry import LAYERS
LAYER_IDS = list(LAYERS.keys())

def build_config_from_indices(indices, num_classes=10, stem_out=64, drop_path_rate=0.1):
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

def main():
    parser = argparse.ArgumentParser(description='Test model with custom layer sequence.')
    parser.add_argument('--seq', nargs='+', type=int, required=True, help='Sequence of layer indices (e.g. 0 1 2)')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--stem_out', type=int, default=64)
    parser.add_argument('--img', nargs=2, type=int, default=[128,128])
    args = parser.parse_args()
    print('Layer registry order:')
    for i, lid in enumerate(LAYER_IDS):
        print(f'{i}: {lid}')
    config = build_config_from_indices(args.seq, num_classes=args.num_classes, stem_out=args.stem_out)
    model = build_model(config)
    x = torch.randn(2, 3, args.img[0], args.img[1])
    out = model(x)
    seg, cls = out['seg'], out['cls']
    print(f'Segmentation output shape: {seg.shape}')
    print(f'Classifier output shape: {cls.shape}')
    print('Test passed.')

if __name__ == "__main__":
    main()
