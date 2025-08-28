"""
CLI to test model assembly from a sequence of layer indices or string pattern.
"""
import argparse
import torch
from models.build import build_model
from models.string_builder import (
    build_model_from_string, 
    parse_string_sequence, 
    print_layer_info,
    LAYER_IDS
)
from layers import convnext_v1, convnext_se, convnext_dil, vit_encoder, vit_rpe, vit_window

def build_config_from_indices(indices, num_classes=10, stem_out=64, drop_path_rate=0.1):
    """Build config from list of indices (legacy function for compatibility)."""
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
    parser.add_argument('--list-layers', action='store_true',
                       help='List available layers and exit')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--seq', nargs='+', type=int, 
                      help='Sequence of layer indices (e.g. 0 1 2)')
    group.add_argument('--pattern', type=str,
                      help='String pattern where each digit is a layer index (e.g. "001045")')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--stem_out', type=int, default=64)
    parser.add_argument('--img', nargs=2, type=int, default=[128,128])
    
    args = parser.parse_args()
    
    if args.list_layers:
        print_layer_info()
        return
    
    if not args.seq and not args.pattern:
        parser.error('Either --seq or --pattern must be provided (unless using --list-layers)')
    
    print('Available layers:')
    print_layer_info()
    print()
    
    try:
        if args.pattern:
            print(f'Building model from string pattern: "{args.pattern}"')
            indices = parse_string_sequence(args.pattern)
            print(f'Parsed indices: {indices}')
            print(f'Layers: {[LAYER_IDS[i] for i in indices]}')
            model = build_model_from_string(args.pattern, 
                                          num_classes=args.num_classes, 
                                          stem_out=args.stem_out)
        else:
            print(f'Building model from index sequence: {args.seq}')
            print(f'Layers: {[LAYER_IDS[i] for i in args.seq]}')
            config = build_config_from_indices(args.seq, 
                                             num_classes=args.num_classes, 
                                             stem_out=args.stem_out)
            model = build_model(config)
        
        print('\nTesting model forward pass...')
        x = torch.randn(2, 3, args.img[0], args.img[1])
        
        with torch.no_grad():
            out = model(x)
            
        seg, cls = out['seg'], out['cls']
        print(f'✓ Segmentation output shape: {seg.shape}')
        print(f'✓ Classifier output shape: {cls.shape}')
        print('✓ Test passed successfully!')
        
    except Exception as e:
        print(f'✗ Error: {e}')
        return 1

if __name__ == "__main__":
    main()
