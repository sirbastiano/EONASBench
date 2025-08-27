#!/usr/bin/env python3
"""
Demo script showcasing the string-based model builder functionality.

This script demonstrates how to build neural network models using string
patterns where each character represents a layer type.
"""

import torch
from models.string_builder import (
    build_model_from_string,
    parse_string_sequence,
    print_layer_info,
    get_layer_info
)


def demo_basic_usage():
    """Demonstrate basic string-based model building."""
    print("=" * 60)
    print("DEMO: Basic String-Based Model Building")
    print("=" * 60)
    
    # Show available layers
    print("Available layers:")
    print_layer_info()
    print()
    
    # Build a simple model
    pattern = "012"
    print(f"Building model with pattern '{pattern}':")
    indices = parse_string_sequence(pattern)
    layer_names = [list(get_layer_info().values())[i] for i in indices]
    print(f"  - Parsed indices: {indices}")
    print(f"  - Layer sequence: {layer_names}")
    
    model = build_model_from_string(pattern, num_classes=5, stem_out=32)
    
    # Test the model
    print(f"  - Model built successfully!")
    print(f"  - Number of stages: {len(model.backbone.stages)}")
    print(f"  - Output classes: {model.num_classes}")
    
    # Forward pass
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        output = model(x)
    
    seg, cls = output['seg'], output['cls']
    print(f"  - Segmentation shape: {seg.shape}")
    print(f"  - Classification shape: {cls.shape}")
    print("  ✅ Success!")
    

def demo_complex_pattern():
    """Demonstrate complex patterns with repeated layers."""
    print("\n" + "=" * 60)
    print("DEMO: Complex Pattern with Repeated Layers")
    print("=" * 60)
    
    pattern = "001201"  # Using only conv layers to avoid ViT bugs
    print(f"Building model with pattern '{pattern}':")
    indices = parse_string_sequence(pattern)
    layer_names = [list(get_layer_info().values())[i] for i in indices]
    print(f"  - Parsed indices: {indices}")
    print(f"  - Layer sequence: {layer_names}")
    print(f"  - Pattern length: {len(pattern)} stages")
    
    model = build_model_from_string(pattern, num_classes=10, stem_out=64)
    
    print(f"  - Model built successfully!")
    print(f"  - Number of stages: {len(model.backbone.stages)}")
    
    # Test with different input size
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        output = model(x)
    
    seg, cls = output['seg'], output['cls']
    print(f"  - Segmentation shape: {seg.shape}")
    print(f"  - Classification shape: {cls.shape}")
    print("  ✅ Success!")


def demo_validation():
    """Demonstrate input validation."""
    print("\n" + "=" * 60)
    print("DEMO: Input Validation")
    print("=" * 60)
    
    print("Testing valid patterns:")
    valid_patterns = ["0", "12", "012", "001122"]
    for pattern in valid_patterns:
        try:
            indices = parse_string_sequence(pattern)
            print(f"  ✅ '{pattern}' → {indices}")
        except ValueError as e:
            print(f"  ❌ '{pattern}' → Error: {e}")
    
    print("\nTesting invalid patterns:")
    invalid_patterns = ["", "01a", "017", "0-1", "  01"]
    for pattern in invalid_patterns:
        try:
            indices = parse_string_sequence(pattern)
            print(f"  ❌ '{pattern}' should have failed but got: {indices}")
        except ValueError as e:
            print(f"  ✅ '{pattern}' → Correctly rejected: {e}")


if __name__ == "__main__":
    print("EONASBench String-Based Model Builder Demo")
    print("==========================================")
    
    try:
        demo_basic_usage()
        demo_complex_pattern()
        demo_validation()
        
        print("\n" + "=" * 60)
        print("🎉 All demos completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()