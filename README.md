# EONASBench Modular Benchmarking Framework

## Overview
This framework enables benchmarking of macro-architectures composed of stackable ConvNeXt and ViT-style cells, with a fixed stem/head, multi-scale feature collection, and a registry of six selectable cell types. Architectures are defined by YAML/JSON configs and assembled with deterministic, device-agnostic logic.

## Features
- Six cell types: convnext_v1, convnext_se, convnext_dil, vit_encoder, vit_rpe, vit_window
- Macro-architecture from config: stem, 3 stages, downsampling, multi-scale head
- UPerNet segmentation head + GAP classifier
- String-based model builder: specify architectures using patterns like '001045'
- AMP, CUDA, deterministic seeding
- Profiling: params, FLOPs, latency
- Unit tests for shape invariants and registry coverage

## Usage

### Install dependencies
```
pip install torch pyyaml pytest
# For FLOPs: pip install ptflops
```

### Run profiling
```
python -m bench.profile --config configs/variant_a.yaml --img 256 256
```

### Run tests
```
pytest tests/
```

### Build a model in Python
```python
from models.build import build_model
model = build_model('configs/variant_a.yaml')
```

### Build a model from string patterns
```python
from models.string_builder import build_model_from_string

# Build model using string pattern where each digit represents a layer index
# 0=convnext_v1, 1=convnext_se, 2=convnext_dil, 3=vit_encoder, 4=vit_rpe, 5=vit_window
model = build_model_from_string('012', num_classes=10, stem_out=64)

# Example with repeated layers
model = build_model_from_string('001201', num_classes=5, stem_out=32)
```

### CLI with string patterns
```bash
# List available layers and their indices
python -m bench.test_sequence_cli --list-layers

# Build model from string pattern
python -m bench.test_sequence_cli --pattern "012" --img 64 64 --num_classes 5

# Build model from index list (legacy)
python -m bench.test_sequence_cli --seq 0 1 2 --img 64 64
```

## Config examples
See `configs/variant_a.yaml` and `configs/variant_b.yaml` for macro-architecture variants.

## String-Based Model Builder

The framework supports a convenient string-based interface for defining model architectures. Each character in the string represents a layer type index:

| Index | Layer Type    | Description                           |
|-------|---------------|---------------------------------------|
| 0     | convnext_v1   | ConvNeXt V1 cell                     |
| 1     | convnext_se   | ConvNeXt with Squeeze-and-Excitation |
| 2     | convnext_dil  | ConvNeXt with dilation               |
| 3     | vit_encoder   | Vision Transformer encoder           |
| 4     | vit_rpe       | ViT with relative positional encoding|
| 5     | vit_window    | ViT with windowed attention          |

### Examples
- `"012"` → convnext_v1, convnext_se, convnext_dil (3 stages)
- `"001045"` → convnext_v1, convnext_v1, convnext_se, convnext_v1, vit_rpe, vit_window (6 stages)
- `"000"` → three stages of convnext_v1

### Pipeline Verification
The string-based builder has been verified for correctness:
- ✅ Parses string patterns correctly
- ✅ Validates layer indices are within range
- ✅ Builds equivalent configurations to index-based builder  
- ✅ Produces functional models with correct output shapes
- ✅ Supports arbitrary layer repetition and combinations

## CLI Example
```
python -m bench.profile --config configs/variant_b.yaml --img 256 256
```

## Acceptance tests
- All six cell types build and run
- Backbone outputs correct feature shapes
- Head outputs correct segmentation and classification shapes
- DropPath schedule is correct
- Determinism: seeding yields identical outputs
