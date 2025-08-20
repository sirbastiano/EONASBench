# EONASBench Modular Benchmarking Framework

## Overview
This framework enables benchmarking of macro-architectures composed of stackable ConvNeXt and ViT-style cells, with a fixed stem/head, multi-scale feature collection, and a registry of six selectable cell types. Architectures are defined by YAML/JSON configs and assembled with deterministic, device-agnostic logic.

## Features
- Six cell types: convnext_v1, convnext_se, convnext_dil, vit_encoder, vit_rpe, vit_window
- Macro-architecture from config: stem, 3 stages, downsampling, multi-scale head
- UPerNet segmentation head + GAP classifier
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

## Config examples
See `configs/variant_a.yaml` and `configs/variant_b.yaml` for macro-architecture variants.

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
