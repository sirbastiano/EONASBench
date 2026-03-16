Changelog
=========

Unreleased
----------

Added
~~~~~

- Added a reduced HPC search macro-profile in ``configs/search_macro_reduced.yaml``.
- Added support for explicit macro controls in model configs:

  - ``stem_stride`` for early downsampling before stage 1
  - ``channels`` for per-stage channel schedules
  - ``head.lateral_dim`` for configurable UPerNet width
  - ``cell_kwargs`` for passing stage-specific layer arguments

- Added ``bench.verify_hpc_profile`` to:

  - record timestamped baseline artifacts under ``outputs/baseline/<timestamp>/``
  - save environment metadata and the exact config used
  - write an analytic worst-case VRAM estimate for the reduced macro-profile
  - measure training-step memory, step time, parameter count, and optional FLOPs on CUDA systems

Changed
~~~~~~~

- Updated the backbone builder to support configurable stem stride and explicit per-stage channel schedules.
- Fixed the drop-path schedule to scale with the total number of cells instead of assuming a fixed depth per stage.
- Updated the classifier input width to follow the actual post-downsampling backbone output channels.
- Reduced the UPerNet head width for the HPC search profile to ``64`` while keeping the segmentation task and output contract intact.
- Updated documentation in ``README.md`` to describe the reduced macro-profile and the verification workflow.

Verification
~~~~~~~~~~~~

- Added shape coverage for the reduced macro-profile in ``tests/test_shapes.py``.
- Updated sequence-based tests so segmentation output size is checked against the highest-resolution backbone feature map.
- Corrected the repeated-layer string-builder expectation to match the builder's actual three-stage behavior.
- Verified Python syntax for the changed files with ``python -m py_compile``.
- Verified the reduced profile forward path in the ``py312`` Conda environment:

  - segmentation output: ``(1, 150, 32, 32)``
  - classification output: ``(1, 150)``
  - feature shapes: ``(1, 32, 32, 32)``, ``(1, 64, 16, 16)``, ``(1, 128, 8, 8)``
  - final backbone tensor: ``(1, 256, 4, 4)``

Known limitations
~~~~~~~~~~~~~~~~~

- Full GPU VRAM verification requires a CUDA-capable environment; the local ``py312`` environment used during development had PyTorch available but no CUDA device.
- The full ``pytest`` suite is expensive on CPU for the largest legacy ViT configs at ``256x256`` input.
