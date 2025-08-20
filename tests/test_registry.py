"""
Unit test for registry coverage: all six cell types build and run.
"""
import torch
import pytest
from layers.registry import build_cell, LAYERS

@pytest.mark.parametrize("cell_id", list(LAYERS.keys()))
def test_cell_registry(cell_id):
    C, H, W = 32, 16, 16
    x = torch.randn(2, C, H, W)
    cell = build_cell(cell_id, C)
    y = cell(x)
    assert y.shape == (2, C, H, W)
