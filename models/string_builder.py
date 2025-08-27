"""
String-based model builder for EONASBench.

This module provides functionality to build models from string patterns where
each character represents a layer type index.
"""

import torch
from typing import List, Dict, Any
from models.build import build_model
from layers.registry import LAYERS

# Map indices to layer IDs in registry order
LAYER_IDS = list(LAYERS.keys())

def parse_string_sequence(sequence: str) -> List[int]:
    """
    Parse a string sequence into layer indices.
    
    Args:
        sequence (str): String pattern where each character is a digit 
                       representing a layer index (e.g., '001045')
    
    Returns:
        List[int]: List of layer indices
        
    Raises:
        ValueError: If sequence contains invalid characters or indices
        
    Examples:
        >>> parse_string_sequence('001045')
        [0, 0, 1, 0, 4, 5]
        >>> parse_string_sequence('012')
        [0, 1, 2]
    """
    if not sequence:
        raise ValueError("Sequence cannot be empty")
    
    if not sequence.isdigit():
        raise ValueError(f"Sequence must contain only digits, got: {sequence}")
    
    indices = [int(char) for char in sequence]
    
    # Validate indices are within range
    max_index = len(LAYER_IDS) - 1
    invalid_indices = [i for i in indices if i > max_index]
    if invalid_indices:
        raise ValueError(
            f"Invalid layer indices {invalid_indices}. "
            f"Valid range is 0-{max_index} for layers: {LAYER_IDS}"
        )
    
    return indices

def build_config_from_string(sequence: str, num_classes: int = 10, 
                           stem_out: int = 64, drop_path_rate: float = 0.1) -> Dict[str, Any]:
    """
    Build a model config dictionary from a string sequence.
    
    Args:
        sequence (str): String pattern where each character is a digit 
                       representing a layer index (e.g., '001045')
        num_classes (int): Number of output classes
        stem_out (int): Number of output channels from the stem
        drop_path_rate (float): Maximum drop path rate for regularization
        
    Returns:
        Dict[str, Any]: Model configuration dictionary
        
    Examples:
        >>> config = build_config_from_string('012', num_classes=5, stem_out=32)
        >>> # This creates a model with layers: convnext_v1, convnext_se, convnext_dil
    """
    indices = parse_string_sequence(sequence)
    
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

def build_model_from_string(sequence: str, num_classes: int = 10,
                          stem_out: int = 64, drop_path_rate: float = 0.1):
    """
    Build a complete model from a string sequence.
    
    Args:
        sequence (str): String pattern where each character is a digit 
                       representing a layer index (e.g., '001045')
        num_classes (int): Number of output classes
        stem_out (int): Number of output channels from the stem
        drop_path_rate (float): Maximum drop path rate for regularization
        
    Returns:
        torch.nn.Module: Assembled model ready for training/inference
        
    Examples:
        >>> model = build_model_from_string('001')
        >>> x = torch.randn(2, 3, 128, 128)
        >>> output = model(x)
        >>> seg, cls = output['seg'], output['cls']
    """
    config = build_config_from_string(sequence, num_classes, stem_out, drop_path_rate)
    return build_model(config)

def get_layer_info() -> Dict[int, str]:
    """
    Get information about available layers and their indices.
    
    Returns:
        Dict[int, str]: Mapping of layer indices to layer names
    """
    return {i: layer_id for i, layer_id in enumerate(LAYER_IDS)}

def print_layer_info():
    """Print available layers and their corresponding indices."""
    print("Available layers:")
    for i, layer_id in enumerate(LAYER_IDS):
        print(f"  {i}: {layer_id}")