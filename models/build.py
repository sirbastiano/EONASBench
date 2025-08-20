"""
Model builder entry point for benchmarking framework.
"""
import yaml
import json
from models.backbone import *
from models.upernet import *
from layers.registry import build_cell
from typing import Any, Dict


import torch
import torch.nn as nn
from models.backbone import Backbone
from models.upernet import UPerNet

class Model(nn.Module):
    """
    Full model: backbone, UPerNet head, GAP classifier.
    Args:
        backbone (nn.Module): Backbone network.
        head (nn.Module): UPerNet head.
        gap_cls (nn.Module): GAP classifier.
    """
    def __init__(self, backbone: nn.Module, num_classes: int, out_channels: list):
        super().__init__()
        self.backbone = backbone
        self.upernet = UPerNet(in_channels_list=out_channels, num_classes=num_classes)
        self._final_C = out_channels[-1] if out_channels else 1
        self.num_classes = num_classes
        self.gap_cls = nn.Linear(self._final_C, self.num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        feats, x_last = self.backbone(x)
        # Defensive: check x_last shape and re-init classifier if needed
        if x_last.shape[1] != self._final_C:
            self._final_C = x_last.shape[1]
            self.gap_cls = nn.Linear(self._final_C, self.num_classes)
        seg = self.upernet(feats)
        cls = self.gap_cls(x_last.mean(dim=[2,3]))
        return {'seg': seg, 'cls': cls}

def build_model(config: Any) -> nn.Module:
    """
    Build the full model from YAML/JSON config.
    Args:
        config (dict or str): Config dict or path to YAML/JSON file.
    Returns:
        nn.Module: Assembled model.
    """
    if isinstance(config, str):
        if config.endswith('.yaml') or config.endswith('.yml'):
            with open(config, 'r') as f:
                cfg = yaml.safe_load(f)
        elif config.endswith('.json'):
            with open(config, 'r') as f:
                cfg = json.load(f)
        else:
            raise ValueError('Unknown config file type')
    else:
        cfg = config
    model_cfg = cfg['model'] if 'model' in cfg else cfg
    backbone = Backbone(model_cfg)
    out_channels = backbone.out_channels
    num_classes = model_cfg['num_classes']
    return Model(backbone, num_classes, out_channels)
