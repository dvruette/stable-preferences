import re
from typing import List

import torch
import torch.nn as nn
from dataclasses import dataclass


class IA3Adapter(nn.Module):
    def __init__(self, layer: nn.Module, init_scale: float = 0.0):
        self.layer = layer
        self.out_features = layer.out_features

        _init_weights = init_scale * torch.randn(self.out_features)
        self.weights = nn.Parameter(_init_weights)

    def forward(self, x):
        x = self.layer(x)
        x = (1 + self.weights) * x
        return x


@dataclass
class IA3Config:
    target_modules: List[str] | str


class IA3Model(nn.Module):
    def __init__(self, model: nn.Moduel, config: IA3Config):
        self.base_model = model
        self.config = config
        
        # adopt forward pass of the base model
        self.forward = self.base_model.forward

        self._freeze_base_model()
        self._patch_base_model()

    def _freeze_base_model(self):
        for p in self.base_model.parameter():
            p.requires_grad_(False)

    def _patch_base_model(self):
        has_target_module = False

        for key, target in self.base_model.named_modules():
            if isinstance(self.peft_config.target_modules, str):
                is_target = re.fullmatch(self.config.target_modules, key)
            else:
                is_target = any(key.endswith(target_key) for target_key in self.config.target_modules)
            
            if is_target:
                has_target_module = True

                parent_key = ".".join(key.split(".")[:-1])
                target_key = key.split(".")[-1]
                parent = self.base_model.get_submodule(parent_key)

                if hasattr(target, "out_features"):
                    dim = target.out_features
                elif hasattr(target, "proj") and hasattr(target.proj, "out_features"):
                    dim = target.proj.out_features
                dim = target.out_features
                new_module = IA3Adapter(target)
                setattr(parent, target_key, new_module)

        if not has_target_module:
            raise ValueError(f"No target modules found: {self.config.target_modules}")

    @property
    def _adapter_layers(self):
        for module in self.modules():
            if isinstance(module, IA3Adapter):
                yield module

    @property
    def modules_to_save(self):
        return list(self._adapter_layers)
