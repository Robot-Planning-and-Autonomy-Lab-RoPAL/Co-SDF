#!/usr/bin/env python3
"""
Utility functions for Jaco Arm NDF
"""
import os
import sys
import torch
import torch.nn as nn

# Add 2Dexample to path for common modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '2Dexample'))
from common.models import select_model


class PeriodicEmbedding(nn.Module):
    """
    Periodic embedding layer for angular inputs.
    Maps each angle θ to [cos(k*θ), sin(k*θ)] for multiple frequencies k.
    """
    def __init__(self, frequencies=[1]):
        super().__init__()
        self.frequencies = frequencies
    
    def forward(self, x):
        # x: (N, D) -> (N, 2*len(frequencies)*D)
        out = []
        for f in self.frequencies:
            out.append(torch.cos(f * x))
            out.append(torch.sin(f * x))
        return torch.cat(out, dim=-1)


def load_model(model_path, device, n_dim=3, n_layers=20, n_hidden=256, model_type="sll", frequencies=[1, 2]):
    """
    Load a trained Jaco NDF model.
    """
    embedding = PeriodicEmbedding(frequencies=frequencies)
    # The input dimension to the core model depends on frequencies
    core_in_dim = n_dim * 2 * len(frequencies)
    model_core = select_model(model_type, core_in_dim, n_layers, n_hidden)
    model = nn.Sequential(embedding, model_core).to(device)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    # Filter out metadata keys
    model_keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
