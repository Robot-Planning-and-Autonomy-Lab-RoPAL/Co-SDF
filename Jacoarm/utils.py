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
    Maps each angle θ to [cos(θ), sin(θ)] to handle periodicity.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # x: (N, D) -> (N, 2D) where D is number of angles
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


def load_model(model_path, device, n_dim=3, n_layers=20, n_hidden=128, model_type="sll"):
    """
    Load a trained Jaco NDF model.
    
    Args:
        model_path: Path to the saved model state dict
        device: Device to load model on ('cuda' or 'cpu')
        n_dim: Input dimension (number of joints, default 3)
        n_layers: Number of layers in the network (default 20)
        n_hidden: Hidden layer size (default 128)
        model_type: Type of Lipschitz architecture ('sll' or 'ortho', default 'sll')
    
    Returns:
        Loaded model in eval mode
    """
    embedding = PeriodicEmbedding()
    model_core = select_model(model_type, n_dim * 2, n_layers, n_hidden)
    model = nn.Sequential(embedding, model_core).to(device)
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
