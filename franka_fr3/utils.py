#!/usr/bin/env python3
"""
Utility functions for Franka FR3 NDF
"""
import os
import sys
import torch
import torch.nn as nn

# Add 2Dexample to path for common modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '2Dexample'))
from common.models import select_model


def load_model(model_path, device, n_dim=7, n_layers=20, n_hidden=256, model_type="sll", n_points_per_link=2):
    """
    Load a trained Franka FR3 NDF model.
    """
    if model_type.lower() == "gndf":
        from common.models.gndf import G_NDF
        model = G_NDF(n_layers=n_layers, n_hidden=n_hidden, 
                      n_points_per_link=n_points_per_link, device=device).to(device)
    else:
        # Standard SLL model
        model = select_model(model_type, n_dim, n_layers, n_hidden).to(device)
    
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

