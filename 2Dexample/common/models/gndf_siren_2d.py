#!/usr/bin/env python3
import torch
import torch.nn as nn
import sys
import os

# Import geometry tools from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from geometry_tools_2d import ControlPointManager2D
from common.models.siren import SirenNet
from common.models.gndf_2d import GNDF_Preprocessing_2D

def G_NDF_SIREN_2D(n_layers=5, n_hidden=128, n_points_per_link=2,
                  base_position=[5.0, 5.0], 
                  link_lengths=[2.1, 2.1], device="cuda"):
    """
    Model builder for G-NDF 2D using SIREN backbone.
    """
    # 1. Preprocessing (Same as GNDF_2D)
    preprocess = GNDF_Preprocessing_2D(
        n_points_per_link=n_points_per_link,
        base_position=base_position,
        link_lengths=link_lengths,
        device=device
    )
    
    # 2. Dimensions
    n_points = preprocess.cp_manager.get_n_points()
    dim_in = n_points * 2 
        
    # 3. SIREN Backbone
    siren_backbone = SirenNet(
        dim_in=dim_in,
        dim_hidden=n_hidden,
        n_layers=n_layers,
        w0=1.0, # Lower w0 for distance fields to avoid over-oscillating
        w0_first_layer=30.0
    )
    
    # 4. Assemble
    model = nn.Sequential(
        preprocess,
        siren_backbone
    ).to(device)
    
    # Standard metadata tags
    model.id = "G-NDF-2D-SIREN"
    model.meta = [n_layers, n_hidden, n_points_per_link, base_position, link_lengths]
    
    return model

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = G_NDF_SIREN_2D(device=device)
    print(f"Model ID: {model.id}")
    q = torch.randn(10, 2).to(device)
    out = model(q)
    print(f"Input (10, 2) -> Output shape: {out.shape}")
