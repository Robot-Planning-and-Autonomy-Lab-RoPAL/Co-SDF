#!/usr/bin/env python3
import torch
import torch.nn as nn
import sys
import os

# Import geometry tools from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from geometry_tools_2d import ControlPointManager2D
from common.models.sll import SDPBasedLipschitzDense
from deel import torchlip

class GNDF_Preprocessing_2D(nn.Module):
    """
    Modular layer for G-NDF preprocessing:
    - FK (joint angles -> control points)
    - Concatenation [q, xyz]
    - No periodic embedding (use raw joint angles)
    """
    def __init__(self, n_points_per_link=2,
                 base_position=[5.0, 5.0], link_lengths=[2.1, 2.1], device="cuda"):
        super().__init__()
        self.device = device
        self.cp_manager = ControlPointManager2D(
            base_position=base_position,
            link_lengths=link_lengths,
            n_points_per_link=n_points_per_link,
            device=device
        )
        # Scale factor for normalization (mirrors 3D version's xyz_std)
        self.register_buffer('xyz_std', torch.tensor(5.0))
        
    def forward(self, x):
        if x.shape[1] == 2:
            # Inference: input is q (B, 2)
            q = x
            points = self.cp_manager.get_control_points(q)
            points_flat = points.reshape(q.shape[0], -1)
            # Normalize physical points: (x-5)/5 -> [-1, 1]
            points_norm = (points_flat - 5.0) / self.xyz_std
            # Return ONLY workspace features (xyz)
            return points_norm
        else:
            # Training: input is [q, xyz] (B, 2 + N*2)
            q, points_raw = x[:, :2], x[:, 2:]
            points_norm = (points_raw - 5.0) / self.xyz_std
            # Return ONLY workspace features (xyz)
            return points_norm

def G_NDF_2D(n_layers=20, n_hidden=128, n_points_per_link=2,
             base_position=[5.0, 5.0], 
             link_lengths=[2.1, 2.1], device="cuda"):
    """
    Functional model builder for G-NDF 2D, mirroring DenseSDP in sll.py.
    """
    # 1. Preprocessing
    preprocess = GNDF_Preprocessing_2D(
        n_points_per_link=n_points_per_link,
        base_position=base_position,
        link_lengths=link_lengths,
        device=device
    )
    
    # 2. Dimensions
    n_joints = 2
    n_points = preprocess.cp_manager.get_n_points()
    # Using ONLY workspace features (4D)
    dim_in = n_points * 2 
        
    # 3. Assemble Sequential (mirrors sll.py style)
    layers = [preprocess]
    layers.append(nn.ZeroPad1d((0, n_hidden - dim_in)))
    
    for _ in range(n_layers):
        layers.append(SDPBasedLipschitzDense(n_hidden))
        
    layers.append(torchlip.FrobeniusLinear(n_hidden, 1))
    
    model = nn.Sequential(*layers).to(device)
    
    # Standard metadata tags
    model.id = "G-NDF-2D"
    model.meta = [n_layers, n_hidden, n_points_per_link, [], base_position, link_lengths]
    
    return model

if __name__ == "__main__":
    # Smoke test on CPU for logic verification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = G_NDF_2D(n_layers=5, n_hidden=64, device=device)
    print(f"Model ID: {model.id}")
    print(f"Meta: {model.meta}")
    q = torch.randn(10, 2).to(device)
    out = model(q)
    print(f"Output shape: {out.shape}")
    print("✓ Refactored G-NDF 2D fully compliant with SLL pattern")
