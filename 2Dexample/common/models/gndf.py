#!/usr/bin/env python3
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from geometry import ControlPoint
from common.models.sll import SDPBasedLipschitzDense
from deel import torchlip

class GNDF(nn.Module):
    def __init__(self, n_points_per_link=2, base_position=[5.0, 5.0], link_lengths=[2.1, 2.1], device="cuda"):
        super().__init__()
        self.device = device
        self.cp_manager = ControlPoint(
            base_position=base_position,
            link_lengths=link_lengths,
            n_points_per_link=n_points_per_link,
            device=device
        )
        self.register_buffer('xyz_std', torch.tensor(5.0))
        
    def forward(self, x):
        if x.shape[1] == 2:
            q = x
            points = self.cp_manager.get_control_points(q)
            points_flat = points.reshape(q.shape[0], -1)
            points_norm = (points_flat - 5.0) / self.xyz_std
            return points_norm
        else:
            points_norm = (x - 5.0) / self.xyz_std
            return points_norm

def G_NDF(n_layers=20, n_hidden=128, n_points_per_link=2, base_position=[5.0, 5.0], link_lengths=[2.1, 2.1], device="cuda"):
    preprocess = GNDF(
        n_points_per_link=n_points_per_link,
        base_position=base_position,
        link_lengths=link_lengths,
        device=device
    )
    
    n_joints = 2
    n_points = preprocess.cp_manager.get_n_points()
    dim_in = n_points * 2 
    layers = [preprocess]
    layers.append(nn.ZeroPad1d((0, n_hidden - dim_in)))
    
    for _ in range(n_layers):
        layers.append(SDPBasedLipschitzDense(n_hidden))
        
    layers.append(torchlip.FrobeniusLinear(n_hidden, 1))
    
    model = nn.Sequential(*layers).to(device)
    
    model.id = "GNDF"
    model.meta = [n_layers, n_hidden, n_points_per_link, [], base_position, link_lengths]
    
    return model
