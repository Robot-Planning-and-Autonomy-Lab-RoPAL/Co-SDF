#!/usr/bin/env python3
import torch
import torch.nn as nn
import sys
import os

from .gndf import GNDF
from .mlp import MultiLayerPerceptron, MultiLayerPerceptronSkips

def G_NDF_MLP(n_layers=8, n_hidden=256, n_points_per_link=2, base_position=[5.0, 5.0], link_lengths=[2.1, 2.1], device="cuda", model_type="mlps", skips=[4]):
    preprocess = GNDF(
        n_points_per_link=n_points_per_link,
        base_position=base_position,
        link_lengths=link_lengths,
        device=device
    )
    
    n_points = preprocess.cp_manager.get_n_points()
    dim_in = n_points * 2 
    
    if model_type == "mlps":
        backbone = MultiLayerPerceptronSkips(dim_in, n_hidden, n_layers, skips=skips)
    else:
        backbone = MultiLayerPerceptron(dim_in, n_hidden, n_layers)
        
    model = nn.Sequential(preprocess, backbone).to(device)
    
    model.id = "G-NDF-MLP"
    model.meta = [n_layers, n_hidden, n_points_per_link, base_position, link_lengths, model_type, skips]
    
    return model
