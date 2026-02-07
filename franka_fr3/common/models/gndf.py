import torch
import torch.nn as nn
from utils import ControlPoint
from common.models.sll import DenseSDP, SDPBasedLipschitzDense
from deel import torchlip


class GNDF_Preprocessing_7D(nn.Module):
    """
    Modular layer for G-NDF preprocessing (aligned with 2D version):
    - FK (joint angles -> control points)
    - Coordinate normalization
    - NO periodic embedding
    """
    def __init__(self, n_points_per_link=1, device="cuda"):
        super().__init__()
        self.device = device
        self.cp_manager = ControlPoint(device=device, n_points_per_link=n_points_per_link)
        
        self.register_buffer('xyz_mean', torch.tensor(0.116775))
        self.register_buffer('xyz_std', torch.tensor(0.272722))
        
    def forward(self, x):
        if x.shape[1] == 7:
            q = x
            points = self.cp_manager.get_control_points(q)
            points_flat = points.reshape(q.shape[0], -1)
            points_norm = (points_flat - self.xyz_mean) / self.xyz_std
            return points_norm
        else:
            points_norm = (x - self.xyz_mean) / self.xyz_std
            return points_norm


def G_NDF(n_layers=30, n_hidden=256, n_points_per_link=1, device="cuda"):
    """
    Functional model builder for G-NDF 7D. 
    Maintains 1-Lipschitz property through SDP-based layers.
    """
    preprocess = GNDF_Preprocessing_7D(
        n_points_per_link=n_points_per_link,
        device=device
    )
    
    n_points = 9 * n_points_per_link
    dim_in = n_points * 3 
    
    class RobustPad(nn.Module):
        def __init__(self, target):
            super().__init__()
            self.target = target
        def forward(self, x):
            if x.shape[1] >= self.target: return x
            return torch.nn.functional.pad(x, (0, self.target - x.shape[1]))

    layers = [preprocess, RobustPad(n_hidden)]
    
    # Core Lipschitz Layers
    for _ in range(n_layers):
        layers.append(SDPBasedLipschitzDense(n_hidden))
        
    layers.append(torchlip.FrobeniusLinear(n_hidden, 1))
    
    model = nn.Sequential(*layers).to(device)
    model.id = "G-NDF-7D"
    model.meta = [n_layers, n_hidden, n_points_per_link]
    
    return model

class G_NDF_Legacy(nn.Module):
    def __init__(self, n_layers=20, n_hidden=256, device="cuda"):
        super().__init__()
        self.device = device
        self.cp_manager = ControlPointManager(device=device, n_points_per_link=2)
        in_dim = 7 + 42
        self.core = DenseSDP(in_dim, n_hidden, n_layers)

    def forward(self, x):
        if x.shape[1] == 7:
            q = x
            points = self.cp_manager.get_control_points(q)
            points_flat = points.reshape(q.shape[0], -1)
            full_x = torch.cat([q, points_flat], dim=-1)
            return self.core(full_x)
        else:
            return self.core(x)
