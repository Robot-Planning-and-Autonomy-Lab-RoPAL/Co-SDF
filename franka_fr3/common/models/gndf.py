import torch
import torch.nn as nn
from geometry_tools import ControlPointManager
from common.models.sll import DenseSDP, SDPBasedLipschitzDense
from deel import torchlip


class GNDF_Preprocessing_7D(nn.Module):
    """
    Modular layer for G-NDF preprocessing (aligned with 2D version):
    - FK (joint angles -> control points)
    - Coordinate normalization
    - NO periodic embedding (use raw joint angles directly)
    """
    def __init__(self, n_points_per_link=2, device="cuda"):
        super().__init__()
        self.device = device
        self.cp_manager = ControlPointManager(device=device, n_points_per_link=n_points_per_link)
        
        # Statistics for normalization (computed from training data)
        # Data range: [-0.85, 1.19], mean=0.1472, std=0.3259
        self.register_buffer('xyz_mean', torch.tensor(0.147218))
        self.register_buffer('xyz_std', torch.tensor(0.325865))
        
    def forward(self, x):
        if x.shape[1] == 7:
            # Inference: input is q (B, 7)
            q = x
            points = self.cp_manager.get_control_points(q)
            points_flat = points.reshape(q.shape[0], -1)
            # Normalize physical points
            points_norm = (points_flat - self.xyz_mean) / self.xyz_std
            return torch.cat([q, points_norm], dim=-1)
        else:
            # Training: input is [q, xyz] (B, 7 + 42)
            q, points_raw = x[:, :7], x[:, 7:]
            points_norm = (points_raw - self.xyz_mean) / self.xyz_std
            return torch.cat([q, points_norm], dim=-1)


def G_NDF(n_layers=30, n_hidden=256, n_points_per_link=2, device="cuda"):
    """
    Functional model builder for G-NDF 7D, aligned with 2D style.
    Returns an nn.Sequential model with preprocessing + core SLL network.
    NO frequency encoding or periodic embedding.
    """
    # 1. Preprocessing layer
    preprocess = GNDF_Preprocessing_7D(
        n_points_per_link=n_points_per_link,
        device=device
    )
    
    # 2. Calculate dimensions
    n_joints = 7
    n_points = 7 * n_points_per_link  # 7 links, each with n_points_per_link control points
    dim_in = n_joints + n_points * 3  # 7 + 42 = 49 (for n_points_per_link=2)
    
    # 3. Assemble Sequential model (matching 2D pattern)
    layers = [preprocess]
    layers.append(nn.ZeroPad1d((0, n_hidden - dim_in)))
    
    for _ in range(n_layers):
        layers.append(SDPBasedLipschitzDense(n_hidden))
        
    layers.append(torchlip.FrobeniusLinear(n_hidden, 1))
    
    model = nn.Sequential(*layers).to(device)
    
    # 4. Attach metadata for saving/loading
    model.id = "G-NDF-7D"
    model.meta = [n_layers, n_hidden, n_points_per_link]
    
    return model


# Legacy class-based G_NDF for backward compatibility
class G_NDF_Legacy(nn.Module):
    """Legacy G-NDF (will be deprecated)."""
    
    def __init__(self, n_layers=20, n_hidden=256, device="cuda"):
        super().__init__()
        self.device = device
        self.cp_manager = ControlPointManager(device=device, n_points_per_link=2)
        
        # 7 joints (q) + 14 points * 3 coords (xyz) = 49 dimensions.
        in_dim = 7 + 42
        
        self.core = DenseSDP(in_dim, n_hidden, n_layers)
        self.id = "G-NDF-Legacy"
        self.meta = [n_layers, n_hidden]

    def forward(self, x):
        """Forward pass.
        If x.shape[1] == 7, it calculates xyz internally (for inference/optim).
        If x.shape[1] == 49, it uses precomputed [q, xyz] (for training).
        """
        if x.shape[1] == 7:
            q = x
            points = self.cp_manager.get_control_points(q)
            points_flat = points.reshape(q.shape[0], -1)
            full_x = torch.cat([q, points_flat], dim=-1)
            return self.core(full_x)
        else:
            return self.core(x)
