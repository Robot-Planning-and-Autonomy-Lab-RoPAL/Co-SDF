import torch
import torch.nn as nn
from utils import ControlPoint
from common.models.sll import SDPBasedLipschitzDense
from deel import torchlip


class COSDFPreprocessor(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.cp_manager = ControlPoint(device=device)
        self.register_buffer('xyz_mean', torch.tensor(0.116775))
        self.register_buffer('xyz_std', torch.tensor(0.272722))

    def forward(self, x):
        if x.shape[1] == 7:
            points = self.cp_manager.get_control_points(x)
            x = points.reshape(x.shape[0], -1)
        return (x - self.xyz_mean) / self.xyz_std


def COSDF(n_layers=30, n_hidden=256, device="cuda"):
    preprocess = COSDFPreprocessor(device=device)

    dim_in = 9 * 3
    layers = [preprocess]
    layers.append(nn.ZeroPad1d((0, n_hidden - dim_in)))

    for _ in range(n_layers):
        layers.append(SDPBasedLipschitzDense(n_hidden))

    layers.append(torchlip.FrobeniusLinear(n_hidden, 1))

    model = nn.Sequential(*layers).to(device)
    model.id = "COSDF"
    model.meta = [n_layers, n_hidden]
    return model
