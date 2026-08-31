#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np

class ForwardKinematics(nn.Module):
    def __init__(self, base_position=[5.0, 5.0], link_lengths=[2.1, 2.1], device="cuda"):
        super().__init__()
        self.device = device
        self.base = torch.tensor(base_position, dtype=torch.float32, device=device)
        self.lengths = torch.tensor(link_lengths, dtype=torch.float32, device=device)
        
    def forward(self, q):
        B = q.shape[0]
        device = q.device
        
        p0 = self.base.unsqueeze(0).expand(B, -1)
        
        x1 = p0[:, 0] + self.lengths[0] * torch.cos(q[:, 0])
        y1 = p0[:, 1] + self.lengths[0] * torch.sin(q[:, 0])
        p1 = torch.stack([x1, y1], dim=-1)
        
        x2 = p1[:, 0] + self.lengths[1] * torch.cos(q[:, 0] + q[:, 1])
        y2 = p1[:, 1] + self.lengths[1] * torch.sin(q[:, 0] + q[:, 1])
        p2 = torch.stack([x2, y2], dim=-1)
        
        positions = torch.stack([p0, p1, p2], dim=1)
        
        return positions


class ControlPoint(nn.Module):
    def __init__(self, base_position=[5.0, 5.0], link_lengths=[2.1, 2.1], n_points_per_link=1, device="cuda"):
        super().__init__()
        self.device = device
        self.n_points_per_link = n_points_per_link
        self.fk = ForwardKinematics(base_position, link_lengths, device)
        
    def get_control_points(self, q):
        B = q.shape[0]
        
        positions = self.fk(q)
        p0 = positions[:, 0, :]
        p1 = positions[:, 1, :]
        p2 = positions[:, 2, :]
        
        control_points = []
        
        for i in range(1, self.n_points_per_link + 1):
            ratio = i / (self.n_points_per_link + 1)
            point = p0 + ratio * (p1 - p0)
            control_points.append(point)
        
        for i in range(1, self.n_points_per_link + 1):
            ratio = i / (self.n_points_per_link + 1)
            point = p1 + ratio * (p2 - p1)
            control_points.append(point)
        
        points = torch.stack(control_points, dim=1)
        
        return points
    
    def get_n_points(self):
        return 2 * self.n_points_per_link
