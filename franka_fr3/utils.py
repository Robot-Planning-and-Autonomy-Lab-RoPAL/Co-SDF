#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import torch.nn as nn

class ForwardKinematics(torch.nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device

        self.offsets = [
            (0, 0, 0.333, 1, 0, 0, 0),
            (0, 0, 0, 0.7071, -0.7071, 0, 0),
            (0, -0.316, 0, 0.7071, 0.7071, 0, 0),
            (0.0825, 0, 0, 0.7071, 0.7071, 0, 0),
            (-0.0825, 0.384, 0, 0.7071, -0.7071, 0, 0),
            (0, 0, 0, 0.7071, 0.7071, 0, 0),
            (0.088, 0, 0, 0.7071, 0.7071, 0, 0),
            (0, 0, 0.107, 0.9238795, 0, 0, -0.3826834),
            (0, 0, 0.0584, 1, 0, 0, 0),
            (0, 0, 0.0584, 0, 0, 0, 1)
        ]
        self.offsets = torch.tensor(self.offsets, device=device).float()
        
    def _quat_to_matrix(self, quat):
        w, x, y, z = quat.unbind(-1)
        tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
        twx, twy, twz = tx * w, ty * w, tz * w
        txx, txy, txz = tx * x, ty * x, tz * x
        tyy, tyz, tzz = ty * y, ty * z, tz * z
        
        row1 = torch.stack([1.0 - (tyy + tzz), txy - twz, txz + twy], dim=-1)
        row2 = torch.stack([txy + twz, 1.0 - (txx + tzz), tyz - twx], dim=-1)
        row3 = torch.stack([txz - twy, tyz + twx, 1.0 - (txx + tyy)], dim=-1)
        return torch.stack([row1, row2, row3], dim=-2)

    def _get_joint_transform(self, angle):
        cos, sin = torch.cos(angle), torch.sin(angle)
        zero, one = torch.zeros_like(angle), torch.ones_like(angle)
        row1 = torch.stack([cos, -sin, zero], dim=-1)
        row2 = torch.stack([sin, cos, zero], dim=-1)
        row3 = torch.stack([zero, zero, one], dim=-1)
        return torch.stack([row1, row2, row3], dim=-2)

    def forward(self, q):
        import torch.func as tf
        def single_fk(q_single):
            transforms = []
            T_curr = torch.eye(4, device=q_single.device)
            # Arm Joints 1-7
            for i in range(7):
                R_stat = self._quat_to_matrix(self.offsets[i, 3:])
                t_stat = self.offsets[i, :3]
                T_stat = torch.cat([torch.cat([R_stat, t_stat.unsqueeze(1)], dim=1), 
                                  torch.tensor([[0,0,0,1]], device=q_single.device)], dim=0)
                T_curr = T_curr @ T_stat
                R_j = self._get_joint_transform(q_single[i])
                T_j = torch.cat([torch.cat([R_j, torch.zeros((3,1), device=q_single.device)], dim=1),
                                torch.tensor([[0,0,0,1]], device=q_single.device)], dim=0)
                T_curr = T_curr @ T_j
                transforms.append(T_curr)
            
            # Hand index 7
            R_h = self._quat_to_matrix(self.offsets[7, 3:])
            T_h_stat = torch.cat([torch.cat([R_h, self.offsets[7, :3].unsqueeze(1)], dim=1),
                                 torch.tensor([[0,0,0,1]], device=q_single.device)], dim=0)
            T_hand = T_curr @ T_h_stat
            transforms.append(T_hand)
            
            # Fingers index 8, 9 (aligned with MJ ctrlrange 0.04)
            t_lf = self.offsets[8, :3] + torch.tensor([0, 0.04, 0], device=q_single.device)
            T_lf = torch.cat([torch.cat([torch.eye(3, device=q_single.device), t_lf.unsqueeze(1)], dim=1),
                            torch.tensor([[0,0,0,1]], device=q_single.device)], dim=0)
            transforms.append(T_hand @ T_lf)
            
            R_rf = self._quat_to_matrix(self.offsets[9, 3:])
            t_rf = self.offsets[9, :3] + torch.tensor([0, -0.04, 0], device=q_single.device)
            T_rf = torch.cat([torch.cat([R_rf, t_rf.unsqueeze(1)], dim=1),
                            torch.tensor([[0,0,0,1]], device=q_single.device)], dim=0)
            transforms.append(T_hand @ T_rf)
            return torch.stack(transforms) # (10, 4, 4)

        return tf.vmap(single_fk)(q)

class ControlPoint:
    def __init__(self, device="cuda", n_points_per_link=1):
        self.fk = ForwardKinematics(device=device)
        self.device = device
        self.n_points_per_link = 1
        self.local_points = []
        
        lengths = [0.166, -0.158, 0.041, 0.192, 0.192, 0.044, 0.053]
        axes = [2, 1, 0, 1, 1, 0, 2]
        for l, a in zip(lengths, axes):
            p = torch.zeros(1, 3, device=device); p[0, a] = l 
            self.local_points.append(p)
            
        for _ in range(2):
            p = torch.zeros(1, 3, device=device); p[0, 2] = 0.0225 
            self.local_points.append(p)
            
        self.seg_indices = [0, 1, 2, 3, 4, 5, 6, 8, 9]

    def get_jacobian_determinant(self, q):
        import torch.func as tf
        def get_flat_points_single(q_single):
            transforms = self.fk.forward(q_single.unsqueeze(0)).squeeze(0)
            all_pts = []
            for i, seg_idx in enumerate(self.seg_indices):
                lp = self.local_points[i]
                T = transforms[seg_idx]; R = T[:3, :3]; t = T[:3, 3]
                all_pts.append((R @ lp.t()).t() + t)
            return torch.cat(all_pts).view(-1)
        jac = tf.vmap(tf.jacrev(get_flat_points_single))(q)
        jtj = torch.matmul(jac.transpose(1, 2), jac)
        det = torch.linalg.det(jtj + 1e-4 * torch.eye(7, device=q.device))
        return torch.sqrt(torch.abs(det))

    def get_control_points(self, q):
        transforms = self.fk(q)
        all_points = []
        for i, seg_idx in enumerate(self.seg_indices):
            lp = self.local_points[i]
            T = transforms[:, seg_idx]; R = T[:, :3, :3]; t = T[:, :3, 3].unsqueeze(1)
            all_points.append(torch.matmul(R, lp.t()).transpose(1, 2) + t)
        return torch.cat(all_points, dim=1)


def load_model(model_path, device, n_dim=7, n_layers=20, n_hidden=256, model_type="sll", n_points_per_link=2):
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
    return sum(p.numel() for p in model.parameters() if p.requires_grad)