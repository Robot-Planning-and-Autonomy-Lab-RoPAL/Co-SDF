#!/usr/bin/env python3
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import robot_tools
from common.models import load_model as global_load_model
from common.escape_utils import sample_near_boundary_collision, check_2d_collision, get_device

def load_2d_model(path, device, n_layers=20, n_hidden=128):
    return global_load_model(path, device)

class GradientEscaper:
    def __init__(self, model, problem, device="cuda", safety_margin=0.0):
        self.model = model
        self.problem = problem
        self.device = device
        self.safety_margin = safety_margin

    def run(self, q0, lr=0.1, max_steps=100):
        q_curr = torch.tensor(q0, dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([q_curr], lr=lr)
        
        trajectory = [q0.copy()]
        ndf_history = []
        ndf_success = False
        
        for step in range(max_steps):
            optimizer.zero_grad()
            ndf_val = self.model(q_curr.reshape(1, -1)).squeeze()
            ndf_history.append(ndf_val.item())
            
            if ndf_val.item() >= self.safety_margin:
                ndf_success = True
                break
            
            loss = torch.nn.functional.relu(self.safety_margin - ndf_val)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                q_curr.data %= (2 * np.pi)
                
                if step >= 10:
                    recent_avg = np.mean(ndf_history[-5:])
                    prev_avg = np.mean(ndf_history[-10:-5]) if len(ndf_history) >= 10 else ndf_history[0]
                    if abs(recent_avg - prev_avg) < 1e-4:
                        q_curr.add_(torch.randn_like(q_curr) * 0.01)
            
            trajectory.append(q_curr.detach().cpu().numpy().copy())
            
        final_q = q_curr.detach().cpu().numpy()
        gt_success = not check_2d_collision(self.problem, final_q)
            
        return {
            "method": "adam_gradient",
            "ndf_success": ndf_success,
            "gt_success": gt_success,
            "gt_steps": (step + 1),
            "ndf_steps": (step + 1),
            "trajectory": np.array(trajectory)
        }

class SamplingEscaper:
    def __init__(self, model, problem, device="cuda", safety_margin=0.0):
        self.model = model
        self.problem = problem
        self.device = device
        self.safety_margin = safety_margin

    def run(self, q0, step_size=0.1, max_steps=100, K=50):
        q_curr = torch.tensor(q0, dtype=torch.float32, device=self.device)
        
        trajectory = [q0.copy()]
        ndf_history = []
        ndf_success = False
        
        for step in range(max_steps):
            with torch.no_grad():
                ndf_curr = self.model(q_curr.reshape(1, -1)).item()
                ndf_history.append(ndf_curr)
                
                if ndf_curr >= self.safety_margin:
                    ndf_success = True
                    break
                
                noise = torch.randn(K, 2, device=self.device) * step_size
                candidates = (q_curr + noise) % (2 * np.pi)
                
                ndf_candidates = self.model(candidates).squeeze()
                best_idx = torch.argmax(ndf_candidates)
                q_curr = candidates[best_idx]
                
                trajectory.append(q_curr.cpu().numpy().copy())
                
        final_q = q_curr.cpu().numpy()
        gt_success = not check_2d_collision(self.problem, final_q)
            
        return {
            "method": "sampling",
            "ndf_success": ndf_success,
            "gt_success": gt_success,
            "gt_steps": (step + 1),
            "ndf_steps": (step + 1),
            "trajectory": np.array(trajectory),
            "ndf_history": ndf_history
        }