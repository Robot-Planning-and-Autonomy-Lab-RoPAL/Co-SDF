#!/usr/bin/env python3
import os
import torch
import numpy as np
import mujoco
from jaco_arm.jaco import JacoEnv
from utils import load_model

def get_gt_report(q, env):
    # 1. Check Joint Limits
    for i in range(len(q)):
        lo, hi = env.model.jnt_range[i]
        if q[i] < lo - 1e-4 or q[i] > hi + 1e-4:
            return "COLLISION (Joint Limit)"
    
    # 2. Check Physical Contacts
    env.data.qpos[:3] = q
    mujoco.mj_forward(env.model, env.data)
    for i in range(env.data.ncon):
        c = env.data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if g1 in {0,1,2,12} and g2 in {0,1,2,12}: continue
        b1, b2 = env.model.geom_bodyid[g1], env.model.geom_bodyid[g2]
        if env.model.body_parentid[b1] == b2 or env.model.body_parentid[b2] == b1: continue
        return "COLLISION (Physical)"
    
    return "FREE"

def test_points():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_path = "output/ndf/model_final.pt"
    
    # Load environment
    env = JacoEnv(width=100, height=100, model_path="jaco_3dof.xml")
    
    # Load NDF model
    model = load_model(model_path, device, frequencies=[1, 2, 4, 8])
    
    test_points = [
        [0.0, 0.0, 0.0],
        [0.0, 3.14, 0.0],
        [0.0, 2.40, 0.0],
        [0.0, -2.40, 0.0],
        [0.0, 0.0, 3.14],
        [0.0, 0.0, 2.80],
        [1.5, 0.5, -0.5],
        [0.0, 2.32, 0.0],
        [0.0, -2.30, 0.0],    
    ]
    
    print(f"{'Pose [q1, q2, q3]':<30} | {'GT Oracle':<20} | {'NDF Pred':<15} | {'NDF Raw'}")
    print("-" * 85)
    
    for q in test_points:
        # GT Result
        gt_res = get_gt_report(q, env)
        
        # NDF Result
        q_tensor = torch.Tensor([q]).to(device)
        with torch.no_grad():
            y_raw = model(q_tensor).item()
        # In HKR, free is positive, collision is negative
        ndf_res = "FREE" if y_raw > 0 else "COLLISION"
        
        print(f"{str(q):<30} | {gt_res:<20} | {ndf_res:<15} | {y_raw:.4f}")

if __name__ == "__main__":
    test_points()
