#!/usr/bin/env python3
import os
import numpy as np
import mujoco
from jaco_arm.jaco import JacoEnv

def verify_collisions(n_check=100):
    # Load sampled collision points
    collapsed_path = "inputs/jaco_Xtrain_in.npy"
    if not os.path.exists(collapsed_path):
        print(f"Error: {collapsed_path} not found. Please run sampling.py first.")
        return
    
    X_in = np.load(collapsed_path)
    print(f"Loaded {len(X_in)} collision points.")
    
    # Initialize environment
    env = JacoEnv(width=101, height=101, model_path="jaco_3dof.xml")
    
    # Randomly select a subset to check if there are many
    if len(X_in) > n_check:
        indices = np.random.choice(len(X_in), n_check, replace=False)
        pts_to_check = X_in[indices]
    else:
        pts_to_check = X_in

    counts = {
        "joint_limit": 0,
        "self_collision": 0,
        "obstacle_collision": 0,
        "ground_collision": 0,
        "other_collision": 0,
        "no_collision_found": 0
    }

    print(f"Checking {len(pts_to_check)} points...")

    for i, q in enumerate(pts_to_check):
        is_collision = False
        collision_reasons = []

        # 1. Check Joint Limits
        for j in range(len(q)):
            if env.model.jnt_limited[j]:
                lo, hi = env.model.jnt_range[j]
                if q[j] < lo or q[j] > hi:
                    is_collision = True
                    collision_reasons.append(f"Joint {j+1} limit [{lo:.3f}, {hi:.3f}] violated (val: {q[j]:.3f})")
        
        if is_collision:
            counts["joint_limit"] += 1
            # Note: we continue checking even if joint limit is hit to see if there are other collisions
        
        # 2. Check Physical Collisions
        env.data.qpos[:3] = q
        mujoco.mj_forward(env.model, env.data)
        
        phys_collision = False
        for j in range(env.data.ncon):
            c = env.data.contact[j]
            g1, g2 = c.geom1, c.geom2
            
            # Ground plane ID is 0, Base geoms are 1, 2, Obstacle is 12 (verified in sampling.py)
            ground_id = 0
            base_ids = {1, 2}
            obstacle_id = 12
            
            # Identify bodies for self-collision check
            b1, b2 = env.model.geom_bodyid[g1], env.model.geom_bodyid[g2]
            name1 = env.model.geom(g1).name or f"geom_{g1}"
            name2 = env.model.geom(g2).name or f"geom_{g2}"

            # Ignore Ground vs Base/Obstacle
            fixed_geoms = {ground_id, 1, 2, obstacle_id}
            if g1 in fixed_geoms and g2 in fixed_geoms:
                continue
                
            # Ignore parent-child body contacts
            if env.model.body_parentid[b1] == b2 or env.model.body_parentid[b2] == b1:
                continue
            
            # Determine cause
            is_collision = True
            phys_collision = True
            
            reason = f"Collision between {name1} and {name2}"
            if g1 == ground_id or g2 == ground_id:
                counts["ground_collision"] += 1
            elif g1 == obstacle_id or g2 == obstacle_id:
                counts["obstacle_collision"] += 1
            elif env.model.geom_bodyid[g1] != 0 and env.model.geom_bodyid[g2] != 0:
                counts["self_collision"] += 1
            else:
                counts["other_collision"] += 1
            
            collision_reasons.append(reason)
            break # Only report the first physical collision found per point for this summary
            
        if not is_collision:
            counts["no_collision_found"] += 1
            if i < 5: # Print first few cases if they exist
                 print(f"Point {i} q={q} - ERROR: No collision found by script logic!")
        # else:
        #     if i < 5:
        #         print(f"Point {i} q={q} - Reasons: {', '.join(collision_reasons)}")

    print("\n" + "="*30)
    print("Verification Summary")
    print("="*30)
    print(f"Total checked: {len(pts_to_check)}")
    for cause, count in counts.items():
        print(f"{cause:20}: {count}")
    print("="*30)

if __name__ == "__main__":
    verify_collisions(n_check=500)
