#!/usr/bin/env python3
import os
import time
import numpy as np
import mujoco
import mujoco.viewer
from jaco_arm.jaco import JacoEnv

def verify_physically():
    # Load sampled collision points
    collapsed_path = "inputs/jaco_Xtrain_in.npy"
    if not os.path.exists(collapsed_path):
        print(f"Error: {collapsed_path} not found. Please run sampling.py first.")
        return
    
    X_in = np.load(collapsed_path)
    print(f"Loaded {len(X_in)} collision points.")
    
    # Initialize environment
    env = JacoEnv(width=800, height=800, model_path="jaco_3dof.xml")
    
    print("\n" + "="*50)
    print("Physical Workspace Verification Tool")
    print("="*50)
    print("Controls:")
    print("  - Press 'n' in the viewer to view the NEXT collision point.")
    print("  - Press 'r' in the viewer to view a RANDOM collision point.")
    print("  - Press 'q' or close the window to quit.")
    print("="*50)

    current_idx = 0
    total_points = len(X_in)

    def get_collision_report(q):
        report = []
        # 1. Check Joint Limits
        for i in range(3):
            lo, hi = env.model.jnt_range[i]
            if q[i] < lo - 1e-4 or q[i] > hi + 1e-4:
                report.append(f"Joint {i+1} LIMIT VIOLATION: {q[i]:.3f} rad (Range: [{lo:.3f}, {hi:.3f}])")
        
        # 2. Check Physical Contacts
        env.data.qpos[:3] = q
        mujoco.mj_forward(env.model, env.data)
        
        found_phys = False
        for i in range(env.data.ncon):
            c = env.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            
            # Filter as per sampling.py
            if g1 in {0,1,2,12} and g2 in {0,1,2,12}: continue
            b1, b2 = env.model.geom_bodyid[g1], env.model.geom_bodyid[g2]
            if env.model.body_parentid[b1] == b2 or env.model.body_parentid[b2] == b1: continue
            
            found_phys = True
            n1 = env.model.geom(g1).name or f"id_{g1}"
            n2 = env.model.geom(g2).name or f"id_{g2}"
            report.append(f"PHYSICAL COLLISION: '{n1}' and '{n2}'")
            break # Just show the first one
            
        if not report:
            return "No collision detected by oracle."
        return "\n".join(report)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # Initialize with first point
        print(f"\n[Point {current_idx}/{total_points}] q={X_in[current_idx]}")
        print(get_collision_report(X_in[current_idx]))
        env.data.qpos[:3] = X_in[current_idx]
        viewer.sync()

        while viewer.is_running():
            key_pressed = None
            # Note: Passive viewer doesn't have a simple blocking key input, 
            # so we use terminal input in a non-blocking-ish or simple loop way,
            # or just instruct the user to use the terminal.
            # However, for simplicity, we'll just wait for the user to press 'n' in terminal
            # to keep it robust across different MuJoCo versions.
            
            cmd = input("\nEnter command [n (next), r (random), q (quit), index number]: ").strip().lower()
            
            if cmd == 'n':
                current_idx = (current_idx + 1) % total_points
            elif cmd == 'r':
                current_idx = np.random.randint(total_points)
            elif cmd == 'q':
                break
            elif cmd.isdigit():
                idx = int(cmd)
                if 0 <= idx < total_points:
                    current_idx = idx
                else:
                    print(f"Invalid index. Max index is {total_points-1}.")
                    continue
            else:
                print("Unknown command.")
                continue

            q = X_in[current_idx]
            print(f"\n[Point {current_idx}/{total_points}] q={q}")
            print(get_collision_report(q))
            
            # Update viewer
            env.data.qpos[:3] = q
            mujoco.mj_forward(env.model, env.data)
            viewer.sync()

if __name__ == "__main__":
    verify_physically()
