#!/usr/bin/env python3
import numpy as np
import mujoco
import os
import argparse

def verify_trajectory(traj_path, scene_xml):
    if not os.path.exists(traj_path):
        print(f"Error: Trajectory file '{traj_path}' not found.")
        return

    path = np.load(traj_path)
    print(f" PATH VERIFICATION: {os.path.basename(traj_path)}")
    print(f" Total Waypoints: {len(path)}")

    abs_xml = os.path.abspath(scene_xml)
    model = mujoco.MjModel.from_xml_path(abs_xml)
    data = mujoco.MjData(model)

    robot_body_ids = set()
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    if base_id == -1:
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fr3_link0")
    
    if base_id != -1:
        for i in range(model.nbody):
            curr = i
            while curr > 0:
                if curr == base_id:
                    robot_body_ids.add(i)
                    break
                curr = model.body_parentid[curr]
    
    is_robot_body = np.zeros(model.nbody, dtype=bool)
    if robot_body_ids:
        is_robot_body[list(robot_body_ids)] = True

    collisions = []
    for idx, q in enumerate(path):
        data.qpos[:7] = q[:7] if len(q) >= 7 else q
        if model.nq > 7:
            data.qpos[7:] = 0.04
        mujoco.mj_forward(model, data)

        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            
            is_b1_robot = is_robot_body[b1]
            is_b2_robot = is_robot_body[b2]
            
            if not (is_b1_robot or is_b2_robot): continue
            
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1)
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2)
            collisions.append((idx, n1, n2))
            break
    
    if not collisions:
        print("\nSUCCESS: Path is 100% COLLISION-FREE!")
    else:
        print(f"\nFAILURE: Detected collisions at {len(collisions)} points.")
        unique_pairs = {}
        for idx, n1, n2 in collisions:
            pair = tuple(sorted((n1, n2)))
            unique_pairs[pair] = unique_pairs.get(pair, 0) + 1
        
        print("Most common collision pairs:")
        for (n1, n2), count in sorted(unique_pairs.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {n1} <-> {n2}: {count} occurrences")
