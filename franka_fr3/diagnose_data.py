import numpy as np
import mujoco
import os

def check_data_validity(data_path, scene_xml):
    # Load limits
    abs_xml = os.path.abspath(scene_xml)
    mj_model = mujoco.MjModel.from_xml_path(abs_xml)
    mj_data = mujoco.MjData(mj_model)
    lo = mj_model.jnt_range[:7, 0]
    hi = mj_model.jnt_range[:7, 1]
    
    # Identify robot bodies
    robot_body_ids = set()
    base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")
    for i in range(mj_model.nbody):
        curr = i
        while curr > 0:
            if curr == base_id: robot_body_ids.add(i); break
            curr = mj_model.body_parentid[curr]
    is_robot = np.zeros(mj_model.nbody, dtype=bool)
    is_robot[list(robot_body_ids)] = True

    # Load data
    if not os.path.exists(data_path):
        print(f"Path {data_path} not found")
        return
        
    X = np.load(data_path)
    q_all = X[:, :7]
    total = len(X)
    
    limit_violations = 0
    self_collisions = 0
    pure_obstacle_collisions = 0
    
    print(f"\nAnalyzing: {data_path}")
    for i in tqdm(range(min(total, 5000)), desc="Checking samples"): # Sample 5k for speed
        q = q_all[i]
        
        # 1. Check Limits
        if np.any(q < lo) or np.any(q > hi):
            limit_violations += 1
            continue
            
        # 2. Check Collision Types
        mj_data.qpos[:7] = q
        if mj_model.nq > 7:
            mj_data.qpos[7:mj_model.nq] = 0.04 # Match DataGenerator fingers
        mujoco.mj_forward(mj_model, mj_data)
        
        has_self = False
        has_env = False
        for j in range(mj_data.ncon):
            c = mj_data.contact[j]
            b1, b2 = mj_model.geom_bodyid[c.geom1], mj_model.geom_bodyid[c.geom2]
            
            if is_robot[b1] and is_robot[b2]:
                has_self = True
            elif is_robot[b1] ^ is_robot[b2]:
                has_env = True
        
        if has_env:
            pure_obstacle_collisions += 1
        elif has_self:
            self_collisions += 1

    print(f"Results (Sampled 5000):")
    print(f"  Joint Limit Violations: {limit_violations}")
    print(f"  Pure Self-Collisions:  {self_collisions}")
    print(f"  Obstacle Collisions:   {pure_obstacle_collisions}")
    print(f"  Cleanliness Note: If this is an 'In' file, Self-Collisions should be 0.")

if __name__ == "__main__":
    from tqdm import tqdm
    # Check the "In" dataset for any self-collisions (should be 0) or limit violations (should be 0)
    check_data_validity("inputs/bookshelf/fr3_Qtrain_in.npy", "scenes/bookshelf.xml")
    # Check the "Out" dataset (should also be 0 for both)
    check_data_validity("inputs/bookshelf/fr3_Qtrain_out.npy", "scenes/bookshelf.xml")
