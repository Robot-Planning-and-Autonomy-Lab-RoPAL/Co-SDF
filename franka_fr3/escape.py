import torch
import numpy as np
import mujoco
import os
import time
from tqdm import tqdm
from geometry_tools import ControlPointManager
from common.models import load_model

class Escape:
    def __init__(self, model_path, scene_xml, device="cuda"):
        self.device = device
        self.model = load_model(model_path, device)
        self.model.eval()
        self.scene_xml = os.path.abspath(scene_xml)
        self.mj_model = mujoco.MjModel.from_xml_path(self.scene_xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        
        # 1. Standard Initial State (qpos0)
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        if self.mj_model.nkey > 0:
            self.qpos0 = self.mj_model.key_qpos[0].copy()
        else:
            self.qpos0 = self.mj_data.qpos.copy()
        
        # 2. Precise Joint Indexing (Arm 1-7)
        joint_names = [f"fr3_joint{i}" for i in range(1, 8)]
        self.q_idx = []
        self.q_lo = []
        self.q_hi = []
        for name in joint_names:
            jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                available_joints = [self.mj_model.joint(i).name for i in range(self.mj_model.njnt)]
                raise ValueError(f"Joint '{name}' not found! Available joints: {available_joints}")
            self.q_idx.append(self.mj_model.jnt_qposadr[jid])
            self.q_lo.append(self.mj_model.jnt_range[jid][0])
            self.q_hi.append(self.mj_model.jnt_range[jid][1])
        self.q_idx = np.array(self.q_idx)
        self.q_lo = np.array(self.q_lo)
        self.q_hi = np.array(self.q_hi)

        # 3. Dynamic Finger Identification
        self.finger_q_idx = []
        self.finger_open_val = []
        for name in ["finger_joint1", "finger_joint2"]:
            jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid != -1 and self.mj_model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE:
                self.finger_q_idx.append(self.mj_model.jnt_qposadr[jid])
                self.finger_open_val.append(self.mj_model.jnt_range[jid][1])
        print(f"DEBUG: Precisely identified {len(self.finger_q_idx)} gripper sliders.")

        # 4. Tree-based Robot Body Identification
        base_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if base_id == -1: base_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "fr3_link0")
        
        self.robot_body_ids = set()
        def collect_children(bid):
            self.robot_body_ids.add(bid)
            for i in range(self.mj_model.nbody):
                if self.mj_model.body_parentid[i] == bid and i != bid:
                    collect_children(i)
        if base_id != -1: collect_children(base_id)
        
        self.is_robot_body = np.zeros(self.mj_model.nbody, dtype=bool)
        if self.robot_body_ids: self.is_robot_body[list(self.robot_body_ids)] = True
            
        from correlation import _init_gt_worker
        indices = [i for i, b in enumerate(self.is_robot_body) if b]
        _init_gt_worker(self.scene_xml, indices)

    def _set_state(self, q):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:] = self.qpos0
        self.mj_data.qpos[self.q_idx] = q
        for adr, val in zip(self.finger_q_idx, self.finger_open_val):
            self.mj_data.qpos[adr] = val
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def get_collision_status(self, q):
        self._set_state(q)
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            b1, b2 = self.mj_model.geom_bodyid[c.geom1], self.mj_model.geom_bodyid[c.geom2]
            if self.is_robot_body[b1] ^ self.is_robot_body[b2]: return True
        return False

    def get_detailed_info(self, q):
        self._set_state(q)
        is_coll = False
        pairs = []
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            b1, b2 = self.mj_model.geom_bodyid[c.geom1], self.mj_model.geom_bodyid[c.geom2]
            if self.is_robot_body[b1] ^ self.is_robot_body[b2]:
                is_coll = True
                pairs.append(f"{self.mj_model.body(b1).name}<->{self.mj_model.body(b2).name}")
        from correlation import _worker_dist_approx
        _, dist, _, _ = _worker_dist_approx(q)
        return is_coll, dist, list(set(pairs))

    def find_collision_q(self):
        while True:
            q = self.q_lo + np.random.rand(7) * (self.q_hi - self.q_lo)
            if self.get_collision_status(q): return q

    def optim_escape(self, start_cfg, lr=0.05, safety_margin=0.02, max_steps=100):
        t_start = time.time()
        q_lo_t = torch.from_numpy(self.q_lo).float().to(self.device)
        q_hi_t = torch.from_numpy(self.q_hi).float().to(self.device)
        q = torch.tensor(start_cfg, dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([q], lr=lr)
        
        cfg_history = [start_cfg.copy()]
        ndf_history = []
        success = False
        joint_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.2, 1.5, 1.5], device=self.device)
        
        for step in range(max_steps):
            optimizer.zero_grad()
            ndf_val = self.model(q.unsqueeze(0)).squeeze()
            ndf_history.append(ndf_val.item())
            
            if ndf_val.item() >= safety_margin:
                success = True
                cfg_history.append(q.detach().cpu().numpy().copy())
                break
            
            loss = torch.nn.functional.relu(safety_margin - ndf_val)
            loss.backward()
            
            with torch.no_grad():
                q.grad.mul_(joint_weights)
            optimizer.step()
            
            with torch.no_grad():
                if step >= 10:
                    recent_avg = np.mean(ndf_history[-5:])
                    prev_avg = np.mean(ndf_history[-10:-5]) if len(ndf_history) >= 10 else ndf_history[0]
                    # More sensitive: 1e-4 instead of 5e-5
                    if abs(recent_avg - prev_avg) < 1e-4:
                        q.add_(torch.randn_like(q) * 0.01) # Bigger kick
                q.clamp_(q_lo_t, q_hi_t)
            
            cfg_history.append(q.detach().cpu().numpy().copy())
            
        t_escape = time.time() - t_start
        return np.array(cfg_history), success, step + 1, t_escape

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_analysis(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    esc_tool = Escape(args.model, args.scene, device)
    print(f"Sampling {args.samples} collision points...")
    seeds = [esc_tool.find_collision_q() for _ in tqdm(range(args.samples))]
    
    ndf_success_count = 0
    gt_actually_safe_count = 0
    strict_success_count = 0
    total_steps = 0
    total_time = 0
    
    print(f"\nRunning Adam Escape (LR: {args.lr}, Target: {args.safety_margin})...")
    for i in tqdm(range(args.samples), desc="Escaping"):
        q0 = seeds[i]
        history, ndf_success, steps, dt = esc_tool.optim_escape(q0, lr=args.lr, safety_margin=args.safety_margin, max_steps=args.max_steps)
        final_q = history[-1]
        
        gt_is_coll = esc_tool.get_collision_status(final_q)
        
        if ndf_success: ndf_success_count += 1
        if not gt_is_coll: gt_actually_safe_count += 1
        
        if ndf_success and not gt_is_coll:
            strict_success_count += 1
            total_steps += steps
            total_time += dt
            status = "SUCCESS"
        else: status = "FAILED "
        
        if i < 15:
            _, dist, _ = esc_tool.get_detailed_info(final_q)
            print(f"ID {i:2} | {status} | Steps: {steps:2} | Time: {dt*1000:.1f}ms | GT_Dist: {dist:.4f} | Final_NDF: {esc_tool.model(torch.from_numpy(final_q).float().to(device).unsqueeze(0)).item():.4f}")
            
    print(f"\n" + "="*60 + "\nADAM ESCAPE PERFORMANCE (27D Midpoint - Optimized)\n" + "="*60)
    print(f"Total Samples: {args.samples}")
    print(f"NDF Perceived SR: {ndf_success_count/args.samples*100:.1f}%")
    print(f"GT Physical SR:   {gt_actually_safe_count/args.samples*100:.1f}%")
    print(f"Strict Success:   {strict_success_count/args.samples*100:.1f}%")
    if strict_success_count > 0:
        print(f"Avg Steps:        {total_steps/strict_success_count:.1f}")
        print(f"Avg Time:         {total_time/args.samples*1000:.2f} ms")
        print(f"Avg Step Time:    {total_time/total_steps*1000:.2f} ms")
    print("="*60 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="output/table_obs20/model_final.pt")
    parser.add_argument("--scene", default="scenes/table_obs20.xml")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.07)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--safety_margin", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_analysis(args)