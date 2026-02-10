import torch
import numpy as np
import mujoco
import os
import time
from tqdm import tqdm
from common.models import load_model
from common.utils import get_device

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class GradientEscaper:
    def __init__(self, model_path, scene_xml, device):
        self.device = device
        self.model = load_model(model_path, device)
        self.model.eval()
        self.scene_xml = os.path.abspath(scene_xml)
        self.mj_model = mujoco.MjModel.from_xml_path(self.scene_xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        if self.mj_model.nkey > 0:
            self.qpos0 = self.mj_model.key_qpos[0].copy()
        else:
            self.qpos0 = self.mj_data.qpos.copy()
        
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

        self.finger_q_idx = []
        self.finger_open_val = []
        for name in ["finger_joint1", "finger_joint2"]:
            jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid != -1 and self.mj_model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE:
                self.finger_q_idx.append(self.mj_model.jnt_qposadr[jid])
                self.finger_open_val.append(self.mj_model.jnt_range[jid][1])

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

    def run(self, start_cfg, lr=0.05, safety_margin=0.02, max_steps=100):
        t_start = time.time()
        q_lo_t = torch.from_numpy(self.q_lo).float().to(self.device)
        q_hi_t = torch.from_numpy(self.q_hi).float().to(self.device)
        q = torch.tensor(start_cfg, dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([q], lr=lr)
        
        trajectory = [start_cfg.copy()]
        ndf_history = []
        ndf_success_step = None
        joint_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.2, 1.5, 1.5], device=self.device)
        
        for step in range(max_steps):
            optimizer.zero_grad()
            ndf_val = self.model(q.unsqueeze(0)).squeeze()
            ndf_history.append(ndf_val.item())
            
            if ndf_val.item() >= safety_margin:
                ndf_success_step = step
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
                    if abs(recent_avg - prev_avg) < 1e-4:
                        q.add_(torch.randn_like(q) * 0.01)
                q.clamp_(q_lo_t, q_hi_t)
            
            trajectory.append(q.detach().cpu().numpy().copy())
            
        t_escape = time.time() - t_start
        final_q_np = q.detach().cpu().numpy()
        is_safe_gt = not self.get_collision_status(final_q_np)
        
        return {
            "ndf_success": ndf_success_step is not None,
            "gt_success": is_safe_gt,
            "gt_steps": ndf_success_step if ndf_success_step is not None else step,
            "ndf_steps": ndf_success_step if ndf_success_step is not None else step,
            "final_q": final_q_np,
            "trajectory": np.array(trajectory),
            "ndf_history": ndf_history,
            "time": t_escape
        }

class SamplingEscaper:
    def __init__(self, model_path, scene_xml, device):
        self.device = device
        self.model = load_model(model_path, device)
        self.model.eval()
        
        self.mj_model = mujoco.MjModel.from_xml_path(os.path.abspath(scene_xml))
        self.mj_data = mujoco.MjData(self.mj_model)
        
        self.lower = self.mj_model.jnt_range[:7, 0]
        self.upper = self.mj_model.jnt_range[:7, 1]
        
        self.robot_body_ids = set()
        base_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if base_id >= 0:
            for i in range(self.mj_model.nbody):
                curr = i
                while curr > 0:
                    if curr == base_id:
                        self.robot_body_ids.add(i)
                        break
                    curr = self.mj_model.body_parentid[curr]
                    
        self.is_robot_body = np.zeros(self.mj_model.nbody, dtype=bool)
        if self.robot_body_ids:
            self.is_robot_body[list(self.robot_body_ids)] = True

    def check_collision(self, q):
        self.mj_data.qpos[:7] = q
        if self.mj_model.nq > 7:
            self.mj_data.qpos[7:self.mj_model.nq] = 0.04
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            b1, b2 = self.mj_model.geom_bodyid[c.geom1], self.mj_model.geom_bodyid[c.geom2]
            if self.is_robot_body[b1] ^ self.is_robot_body[b2]:
                return True
        return False

    def sample_boundary_collision(self):
        while True:
            q1 = self.lower + np.random.rand(7) * (self.upper - self.lower)
            c1 = self.check_collision(q1)
            
            q2 = np.clip(q1 + np.random.randn(7) * 0.15, self.lower, self.upper)
            c2 = self.check_collision(q2)
            
            if c1 != c2:
                q_coll, q_free = (q1, q2) if c1 else (q2, q1)
                for _ in range(5): 
                    mid = (q_coll + q_free) / 2
                    if self.check_collision(mid): 
                        q_coll = mid
                    else: 
                        q_free = mid
                return q_coll

    def run(self, q0, step_size=0.05, max_steps=100, K=50, safety_margin=0.02):
        t_start = time.time()
        q_curr = torch.tensor(q0, dtype=torch.float32, device=self.device)
        
        trajectory = [q0.copy()]
        ndf_success_step = None
        ndf_history = []
        
        for step in range(max_steps):
            with torch.no_grad():
                ndf_curr = self.model(q_curr.reshape(1, -1)).item()
                ndf_history.append(ndf_curr)
                
                if ndf_curr >= safety_margin:
                    ndf_success_step = step
                    break
                
                noise = torch.randn(K, 7, device=self.device) * step_size
                candidates = torch.clamp(q_curr + noise, 
                                        torch.from_numpy(self.lower).float().to(self.device),
                                        torch.from_numpy(self.upper).float().to(self.device))
                ndf_candidates = self.model(candidates).squeeze()
                best_idx = torch.argmax(ndf_candidates)
                q_curr = candidates[best_idx]
                trajectory.append(q_curr.cpu().numpy().copy())
        
        t_escape = time.time() - t_start
        final_q_np = q_curr.cpu().numpy()
        is_safe_gt = not self.check_collision(final_q_np)
        
        return {
            "ndf_success": ndf_success_step is not None,
            "gt_success": is_safe_gt,
            "gt_steps": ndf_success_step if ndf_success_step is not None else step,
            "ndf_steps": ndf_success_step if ndf_success_step is not None else step,
            "final_q": final_q_np,
            "trajectory": np.array(trajectory),
            "ndf_history": ndf_history,
            "time": t_escape
        }