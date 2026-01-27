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
        
        abs_xml = os.path.abspath(scene_xml)
        self.mj_model = mujoco.MjModel.from_xml_path(abs_xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        
        self.robot_body_ids = set()
        base_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")
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

    def find_near_boundary_collision(self):

        lo, hi = self.mj_model.jnt_range[:7, 0], self.mj_model.jnt_range[:7, 1]
        while True:
            q1 = lo + np.random.rand(7) * (hi - lo)
            c1 = self.check_collision(q1)
            q2 = np.clip(q1 + np.random.randn(7) * 0.15, lo, hi)
            c2 = self.check_collision(q2)
            
            if c1 != c2:
                q_coll, q_free = (q1, q2) if c1 else (q2, q1)
                for _ in range(5):
                    mid = (q_coll + q_free) / 2
                    if self.check_collision(mid): q_coll = mid
                    else: q_free = mid
                return q_coll

    def run_escape(self, q0, lr=0.1, max_steps=100, safety_margin=0.02):
        q_curr = torch.tensor(q0, dtype=torch.float32, device=self.device, requires_grad=True)
        lo_t = torch.from_numpy(self.mj_model.jnt_range[:7, 0]).float().to(self.device)
        hi_t = torch.from_numpy(self.mj_model.jnt_range[:7, 1]).float().to(self.device)
        
        trajectory = [q0.copy()]
        gt_success_step = None
        ndf_success_step = None
        
        for step in range(max_steps):
            ndf_val = self.model(q_curr.unsqueeze(0)).squeeze()
            
            if gt_success_step is None and not self.check_collision(q_curr.detach().cpu().numpy()):
                gt_success_step = step
            
            if ndf_success_step is None and ndf_val.item() >= safety_margin:
                ndf_success_step = step
            
            if gt_success_step is not None and ndf_success_step is not None:
                break

            loss = torch.nn.functional.relu(safety_margin - ndf_val)
            
            if q_curr.grad is not None: q_curr.grad.zero_()
            loss.backward()
            
            grad = q_curr.grad.data.clone()
            grad_norm = grad.norm().item()
            
            if grad_norm < 1e-9:
                break
            
            q_orig = q_curr.detach().clone()
            v_orig = loss.item()
            alpha = lr
            step_taken = False
            
            for _ in range(6):
                q_next = torch.clamp(q_orig - alpha * grad, lo_t, hi_t)
                
                with torch.no_grad():
                    v_next = torch.nn.functional.relu(safety_margin - self.model(q_next.unsqueeze(0)).squeeze()).item()
                    if v_next < v_orig:
                        q_curr.data = q_next
                        step_taken = True
                        break
                alpha *= 0.5
            
            if not step_taken:
                break
                
            trajectory.append(q_curr.detach().cpu().numpy().copy())
            
        return {
            "ndf_success": ndf_success_step is not None,
            "gt_success": gt_success_step is not None,
            "gt_steps": gt_success_step if gt_success_step is not None else step,
            "ndf_steps": ndf_success_step if ndf_success_step is not None else step,
            "trajectory": np.array(trajectory),
            "final_grad_norm": grad_norm
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="output/bookshelf/model_final.pt", help="Path to NDF model")
    parser.add_argument("--scene", default="scenes/bookshelf.xml", help="Path to scene")
    parser.add_argument("--samples", type=int, default=200, help="Number of samples")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=100, help="Max escape steps")
    parser.add_argument("--safety-margin", type=float, default=0.02, help="NDF safety margin")
    args = parser.parse_args()
    
    escape_tool = Escape(args.model, args.scene)
    
    print(f"Sampling {args.samples} near-boundary collision points...")
    collision_points = []
    for _ in tqdm(range(args.samples), desc="Boundary Sampling", ncols=80):
        collision_points.append(escape_tool.find_near_boundary_collision())
    
    results = []
    t0 = time.time()
    for q in tqdm(collision_points, desc="Escaping", ncols=80):
        res = escape_tool.run_escape(q, lr=args.lr, max_steps=args.max_steps, safety_margin=args.safety_margin)
        results.append(res)
    total_time = time.time() - t0
        
    ndf_succ = [r['ndf_success'] for r in results]
    gt_succ = [r['gt_success'] for r in results]
    gt_steps = [r['gt_steps'] for r in results if r['gt_success']]
    ndf_steps = [r['ndf_steps'] for r in results if r['ndf_success']]
    
    print(f"\n{'='*65}")
    print(f"NDF GRADIENT ESCAPE (7D Franka - Intensive Refined)")
    print(f"{'='*65}")
    print(f"Total Samples:    {len(results)}")
    print(f"Goal Condition:   not check_collision(q)")
    print(f"Hinge Loss:       relu(m - f)")
    print(f"Grad Norm Cutoff: 1e-8")
    print(f"-----------------------------------------------------------------")
    print(f"GT Success Rate:  {np.mean(gt_succ)*100:.1f}% (NO COLLISION)")
    print(f"NDF Success Rate: {np.mean(ndf_succ)*100:.1f}% (NDF >= {args.safety_margin})")
    print(f"Avg Steps to GT:  {np.mean(gt_steps):.1f}" if gt_steps else "Avg Steps to GT: N/A")
    print(f"Avg Steps to NDF: {np.mean(ndf_steps):.1f}" if ndf_steps else "Avg Steps to NDF: N/A")
    print(f"Avg Time/Sample:  {total_time/len(results)*1000:.1f} ms")
    print(f"{'='*65}\n")