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
        
        # Initialize FK preprocessing (for SDP models that need it)
        self.cp_manager = ControlPointManager(device=device, n_points_per_link=2)
        
        abs_xml = os.path.abspath(scene_xml)
        self.mj_model = mujoco.MjModel.from_xml_path(abs_xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        
        # Identify robot bodies for collision checking
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
        mujoco.mj_forward(self.mj_model, self.mj_data)
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            b1, b2 = self.mj_model.geom_bodyid[c.geom1], self.mj_model.geom_bodyid[c.geom2]
            if self.is_robot_body[b1] or self.is_robot_body[b2]:
                return True
        return False

    def run_escape(self, q0, lr=0.01, max_steps=100, safety_margin=0.05):
        q_curr = torch.tensor(q0, dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([q_curr], lr=lr)
        
        trajectory = [q0.copy()]
        success = False
        
        for step in range(max_steps):
            # Compute FK features
            points = self.cp_manager.get_control_points(q_curr.unsqueeze(0))
            points_flat = points.reshape(1, -1)
            x_features = torch.cat([q_curr.unsqueeze(0), points_flat], dim=-1)
            
            ndf_val = self.model(x_features).squeeze()
            
            if ndf_val.item() >= safety_margin:
                success = True
                break
                
            optimizer.zero_grad()
            # Maximize NDF value -> Minimize -NDF
            loss = -ndf_val
            loss.backward()
            optimizer.step()
            
            # Clamp to joint limits
            with torch.no_grad():
                lo = torch.from_numpy(self.mj_model.jnt_range[:7, 0]).float().to(self.device)
                hi = torch.from_numpy(self.mj_model.jnt_range[:7, 1]).float().to(self.device)
                q_curr.data = torch.clamp(q_curr.data, lo, hi)
                
            trajectory.append(q_curr.detach().cpu().numpy().copy())
            
        q_final = q_curr.detach().cpu().numpy()
        gt_collision = self.check_collision(q_final)
        
        return {
            "success": success,
            "gt_success": not gt_collision,
            "steps": step,
            "trajectory": np.array(trajectory)
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="output/bookshelf/model_final.pt")
    parser.add_argument("--scene", default="scenes/bookshelf.xml")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for gradient escape")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum steps for gradient escape")
    parser.add_argument("--safety-margin", type=float, default=0.02, help="Safety margin for escape (model outputs ~0.01 in free space)")
    args = parser.parse_args()
    
    escape_tool = Escape(args.model, args.scene)
    
    collision_points = []
    lo, hi = escape_tool.mj_model.jnt_range[:7, 0], escape_tool.mj_model.jnt_range[:7, 1]
    pbar = tqdm(total=args.samples, desc="Finding collision pts", ncols=80)
    while len(collision_points) < args.samples:
        q = lo + np.random.rand(7) * (hi - lo)
        if escape_tool.check_collision(q):
            collision_points.append(q)
            pbar.update(1)
    pbar.close()
    
    n_actual = len(collision_points)
    
    results = []
    t0 = time.time()
    for q in tqdm(collision_points, desc="Escaping", ncols=80):
        res = escape_tool.run_escape(q, lr=args.lr, max_steps=args.max_steps, safety_margin=args.safety_margin)
        results.append(res)
    total_time = time.time() - t0
        
    success_flags = [r['success'] for r in results]
    gt_success_flags = [r['gt_success'] for r in results]
    steps_if_success = [r['steps'] for r in results if r['success']]
    
    print(f"NDF GRADIENT ESCAPE RESULTS (safety_margin={args.safety_margin})")
    print(f"NDF Success Rate: {np.mean(success_flags)*100:.1f}% ({sum(success_flags)}/{n_actual})")
    print(f"GT Success Rate:  {np.mean(gt_success_flags)*100:.1f}% ({sum(gt_success_flags)}/{n_actual})")
    print(f"Avg Steps (if success): {np.mean(steps_if_success):.1f}" if steps_if_success else "Avg Steps: N/A")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Avg Time/Sample: {total_time/n_actual*1000:.1f} ms")