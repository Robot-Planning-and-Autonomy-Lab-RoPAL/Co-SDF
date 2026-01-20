#!/usr/bin/env python3
import os
# Handle High-DPI displays and UI font scale
os.environ['GDK_SCALE'] = '1'
os.environ['GDK_DPI_SCALE'] = '1'
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
os.environ['MUJOCO_FONT_SCALE'] = '150' # Controls font size in left/right UI panels
import time
import threading
import numpy as np
import mujoco
import mujoco.viewer
import torch
# Handle 7D FR3 configuration space
from utils import load_model

class FR3Env:
    def __init__(self, model_path):
        # Use absolute path for model to avoid inclusion errors
        abs_model_path = os.path.abspath(model_path)
        self.model = mujoco.MjModel.from_xml_path(abs_model_path)
        self.data = mujoco.MjData(self.model)
        self.nq = 7 

        # Identify robot bodies using subtree traversal starting from 'base'
        self.robot_body_ids = set()
        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if base_id != -1:
            for i in range(self.model.nbody):
                curr = i
                while curr > 0:
                    if curr == base_id:
                        self.robot_body_ids.add(i)
                        break
                    curr = self.model.body_parentid[curr]
            
            # Logging for sanity check
            names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in self.robot_body_ids]
            print(f"[Workspace FR3Env] Identified {len(self.robot_body_ids)} robot bodies.")
            
            # Optimized lookup mask
            self.is_robot_body = np.zeros(self.model.nbody, dtype=bool)
            if self.robot_body_ids:
                self.is_robot_body[list(self.robot_body_ids)] = True
        else:
            print("[Warning] Robot root body 'base' not found. Subtree identification failed.")
            self.is_robot_body = np.zeros(self.model.nbody, dtype=bool)
        
    def set_joints(self, q):
        # q: (7,)
        for i in range(self.nq):
            self.data.qpos[i] = q[i]
        # Set gripper to open (0.04)
        if self.model.nq > 7:
            for i in range(7, self.model.nq):
                self.data.qpos[i] = 0.04
        mujoco.mj_forward(self.model, self.data)

class WorkspaceVerifier:
    def __init__(self, model_path="output/bookshelf/model_final.pt", mujoco_model="scenes/bookshelf.xml"):
        # 1. Load Data (Optional for dataset browsing)
        self.n_dim = 7
        collapsed_path = "inputs/fr3_Xtrain_in.npy"
        if os.path.exists(collapsed_path):
            self.X_in = np.load(collapsed_path)
            self.total_points = len(self.X_in)
            print(f"Loaded {self.total_points} configurations from dataset.")
        else:
            self.X_in = None
            self.total_points = 0
            print(f"Note: Dataset '{collapsed_path}' not found. Dataset browsing disabled.")
        
        # 2. Setup NDF Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading NDF model from {model_path} on {self.device}...")
        try:
            self.model = load_model(model_path, self.device, n_dim=self.n_dim, n_hidden=512, frequencies=[1, 2, 4, 8])
        except Exception as e:
            print(f"Warning: Could not load NDF model: {e}")
            self.model = None

        # 3. Setup Environment
        self.env = FR3Env(model_path=mujoco_model)
        self.env.model.opt.gravity[:] = 0 
        
        # 4. State Management
        self.lock = threading.Lock()
        self.current_idx = 0
        self.last_synced_idx = -1 
        self.last_q = np.zeros(self.n_dim)
        self.running = True
        self.manual_override = False
        self.manual_q = np.zeros(self.n_dim)
        self.physics_on = False 

    def get_collision_report(self, q):
        report = []
        # Check Joint Limits
        for i in range(self.n_dim):
            if self.env.model.jnt_limited[i]:
                lo, hi = self.env.model.jnt_range[i]
                jnt_name = mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if q[i] < lo - 1e-4:
                    report.append(f"LIMIT VIOLATION [{jnt_name}]: {q[i]:.3f} < MIN {lo:.3f}")
                elif q[i] > hi + 1e-4:
                    report.append(f"LIMIT VIOLATION [{jnt_name}]: {q[i]:.3f} > MAX {hi:.3f}")
        
        # Check Physical Contacts
        for i in range(self.env.data.ncon):
            c = self.env.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            
            b1, b2 = self.env.model.geom_bodyid[g1], self.env.model.geom_bodyid[g2]
            
            # Identify bodies using optimized mask
            is_b1_robot = self.env.is_robot_body[b1]
            is_b2_robot = self.env.is_robot_body[b2]
            
            # Ignore contacts between two environment objects
            if not (is_b1_robot or is_b2_robot):
                continue
            
            # Filter: Ignore parent-child body contacts (adjacent links)
            if is_b1_robot and is_b2_robot:
                if self.env.model.body_parentid[b1] == b2 or self.env.model.body_parentid[b2] == b1:
                    continue
            
            # Identify bodies for a more readable report
            n1 = self.env.model.body(b1).name or f"body_{b1}"
            n2 = self.env.model.body(b2).name or f"body_{b2}"
            report.append(f"PHYSICAL COLLISION: '{n1}' and '{n2}'")
            break
            
        oracle_status = "\n".join(report) if report else "No collision detected."
        
        # --- NDF Prediction ---
        ndf_info = ""
        if self.model is not None:
            with torch.no_grad():
                q_tnsr = torch.Tensor(q).to(self.device).reshape(1, -1)
                val = self.model(q_tnsr).item()
                status = "COLLISION" if val < 0 else "FREE"
                ndf_info = f"\nNDF Output: {val:+.4f} -> [{status}]"
                
        return f"Oracle: {oracle_status}{ndf_info}"

    def _terminal_loop(self):
        """Threaded terminal input handler."""
        while self.running:
            try:
                line = input().strip().lower()
                if not line: continue
                
                # Check for 7 angles
                pts = line.replace(',', ' ').split()
                if len(pts) == self.n_dim:
                    try:
                        q = np.array([float(p) for p in pts])
                        with self.lock:
                            self.manual_q = q
                            self.manual_override = True
                        continue
                    except ValueError: pass

                if line == 'q':
                    self.running = False
                elif line.isdigit():
                    idx = int(line)
                    if self.X_in is not None and 0 <= idx < self.total_points:
                        with self.lock:
                            self.current_idx = idx
                            self.manual_override = False
                            self.physics_on = False 
                    elif self.X_in is None:
                        print("Dataset browsing is disabled (no .npy file found).")
                    else:
                        print(f"Index out of range (0-{self.total_points-1})")
            except EOFError:
                break

    def _key_callback(self, key):
        """MuJoCo viewer key handler - Disabled following user request."""
        pass

    def run(self):
        print("\n" + "="*50)
        print("Configuration Space Point Verifier (Terminal-Only)")
        print("="*50)
        print("CONTROLS:")
        if self.X_in is not None:
            print("  - [Terminal] Enter Index Number (0 to {0})".format(self.total_points-1))
        print("  - [Terminal] Enter 7 Angles (e.g., '0 0 0 -1.5 0 1.5 -0.7')")
        print("  - [Terminal] Type 'q' to quit")
        print("="*50)

        with mujoco.viewer.launch_passive(self.env.model, self.env.data) as viewer:
            threading.Thread(target=self._terminal_loop, daemon=True).start()

            while self.running and viewer.is_running():
                # 1. Sync from Dataset or Manual Input
                needs_sync = False
                with self.lock:
                    if self.manual_override:
                        target_q = self.manual_q.copy()
                        self.manual_override = False
                        print(f"\n[Input] q={target_q}")
                        needs_sync = True
                    elif self.current_idx != self.last_synced_idx:
                        target_q = self.X_in[self.current_idx].copy()
                        self.last_synced_idx = self.current_idx
                        print(f"\n[Dataset Point {self.current_idx}] q={target_q}")
                        needs_sync = True

                # 2. Sync Pose to MuJoCo
                if needs_sync:
                    with self.lock:
                        self.env.set_joints(target_q)
                        print(self.get_collision_report(target_q))
                        self.last_q = target_q.copy()
                    viewer.sync()

                # 3. Live Reporting (for manual pose checking if needed)
                curr_q = self.env.data.qpos[:self.n_dim].copy()
                if np.linalg.norm(curr_q - self.last_q) > 1e-3:
                    report = self.get_collision_report(curr_q)
                    reason = report.splitlines()[0] if report.startswith("COLLISION") or report.startswith("Joint") else "FREE"
                    print(f"\r[Current Pose] q={np.round(curr_q, 3)} | {reason}".ljust(80), end="", flush=True)
                    self.last_q = curr_q

                viewer.sync()
                time.sleep(0.01)

if __name__ == "__main__":
    try:
        verifier = WorkspaceVerifier()
        verifier.run()
    except Exception as e:
        print(f"\nExiting: {e}")
