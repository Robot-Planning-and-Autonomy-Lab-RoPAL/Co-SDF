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
from jaco_arm.jaco import JacoEnv
from utils import load_model

class WorkspaceVerifier:
    def __init__(self, model_path="output/ndf/model_final.pt"):
        # 1. Load Data
        collapsed_path = "inputs/jaco_Xtrain_in.npy"
        if not os.path.exists(collapsed_path):
            raise FileNotFoundError(f"{collapsed_path} not found. Run sampling.py first.")
        
        self.X_in = np.load(collapsed_path)
        self.total_points = len(self.X_in)
        print(f"Loaded {self.total_points} collision points.")
        
        # 2. Setup NDF Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading NDF model from {model_path} on {self.device}...")
        try:
            self.model = load_model(model_path, self.device, frequencies=[1, 2])
        except Exception as e:
            print(f"Warning: Could not load NDF model: {e}")
            self.model = None

        # 3. Setup Environment
        self.env = JacoEnv(width=800, height=800, model_path="jaco_3dof.xml")
        self.env.model.opt.gravity[:] = 0 # No gravity for easy dragging
        
        # 3. State Management
        self.lock = threading.Lock()
        self.current_idx = 0
        self.last_synced_idx = -1 
        self.last_q = np.zeros(3)
        self.running = True
        self.manual_override = False
        self.manual_q = np.zeros(3)
        self.physics_on = False # Start in static mode to see overlaps

    def get_collision_report(self, q):
        report = []
        # Check Joint Limits
        for i in range(3):
            lo, hi = self.env.model.jnt_range[i]
            if q[i] < lo - 1e-4 or q[i] > hi + 1e-4:
                report.append(f"Joint {i+1} LIMIT: {q[i]:.3f} rad (Range: [{lo:.3f}, {hi:.3f}])")
        
        # Check Physical Contacts
        for i in range(self.env.data.ncon):
            c = self.env.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in {0,1,2,12} and g2 in {0,1,2,12}: continue
            # Identify bodies for a more readable report
            b1, b2 = self.env.model.geom_bodyid[g1], self.env.model.geom_bodyid[g2]
            n1 = self.env.model.body(b1).name or f"body_{b1}"
            n2 = self.env.model.body(b2).name or f"body_{b2}"
            report.append(f"COLLISION: '{n1}' and '{n2}'")
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
                
                # Check for 3 angles
                pts = line.replace(',', ' ').split()
                if len(pts) == 3:
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
                    if 0 <= idx < self.total_points:
                        with self.lock:
                            self.current_idx = idx
                            self.manual_override = False
                            self.physics_on = False # Auto-switch to static to see overlap
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
        print("  - [Terminal] Enter Index Number (0 to {0})".format(self.total_points-1))
        print("  - [Terminal] Enter 3 Angles (e.g., '1.2 0.5 -0.3')")
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

                if needs_sync:
                    with self.lock:
                        self.env.data.qpos[:3] = target_q
                        self.env.data.ctrl[:3] = target_q
                        self.env.data.qvel[:3] = 0
                        mujoco.mj_forward(self.env.model, self.env.data)
                        print(self.get_collision_report(target_q))
                        self.last_q = target_q.copy()
                    viewer.sync()

                # 3. Live Reporting (for manual pose checking if needed)
                curr_q = self.env.data.qpos[:3].copy()
                if np.linalg.norm(curr_q - self.last_q) > 1e-3:
                    report = self.get_collision_report(curr_q)
                    reason = report.splitlines()[0] if report.startswith("COLLISION") or report.startswith("Joint") else "FREE"
                    print(f"\r[Current Pose] q={np.round(curr_q, 3)} | {reason}".ljust(60), end="", flush=True)
                    self.last_q = curr_q

                viewer.sync()
                time.sleep(0.01)

if __name__ == "__main__":
    try:
        verifier = WorkspaceVerifier()
        verifier.run()
    except Exception as e:
        print(f"\nExiting: {e}")
