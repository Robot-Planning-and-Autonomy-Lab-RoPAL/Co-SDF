#!/usr/bin/env python3
import os
import time
import threading
import numpy as np
import mujoco
import mujoco.viewer
from jaco_arm.jaco import JacoEnv

class WorkspaceVerifier:
    def __init__(self):
        # 1. Load Data
        collapsed_path = "inputs/jaco_Xtrain_in.npy"
        if not os.path.exists(collapsed_path):
            raise FileNotFoundError(f"{collapsed_path} not found. Run sampling.py first.")
        
        self.X_in = np.load(collapsed_path)
        self.total_points = len(self.X_in)
        print(f"Loaded {self.total_points} collision points.")
        
        # 2. Setup Environment
        self.env = JacoEnv(width=800, height=800, model_path="jaco_3dof.xml")
        self.env.model.opt.gravity[:] = 0 # No gravity for easy dragging
        
        # 3. State Management
        self.current_idx = 0
        self.last_synced_idx = -1
        self.last_q = np.zeros(3)
        self.running = True
        self.manual_override = False

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
            b1, b2 = self.env.model.geom_bodyid[g1], self.env.model.geom_bodyid[g2]
            if self.env.model.body_parentid[b1] == b2 or self.env.model.body_parentid[b2] == b1: continue
            
            n1 = self.env.model.geom(g1).name or f"id_{g1}"
            n2 = self.env.model.geom(g2).name or f"id_{g2}"
            report.append(f"COLLISION: '{n1}' and '{n2}'")
            break
            
        return "\n".join(report) if report else "No collision detected."

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
                        self.manual_q = np.array([float(p) for p in pts])
                        self.manual_override = True
                        continue
                    except ValueError: pass

                if line == 'q':
                    self.running = False
                elif line.isdigit():
                    idx = int(line)
                    if 0 <= idx < self.total_points:
                        self.current_idx = idx
                        self.manual_override = False
                    else:
                        print(f"Index out of range (0-{self.total_points-1})")
            except EOFError:
                break

    def _key_callback(self, key):
        """MuJoCo viewer key handler."""
        if key in [78, 110]: # N
            self.current_idx = (self.current_idx + 1) % self.total_points
            self.manual_override = False
        elif key in [82, 114]: # R
            self.current_idx = np.random.randint(self.total_points)
            self.manual_override = False
        elif key in [76, 108]: # L (Reload)
            self.last_synced_idx = -1
            self.manual_override = False
        elif key in [81, 113]: # Q
            self.running = False

    def run(self):
        print("\n" + "="*50)
        print("Physical Workspace Verification Tool (Refactored)")
        print("="*50)
        print("CONTROLS:")
        print("  - [Window] 'N': Next | 'R': Random | 'L': Reload | 'Q': Quit")
        print("  - [Window] Mouse Drag: Test any pose")
        print("  - [Terminal] Enter Index Number (e.g., '105')")
        print("  - [Terminal] Enter 3 Angles (e.g., '1.2 0.5 -0.3')")
        print("="*50)

        with mujoco.viewer.launch_passive(self.env.model, self.env.data) as viewer:
            viewer.on_key = self._key_callback
            
            threading.Thread(target=self._terminal_loop, daemon=True).start()

            while self.running and viewer.is_running():
                # 1. Sync from Dataset or Manual Input
                needs_sync = False
                if self.manual_override:
                    target_q = self.manual_q
                    self.manual_override = False
                    print(f"\n[Manual Input] q={target_q}")
                    needs_sync = True
                elif self.current_idx != self.last_synced_idx:
                    target_q = self.X_in[self.current_idx]
                    self.last_synced_idx = self.current_idx
                    print(f"\n[Point {self.current_idx}/{self.total_points}] q={target_q}")
                    needs_sync = True

                if needs_sync:
                    self.env.data.qpos[:3] = target_q
                    self.env.data.ctrl[:3] = target_q  # Set the 'spring' target to the collision point
                    self.env.data.qvel[:3] = 0
                    mujoco.mj_forward(self.env.model, self.env.data)
                    print(self.get_collision_report(target_q))
                    self.last_q = target_q.copy()
                    viewer.sync()

                # 2. Physics Step (allows dragging)
                mujoco.mj_step(self.env.model, self.env.data)

                # Keep target synced during dragging to prevent spring-back to zero,
                # but only if NOT in collision to allow it to "jam".
                has_collision = False
                for i in range(self.env.data.ncon):
                    c = self.env.data.contact[i]
                    g1, g2 = c.geom1, c.geom2
                    if g1 in {0,1,2,12} and g2 in {0,1,2,12}: continue
                    b1, b2 = self.env.model.geom_bodyid[g1], self.env.model.geom_bodyid[g2]
                    if self.env.model.body_parentid[b1] == b2 or self.env.model.body_parentid[b2] == b1: continue
                    has_collision = True
                    break
                
                if not has_collision:
                    self.env.data.ctrl[:3] = self.env.data.qpos[:3]

                # 3. Live Reporting (for dragging)
                curr_q = self.env.data.qpos[:3].copy()
                if np.linalg.norm(curr_q - self.last_q) > 5e-3:
                    report = self.get_collision_report(curr_q)
                    print(f"\r[Live Pose] q={np.round(curr_q, 3)} | {report.splitlines()[0]}".ljust(100), end="", flush=True)
                    self.last_q = curr_q

                viewer.sync()
                time.sleep(self.env.model.opt.timestep or 0.005)

if __name__ == "__main__":
    try:
        verifier = WorkspaceVerifier()
        verifier.run()
    except KeyboardInterrupt:
        pass
