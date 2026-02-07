#!/usr/bin/env python3
import casadi as ca
import numpy as np
import torch
import l4casadi as l4c
import time
import os
import contextlib
import mujoco
from sdf_grid import WorkspaceGridSDF

class WorkspaceSDFPlanner:
    def __init__(self, scene_xml, n_waypoints=30, grid_resolution=32, device="cuda"):
        self.mj_model = mujoco.MjModel.from_xml_path(os.path.abspath(scene_xml))
        self.n = n_waypoints
        self.lower = self.mj_model.jnt_range[:7, 0]
        self.upper = self.mj_model.jnt_range[:7, 1]
        self.device = device
        
        self.workspace_sdf = WorkspaceGridSDF(grid_resolution=grid_resolution, device=device)
        
        self.workspace_sdf.compute_grid(scene_xml, samples_per_dim=grid_resolution)
        
        self.l4c_sdf = l4c.L4CasADi(
            self.workspace_sdf,
            device=device,
            batched=True,
            name="workspace_grid_sdf"
        )
    
    def solve(self, start, goal, initial_path=None, solver='ipopt', 
              safety_margin=0.02, verbose=False, v_max=1.5, a_max=3.0, dt=0.1, 
              n_waypoints=None, max_iter=50, w_coll=1e5, w_clr=1.0):
        if initial_path is not None:
            n = initial_path.shape[0]
        else:
            n = n_waypoints if n_waypoints is not None else self.n
            initial_path = np.linspace(start, goal, n)
        
        opti = ca.Opti()
        Q = opti.variable(n, 7)
        opti.set_initial(Q, initial_path)
        
        # --- Debug: Check initial SDF ---
        if verbose:
            try:
                init_sdf = np.array(self.workspace_sdf(torch.tensor(initial_path, device=self.device, dtype=torch.float32)).detach().cpu())
                print(f"[SDF] Initial min dist: {init_sdf.min():.4f}, Margin: {safety_margin:.4f}")
            except:
                pass
        
        diff1 = Q[1:, :] - Q[:-1, :]
        diff2 = Q[2:, :] + Q[:-2, :] - 2*Q[1:-1, :]
        cost_smooth = (1.0*ca.sumsqr(diff1) + 20.0*ca.sumsqr(diff2))/n
        
        # --- 2. Active Clearance / Collision Constraints ---
        sdf_dists = self.l4c_sdf(Q)
        
        # Use Slack Variables for Robustness (Soft Constraints)
        slack = opti.variable(n)
        opti.subject_to(slack >= 0)
        opti.set_initial(slack, 0.0)
        
        # Violation cost (Hinge Loss)
        # Using quadratic penalty for smoother gradients compared to exp
        violation = ca.fmax(0.0, safety_margin - sdf_dists)
        cost_clearance = w_clr * ca.sumsqr(violation)
        
        # Penalize slack heavily to keep it close to zero if possible
        cost_slack = 1e6 * ca.sumsqr(slack)
            
        opti.minimize(cost_smooth + cost_clearance + cost_slack)
        
        opti.subject_to(Q[0, :].T == ca.DM(start))
        opti.subject_to(Q[-1, :].T == ca.DM(goal))
        
        for j in range(7):
            opti.subject_to(opti.bounded(self.lower[j], Q[:, j], self.upper[j]))
        
        # Soft constraint: dist + slack >= margin
        opti.subject_to(sdf_dists + slack >= safety_margin)
        
        # Velocity and Acceleration limits (aligned with NDF planner)
        opti.subject_to(opti.bounded(-v_max * dt, diff1, v_max * dt))
        opti.subject_to(opti.bounded(-a_max * dt**2, diff2, a_max * dt**2))
        
        if solver == 'ipopt':
            solver_name = 'ipopt'
            opts = {
                "ipopt.print_level": 5 if verbose else 0,
                "ipopt.max_iter": max_iter,
                "ipopt.tol": 1e-4,
                "ipopt.acceptable_tol": 1e-3,
                "ipopt.acceptable_iter": 5, 
                "ipopt.hessian_approximation": "limited-memory",
                "print_time": False,
            }
        else:
            solver_name = 'sqpmethod'
            opts = {
                "qpsol": solver,
                "max_iter": max_iter,
                "hessian_approximation": "limited-memory",
                "print_header": False,
                "print_iteration": False,
                "print_status": False,
                "qpsol_options": {
                    "verbose": False
                },
                "tol_pr": 1e-4,
                "tol_du": 1e-4,
                "print_time": False,
            }
        
        if solver == 'qpoases':
            opts["qpsol_options"] = {"printLevel": "none"}
        elif solver == 'osqp':
            opts["qpsol_options"] = {"warm_start_primal": True, "print_out": False, "verbose": False}

        opti.solver(solver_name, opts)
        
        t0 = time.time()
        try:
            with _suppress_fd_output(solver == 'osqp' and not verbose):
                sol = opti.solve()
            Qsol = sol.value(Q)
            s = sol.stats()
            rs = str(s.get("return_status", ""))
            ok = bool(s.get("success", False)) \
                or ("Solve_Succeeded" in rs) \
                or ("Solved_To_Acceptable_Level" in rs)
            meta = {
                "return_status": s.get("return_status", None),
                "iter_count": s.get("iter_count", s.get("iterations", s.get("iter", None))),
                "success_flag": s.get("success", None),
            }
            return ok, Qsol, time.time()-t0, meta
        except Exception as e:
            if verbose:
                print(f"[SDF] Optimization failed ({solver}): {e}")
            try:
                final_path = opti.debug.value(Q)
            except:
                final_path = initial_path if initial_path is not None else np.vstack([start, goal])
            return False, final_path, time.time()-t0, {"error": str(e)}


@contextlib.contextmanager
def _suppress_fd_output(enabled: bool):
    if not enabled:
        yield
        return

    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(old_stdout)
        os.close(old_stderr)
