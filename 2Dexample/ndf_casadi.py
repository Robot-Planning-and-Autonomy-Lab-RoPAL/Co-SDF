#!/usr/bin/env python3
import casadi as ca
import numpy as np
import torch
import l4casadi as l4c
import time
import os
import contextlib
from trajectory_optim import NDFOptimizer2D

class CasADiPlanner:
    def __init__(self, model_path, scene_xml, n_waypoints=30, device="cuda"):
        # Reuse existing optimizer for model loading and evaluation
        self.opt_base = NDFOptimizer2D(model_path, scene_xml, device=device, n_layers=20, n_hidden=128)
        self.device = device
        self.default_n = n_waypoints
        self.dof = 2
        self.lower = self.opt_base.lower
        self.upper = self.opt_base.upper
        self.limits = np.stack([self.lower, self.upper], axis=1) # (7, 2)
        
        # Cache for l4c models with different waypoint counts to avoid recompilation
        self.l4c_models = {}
        
    def get_l4c_model(self, n):
        if n not in self.l4c_models:
            print(f"Creating/Compiling L4CasADi model for n_waypoints={n}...")
            # We use a unique name for each dimension to avoid internal cache collisions
            model = l4c.L4CasADi(
                self.opt_base.model, 
                device=self.device, 
                batched=True, 
                name=f'ndf_planner_{n}'
            )
            # Pre-compile
            dummy_q = ca.MX.sym('dummy_q', n, self.dof)
            _ = model(dummy_q)
            self.l4c_models[n] = model
        return self.l4c_models[n]

    def solve(self, start, goal, formulation='hard', solver='ipopt', initial_path=None, 
              safety_margin=0.02, verbose=False, v_max=1.5, a_max=3.0, dt=0.1, 
              n_waypoints=None, max_iter=50, w_coll=1e5, w_clr=1.0):
        """
        Solve the path optimization problem.
        
        [算法框架 Solver Frameworks]:
        - 'ipopt':   Interior Point Optimizer (Global NLP solver)
        - 'qpoases': Sequential Quadratic Programming (SQP) using qpOASES as sub-solver
        - 'osqp':    Sequential Quadratic Programming (SQP) using OSQP as sub-solver
        
        [约束处理方式 Formulation]:
        - 'hard':    Strict constraints (dists >= margin). No slacks.
        - 'relaxed': Hard constraints with slack variables (dists + slack >= margin). 
                     Large penalty on slack (Relaxed-Hard).
        - 'soft':    Soft constraints using hinge penalty (min w * max(0, margin - dists)^2).
        """
        n = n_waypoints if n_waypoints is not None else self.default_n
        l4c_model = self.get_l4c_model(n)
        
        opti = ca.Opti()
        Q = opti.variable(n, self.dof)
        
        # --- 1. 定义目标函数 (NLP Objective) ---
        diff1 = Q[1:, :] - Q[:-1, :]
        diff2 = Q[2:, :] + Q[:-2, :] - 2*Q[1:-1, :]
        cost_smooth = (1.0 * ca.sumsqr(diff1) + 20.0 * ca.sumsqr(diff2)) / n
        
        # --- 2. 约束处理 (Constraint Handling) ---
        dists = l4c_model(Q) # (n, 1)
        slack = None
        cost_coll = 0
        
        if formulation == 'hard':
            # Strict hard constraints: NO slacks
            opti.subject_to(dists >= safety_margin)
        elif formulation == 'relaxed':
            # Relaxed hard constraints: slacks + strong penalty
            slack = opti.variable(n, 1)
            opti.subject_to(slack >= 0)
            opti.subject_to(dists + slack >= safety_margin)
            cost_coll = 1e6 * ca.sumsqr(slack)
        elif formulation == 'soft':
            # Pure soft penalty (hinge)
            viol = ca.fmax(0, safety_margin - dists)
            cost_coll = w_coll * ca.sumsqr(viol)
        else:
            raise ValueError(f"Unknown formulation: {formulation}")
            
        # --- 3. Active Clearance Objective (Repulsion Force) ---
        z = ca.fmin(50, -dists / safety_margin)
        cost_clearance = w_clr * ca.sum1(ca.exp(z))

        opti.minimize(cost_smooth + cost_coll + cost_clearance)
        
        # --- 3. 基础约束 (Joint/Kinematic) ---
        opti.subject_to(Q[0, :].T == ca.DM(start))
        opti.subject_to(Q[-1, :].T == ca.DM(goal))
        for j in range(self.dof):
            opti.subject_to(opti.bounded(self.limits[j, 0], Q[:, j], self.limits[j, 1]))
        opti.subject_to(opti.bounded(-v_max * dt, diff1, v_max * dt))
        opti.subject_to(opti.bounded(-a_max * dt**2, diff2, a_max * dt**2))
        
        # --- 4. 算法框架配置 (Algorithm Framework Configuration) ---
        if solver == 'ipopt':
            solver_name = 'ipopt'
            opts = {
                "ipopt.print_level": 5 if verbose else 0,
                "ipopt.max_iter": max_iter,
                "ipopt.tol": 1e-4,     # Realistic IROS precision
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
                "tol_du": 1e-4,
                "print_time": False,
            }
        
        if solver == 'qpoases':
            opts["qpsol_options"] = {"printLevel": "none"}
        elif solver == 'osqp':
            opts["qpsol_options"] = {"warm_start_primal": True, "print_out": False, "verbose": False}

        with _suppress_fd_output(not verbose):
            opti.solver(solver_name, opts)
        
        # Initial Guess and Solve
        if initial_path is not None:
            if len(initial_path) != n:
                initial_path = self.opt_base.resample_path(initial_path, n)
            opti.set_initial(Q, initial_path)
            
            # IMPROVEMENT: Pre-initialize slack to satisfy constraints at the start
            if slack is not None:
                with torch.no_grad():
                    q_tensor = torch.from_numpy(initial_path).float().to(self.device)
                    d0 = self.opt_base.model(q_tensor).cpu().numpy().squeeze()
                
                if len(d0.shape) == 0: d0 = np.array([d0]) # handle single point
                s0 = np.maximum(0, safety_margin - d0).reshape(-1, 1)
                opti.set_initial(slack, s0)
            
        t0 = time.time()
        stats = {
            "framework": solver_name, 
            "qpsol": solver if solver_name == "sqpmethod" else None,
            "exception": False, 
            "return_status": None, 
            "iter_count": None, 
            "success": False
        }

        try:
            with _suppress_fd_output(not verbose):
                sol = opti.solve()
            Qsol = sol.value(Q)
            final_path = Qsol
            s = sol.stats()
            rs = str(s.get("return_status", ""))
            ok = bool(s.get("success", False)) or ("Solve_Succeeded" in rs) or ("Solved_To_Acceptable_Level" in rs)
            stats["success"] = ok
            stats["return_status"] = s.get("return_status", None)
            stats["iter_count"] = s.get("iter_count", s.get("iterations", s.get("iter", None)))
        except Exception as e:
            stats["exception"] = True
            stats["return_status"] = str(e)
            if verbose: 
                print(f"Optimization failed ({formulation}_{solver}): {e}")
            
            # CRITICAL: Return the best iterate found so far (even if not converged)
            # rather than reverting to the initial path.
            try:
                final_path = opti.debug.value(Q)
            except:
                final_path = initial_path if initial_path is not None else np.vstack([start, goal])
            
            stats["success"] = False
                
        return stats["success"], final_path, time.time() - t0, stats

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
