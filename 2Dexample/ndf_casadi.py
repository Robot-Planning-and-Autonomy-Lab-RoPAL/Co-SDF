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
        self.opt_base = NDFOptimizer2D(model_path, scene_xml, device=device, n_layers=20, n_hidden=128)
        self.device = device
        self.default_n = n_waypoints
        self.dof = 2
        self.lower = self.opt_base.lower
        self.upper = self.opt_base.upper
        self.limits = np.stack([self.lower, self.upper], axis=1)
        
        self.l4c_models = {}
        
    def get_l4c_model(self, n):
        if n not in self.l4c_models:
            model = l4c.L4CasADi(
                self.opt_base.model, 
                device=self.device, 
                batched=True, 
                name=f'ndf_planner_{n}'
            )
            dummy_q = ca.MX.sym('dummy_q', n, self.dof)
            _ = model(dummy_q)
            self.l4c_models[n] = model
        return self.l4c_models[n]

    def solve(self, start, goal, formulation='hard', solver='ipopt', initial_path=None, 
              safety_margin=0.02, verbose=False, v_max=1.5, a_max=3.0, dt=0.1, 
              n_waypoints=None, max_iter=50, w_coll=1e5, w_clr=0.1):
        n = n_waypoints if n_waypoints is not None else self.default_n
        l4c_model = self.get_l4c_model(n)
        
        opti = ca.Opti()
        Q = opti.variable(n, self.dof)
        
        diff1 = Q[1:, :] - Q[:-1, :]
        diff2 = Q[2:, :] + Q[:-2, :] - 2*Q[1:-1, :]
        cost_smooth = (1.0 * ca.sumsqr(diff1) + 20.0 * ca.sumsqr(diff2)) / n
        
        dists = l4c_model(Q)
        slack = None
        cost_coll = 0
        
        if formulation == 'hard':
            opti.subject_to(dists >= safety_margin)
        elif formulation == 'relaxed':
            slack = opti.variable(n, 1)
            opti.subject_to(slack >= 0)
            opti.subject_to(dists + slack >= safety_margin)
            cost_coll = 1e6 * ca.sumsqr(slack)
        elif formulation == 'soft':
            viol = ca.fmax(0, safety_margin - dists)
            cost_coll = w_coll * ca.sumsqr(viol)
        else:
            raise ValueError(f"Unknown formulation: {formulation}")
            
        z = ca.fmin(50, -dists / safety_margin)
        cost_clearance = w_clr * ca.sum1(ca.exp(z))

        opti.minimize(cost_smooth + cost_coll + cost_clearance)
        
        opti.subject_to(Q[0, :].T == ca.DM(start))
        opti.subject_to(Q[-1, :].T == ca.DM(goal))
        for j in range(self.dof):
            opti.subject_to(opti.bounded(self.limits[j, 0], Q[:, j], self.limits[j, 1]))
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
                "tol_du": 1e-4,
                "print_time": False,
            }
        
        if solver == 'qpoases':
            opts["qpsol_options"] = {"printLevel": "none"}
        elif solver == 'osqp':
            opts["qpsol_options"] = {"warm_start_primal": True, "print_out": False, "verbose": False}

        with _suppress_fd_output(not verbose):
            opti.solver(solver_name, opts)
        
        if initial_path is not None:
            if len(initial_path) != n:
                initial_path = self.opt_base.resample_path(initial_path, n)
            opti.set_initial(Q, initial_path)
            
            if slack is not None:
                with torch.no_grad():
                    q_tensor = torch.from_numpy(initial_path).float().to(self.device)
                    d0 = self.opt_base.model(q_tensor).cpu().numpy().squeeze()
                
                if len(d0.shape) == 0: d0 = np.array([d0])
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
