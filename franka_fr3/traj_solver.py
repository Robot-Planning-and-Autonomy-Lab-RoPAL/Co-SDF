#!/usr/bin/env python3
import os
import time
import numpy as np
import torch
import mujoco
import nlopt
import casadi as ca
import l4casadi as l4c
from common.models import load_model

class NDFOptimizerFranka:
    def __init__(self, model_path: str, scene_xml: str, device: str = "cuda"):
        self.nq = 7
       
        self.model = load_model(model_path, device)
        self.model.eval()
        
        abs_xml = os.path.abspath(scene_xml)
        mj_model = mujoco.MjModel.from_xml_path(abs_xml)
        self.lower = mj_model.jnt_range[:self.nq, 0].copy()
        self.upper = mj_model.jnt_range[:self.nq, 1].copy()

    def resample_path(self, path: np.ndarray, num_waypoints: int) -> np.ndarray:
        N = len(path)
        if N == num_waypoints:
            return path
        old_indices = np.arange(N)
        new_indices = np.linspace(0, N - 1, num_waypoints)
        resampled = np.zeros((num_waypoints, self.nq))
        for j in range(self.nq):
            resampled[:, j] = np.interp(new_indices, old_indices, path[:, j])
        return resampled

class NLoptPlanner:
    def __init__(self, opt_base, n_waypoints=30, device="cpu"):
        if opt_base is not None:
            self.opt_base = opt_base
            self.dist_model = opt_base.model
        else:
            raise ValueError("opt_base must be provided")
        self.device = device
        self.default_n = n_waypoints
        self.dof = 7
        self.lower = self.opt_base.lower
        self.upper = self.opt_base.upper

    def solve(self, start, goal, solver='LD_SLSQP',
              initial_path=None, safety_margin=0.02, verbose=False,
              v_max=1.5, a_max=3.0, dt=0.1,
              n_waypoints=None, max_iter=50, w_clr=1.0):

        n = n_waypoints if n_waypoints is not None else self.default_n
        dof = self.dof
        n_vars = n * dof

        if initial_path is None:
            initial_path = np.linspace(start, goal, n)
        elif len(initial_path) != n:
            initial_path = self.opt_base.resample_path(initial_path, n)
        x_init = initial_path.astype(np.float64).flatten()

        solver_map = {"slsqp": "LD_SLSQP", "cobyla": "LN_COBYLA"}
        algo_name = solver_map.get(solver.lower(), solver)

        try:
            algo = getattr(nlopt, algo_name)
        except AttributeError:
            algo = nlopt.LD_SLSQP

        opt = nlopt.opt(algo, n_vars)

        if "LN_" in algo_name or "GN_" in algo_name:
             opt.set_initial_step(1e-2)

        opt.set_lower_bounds(np.tile(self.lower, n).astype(np.float64))
        opt.set_upper_bounds(np.tile(self.upper, n).astype(np.float64))

        best = {"x": x_init.copy(), "f": float("inf")}
        self.iter_count = 0
        cache = {"x": None, "f": None, "obj_grad": None, "dists": None, "q_jac_diag": None}

        def compute_everything(x_np):
            if cache["x"] is not None and np.array_equal(x_np, cache["x"]):
                return

            Q = torch.tensor(x_np.reshape(n, dof), dtype=torch.float32, 
                             device=self.device, requires_grad=True)
            
            diff1 = Q[1:] - Q[:-1]
            diff2 = Q[2:] + Q[:-2] - 2 * Q[1:-1]
            cost_smooth = (1.0 * torch.sum(diff1**2) + 20.0 * torch.sum(diff2**2)) / n

            dists = self.dist_model(Q).reshape(n)

            z = torch.clamp(-dists / safety_margin, max=10.0)
            cost_clr = w_clr * torch.sum(torch.exp(z))
            
            total_f = cost_smooth + cost_clr
            
            g_obj = torch.autograd.grad(total_f, Q, retain_graph=True)[0]
            obj_grad = g_obj.detach().cpu().numpy().flatten().copy()

            g_jac = torch.autograd.grad(dists.sum(), Q, retain_graph=False)[0]
            q_jac_diag = g_jac.detach().cpu().numpy().copy()

            cache["x"] = x_np.copy()
            cache["f"] = total_f.item()
            cache["obj_grad"] = obj_grad
            cache["dists"] = dists.detach().cpu().numpy().copy()
            cache["q_jac_diag"] = q_jac_diag

        compute_everything(x_init)
        init_dist_min = cache["dists"].min()
        c_tol = 1e-2 if init_dist_min < safety_margin else 1e-4

        def objective(x, grad):
            compute_everything(x)
            if grad.size > 0:
                grad[:] = cache["obj_grad"]
            
            f_val = cache["f"]
            if f_val < best["f"]:
                best["f"] = f_val
                best["x"] = x.copy()
            
            if verbose and self.iter_count % 10 == 0:
                print(f"  [Iter {self.iter_count}] f={f_val:.4f}")
            self.iter_count += 1
            return f_val

        opt.set_min_objective(objective)

        def constraint_start(result, x, grad):
            result[:] = x[:dof] - start
            if grad.size > 0:
                grad[:] = 0.0
                for i in range(dof):
                    grad[i, i] = 1.0

        def constraint_goal(result, x, grad):
            result[:] = x[-dof:] - goal
            if grad.size > 0:
                grad[:] = 0.0
                for i in range(dof):
                    grad[i, (n - 1) * dof + i] = 1.0

        opt.add_equality_mconstraint(constraint_start, np.full(dof, 1e-8))
        opt.add_equality_mconstraint(constraint_goal, np.full(dof, 1e-8))

        def constraint_collision(result, x, grad):
            compute_everything(x)
            result[:] = (safety_margin - cache["dists"]).astype(np.float64)

            if grad.size > 0:
                g = -cache["q_jac_diag"]
                jac = np.zeros((n, n_vars))
                for i in range(n):
                    jac[i, i*dof : (i+1)*dof] = g[i]
                grad[:] = jac

        opt.add_inequality_mconstraint(constraint_collision, np.full(n, c_tol))
        v_bound = v_max * dt
        n_vel = 2 * (n - 1) * dof

        J_vel = np.zeros((n_vel, n_vars))
        for i in range(n - 1):
            for j in range(dof):
                row_u = i * dof + j
                row_l = (n - 1) * dof + i * dof + j
                J_vel[row_u, (i + 1) * dof + j] = 1.0
                J_vel[row_u, i * dof + j] = -1.0
                J_vel[row_l, (i + 1) * dof + j] = -1.0
                J_vel[row_l, i * dof + j] = 1.0

        def constraint_velocity(result, x, grad):
            Q = x.reshape(n, dof)
            diff1 = Q[1:] - Q[:-1]
            result[: (n - 1) * dof] = (diff1 - v_bound).flatten()
            result[(n - 1) * dof:] = (-diff1 - v_bound).flatten()
            if grad.size > 0:
                grad[:] = J_vel

        opt.add_inequality_mconstraint(constraint_velocity, np.full(n_vel, 1e-4))

        a_bound = a_max * dt ** 2
        n_acc = 2 * (n - 2) * dof

        J_acc = np.zeros((n_acc, n_vars))
        for i in range(n - 2):
            for j in range(dof):
                row_u = i * dof + j
                row_l = (n - 2) * dof + i * dof + j
                J_acc[row_u, (i + 2) * dof + j] = 1.0
                J_acc[row_u, i * dof + j] = 1.0
                J_acc[row_u, (i + 1) * dof + j] = -2.0
                J_acc[row_l, (i + 2) * dof + j] = -1.0
                J_acc[row_l, i * dof + j] = -1.0
                J_acc[row_l, (i + 1) * dof + j] = 2.0

        def constraint_acceleration(result, x, grad):
            Q = x.reshape(n, dof)
            diff2 = Q[2:] + Q[:-2] - 2 * Q[1:-1]
            result[: (n - 2) * dof] = (diff2 - a_bound).flatten()
            result[(n - 2) * dof:] = (-diff2 - a_bound).flatten()
            if grad.size > 0:
                grad[:] = J_acc

        opt.add_inequality_mconstraint(constraint_acceleration, np.full(n_acc, 1e-4))

        opt.set_ftol_rel(1e-1)
        opt.set_maxeval(max_iter)

        t0 = time.time()
        stats = {"framework": f"nlopt_{algo_name}", "return_status": None,
                 "exception": False, "f": best["f"]}

        try:
            x_opt = opt.optimize(x_init)
            status = opt.last_optimize_result()
            stats["return_status"] = status
            success = (status > 0)
            stats["f"] = opt.last_optimum_value()
        except nlopt.RoundoffLimited:
            x_opt = best["x"]
            stats["return_status"] = "RoundoffLimited"
            stats["f"] = best["f"]
            success = True
        except Exception as e:
            if verbose:
                print(f"NLopt crash: {type(e).__name__}: {str(e)}")
            stats["return_status"] = str(e)
            stats["exception"] = True
            return False, initial_path.reshape(n, dof), time.time() - t0, stats

        result_path = x_opt.reshape(n, dof)
        del opt
        return success, result_path, time.time() - t0, stats

class CasADiPlanner:
    def __init__(self, model_path, scene_xml, n_waypoints=30, device="cuda"):
        self.device = device
        self.default_n = n_waypoints
        self.opt_base = NDFOptimizerFranka(model_path, scene_xml, device=device)
        self.model = self.opt_base.model
        self.dof = 7
        self.lower = self.opt_base.lower
        self.upper = self.opt_base.upper
        
        self.limits = np.stack([self.lower, self.upper], axis=1)
        
        self.l4c_models = {}

    def get_l4c_model(self, n):
        if n not in self.l4c_models:
            model_name = f'casadi_cosdf_{n}_{self.device}'
            model = l4c.L4CasADi(
                self.model, 
                device=self.device, 
                batched=True, 
                name=model_name,
                generate_jac=True,
                generate_jac_jac=False
            )
            
            dummy_q = ca.MX.sym('q', n, self.dof)
            try:
                _ = model(dummy_q)
            except Exception:
                pass
                
            self.l4c_models[n] = model
            
        return self.l4c_models[n]

    def _resample_path(self, path, num_waypoints):
        N = len(path)
        if N == num_waypoints:
            return path
        old_idx = np.arange(N)
        new_idx = np.linspace(0, N - 1, num_waypoints)
        resampled = np.zeros((num_waypoints, self.dof))
        for j in range(self.dof):
            resampled[:, j] = np.interp(new_idx, old_idx, path[:, j])
        return resampled

    def solve(self, start, goal, initial_path=None, solver='ipopt',
              safety_margin=0.02, verbose=False, v_max=1.5, a_max=3.0, dt=0.1, 
              n_waypoints=None, max_iter=50, w_clr=1.0):
        n = n_waypoints if n_waypoints is not None else self.default_n
        l4c_model = self.get_l4c_model(n)
        
        opti = ca.Opti()
        Q = opti.variable(n, self.dof)
        
        diff1 = Q[1:, :] - Q[:-1, :]
        diff2 = Q[2:, :] + Q[:-2, :] - 2*Q[1:-1, :]
        cost_smooth = (1.0 * ca.sumsqr(diff1) + 20.0 * ca.sumsqr(diff2)) / n
        
        dists = l4c_model(Q)
        opti.subject_to(dists >= safety_margin)
            
        z = ca.fmin(50, -dists / safety_margin)
        cost_clearance = w_clr * ca.sum1(ca.exp(z))

        opti.minimize(cost_smooth + cost_clearance)
        
        opti.subject_to(Q[0, :].T == ca.DM(start))
        opti.subject_to(Q[-1, :].T == ca.DM(goal))
        
        for j in range(self.dof):
            opti.subject_to(opti.bounded(self.limits[j, 0], Q[:, j], self.limits[j, 1]))
            
        if dt > 0:
            opti.subject_to(opti.bounded(-v_max * dt, diff1, v_max * dt))
            opti.subject_to(opti.bounded(-a_max * dt**2, diff2, a_max * dt**2))
        
        if initial_path is None:
            initial_path = np.linspace(start, goal, n)
        else:
            initial_path = self._resample_path(initial_path, n)
            
        opti.set_initial(Q, initial_path)
        
        p_opts = {"expand": True, "print_time": False}
        s_opts = {
            "print_level": 0,
            "max_iter": max_iter,
            "sb": "yes",
            "tol": 1e-4,
            "acceptable_tol": 1e-1,
            "acceptable_iter": 3, 
            "mu_strategy": "adaptive",
            "hessian_approximation": "limited-memory",
        }
        
        opti.solver(solver, p_opts, s_opts)
        
        t0 = time.time()
        try:
            sol = opti.solve()
            q_opt = sol.value(Q)
            success = True
            status = sol.stats()['return_status']
        except Exception as e:
            stats = opti.debug.stats()
            status = stats.get('return_status', 'failed')
            
            if status == 'Solved_To_Acceptable_Level':
                if verbose:
                    print("Optimization reached acceptable tolerance (Early Stopping).")
                q_opt = opti.debug.value(Q)
                success = True
            else:
                if verbose:
                    print(f"CasADi Optimization Failed: {e} (Status: {status})")
                try:
                    q_opt = opti.debug.value(Q)
                except Exception:
                    q_opt = initial_path
                success = False
            
        t_solve = time.time() - t0
        return success, q_opt, t_solve, {"status": status}
