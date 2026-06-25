#!/usr/bin/env python3
import os
import time
import numpy as np
import torch
import mujoco
import nlopt
import casadi as ca
import l4casadi as l4c
from typing import Dict
from common.models import load_model
from utils import ControlPoint
from robot_sdf import RobotSDF
from mujoco_env_sampler import MujocoEnvSampler
from trajectory_verify import TrajectoryEvaluator

class NDFOptimizerFranka:
    def __init__(self, model_path: str, scene_xml: str, device: str = "cuda"):
        self.device = device
        self.nq = 7
       
        self.model = load_model(model_path, device)
        self.model.eval()
        
        self.cp_manager = ControlPoint(device=device)

        self.evaluator = TrajectoryEvaluator(scene_xml, model=self.model, device=device)
        
        abs_xml = os.path.abspath(scene_xml)
        self.scene_xml = abs_xml
        self.mj_model = mujoco.MjModel.from_xml_path(abs_xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        
        self.lower = self.mj_model.jnt_range[:self.nq, 0].copy()
        self.upper = self.mj_model.jnt_range[:self.nq, 1].copy()
        
        self.robot_body_ids = set()
        base_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if base_id != -1:
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

    def get_interpolated_path(self, q_full: torch.Tensor, interp_steps: int) -> torch.Tensor:
        if interp_steps <= 1:
            return q_full
        
        B, NQ = q_full.shape
        diffs = q_full[1:] - q_full[:-1]
        
        weights = torch.linspace(0, 1, interp_steps + 1, device=q_full.device)[:-1].reshape(1, -1, 1)
        segments = q_full[:-1].unsqueeze(1) + weights * diffs.unsqueeze(1)
        
        q_interp = segments.reshape(-1, NQ)
        q_interp = torch.cat([q_interp, q_full[-1:]], dim=0)
        return q_interp

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

    def is_collision_free(self, q: np.ndarray) -> bool:
        self.mj_data.qpos[:self.nq] = q
        if self.mj_model.nq > self.nq:
            self.mj_data.qpos[self.nq:self.mj_model.nq] = 0.04
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            b1, b2 = self.mj_model.geom_bodyid[c.geom1], self.mj_model.geom_bodyid[c.geom2]
            if self.is_robot_body[b1] ^ self.is_robot_body[b2]:
                return False
        return True

class NLoptPlanner:
    def __init__(self, model_path, scene_xml, n_waypoints=30, device="cpu", opt_base=None):
        if opt_base is not None:
            self.opt_base = opt_base
            if hasattr(opt_base, 'model'):
                self.dist_model = opt_base.model
            elif hasattr(opt_base, 'nn_sdf_robot'):
                self.dist_model = opt_base.nn_sdf_robot
            else:
                raise ValueError("opt_base must have .model or .nn_sdf_robot")
        else:
            raise ValueError("opt_base (NDF/NNSDF planner instance) must be provided")
        self.device = device
        self.default_n = n_waypoints
        self.dof = 7
        self.lower = self.opt_base.lower
        self.upper = self.opt_base.upper

        model_type = getattr(opt_base, 'model_type', 'G-NDF')
        if model_type == 'NN-SDF' or hasattr(opt_base, 'nn_sdf_robot') or (hasattr(opt_base, 'model_type') and opt_base.model_type == 'NN-SDF'):
            self.dist_scale = 0.05
        else:
            self.dist_scale = 1.0

    def solve(self, start, goal, solver='LD_SLSQP',
              initial_path=None, safety_margin=0.02, verbose=False,
              v_max=1.5, a_max=3.0, dt=0.1,
              n_waypoints=None, max_iter=50, w_clr=1.0):

        if nlopt is None:
            raise RuntimeError("nlopt not installed. pip install nlopt")

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

            raw_dists = self.dist_model(Q).reshape(n)
            dists = raw_dists * self.dist_scale

            z = torch.clamp(-dists / safety_margin, max=10.0)
            cost_clr = w_clr * torch.sum(torch.exp(z))
            
            total_f = cost_smooth + cost_clr
            
            if self.dist_scale < 0:
                g_smooth = torch.autograd.grad(cost_smooth, Q, retain_graph=True)[0]
                
                delta = 1e-3
                q_jac_diag = np.zeros((n, dof))
                grad_clr_fd = torch.zeros_like(Q)
                
                with torch.no_grad():
                    d_base = dists.detach()
                    
                    term_grad = (-self.dist_scale / safety_margin) * torch.exp(z).detach()
                    for j in range(dof):
                        Q_plus = Q.detach().clone()
                        Q_plus[:, j] += delta
                        
                        d_plus = self.dist_model(Q_plus).reshape(n) * self.dist_scale
                        
                        jac_col = (d_plus - d_base) / delta
                        q_jac_diag[:, j] = jac_col.cpu().numpy()
                        
                        grad_clr_fd[:, j] = term_grad * jac_col

                total_grad = g_smooth + grad_clr_fd
                obj_grad = total_grad.detach().cpu().numpy().flatten().copy()
            else:
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
        if verbose and init_dist_min < safety_margin:
            compute_everything(x_init)

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
                for i in range(dof): grad[i, i] = 1.0

        def constraint_goal(result, x, grad):
            result[:] = x[-dof:] - goal
            if grad.size > 0:
                grad[:] = 0.0
                for i in range(dof): grad[i, (n - 1) * dof + i] = 1.0

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
    def __init__(self, model_path, scene_xml, n_waypoints=30, device="cuda", model_type="G-NDF", verbose=False):
        """
        Unified CasADi Planner for both G-NDF and NN-SDF.
        """
        self.device = device
        self.default_n = n_waypoints
        self.model_type = model_type
        
        self.use_autodiff = True # Always use autodiff for L4CasADi
        
        if model_type == "NN-SDF":
            abs_xml = os.path.abspath(scene_xml)
            self.scene_xml = abs_xml
            self.mj_model = mujoco.MjModel.from_xml_path(abs_xml)
            self.dof = 7
            self.lower = self.mj_model.jnt_range[:self.dof, 0].copy()
            self.upper = self.mj_model.jnt_range[:self.dof, 1].copy()
            
            self.nn_sdf_robot = RobotSDF("NN-SDF", model_path, device=device)
            self.env_sampler = MujocoEnvSampler(abs_xml, device=device)
            self.model = self.nn_sdf_robot
        else:
            self.opt_base = NDFOptimizerFranka(model_path, scene_xml, device=device)
            self.model = self.opt_base.model
            self.dof = 7
            self.lower = self.opt_base.lower
            self.upper = self.opt_base.upper
        
        self.limits = np.stack([self.lower, self.upper], axis=1)
        
        if model_type == 'NN-SDF' or hasattr(self, 'nn_sdf_robot'):
            self.dist_scale = 0.05
        else:
            self.dist_scale = 1.0
            
        self.l4c_models = {}

    def resample_path(self, path, num_waypoints):
        return self._resample_path(path, num_waypoints)

    def get_l4c_model(self, n):
        if n not in self.l4c_models:
            model_name = f'casadi_{self.model_type}_{n}_{self.device}'.replace('-', '_')
            
            # Use lower optimization for NN-SDF large graphs to avoid GCC OOM
            if self.model_type == 'NN-SDF':
                # -O0 or -O1 drastically reduces memory usage during compilation
                # Set it via environment variables before invoking l4c
                import os
                os.environ['CXXFLAGS'] = '-O0'
                os.environ['CFLAGS'] = '-O0'
                
            model = l4c.L4CasADi(
                self.model, 
                device=self.device, 
                batched=True, 
                name=model_name,
                generate_jac=self.use_autodiff, 
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
        if self.model_type == "NN-SDF":
            # Auto-sample points if not already set
            if self.nn_sdf_robot.obs_points is None:
                if verbose: print("Warning: NN-SDF obs_points not set.")
        
        n = n_waypoints if n_waypoints is not None else self.default_n
        l4c_model = self.get_l4c_model(n)
        
        opti = ca.Opti()
        Q = opti.variable(n, self.dof)
        
        diff1 = Q[1:, :] - Q[:-1, :]
        diff2 = Q[2:, :] + Q[:-2, :] - 2*Q[1:-1, :]
        cost_smooth = (1.0 * ca.sumsqr(diff1) + 20.0 * ca.sumsqr(diff2)) / n
        
        raw_dists = l4c_model(Q)
        dists = raw_dists * self.dist_scale
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
            "print_level": 5 if verbose else 0,
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
            
            # Treat "Solved To Acceptable Level" (Early Stopping) as success
            if status == 'Solved_To_Acceptable_Level':
                if verbose: print("Optimization reached acceptable tolerance (Early Stopping).")
                q_opt = opti.debug.value(Q)
                success = True
            else:
                if verbose: print(f"CasADi Optimization Failed: {e} (Status: {status})")
                try:
                    q_opt = opti.debug.value(Q)
                except:
                    q_opt = initial_path
                success = False
            
        t_solve = time.time() - t0
        return success, q_opt, t_solve, {"status": status}


# ── DiffCo Planner ────────────────────────────────────────────────────────────

class DiffCoModule(torch.nn.Module):
    """Wraps DiffCo checker.poly_score as a nn.Module for L4CasADi compatibility.

    poly_score output: +1 = free, -1 = collision (same sign convention as CoSDF).
    """
    def __init__(self, checker):
        super().__init__()
        self.checker = checker
        # Register support tensors as buffers so L4CasADi can trace them
        self.register_buffer('support_points', checker.support_points)
        self.register_buffer('support_transformed', checker.support_transformed)
        self.register_buffer('rbf_nodes', checker.rbf_nodes)

    def forward(self, q):
        # q: (N, 7) — joint angles
        return self.checker.poly_score(q)   # (N, 1)


class DiffCoPlanner:
    """Trajectory optimizer backed by a pre-trained DiffCo checker.

    Supports both NLopt (slsqp) and CasADi (ipopt) via L4CasADi.

    poly_score returns values in roughly [-1, +1]:
      +1  → clearly free
       0  → collision boundary
      -1  → clearly in collision
    """

    def __init__(self, checker_path: str, scene_xml: str, n_waypoints: int = 30,
                 device: str = "cpu"):
        import dill
        import sys
        _diffco_path = os.path.join(os.path.dirname(__file__), "diffco")
        if _diffco_path not in sys.path:
            sys.path.insert(0, _diffco_path)

        with open(checker_path, "rb") as f:
            self.checker = dill.load(f)

        # DiffCo inference is fast on CPU; L4CasADi also requires CPU
        self.device = "cpu"
        for attr in ("support_points", "support_transformed", "rbf_nodes", "gains"):
            t = getattr(self.checker, attr, None)
            if isinstance(t, torch.Tensor):
                setattr(self.checker, attr, t.cpu())
        self.model = DiffCoModule(self.checker).cpu()
        self.model.eval()

        abs_xml = os.path.abspath(scene_xml)
        mj_model = mujoco.MjModel.from_xml_path(abs_xml)
        self.lower = mj_model.jnt_range[:7, 0].copy()
        self.upper = mj_model.jnt_range[:7, 1].copy()
        self.limits = np.stack([self.lower, self.upper], axis=1)
        self.dof = 7
        self.default_n = n_waypoints
        self.dist_scale = 1.0
        self._l4c_models = {}

    def _resample_path(self, path: np.ndarray, n: int) -> np.ndarray:
        N = len(path)
        if N == n:
            return path
        old_idx = np.arange(N)
        new_idx = np.linspace(0, N - 1, n)
        out = np.zeros((n, self.dof))
        for j in range(self.dof):
            out[:, j] = np.interp(new_idx, old_idx, path[:, j])
        return out

    def _get_l4c_model(self, n):
        if n not in self._l4c_models:
            model_name = f'casadi_diffco_{n}_{self.device}'
            l4c_model = l4c.L4CasADi(
                self.model, device=self.device, batched=True,
                name=model_name, generate_jac=True, generate_jac_jac=False,
            )
            dummy = ca.MX.sym('q', n, self.dof)
            try:
                _ = l4c_model(dummy)
            except Exception:
                pass
            self._l4c_models[n] = l4c_model
        return self._l4c_models[n]

    def solve(self, start: np.ndarray, goal: np.ndarray,
              initial_path: np.ndarray = None,
              solver: str = "ipopt",
              safety_margin: float = 0.05,
              verbose: bool = False,
              v_max: float = 1.5, a_max: float = 3.0, dt: float = 0.1,
              n_waypoints: int = None, max_iter: int = 50,
              w_clr: float = 1.0):

        n = n_waypoints if n_waypoints is not None else self.default_n
        dof = self.dof
        n_vars = n * dof

        if initial_path is None:
            initial_path = np.linspace(start, goal, n)
        else:
            initial_path = self._resample_path(initial_path, n)

        # ── CasADi / ipopt path ───────────────────────────────────────────────
        if solver.lower() == "ipopt":
            l4c_model = self._get_l4c_model(n)
            opti  = ca.Opti()
            Q     = opti.variable(n, dof)
            diff1 = Q[1:, :] - Q[:-1, :]
            diff2 = Q[2:, :] + Q[:-2, :] - 2 * Q[1:-1, :]
            cost_smooth = (ca.sumsqr(diff1) + 20.0 * ca.sumsqr(diff2)) / n

            dists = l4c_model(Q) * self.dist_scale
            opti.subject_to(dists >= safety_margin)
            z = ca.fmin(50, -dists / safety_margin)
            cost_clr = w_clr * ca.sum1(ca.exp(z))
            opti.minimize(cost_smooth + cost_clr)

            opti.subject_to(Q[0, :].T  == ca.DM(start))
            opti.subject_to(Q[-1, :].T == ca.DM(goal))
            for j in range(dof):
                opti.subject_to(opti.bounded(self.limits[j, 0], Q[:, j], self.limits[j, 1]))
            if dt > 0:
                opti.subject_to(opti.bounded(-v_max * dt, diff1, v_max * dt))
                opti.subject_to(opti.bounded(-a_max * dt**2, diff2, a_max * dt**2))

            opti.set_initial(Q, initial_path)
            p_opts = {"expand": True, "print_time": False}
            s_opts = {
                "print_level": 5 if verbose else 0, "max_iter": max_iter,
                "sb": "yes", "tol": 1e-4, "acceptable_tol": 1e-1,
                "acceptable_iter": 3, "mu_strategy": "adaptive",
                "hessian_approximation": "limited-memory",
            }
            opti.solver("ipopt", p_opts, s_opts)
            t0 = time.time()
            try:
                sol    = opti.solve()
                q_opt  = sol.value(Q)
                status = sol.stats()['return_status']
                success = True
            except Exception as e:
                stats  = opti.debug.stats()
                status = stats.get('return_status', 'failed')
                if status == 'Solved_To_Acceptable_Level':
                    q_opt   = opti.debug.value(Q)
                    success = True
                else:
                    if verbose: print(f"DiffCo CasADi failed: {e}")
                    try:
                        q_opt = opti.debug.value(Q)
                    except Exception:
                        q_opt = initial_path
                    success = False
            return success, q_opt, time.time() - t0, {"status": status}

        if initial_path is None:
            initial_path = np.linspace(start, goal, n)
        else:
            initial_path = self._resample_path(initial_path, n)
        x_init = initial_path.astype(np.float64).flatten()

        algo_map = {"slsqp": nlopt.LD_SLSQP, "cobyla": nlopt.LN_COBYLA}
        algo = algo_map.get(solver.lower(), nlopt.LD_SLSQP)
        opt = nlopt.opt(algo, n_vars)
        opt.set_lower_bounds(np.tile(self.lower, n).astype(np.float64))
        opt.set_upper_bounds(np.tile(self.upper, n).astype(np.float64))

        best = {"x": x_init.copy(), "f": float("inf")}
        cache = {"x": None, "f": None, "obj_grad": None, "dists": None, "q_jac_diag": None}

        def compute_everything(x_np):
            if cache["x"] is not None and np.array_equal(x_np, cache["x"]):
                return
            Q = torch.tensor(x_np.reshape(n, dof), dtype=torch.float32, requires_grad=True)

            diff1 = Q[1:] - Q[:-1]
            diff2 = Q[2:] + Q[:-2] - 2 * Q[1:-1]
            cost_smooth = (torch.sum(diff1 ** 2) + 20.0 * torch.sum(diff2 ** 2)) / n

            scores = self.checker.poly_score(Q).reshape(n)
            z = torch.clamp(-scores / safety_margin, max=10.0)
            cost_clr = w_clr * torch.sum(torch.exp(z))

            total_f = cost_smooth + cost_clr
            g = torch.autograd.grad(total_f, Q, retain_graph=True)[0]
            obj_grad = g.detach().cpu().numpy().flatten().copy()

            g_jac = torch.autograd.grad(scores.sum(), Q, retain_graph=False)[0]
            q_jac_diag = g_jac.detach().cpu().numpy().copy()

            cache["x"] = x_np.copy()
            cache["f"] = total_f.item()
            cache["obj_grad"] = obj_grad
            cache["dists"] = scores.detach().cpu().numpy().copy()
            cache["q_jac_diag"] = q_jac_diag

        compute_everything(x_init)

        def objective(x, grad):
            compute_everything(x)
            if grad.size > 0:
                grad[:] = cache["obj_grad"]
            f_val = cache["f"]
            if f_val < best["f"]:
                best["f"] = f_val
                best["x"] = x.copy()
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
                    jac[i, i * dof:(i + 1) * dof] = g[i]
                grad[:] = jac

        opt.add_inequality_mconstraint(constraint_collision, np.full(n, 1e-2))

        v_bound = v_max * dt
        n_vel = 2 * (n - 1) * dof
        J_vel = np.zeros((n_vel, n_vars))
        for i in range(n - 1):
            for j in range(dof):
                J_vel[i * dof + j, (i + 1) * dof + j] = 1.0
                J_vel[i * dof + j, i * dof + j] = -1.0
                J_vel[(n - 1) * dof + i * dof + j, (i + 1) * dof + j] = -1.0
                J_vel[(n - 1) * dof + i * dof + j, i * dof + j] = 1.0

        def constraint_velocity(result, x, grad):
            Q = x.reshape(n, dof)
            d1 = Q[1:] - Q[:-1]
            result[:(n - 1) * dof] = (d1 - v_bound).flatten()
            result[(n - 1) * dof:] = (-d1 - v_bound).flatten()
            if grad.size > 0:
                grad[:] = J_vel

        opt.add_inequality_mconstraint(constraint_velocity, np.full(n_vel, 1e-4))

        opt.set_ftol_rel(1e-1)
        opt.set_maxeval(max_iter)

        t0 = time.time()
        stats = {"framework": "diffco_nlopt", "return_status": None, "exception": False}
        try:
            x_opt = opt.optimize(x_init)
            status = opt.last_optimize_result()
            stats["return_status"] = status
            success = (status > 0)
        except nlopt.RoundoffLimited:
            x_opt = best["x"]
            stats["return_status"] = "RoundoffLimited"
            success = True
        except Exception as e:
            if verbose:
                print(f"DiffCo NLopt error: {e}")
            stats["return_status"] = str(e)
            stats["exception"] = True
            return False, initial_path.reshape(n, dof), time.time() - t0, stats

        del opt
        return success, x_opt.reshape(n, dof), time.time() - t0, stats