#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict
from scipy.optimize import minimize, NonlinearConstraint
import time

import robot_tools
import trajectory_verify
from geometry_tools_2d import ControlPointManager2D
from rrt import RRT
from common.utils import get_device
from common.models import load_model as global_load_model

def load_2d_model(path, device, n_layers=20, n_hidden=128):
    return global_load_model(path, device)


def wrap_angles_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x, 2 * np.pi)


class NDFOptimizer2D:
    """NDF-based trajectory optimizer for 2D joint space using constrained optimization."""

    def __init__(self, model_path, world_file, device, n_layers, n_hidden):
        self.device = device
        self.nq = 2
        self.model = load_2d_model(model_path, self.device, n_layers=n_layers, n_hidden=n_hidden)
        self.model.eval()
        self.problem = robot_tools.PlanningProblem(world_file)
        self.cp_manager = ControlPointManager2D(
            base_position=[self.problem.robot._base_position.x, self.problem.robot._base_position.y],
            link_lengths=self.problem.robot._link_lengths.tolist(),
            device=self.device
        )
        self.lower = np.array([0.0, 0.0])
        self.upper = np.array([2 * np.pi, 2 * np.pi])
        
        # Unified evaluation cache (Aligned with 7D pattern)
        self.last_p_hash = None
        self.curr_var_p = None
        self.curr_q_interp = None
        self.curr_ndf_vals = None
        self.curr_cp = None
        self.curr_interp_steps = None
        
        # Cached costs and gradients
        self.obj = None
        self.collision_cost = None
        self.speed_cost = None
        
        # Hyperparameters for grad recomputation
        self.init_path_cache = None
        self.safety_margin_cache = None
        self.interp_steps_cache = None
        self.max_speed_cache = None
        self.smoothness_type_cache = None
        self.smoothness_weight_cache = None
        self.path_length_weight_cache = 0.0
        self.ndf_pot_weight_cache = 0.0

    def _evaluate_unified(self, p: np.ndarray, init_path: np.ndarray, interp_steps: int):
        """Unified forward and backward pass (Maximum Performance)."""
        interp_steps = max(1, interp_steps)
        p_hash = hash(p.tobytes())
        if p_hash == self.last_p_hash and interp_steps == self.curr_interp_steps:
            return
            
        self.last_p_hash = p_hash
        self.curr_interp_steps = interp_steps
        
        # 1. Forward Pass
        self.curr_var_p = self.pre_process(p, init_path)
        self.curr_q_interp = self.get_interpolated_path(self.curr_var_p, interp_steps)
        self.cnt_check += len(self.curr_q_interp)
        self.curr_ndf_vals = self.model(self.curr_q_interp).squeeze()
        self.curr_cp = self.cp_manager.get_control_points(self.curr_var_p)
        
        # 2. Compute Costs
        margin = self.safety_margin_cache or 0.02
        self.collision_cost = -torch.sum(torch.clamp(margin - self.curr_ndf_vals, min=0))
        
        sw = self.smoothness_weight_cache
        pw = self.path_length_weight_cache
        nw = self.ndf_pot_weight_cache
        st = self.smoothness_type_cache
        
        if st == "joint":
            accel = self.curr_var_p[2:] - 2 * self.curr_var_p[1:-1] + self.curr_var_p[:-2]
            self.obj = sw * torch.sum(accel ** 2) if len(self.curr_var_p) >= 3 else torch.tensor(0.0, device=self.device)
        else:
            diffs = self.curr_cp[1:] - self.curr_cp[:-1]
            self.obj = sw * torch.sum(diffs ** 2)
            
        if pw > 0:
            # Sum of squared distances between adjacent waypoints
            diffs_p = (self.curr_var_p[1:] - self.curr_var_p[:-1] + np.pi) % (2 * np.pi) - np.pi
            self.obj += pw * torch.sum(diffs_p ** 2)
            
        if nw > 0:
            # NDF potential cost (workspace potential)
            step = max(1, self.curr_interp_steps)
            ndf_waypoints = self.curr_ndf_vals[::step]
            self.obj += nw * torch.clamp(margin - ndf_waypoints, min=0).sum()

    def optimize_rrt(self, start: np.ndarray, goal: np.ndarray, max_iterations: int = 5000) -> Tuple[bool, np.ndarray, float]:
        """
        Run 2D RRT (OMPL) to generate initial path.
        
        Returns:
            (success, path, solve_time) where path is Nx2 array
        """
        planner = RRT(
            self.problem,
            max_iter=max_iterations,
            step_size=0.15
        )
        t0 = time.time()
        path = planner.plan(start, goal)
        solve_time = time.time() - t0
        
        success = path is not None
        if success:
            path = np.array(path)
        else:
            path = np.array([start, goal])
        return success, path, solve_time

    def resample_path(self, path: np.ndarray, num_waypoints: int) -> np.ndarray:
        N = len(path)
        if N == num_waypoints:
            return path
        
        # 对路径进行解离(Unwrap)，确保插值时不会在 0/2pi 处跳跃
        path_unwrapped = np.unwrap(path, axis=0)
        
        old_indices = np.arange(N)
        new_indices = np.linspace(0, N - 1, num_waypoints)
        resampled = np.zeros((num_waypoints, self.nq))
        for j in range(self.nq):
            resampled[:, j] = np.interp(new_indices, old_indices, path_unwrapped[:, j])
        return resampled

    def pre_process(self, p: np.ndarray, init_path: np.ndarray) -> torch.Tensor:
        """Convert optimization variable to full path with fixed endpoints.
        NOTE: No remainder/wrap here to keep gradients continuous.
        """
        p = torch.FloatTensor(p).reshape([-1, self.nq]).to(self.device)
        init_start = torch.from_numpy(init_path[:1]).float().to(self.device)
        init_end = torch.from_numpy(init_path[-1:]).float().to(self.device)
        full_path = torch.cat([init_start, p, init_end], dim=0).requires_grad_(True)
        return full_path

    def get_interpolated_path(self, q_full: torch.Tensor, interp_steps: int) -> torch.Tensor:
        """Interpolate between waypoints for denser collision checking."""
        if interp_steps <= 1:
            return q_full
        
        B, NQ = q_full.shape
        diffs = q_full[1:] - q_full[:-1]
        
        # (B-1, interp_steps, NQ)
        weights = torch.linspace(0, 1, interp_steps + 1, device=q_full.device)[:-1].reshape(1, -1, 1)
        segments = q_full[:-1].unsqueeze(1) + weights * diffs.unsqueeze(1)
        
        # Flatten: ( (B-1)*interp_steps, NQ )
        q_interp = segments.reshape(-1, NQ)
        
        # Add the very last point
        q_interp = torch.cat([q_interp, q_full[-1:]], dim=0)
        return q_interp

    class MockResult:
        def __init__(self, x, fun, success):
            self.x = x
            self.fun = fun
            self.success = success

    def con_collision_free(self, p: np.ndarray, init_path: np.ndarray, safety_margin: float, interp_steps: int = 1) -> float:
        """Collision constraint using unified forward cache."""
        self._evaluate_unified(p, init_path, interp_steps)
        # Re-compute specifically for the provided margin (usually matches cache)
        self.collision_cost = -torch.sum(torch.clamp(safety_margin - self.curr_ndf_vals, min=0))
        return float(self.collision_cost.detach().item())

    def grad_con_collision_free(self, p: np.ndarray) -> np.ndarray:
        """Gradient of collision constraint using unified forward cache."""
        self._evaluate_unified(p, self.init_path_cache, self.curr_interp_steps or 1)
        
        self.curr_var_p.grad = None
        self.collision_cost.backward(retain_graph=True)
        if self.curr_var_p.grad is None:
            return np.zeros(len(p), dtype=np.float64)
        return self.curr_var_p.grad[1:-1].detach().cpu().numpy().reshape(-1).astype(np.float64)

    def cost(
        self, 
        p: np.ndarray, 
        init_path: np.ndarray, 
        smoothness_type: str = "joint", 
        smoothness_weight: float = 1.0,
        path_length_weight: float = 0.0,
        ndf_pot_weight: float = 0.0,
        safety_margin: float = 0.02
    ) -> float:
        """Objective using unified cache."""
        self.smoothness_type_cache = smoothness_type
        self.smoothness_weight_cache = smoothness_weight
        self.path_length_weight_cache = path_length_weight
        self.ndf_pot_weight_cache = ndf_pot_weight
        self.safety_margin_cache = safety_margin
        
        self._evaluate_unified(p, init_path, self.curr_interp_steps or 1)
        return float(self.obj.detach().item())

    def grad_cost(self, p: np.ndarray) -> np.ndarray:
        """Gradient of objective using unified forward cache."""
        self._evaluate_unified(p, self.init_path_cache, self.curr_interp_steps or 1)
        
        self.curr_var_p.grad = None
        self.obj.backward(retain_graph=True)
        if self.curr_var_p.grad is None:
            return np.zeros(len(p), dtype=np.float64)
        return self.curr_var_p.grad[1:-1].detach().cpu().numpy().reshape(-1).astype(np.float64)

    def con_max_speed(self, p: np.ndarray, init_path: np.ndarray, max_speed: float) -> float:
        """Velocity constraint using unified cache."""
        self._evaluate_unified(p, init_path, self.curr_interp_steps or 1)
        diffs = (self.curr_var_p[1:] - self.curr_var_p[:-1] + np.pi) % (2 * np.pi) - np.pi
        dist_sq = torch.sum(diffs ** 2, dim=1)
        self.speed_cost = -torch.sum(torch.clamp(dist_sq - max_speed**2, min=0))
        return float(self.speed_cost.detach().item())

    def grad_con_max_speed(self, p: np.ndarray) -> np.ndarray:
        self._evaluate_unified(p, self.init_path_cache, self.curr_interp_steps or 1)
        self.curr_var_p.grad = None
        self.speed_cost.backward(retain_graph=True)
        if self.curr_var_p.grad is None:
            return np.zeros(len(p), dtype=np.float64)
        return self.curr_var_p.grad[1:-1].detach().cpu().numpy().reshape(-1).astype(np.float64)

    def optimize(
        self,
        initial_path: np.ndarray,
        num_waypoints: int = 50,
        safety_margin: float = 0.02,
        max_iter: int = 200,
        num_trials: int = 5,
        optimizer_method: str = "SLSQP",
        smoothness_type: str = "joint",
        max_speed: float = 0.5,
        interp_steps: int = 10,
        verbose: bool = True
    ) -> Tuple[bool, np.ndarray, Dict]:
        if verbose:
            print(f"\n{'='*60}")
            print(f"NDF TRAJECTORY OPTIMIZATION (2D - {optimizer_method})")
            print(f"{'='*60}")
            print(f"  Target waypoints: {num_waypoints}, Smoothness: {smoothness_type}")
            print(f"  Safety margin: {safety_margin}, Max speed: {max_speed}, Interp: {interp_steps}")

        start_time = time.time()
        self.cnt_check = 0
        self.t_collision = 0.0
        self.t_cost = 0.0
        
        success = False
        best_solution = None
        best_feasible_cost = np.inf
        best_cost = np.inf
        
        for trial in range(num_trials):
            if trial == 0:
                init_path = self.resample_path(initial_path, num_waypoints)
            else:
                init_path = np.random.uniform(self.lower, self.upper, (num_waypoints, self.nq))
                init_path[0] = initial_path[0]
                init_path[-1] = initial_path[-1]
            
            x0 = init_path[1:-1].reshape(-1)
            self.init_path_cache = init_path
            self.safety_margin_cache = safety_margin
            
            # Setup unified callback params (Aligned with 7D/DiffCo)
            self.smoothness_type_cache = smoothness_type
            self.smoothness_weight_cache = 1.0 
            self.path_length_weight_cache = 0.01 
            self.ndf_pot_weight_cache = 0.1
            
            cost_fn = lambda p: self.cost(p, init_path, smoothness_type=smoothness_type)
            grad_cost_fn = lambda p: self.grad_cost(p)
            collision_fn = lambda p: self.con_collision_free(p, init_path, safety_margin, interp_steps=interp_steps)
            grad_collision_fn = lambda p: self.grad_con_collision_free(p)
            speed_fn = lambda p: self.con_max_speed(p, init_path, max_speed)
            grad_speed_fn = lambda p: self.grad_con_max_speed(p)
            
            cost_history = []
            self.last_x = None
            
            class StopOptimization(Exception):
                pass
            
            def early_stopping_callback(pk, *args):
                self.last_x = pk
                # Use cached evaluation
                is_feasible = (self.collision_cost is not None and self.collision_cost >= -1e-3)
                current_cost = float(self.obj.item()) if self.obj is not None else np.inf
                cost_history.append(current_cost)
                if is_feasible and len(cost_history) > 5:
                    if np.std(cost_history[-5:]) < 1e-4:
                        if verbose: print(f"  Trial {trial}: Early stopping (Feasible & Stable)")
                        raise StopOptimization
            
            try:
                opts = {'maxiter': max_iter, 'disp': False, 'ftol': 1e-3}
                result = minimize(
                    cost_fn, x0, jac=grad_cost_fn, method=optimizer_method,
                    constraints=[
                        {'type': 'ineq', 'fun': collision_fn, 'jac': grad_collision_fn},
                        {'type': 'ineq', 'fun': speed_fn, 'jac': grad_speed_fn}
                    ],
                    callback=early_stopping_callback,
                    options=opts
                )
            except StopOptimization:
                # Callback uses self.last_x
                result = self.MockResult(self.last_x, self.obj.item(), True)
            
            # Feasibility check for trial success
            final_p = self.pre_process(result.x, init_path)
            with torch.no_grad():
                q_interp = self.get_interpolated_path(final_p, interp_steps)
                ndf_vals = self.model(q_interp)
                min_ndf = torch.min(ndf_vals).item()
            
            is_feasible = (min_ndf >= safety_margin - 1e-4) # Allow tiny tolerance
            
            if is_feasible:
                # Update best solution based on cost (objective)
                if result.fun < best_feasible_cost:
                    best_feasible_cost = result.fun
                    best_solution = final_p.detach().cpu().numpy()
                    success = True
            
            self.last_p_hash = None # Clear cache for next trial
            if success:
                break

        solve_time = time.time() - start_time
        if best_solution is None:
            best_solution = self.resample_path(initial_path, num_waypoints)
            
        best_solution = np.remainder(best_solution, 2 * np.pi)
        
        # Compute metrics (includes ground truth check)
        metrics = self.evaluate_path(best_solution, safety_margin)
        
        metrics['success'] = success
        metrics['solve_time'] = solve_time
        metrics['cnt_check'] = self.cnt_check
        metrics['optimizer_cost'] = best_feasible_cost
        
        if verbose:
            print(f"\nOptimization complete!")
            print(f"  Success: {success}")
            print(f"  Solve Time (NDF): {metrics['solve_time']:.4f}s")
            print(f"  Collision checks: {metrics['cnt_check']}")
            print(f"{'='*60}\n")
        
        return success, best_solution, metrics

    def evaluate_path(self, path: np.ndarray, safety_margin: float, interp_steps: int = 10) -> Dict:
        metrics = {}
        def p_diff(q1, q2): return (q1 - q2 + np.pi) % (2 * np.pi) - np.pi
        
        # 1. Path length
        diffs = p_diff(path[1:], path[:-1])
        metrics['path_length'] = float(np.sum(np.linalg.norm(diffs, axis=1)))
        
        # 2. Smoothness (Mean squared velocity)
        metrics['smoothness'] = float(np.mean(np.linalg.norm(diffs, axis=1) ** 2))
        
        # 3. Acceleration
        if len(path) >= 3:
            accel = p_diff(path[2:], path[1:-1]) - p_diff(path[1:-1], path[:-2])
            metrics['accel'] = float(np.mean(np.linalg.norm(accel, axis=1) ** 2))
        else:
            metrics['accel'] = 0.0
            
        # 4. Max step
        step_lengths = np.linalg.norm(diffs, axis=1)
        metrics['max_step'] = float(np.max(step_lengths))
        
        # 5. NDF Analysis
        path_wrapped = np.remainder(path, 2 * np.pi)
        with torch.no_grad():
            path_torch = torch.from_numpy(path_wrapped).float().to(self.device)
            # Use interpolated path for NDF metrics matching 7D
            q_interp = self.get_interpolated_path(path_torch, interp_steps)
            ndf_values = self.model(q_interp).squeeze().cpu().numpy()
            
        violations = np.maximum(0, safety_margin - ndf_values)
        metrics['collision_violations'] = int(np.sum(violations > 1e-6))
        metrics['max_violation'] = float(np.max(violations))
        metrics['avg_ndf'] = float(np.mean(ndf_values))
        
        # 6. Clearance (Ground Truth)
        from trajectory_benchmark import get_true_clearance
        min_clr = 1e6
        q_dense = q_interp.cpu().numpy()
        for q in q_dense:
            min_clr = min(min_clr, get_true_clearance(self.problem, q))
        metrics['min_clearance'] = float(min_clr)
        
        # 7. Ground Truth Collision
        metrics['joint_limit_violations'] = 0
        path_collision_free = True
        for q in q_dense:
            self.problem.robot.set_link_angles(q)
            fp = self.problem.robot.get_current_polygon()
            for o in self.problem.workspace.obstacles:
                if (hasattr(o, 'intersect') and o.intersect(fp)) or (hasattr(o, 'intersects') and o.intersects(fp)):
                    path_collision_free = False; break
            if not path_collision_free: break
            
        metrics['collision_free'] = path_collision_free
        metrics['env_collision_free'] = path_collision_free # 2D only has env
        metrics['self_collision_free'] = True # 2D robot is 1-link chain or simple
        metrics['success_rate'] = 1.0 if path_collision_free else 0.0
        
        return metrics

    def optimize_adam(
        self, initial_path: np.ndarray, num_waypoints: int = 50, safety_margin: float = 0.02,
        max_iter: int = 200, num_trials: int = 5, collision_weight: float = 1000.0,
        smooth_weight: float = 1.0, smoothness_type: str = "joint", max_speed: float = 0.5,
        lr: float = 0.01, interp_steps: int = 10, verbose: bool = True
    ) -> Tuple[bool, np.ndarray, Dict]:
        if verbose:
            print(f"\n{'='*60}\nNDF TRAJECTORY OPTIMIZATION (2D - Adam)\n{'='*60}")
        start_time = time.time(); self.cnt_check = 0
        best_solution = None; best_cost = np.inf
        for trial in range(num_trials):
            if trial == 0: init_path = self.resample_path(initial_path, num_waypoints)
            else:
                init_path = np.random.uniform(self.lower, self.upper, (num_waypoints, self.nq))
                init_path[0], init_path[-1] = initial_path[0], initial_path[-1]
            start_tensor = torch.from_numpy(init_path[:1]).float().to(self.device)
            goal_tensor = torch.from_numpy(init_path[-1:]).float().to(self.device)
            x = torch.from_numpy(init_path[1:-1]).float().to(self.device).requires_grad_(True)
            optimizer = torch.optim.Adam([x], lr=lr)
            for step in range(max_iter):
                optimizer.zero_grad()
                q_full = torch.cat([start_tensor, x, goal_tensor], dim=0)
                if len(q_full) < 3: smooth_cost = torch.tensor(0.0, device=self.device)
                else:
                    if smoothness_type == "joint": smooth_cost = torch.sum((q_full[2:] - 2*q_full[1:-1] + q_full[:-2])**2)
                    else:
                        cp = self.cp_manager.get_control_points(q_full)
                        smooth_cost = torch.sum((cp[1:] - cp[:-1])**2)
                q_interp = self.get_interpolated_path(q_full, interp_steps=interp_steps)
                ndf_values = self.model(q_interp).squeeze(); self.cnt_check += len(q_interp)
                collision_penalty = torch.sum(torch.clamp(safety_margin - ndf_values, min=0))
                diffs_q = q_full[1:] - q_full[:-1]
                speed_penalty = torch.sum(torch.clamp(torch.sum(diffs_q**2, dim=1) - max_speed**2, min=0))
                constraint_loss = collision_weight * collision_penalty + 100.0 * speed_penalty
                obj_loss = smooth_weight * smooth_cost
                (obj_loss + constraint_loss).backward(); optimizer.step()
                if constraint_loss.item() < 1e-2:
                    if obj_loss.item() < best_cost:
                        best_cost = obj_loss.item()
                        best_solution = torch.cat([start_tensor, x.detach(), goal_tensor], dim=0)
                        if x.grad is not None and torch.norm(x.grad).item() < 1e-3: break
            if best_solution is not None: break
        end_time = time.time()
        if best_solution is None: best_solution = torch.cat([start_tensor, x.detach(), goal_tensor], dim=0); success = False
        else: success = True
        optimized_path = best_solution.cpu().numpy()
        metrics = self.evaluate_path(optimized_path, safety_margin)
        metrics.update({'success': success, 'time': end_time - start_time, 'cnt_check': self.cnt_check, 'optimizer_cost': best_cost})
        if verbose: print(f"Optimization complete! Time: {metrics['time']:.2f}s\n")
        return success, np.remainder(optimized_path, 2*np.pi), metrics

    def optimize_gradient_free(
        self,
        initial_path: np.ndarray,
        num_waypoints: int = 50,
        safety_margin: float = 0.02,
        max_iter: int = 200,
        num_trials: int = 5,
        optimizer_method: str = "COBYLA",
        smoothness_type: str = "joint",
        max_speed: float = 0.5,
        verbose: bool = True
    ) -> Tuple[bool, np.ndarray, Dict]:
        """Gradient-free scipy minimize (COBYLA) with constraints."""
        if verbose:
            print(f"\n{'='*60}")
            print(f"NDF TRAJECTORY OPTIMIZATION (2D - {optimizer_method})")
            print(f"{'='*60}")
            print(f"  Target waypoints: {num_waypoints}, Smoothness: {smoothness_type}")
            print(f"  Safety margin: {safety_margin}, Max speed: {max_speed}")

        start_time = time.time()
        self.cnt_check = 0
        self.t_collision = 0.0
        self.t_cost = 0.0
        
        success = False
        best_solution = None
        best_cost = np.inf
        
        for trial in range(num_trials):
            if trial == 0:
                init_path = self.resample_path(initial_path, num_waypoints)
            else:
                init_path = np.random.uniform(self.lower, self.upper, (num_waypoints, self.nq))
                init_path[0] = initial_path[0]
                init_path[-1] = initial_path[-1]
            
            x0 = init_path[1:-1].reshape(-1)
            self.init_path_cache = init_path
            self.safety_margin_cache = safety_margin
            
            # Partial function wrappers
            cost_fn = lambda p: self.cost(p, init_path, smoothness_type=smoothness_type)
            collision_fn = lambda p: self.con_collision_free(p, init_path, safety_margin)
            speed_fn = lambda p: self.con_max_speed(p, init_path, max_speed)
            
            result = minimize(
                cost_fn,
                x0,
                method=optimizer_method,
                constraints=[
                    {'type': 'ineq', 'fun': collision_fn},
                    {'type': 'ineq', 'fun': speed_fn}
                ],
                options={'maxiter': max_iter, 'disp': False}
            )
            
            if result.success:
                success = True
                if result.fun < best_cost:
                    best_cost = result.fun
                    best_solution = result
                break
            else:
                if result.fun < best_cost:
                    best_cost = result.fun
                    best_solution = result
            
            if verbose:
                print(f"  Trial {trial}: success={result.success}, cost={result.fun:.6f}")
        
        end_time = time.time()
        
        if best_solution is None:
            return False, initial_path, {'success': False, 'time': end_time - start_time}
        
        optimized_x = best_solution.x.reshape(-1, self.nq)
        optimized_path = np.vstack([init_path[:1], optimized_x, init_path[-1:]])
        
        # Compute metrics on unwrapped
        metrics = self.evaluate_path(optimized_path, safety_margin)
        metrics['success'] = success
        metrics['time'] = end_time - start_time
        metrics['cnt_check'] = self.cnt_check
        metrics['optimizer_cost'] = best_solution.fun
        
        # Wrap for output
        optimized_path = np.remainder(optimized_path, 2 * np.pi)
        
        if verbose:
            print(f"Optimization complete! Time: {metrics['time']:.2f}s\n")
        
        return success, optimized_path, metrics


def print_metrics_comparison(m1, m2):
    print(f"\n{'='*60}\nMETRICS COMPARISON\n{'='*60}")
    def fmt(ov, nv): return f"{ov:.4f} → {nv:.4f} ({ (nv/ov-1)*100:+.1f}%)" if ov != 0 else f"{nv:.4f}"
    print(f"  Path Length: {fmt(m1['path_length'], m2['path_length'])}")
    print(f"  Success:     {'success' if m1['collision_free'] else 'failure'} → {'success' if m2['collision_free'] else 'failure'}")
    print(f"  Time:        {m2['time']:.4f}s")
    if 'optimizer_cost' in m2:
        print(f"  Optim Cost:  {m2['optimizer_cost']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="output/traj/rrt.npy")
    parser.add_argument("--output", type=str, default="output/traj/optim.npy")
    parser.add_argument("--world", type=str, default="config/block_world1.yaml")
    parser.add_argument("--model", type=str, default="output/block_world1/model_final.pt")
    parser.add_argument("--waypoints", type=int, default=30)
    parser.add_argument("--safety-margin", type=float, default=0.01)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--optimizer", choices=["adam", "slsqp"], default="slsqp")
    parser.add_argument("--smoothness-type", choices=["joint", "workspace"], default="workspace")
    parser.add_argument("--max-speed", type=float, default=0.3)
    parser.add_argument("--collision-weight", type=float, default=1000.0, help="Collision weight (Adam only)")
    parser.add_argument("--collision-weight-adam", type=float, default=1000.0, dest="collision_weight_adam", help="Alias for collision weight")
    parser.add_argument("--smooth-weight", type=float, default=1.0, help="Smoothness weight")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (Adam only)")
    parser.add_argument("--n-layers", type=int, default=20, help="NDF layers")
    parser.add_argument("--n-hidden", type=int, default=128, help="NDF hidden units")
    parser.add_argument("--interp", type=int, default=0, help="Interpolation steps for collision checking")
    args = parser.parse_args()

    # Ensure collision_weight_adam is set if collision-weight is used
    if hasattr(args, 'collision_weight') and not hasattr(args, 'collision_weight_adam'):
        args.collision_weight_adam = args.collision_weight

    device = get_device()
    print(f"Using device: {device}")

    initial_path = np.load(args.path)

    opt = NDFOptimizer2D(args.model, args.world, device, args.n_layers, args.n_hidden)
    m_before = opt.evaluate_path(initial_path, args.safety_margin)
    
    if args.optimizer == "adam":
        success, opt_path, m_after = opt.optimize_adam(initial_path, args.waypoints, args.safety_margin, args.max_iter, args.num_trials, interp_steps=args.interp)
    else:
        success, opt_path, m_after = opt.optimize(initial_path, args.waypoints, args.safety_margin, args.max_iter, args.num_trials, interp_steps=args.interp)
    
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, opt_path)
    
    val_interp = max(args.interp, 10)
    print(f"\nValidation (interp={val_interp})...")
    m_before['collision_free'], _ = trajectory_verify.check_path_collision(initial_path, opt.problem, val_interp, False)
    m_after['collision_free'], _ = trajectory_verify.check_path_collision(opt_path, opt.problem, val_interp, False)
    print_metrics_comparison(m_before, m_after)
