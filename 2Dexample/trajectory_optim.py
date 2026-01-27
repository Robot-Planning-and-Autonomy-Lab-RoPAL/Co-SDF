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


from common.models import load_model as global_load_model

def load_2d_model(path, device, n_layers=20, n_hidden=128):
    return global_load_model(path, device)


def wrap_angles_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x, 2 * np.pi)


class NDFOptimizer2D:
    """NDF-based trajectory optimizer for 2D joint space using constrained optimization."""

    def __init__(self, model_path: str, world_file: str, device: str = "cuda", n_layers: int = 20, n_hidden: int = 128):
        self.device = device
        self.nq = 2
        self.model = load_2d_model(model_path, device, n_layers=n_layers, n_hidden=n_hidden)
        self.problem = robot_tools.PlanningProblem(world_file)
        self.cp_manager = ControlPointManager2D(
            base_position=[self.problem.robot._base_position.x, self.problem.robot._base_position.y],
            link_lengths=self.problem.robot._link_lengths.tolist(),
            device=device
        )
        self.lower = np.array([0.0, 0.0])
        self.upper = np.array([2 * np.pi, 2 * np.pi])
        
        # For gradient computation
        self.cnt_check = 0
        self.p_hash_collision = None
        self.p_hash_cost = None
        self.var_p_collision = None
        self.var_p_cost = None
        self.collision_cost = None
        self.obj = None

    def optimize_rrt(self, start: np.ndarray, goal: np.ndarray, max_iterations: int = 1000, goal_sample_rate: float = 0.2) -> Tuple[bool, np.ndarray]:
        """
        Run 2D RRT to generate initial path.
        
        Returns:
            (success, path) where path is Nx2 array
        """
        from rrt import RRT2D
        
        planner = RRT2D(
            self.problem,
            max_iter=max_iterations,
            step_size=0.15
        )
        path = planner.plan(start, goal)
        success = path is not None
        if success:
            path = np.array(path)
        else:
            path = np.array([start, goal])
        return success, path

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

    def con_collision_free(self, p: np.ndarray, init_path: np.ndarray, safety_margin: float, interp_steps: int = 10) -> float:
        """Collision constraint: must be >= 0 for feasibility."""
        p_hash = hash(p.tobytes())
        if p_hash == self.p_hash_collision:
            return float(self.collision_cost.item())

        self.var_p_collision = self.pre_process(p, init_path)
        self.p_hash_collision = p_hash
        
        q_interp = self.get_interpolated_path(self.var_p_collision, interp_steps)
        self.cnt_check += len(q_interp)
        
        ndf_values = self.model(q_interp).squeeze()
        # Max violation across all points (including interpolated ones)
        # We want safety_margin - ndf <= 0
        self.collision_cost = -torch.sum(torch.clamp(safety_margin - ndf_values, min=0))
        return float(self.collision_cost.item())

    def grad_con_collision_free(self, p: np.ndarray) -> np.ndarray:
        """Gradient of collision constraint."""
        p_hash = hash(p.tobytes())
        if p_hash != self.p_hash_collision:
            # This should rarely happen if scipy calls fun then jac
            self.con_collision_free(p, self.init_path_cache, self.safety_margin_cache)
        
        self.var_p_collision.grad = None
        self.collision_cost.backward()
        if self.var_p_collision.grad is None:
            return np.zeros(len(p), dtype=np.float64)
        return self.var_p_collision.grad[1:-1].cpu().numpy().reshape(-1).astype(np.float64)

    def con_joint_limit(self, p: np.ndarray, init_path: np.ndarray) -> float:
        """Joint limit constraint: In unwrapped space, we don't apply hard periodicity limits.
        Instead, we allow the optimizer to find the shortest 'real' distance.
        Note: If you want to force it back to [0, 2pi], do it after optimization.
        """
        return 0.0 # No hard joint limits in periodic C-space

    def grad_con_joint_limit(self, p: np.ndarray) -> np.ndarray:
        return np.zeros(len(p), dtype=np.float64)

    def cost(self, p: np.ndarray, init_path: np.ndarray, smoothness_type: str = "joint", smoothness_weight: float = 1.0) -> float:
        """Objective: path smoothness."""
        p_hash = hash(p.tobytes())
        if p_hash == self.p_hash_cost:
            return float(self.obj.item())

        self.var_p_cost = self.pre_process(p, init_path)
        self.p_hash_cost = p_hash
        
        if len(self.var_p_cost) < 3:
            self.obj = torch.tensor(0.0, device=self.device)
        else:
            if smoothness_type == "joint":
                # Joint space acceleration
                accel = self.var_p_cost[2:] - 2 * self.var_p_cost[1:-1] + self.var_p_cost[:-2]
                self.obj = smoothness_weight * torch.sum(accel ** 2)
            else:
                # Workspace smoothness (sum of squared distances between control points)
                # Similar to diffco's implementation
                cp = self.cp_manager.get_control_points(self.var_p_cost) # (N, CP, 2)
                diffs = cp[1:] - cp[:-1] # (N-1, CP, 2)
                self.obj = smoothness_weight * torch.sum(diffs ** 2)
        
        return float(self.obj.item())

    def grad_cost(self, p: np.ndarray, smoothness_type: str = "joint", smoothness_weight: float = 1.0) -> np.ndarray:
        p_hash = hash(p.tobytes())
        if p_hash != self.p_hash_cost:
            self.cost(p, self.init_path_cache, smoothness_type, smoothness_weight)
            
        self.var_p_cost.grad = None
        self.obj.backward()
        if self.var_p_cost.grad is None:
            return np.zeros(len(p), dtype=np.float64)
        return self.var_p_cost.grad[1:-1].cpu().numpy().reshape(-1).astype(np.float64)

    def con_max_speed(self, p: np.ndarray, init_path: np.ndarray, max_speed: float) -> float:
        """Constraint: max distance between adjacent waypoints in joint space."""
        q = self.pre_process(p, init_path)
        diffs = q[1:] - q[:-1]
        dist_sq = torch.sum(diffs ** 2, dim=1)
        # We want max_speed^2 - dist_sq >= 0
        violation = torch.sum(torch.clamp(dist_sq - max_speed**2, min=0))
        return -float(violation.item())

    def grad_con_max_speed(self, p: np.ndarray, init_path: np.ndarray, max_speed: float) -> np.ndarray:
        q = self.pre_process(p, init_path)
        diffs = q[1:] - q[:-1]
        dist_sq = torch.sum(diffs ** 2, dim=1)
        violation = torch.sum(torch.clamp(dist_sq - max_speed**2, min=0))
        cost_val = -violation
        
        q.grad = None
        cost_val.backward()
        if q.grad is None:
            return np.zeros(len(p), dtype=np.float64)
        return q.grad[1:-1].cpu().numpy().reshape(-1).astype(np.float64)

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
        verbose: bool = True
    ) -> Tuple[bool, np.ndarray, Dict]:
        if verbose:
            print(f"\n{'='*60}")
            print(f"NDF TRAJECTORY OPTIMIZATION (2D - {optimizer_method})")
            print(f"{'='*60}")
            print(f"  Target waypoints: {num_waypoints}, Smoothness: {smoothness_type}")
            print(f"  Safety margin: {safety_margin}, Max speed: {max_speed}")

        start_time = time.time()
        self.cnt_check = 0
        
        success = False
        best_solution = None
        best_cost = np.inf
        best_feasible_cost = np.inf
        
        for trial in range(num_trials):
            if trial == 0:
                init_path = self.resample_path(initial_path, num_waypoints)
            else:
                # Random initialization
                init_path = np.random.uniform(self.lower, self.upper, (num_waypoints, self.nq))
                init_path[0] = initial_path[0]
                init_path[-1] = initial_path[-1]
            
            x0 = init_path[1:-1].reshape(-1)
            
            # Cache for gradient recomputation
            self.init_path_cache = init_path
            self.safety_margin_cache = safety_margin
            
            # Partial function wrappers
            cost_fn = lambda p: self.cost(p, init_path, smoothness_type=smoothness_type)
            grad_cost_fn = lambda p: self.grad_cost(p, smoothness_type=smoothness_type)
            collision_fn = lambda p: self.con_collision_free(p, init_path, safety_margin)
            grad_collision_fn = lambda p: self.grad_con_collision_free(p)
            speed_fn = lambda p: self.con_max_speed(p, init_path, max_speed)
            grad_speed_fn = lambda p: self.grad_con_max_speed(p, init_path, max_speed)
            
            result = minimize(
                cost_fn,
                x0,
                jac=grad_cost_fn,
                method=optimizer_method,
                constraints=[
                    {'type': 'ineq', 'fun': collision_fn, 'jac': grad_collision_fn},
                    {'type': 'ineq', 'fun': speed_fn, 'jac': grad_speed_fn}
                ],
                options={'maxiter': max_iter, 'disp': False}
            )
            
            if verbose:
                print(f"  Trial {trial}: success={result.success}, cost={result.fun:.6f}")
            
            if result.success:
                success = True
                if result.fun < best_feasible_cost:
                    best_feasible_cost = result.fun
                    best_solution = result
                break
            else:
                if result.fun < best_cost:
                    best_cost = result.fun
                    best_solution = result
        
        end_time = time.time()
        
        if best_solution is None:
            return False, initial_path, {'success': False, 'time': end_time - start_time}
        
        # Reconstruct full path
        optimized_x = best_solution.x.reshape(-1, self.nq)
        optimized_path = np.vstack([init_path[:1], optimized_x, init_path[-1:]])
        
        # Compute metrics
        metrics = self.evaluate_path(optimized_path, safety_margin)
        metrics['success'] = success
        metrics['time'] = end_time - start_time
        metrics['cnt_check'] = self.cnt_check
        metrics['optimizer_cost'] = best_solution.fun
        
        # Finally, wrap for output
        optimized_path = np.remainder(optimized_path, 2 * np.pi)
        
        if verbose:
            print(f"\nOptimization complete!")
            print(f"  Success: {success}")
            print(f"  Time: {metrics['time']:.2f}s")
            print(f"  Collision checks: {metrics['cnt_check']}")
            print(f"{'='*60}\n")
        
        return success, optimized_path, metrics

    def evaluate_path(self, path: np.ndarray, safety_margin: float) -> Dict:
        """Compute evaluation metrics for a path. 
        Works best if path is unwrapped, but handles wrapped path too via periodic diffs.
        """
        metrics = {}
        
        # Periodic difference helper
        def p_diff(q1, q2):
            return (q1 - q2 + np.pi) % (2 * np.pi) - np.pi

        # 1. Path length (periodic)
        diffs = p_diff(path[1:], path[:-1])
        metrics['path_length'] = float(np.sum(np.linalg.norm(diffs, axis=1)))
        
        # 2. Smoothness (use local accelerations, but sensitive to wrap if already wrapped)
        # If the path is unwrapped, this is standard. If wrapped, p_diff handles it.
        if len(path) >= 3:
            # Simple approximation of acceleration in periodic space
            v1 = p_diff(path[1:-1], path[:-2])
            v2 = p_diff(path[2:], path[1:-1])
            accel = v2 - v1
            metrics['smoothness'] = float(np.sum(accel ** 2))
        else:
            metrics['smoothness'] = 0.0
        
        # 3. Max speed violation (periodic)
        step_lengths = np.linalg.norm(diffs, axis=1)
        metrics['max_step'] = float(np.max(step_lengths))
        
        # 4. Collision violations (NDF-based)
        # We MUST wrap for NDF if the model doesn't handle it, but our FK does.
        # However, to be safe and consistent with training, we wrap to [0, 2pi]
        path_wrapped = np.remainder(path, 2 * np.pi)
        with torch.no_grad():
            path_torch = torch.from_numpy(path_wrapped).float().to(self.device)
            ndf_values = self.model(path_torch).squeeze().cpu().numpy()
        violations = np.maximum(0, safety_margin - ndf_values)
        metrics['collision_violations'] = int(np.sum(violations > 1e-6))
        metrics['max_violation'] = float(np.max(violations))
        metrics['avg_ndf'] = float(np.mean(ndf_values))
        
        # 5. Joint limit violations (Not applicable for periodic)
        metrics['joint_limit_violations'] = 0
        
        # 6. TRUE COLLISION CHECK (using robot_tools - ground truth)
        path_collision_free = True
        for q in path_wrapped:
            self.problem.robot.set_link_angles(q)
            fp = self.problem.robot.get_current_polygon()
            if any(fp.intersect(o) for o in self.problem.workspace.obstacles):
                path_collision_free = False
                break
        metrics['success_rate'] = 1.0 if path_collision_free else 0.0
        metrics['collision_free'] = path_collision_free
        
        return metrics

    def optimize_adam(
        self,
        initial_path: np.ndarray,
        num_waypoints: int = 50,
        safety_margin: float = 0.02,
        max_iter: int = 200,
        num_trials: int = 5,
        collision_weight: float = 1000.0,
        smooth_weight: float = 1.0,
        smoothness_type: str = "joint",
        max_speed: float = 0.5,
        lr: float = 0.01,
        verbose: bool = True
    ) -> Tuple[bool, np.ndarray, Dict]:
        """Adam optimizer with weighted soft constraints (no hard constraints)."""
        if verbose:
            print(f"\n{'='*60}")
            print("NDF TRAJECTORY OPTIMIZATION (2D - Adam)")
            print(f"{'='*60}")
            print(f"  Target waypoints: {num_waypoints}, Smoothness: {smoothness_type}")
            print(f"  Weights: smooth={smooth_weight}, collision={collision_weight}")
            print(f"  Max speed: {max_speed}, LR: {lr}")

        start_time = time.time()
        self.cnt_check = 0
        
        best_solution = None
        best_cost = np.inf
        
        for trial in range(num_trials):
            if trial == 0:
                init_path = self.resample_path(initial_path, num_waypoints)
            else:
                init_path = np.random.uniform(self.lower, self.upper, (num_waypoints, self.nq))
                init_path[0] = initial_path[0]
                init_path[-1] = initial_path[-1]
            
            start_tensor = torch.from_numpy(init_path[:1]).float().to(self.device)
            goal_tensor = torch.from_numpy(init_path[-1:]).float().to(self.device)
            x = torch.from_numpy(init_path[1:-1]).float().to(self.device).requires_grad_(True)
            
            optimizer = torch.optim.Adam([x], lr=lr)
            
            for step in range(max_iter):
                optimizer.zero_grad()
                
                q_full = torch.cat([start_tensor, x, goal_tensor], dim=0)
                
                # Smoothness
                if len(q_full) < 3:
                    smooth_cost = torch.tensor(0.0, device=self.device)
                else:
                    if smoothness_type == "joint":
                        # Joint space acceleration
                        accel = q_full[2:] - 2 * q_full[1:-1] + q_full[:-2]
                        smooth_cost = torch.sum(accel ** 2)
                    else:
                        # Workspace smoothness (distances between control points)
                        cp = self.cp_manager.get_control_points(q_full)
                        diffs = cp[1:] - cp[:-1]
                        smooth_cost = torch.sum(diffs ** 2)
                
                # NDF collision penalty (interpolated)
                q_interp = self.get_interpolated_path(q_full, interp_steps=10)
                ndf_values = self.model(q_interp).squeeze()
                self.cnt_check += len(q_interp)
                collision_penalty = torch.sum(torch.clamp(safety_margin - ndf_values, min=0))
                
                # Max speed penalty (limit movement between waypoints)
                diffs_q = q_full[1:] - q_full[:-1]
                dist_sq = torch.sum(diffs_q ** 2, dim=1)
                speed_penalty = torch.sum(torch.clamp(dist_sq - max_speed**2, min=0))
                
                loss = smooth_weight * smooth_cost + collision_weight * collision_penalty + 100.0 * speed_penalty
                loss.backward()
                optimizer.step()
                
                if verbose and step % 50 == 0:
                    print(f"  Trial {trial} Step {step}: Loss={loss.item():.6f}")
            
            final_loss = loss.item()
            if final_loss < best_cost:
                best_cost = final_loss
                best_solution = torch.cat([start_tensor, x.detach(), goal_tensor], dim=0)
            
            if verbose:
                print(f"  Trial {trial}: cost={final_loss:.6f}")
        
        end_time = time.time()
        
        optimized_path = best_solution.cpu().numpy()
        
        # Evaluate (on unwrapped best_solution)
        metrics = self.evaluate_path(optimized_path, safety_margin)
        metrics['success'] = True  # Adam always returns a solution
        metrics['time'] = end_time - start_time
        metrics['cnt_check'] = self.cnt_check
        metrics['optimizer_cost'] = best_cost
        
        # Finally wrap for output
        optimized_path = np.remainder(optimized_path, 2 * np.pi)
        
        if verbose:
            print(f"Optimization complete! Time: {metrics['time']:.2f}s\n")
        
        return True, optimized_path, metrics

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


def compute_smoothness(path: np.ndarray) -> float:
    """Compute smoothness metric (sum of squared accelerations)."""
    if len(path) < 3:
        return 0.0
    accel = path[2:] - 2 * path[1:-1] + path[:-2]
    return float(np.sum(accel ** 2))


def compute_path_length(path: np.ndarray) -> float:
    """Compute total path length in joint space."""
    return float(np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1)))


def print_metrics_comparison(metrics_before: Dict, metrics_after: Dict):
    """Pretty print metrics comparison."""
    print(f"\n{'='*60}")
    print("METRICS COMPARISON")
    print(f"{'='*60}")
    
    def format_change(before, after):
        if before == 0:
            return f"{after:.4f}"
        pct = (after / before - 1) * 100
        return f"{before:.4f} → {after:.4f} ({pct:+.1f}%)"
    
    print(f"  Path Length:        {format_change(metrics_before['path_length'], metrics_after['path_length'])}")
    print(f"  Smoothness:         {format_change(metrics_before['smoothness'], metrics_after['smoothness'])}")
    print(f"  Max Step:           {format_change(metrics_before['max_step'], metrics_after['max_step'])}")
    print(f"  Collision Viols:    {metrics_before['collision_violations']} → {metrics_after['collision_violations']}")
    print(f"  Max Violation:      {metrics_before['max_violation']:.6f} → {metrics_after['max_violation']:.6f}")
    print(f"  Avg NDF:            {metrics_before['avg_ndf']:.6f} → {metrics_after['avg_ndf']:.6f}")
    print(f"  Joint Limit Viols:  {metrics_before['joint_limit_violations']} → {metrics_after['joint_limit_violations']}")
    print(f"\n  === TRUE COLLISION CHECK (Ground Truth) ===")
    print(f"  Success Rate:       {'✓ YES' if metrics_before['collision_free'] else '✗ NO'} → {'✓ YES' if metrics_after['collision_free'] else '✗ NO'}")
    print(f"  Collision-Free:     {metrics_before['collision_free']} → {metrics_after['collision_free']}")
    print(f"\n  Optimization Time:  {metrics_after['time']:.2f}s")
    print(f"  Collision Checks:   {metrics_after['cnt_check']}")
    print(f"  Optimizer Cost:     {metrics_after['optimizer_cost']:.6f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D NDF-based trajectory optimizer")
    parser.add_argument("--path", type=str, default="output/traj/rrt.npy", help="Input RRT path")
    parser.add_argument("--output", type=str, default="output/traj/optim.npy", help="Output path")
    parser.add_argument("--world", type=str, default="config/block_world5.yaml", help="World YAML")
    parser.add_argument("--model", type=str, default="output/block_world5/model_final.pt", help="NDF model path")
    parser.add_argument("--waypoints", type=int, default=30, help="Number of waypoints")
    parser.add_argument("--safety-margin", type=float, default=0.01, help="Safety margin for NDF (aligned with loss-margin)")
    parser.add_argument("--max-iter", type=int, default=300, help="Max iterations per trial")
    parser.add_argument("--num-trials", type=int, default=5, help="Number of optimization trials")
    parser.add_argument("--optimizer", choices=["adam", "slsqp", "trust-constr", "cobyla"], default="slsqp", help="Optimizer type")
    parser.add_argument("--smoothness-type", choices=["joint", "workspace"], default="workspace", help="Smoothness objective type")
    parser.add_argument("--max-speed", type=float, default=0.3, help="Maximum joint space movement between waypoints")
    parser.add_argument("--collision-weight", type=float, default=1000.0, help="Collision weight (Adam only)")
    parser.add_argument("--smooth-weight", type=float, default=1.0, help="Smoothness weight")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (Adam only)")
    parser.add_argument("--n-layers", type=int, default=20, help="NDF layers")
    parser.add_argument("--n-hidden", type=int, default=128, help="NDF hidden units")
    parser.add_argument("--interp", type=int, default=10, help="Interpolation steps for collision checking")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Error: Path file not found: {args.path}")
        sys.exit(1)

    initial_path = np.load(args.path)
    print(f"Loaded initial path: {args.path}")
    print(f"  Shape: {initial_path.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    opt = NDFOptimizer2D(
        args.model,
        args.world,
        device=device,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden
    )

    # Evaluate initial path
    print("\nEvaluating initial path...")
    metrics_before = opt.evaluate_path(initial_path, args.safety_margin)

    # Optimize based on selected optimizer
    if args.optimizer == "adam":
        success, optimized_path, metrics_after = opt.optimize_adam(
            initial_path,
            num_waypoints=args.waypoints,
            safety_margin=args.safety_margin,
            max_iter=args.max_iter,
            num_trials=args.num_trials,
            collision_weight=args.collision_weight,
            smooth_weight=args.smooth_weight,
            smoothness_type=args.smoothness_type,
            max_speed=args.max_speed,
            lr=args.lr,
            verbose=True
        )
    elif args.optimizer == "cobyla":
        success, optimized_path, metrics_after = opt.optimize_gradient_free(
            initial_path,
            num_waypoints=args.waypoints,
            safety_margin=args.safety_margin,
            max_iter=args.max_iter,
            num_trials=args.num_trials,
            optimizer_method=args.optimizer.upper(),
            smoothness_type=args.smoothness_type,
            max_speed=args.max_speed,
            verbose=True
        )
    elif args.optimizer in ["slsqp", "trust-constr"]:
        success, optimized_path, metrics_after = opt.optimize(
            initial_path,
            num_waypoints=args.waypoints,
            safety_margin=args.safety_margin,
            max_iter=args.max_iter,
            num_trials=args.num_trials,
            optimizer_method=args.optimizer.upper(),
            smoothness_type=args.smoothness_type,
            max_speed=args.max_speed,
            verbose=True
        )
    else:
        print(f"Unknown optimizer: {args.optimizer}")
        sys.exit(1)

    # Save result
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, optimized_path)
    print(f"Saved optimized path to: {args.output}")

    # Optional: Re-check with interpolation for more accurate validation
    if args.interp > 0:
        print(f"\n⚠ Re-checking with interpolation (interp={args.interp}) for accurate validation...")
        
        # Re-check initial path
        collision_free_before, n_collisions_before = trajectory_verify.check_path_collision(
            initial_path, opt.problem, args.interp, verbose=False
        )
        metrics_before['collision_free'] = collision_free_before
        metrics_before['success_rate'] = 1.0 if collision_free_before else 0.0
        if n_collisions_before > 0:
            metrics_before['total_collision_points'] = n_collisions_before
            print(f"  Initial path: {n_collisions_before} collision points detected")
        
        # Re-check optimized path
        collision_free_after, n_collisions_after = trajectory_verify.check_path_collision(
            optimized_path, opt.problem, args.interp, verbose=False
        )
        metrics_after['collision_free'] = collision_free_after
        metrics_after['success_rate'] = 1.0 if collision_free_after else 0.0
        if n_collisions_after > 0:
            metrics_after['total_collision_points'] = n_collisions_after
            print(f"  Optimized path: {n_collisions_after} collision points detected")

    # Print comparison
    print_metrics_comparison(metrics_before, metrics_after)
