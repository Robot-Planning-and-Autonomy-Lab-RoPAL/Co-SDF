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
        self.lower = np.array([0.0, 0.0])
        self.upper = np.array([2 * np.pi, 2 * np.pi])
        
        # For gradient computation
        self.cnt_check = 0
        self.latest_p_collision = None
        self.latest_p_limit = None
        self.latest_p_cost = None
        self.var_p_collision = None
        self.var_p_limit = None
        self.var_p_cost = None
        self.collision_cost = None
        self.joint_limit_cost = None
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
        old_indices = np.arange(N)
        new_indices = np.linspace(0, N - 1, num_waypoints)
        resampled = np.zeros((num_waypoints, self.nq))
        for j in range(self.nq):
            resampled[:, j] = np.interp(new_indices, old_indices, path[:, j])
        return resampled

    def pre_process(self, p: np.ndarray, init_path: np.ndarray) -> torch.Tensor:
        """Convert optimization variable to full path with fixed endpoints."""
        p = torch.FloatTensor(p).reshape([-1, self.nq]).to(self.device)
        p = torch.remainder(p, 2 * np.pi)  # wrap to [0, 2π)
        init_start = torch.from_numpy(init_path[:1]).float().to(self.device)
        init_end = torch.from_numpy(init_path[-1:]).float().to(self.device)
        full_path = torch.cat([init_start, p, init_end], dim=0).requires_grad_(True)
        return full_path

    def con_collision_free(self, p: np.ndarray, init_path: np.ndarray, safety_margin: float) -> float:
        """Collision constraint: must be >= 0 for feasibility."""
        self.var_p_collision = self.pre_process(p, init_path)
        self.latest_p_collision = self.var_p_collision.data[1:-1].cpu().numpy().reshape(-1)
        self.cnt_check += len(self.var_p_collision)
        
        ndf_values = self.model(self.var_p_collision).squeeze()
        self.collision_cost = -torch.sum(torch.clamp(safety_margin - ndf_values, min=0))
        return float(self.collision_cost.item())

    def grad_con_collision_free(self, p: np.ndarray) -> np.ndarray:
        """Gradient of collision constraint."""
        if not np.allclose(p, self.latest_p_collision):
            # Scipy may copy arrays, recompute forward pass
            self.con_collision_free(p, self.init_path_cache, self.safety_margin_cache)
        self.var_p_collision.grad = None
        self.collision_cost.backward()
        if self.var_p_collision.grad is None:
            return np.zeros(len(p), dtype=np.float64)
        return self.var_p_collision.grad[1:-1].cpu().numpy().reshape(-1).astype(np.float64)

    def con_joint_limit(self, p: np.ndarray, init_path: np.ndarray) -> float:
        """Joint limit constraint: must be >= 0 for feasibility."""
        self.var_p_limit = self.pre_process(p, init_path)
        self.latest_p_limit = self.var_p_limit.data[1:-1].cpu().numpy().reshape(-1)
        
        lower_torch = torch.from_numpy(self.lower).float().to(self.device)
        upper_torch = torch.from_numpy(self.upper).float().to(self.device)
        
        self.joint_limit_cost = -torch.sum(
            torch.clamp(lower_torch - self.var_p_limit, min=0) + 
            torch.clamp(self.var_p_limit - upper_torch, min=0)
        )
        return float(self.joint_limit_cost.item())

    def grad_con_joint_limit(self, p: np.ndarray) -> np.ndarray:
        """Gradient of joint limit constraint."""
        if not np.allclose(p, self.latest_p_limit):
            self.con_joint_limit(p, self.init_path_cache)
        self.var_p_limit.grad = None
        self.joint_limit_cost.backward()
        if self.var_p_limit.grad is None:
            return np.zeros(len(p), dtype=np.float64)
        return self.var_p_limit.grad[1:-1].cpu().numpy().reshape(-1).astype(np.float64)

    def cost(self, p: np.ndarray, init_path: np.ndarray) -> float:
        """Objective: path smoothness (sum of squared accelerations)."""
        self.var_p_cost = self.pre_process(p, init_path)
        self.latest_p_cost = self.var_p_cost.data[1:-1].cpu().numpy().reshape(-1)
        
        # Smoothness: acceleration penalty
        if len(self.var_p_cost) < 3:
            self.obj = torch.tensor(0.0, device=self.device)
        else:
            accel = self.var_p_cost[2:] - 2 * self.var_p_cost[1:-1] + self.var_p_cost[:-2]
            self.obj = torch.sum(accel ** 2)
        
        return float(self.obj.item())

    def grad_cost(self, p: np.ndarray) -> np.ndarray:
        """Gradient of objective function."""
        if not np.allclose(p, self.latest_p_cost):
            self.cost(p, self.init_path_cache)
        self.var_p_cost.grad = None
        self.obj.backward()
        if self.var_p_cost.grad is None:
            return np.zeros(len(p), dtype=np.float64)
        return self.var_p_cost.grad[1:-1].cpu().numpy().reshape(-1).astype(np.float64)

    def optimize(
        self,
        initial_path: np.ndarray,
        num_waypoints: int = 50,
        safety_margin: float = 0.02,
        max_iter: int = 200,
        num_trials: int = 5,
        optimizer_method: str = "SLSQP",
        verbose: bool = True
    ) -> Tuple[bool, np.ndarray, Dict]:
        if verbose:
            print(f"\n{'='*60}")
            print("NDF TRAJECTORY OPTIMIZATION (2D - Constrained)")
            print(f"{'='*60}")
            print(f"  Initial waypoints: {len(initial_path)}")
            print(f"  Target waypoints: {num_waypoints}")
            print(f"  Safety margin: {safety_margin}")
            print(f"  Optimizer: {optimizer_method}")
            print(f"  Max iterations: {max_iter}")
            print(f"  Num trials: {num_trials}")

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
            cost_fn = lambda p: self.cost(p, init_path)
            grad_cost_fn = lambda p: self.grad_cost(p)
            collision_fn = lambda p: self.con_collision_free(p, init_path, safety_margin)
            grad_collision_fn = lambda p: self.grad_con_collision_free(p)
            limit_fn = lambda p: self.con_joint_limit(p, init_path)
            grad_limit_fn = lambda p: self.grad_con_joint_limit(p)
            
            result = minimize(
                cost_fn,
                x0,
                jac=grad_cost_fn,
                method=optimizer_method,
                constraints=[
                    {'type': 'ineq', 'fun': collision_fn, 'jac': grad_collision_fn},
                    {'type': 'ineq', 'fun': limit_fn, 'jac': grad_limit_fn}
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
        optimized_path = np.remainder(optimized_path, 2 * np.pi)
        
        # Compute metrics
        metrics = self.evaluate_path(optimized_path, safety_margin)
        metrics['success'] = success
        metrics['time'] = end_time - start_time
        metrics['cnt_check'] = self.cnt_check
        metrics['optimizer_cost'] = best_solution.fun
        
        if verbose:
            print(f"\nOptimization complete!")
            print(f"  Success: {success}")
            print(f"  Time: {metrics['time']:.2f}s")
            print(f"  Collision checks: {metrics['cnt_check']}")
            print(f"{'='*60}\n")
        
        return success, optimized_path, metrics

    def evaluate_path(self, path: np.ndarray, safety_margin: float) -> Dict:
        """Compute evaluation metrics for a path."""
        metrics = {}
        
        # 1. Path length
        diffs = path[1:] - path[:-1]
        metrics['path_length'] = float(np.sum(np.linalg.norm(diffs, axis=1)))
        
        # 2. Smoothness (sum of squared accelerations)
        if len(path) >= 3:
            accel = path[2:] - 2 * path[1:-1] + path[:-2]
            metrics['smoothness'] = float(np.sum(accel ** 2))
        else:
            metrics['smoothness'] = 0.0
        
        # 3. Max speed violation
        step_lengths = np.linalg.norm(diffs, axis=1)
        metrics['max_step'] = float(np.max(step_lengths))
        
        # 4. Collision violations (NDF-based)
        with torch.no_grad():
            path_torch = torch.from_numpy(path).float().to(self.device)
            ndf_values = self.model(path_torch).squeeze().cpu().numpy()
        violations = np.maximum(0, safety_margin - ndf_values)
        metrics['collision_violations'] = int(np.sum(violations > 1e-6))
        metrics['max_violation'] = float(np.max(violations))
        metrics['avg_ndf'] = float(np.mean(ndf_values))
        
        # 5. Joint limit violations
        lower_viol = np.maximum(0, self.lower - path)
        upper_viol = np.maximum(0, path - self.upper)
        metrics['joint_limit_violations'] = int(np.sum((lower_viol + upper_viol) > 1e-6))
        
        # 6. TRUE COLLISION CHECK (using robot_tools - ground truth)
        path_collision_free = True
        for q in path:
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
        collision_weight: float = 100.0,
        smooth_weight: float = 1.0,
        lr: float = 0.01,
        verbose: bool = True
    ) -> Tuple[bool, np.ndarray, Dict]:
        """Adam optimizer with weighted soft constraints (no hard constraints)."""
        if verbose:
            print(f"\n{'='*60}")
            print("NDF TRAJECTORY OPTIMIZATION (2D - Adam)")
            print(f"{'='*60}")
            print(f"  Initial waypoints: {len(initial_path)}")
            print(f"  Target waypoints: {num_waypoints}")
            print(f"  Smooth weight: {smooth_weight}, Collision weight: {collision_weight}")
            print(f"  Max iterations: {max_iter}, Num trials: {num_trials}")

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
                q_full = torch.remainder(q_full, 2 * np.pi)
                
                # Smoothness
                if len(q_full) >= 3:
                    accel = q_full[2:] - 2 * q_full[1:-1] + q_full[:-2]
                    smooth_cost = torch.sum(accel ** 2)
                else:
                    smooth_cost = torch.tensor(0.0, device=self.device)
                
                # NDF collision penalty
                ndf_values = self.model(q_full).squeeze()
                self.cnt_check += len(q_full)
                collision_penalty = torch.sum(torch.clamp(safety_margin - ndf_values, min=0))
                
                # Joint limit penalty
                lower_torch = torch.from_numpy(self.lower).float().to(self.device)
                upper_torch = torch.from_numpy(self.upper).float().to(self.device)
                joint_penalty = torch.sum(
                    torch.clamp(lower_torch - q_full, min=0) ** 2 +
                    torch.clamp(q_full - upper_torch, min=0) ** 2
                )
                
                loss = smooth_weight * smooth_cost + collision_weight * collision_penalty + 1000.0 * joint_penalty
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
        optimized_path = np.remainder(optimized_path, 2 * np.pi)
        
        # Evaluate
        metrics = self.evaluate_path(optimized_path, safety_margin)
        metrics['success'] = True  # Adam always returns a solution
        metrics['time'] = end_time - start_time
        metrics['cnt_check'] = self.cnt_check
        metrics['optimizer_cost'] = best_cost
        
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
        verbose: bool = True
    ) -> Tuple[bool, np.ndarray, Dict]:
        """Gradient-free scipy minimize (COBYLA) with constraints."""
        if verbose:
            print(f"\n{'='*60}")
            print("NDF TRAJECTORY OPTIMIZATION (2D - Gradient Free)")
            print(f"{'='*60}")
            print(f"  Initial waypoints: {len(initial_path)}")
            print(f"  Target waypoints: {num_waypoints}")
            print(f"  Max iterations: {max_iter}, Num trials: {num_trials}")

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
            
            cost_fn = lambda p: self.cost(p, init_path)
            collision_fn = lambda p: self.con_collision_free(p, init_path, safety_margin)
            limit_fn = lambda p: self.con_joint_limit(p, init_path)
            
            result = minimize(
                cost_fn,
                x0,
                method='COBYLA',
                constraints=[
                    {'type': 'ineq', 'fun': collision_fn},
                    {'type': 'ineq', 'fun': limit_fn}
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
        optimized_path = np.remainder(optimized_path, 2 * np.pi)
        
        metrics = self.evaluate_path(optimized_path, safety_margin)
        metrics['success'] = success
        metrics['time'] = end_time - start_time
        metrics['cnt_check'] = self.cnt_check
        metrics['optimizer_cost'] = best_solution.fun
        
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
    parser.add_argument("--waypoints", type=int, default=10, help="Number of waypoints")
    parser.add_argument("--safety-margin", type=float, default=0.02, help="Safety margin for NDF")
    parser.add_argument("--max-iter", type=int, default=200, help="Max iterations per trial")
    parser.add_argument("--num-trials", type=int, default=5, help="Number of optimization trials")
    parser.add_argument("--optimizer", choices=["adam", "slsqp", "trust-constr", "cobyla"], default="slsqp", help="Optimizer type")
    parser.add_argument("--collision-weight", type=float, default=1000.0, help="Collision weight")
    parser.add_argument("--smooth-weight", type=float, default=1.0, help="Smoothness weight")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
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
