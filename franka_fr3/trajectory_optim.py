#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import mujoco
from typing import Tuple, Dict
from scipy.optimize import minimize
import time
from common.models import load_model
from trajecor_verify import check_path_collision_simple
from geometry_tools import ControlPointManager

class NDFOptimizerFranka:

    def __init__(self, model_path: str, scene_xml: str, device: str = "cuda"):
        self.device = device
        self.nq = 7
       
        # Load NDF model
        self.model = load_model(model_path, device)
        self.model.eval()
        
        # Initialize control point manager for geometry features
        self.cp_manager = ControlPointManager(device=device)

        
        # Load MuJoCo scene
        abs_xml = os.path.abspath(scene_xml)
        self.mj_model = mujoco.MjModel.from_xml_path(abs_xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        
        # Joint limits
        self.lower = self.mj_model.jnt_range[:self.nq, 0].copy()
        self.upper = self.mj_model.jnt_range[:self.nq, 1].copy()
        
        # Identify robot bodies for collision checking
        self.robot_body_ids = set()
        base_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")
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
        
        # For gradient computation caching
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

    def is_collision_free(self, q: np.ndarray) -> bool:
        """Check if configuration q is collision-free using MuJoCo."""
        self.mj_data.qpos[:self.nq] = q
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            b1, b2 = self.mj_model.geom_bodyid[c.geom1], self.mj_model.geom_bodyid[c.geom2]
            if self.is_robot_body[b1] or self.is_robot_body[b2]:
                return False
        return True

    def joints_to_features(self, q_batch: torch.Tensor) -> torch.Tensor:
        """
        Convert joint configurations to full features for SDP model.
        q_batch: (B, 7) joint angles
        returns: (B, 49) features = [q(7) + control_points(42)]
        """
        # Get control points (B, 14, 3) -> flatten to (B, 42)
        points = self.cp_manager.get_control_points(q_batch)  # (B, 14, 3)
        points_flat = points.reshape(q_batch.shape[0], -1)  # (B, 42)
        
        # Concatenate: [joints(7) + points(42)] = 49-dim
        features = torch.cat([q_batch, points_flat], dim=-1)  # (B, 49)
        return features


    def resample_path(self, path: np.ndarray, num_waypoints: int) -> np.ndarray:
        """Resample path to specified number of waypoints."""
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
        init_start = torch.from_numpy(init_path[:1]).float().to(self.device)
        init_end = torch.from_numpy(init_path[-1:]).float().to(self.device)
        full_path = torch.cat([init_start, p, init_end], dim=0).requires_grad_(True)
        return full_path

    def con_collision_free(self, p: np.ndarray, init_path: np.ndarray, safety_margin: float) -> float:
        """Collision constraint: must be >= 0 for feasibility."""
        self.var_p_collision = self.pre_process(p, init_path)
        self.latest_p_collision = self.var_p_collision.data[1:-1].cpu().numpy().reshape(-1)
        self.cnt_check += len(self.var_p_collision)
        
        # Convert joints to features for SDP model
        features = self.joints_to_features(self.var_p_collision)
        ndf_values = self.model(features).squeeze()
        
        self.collision_cost = -torch.sum(torch.clamp(safety_margin - ndf_values, min=0))
        return float(self.collision_cost.item())

    def grad_con_collision_free(self, p: np.ndarray) -> np.ndarray:
        """Gradient of collision constraint."""
        if not np.allclose(p, self.latest_p_collision):
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
        num_waypoints: int = 30,
        safety_margin: float = 0.010,
        max_iter: int = 200,
        num_trials: int = 3,
        optimizer_method: str = "SLSQP",
        verbose: bool = True
    ) -> Tuple[bool, np.ndarray, Dict]:
        """Optimize trajectory using constrained optimization."""
        if verbose:
            print(f"\n{'='*60}")
            print("NDF TRAJECTORY OPTIMIZATION (Franka 7D - Constrained)")
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
        optimized_path = np.clip(optimized_path, self.lower, self.upper)
        
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
        
        metrics = {}
        
        # 1. Path length
        diffs = path[1:] - path[:-1]
        metrics['path_length'] = float(np.sum(np.linalg.norm(diffs, axis=1)))
        
        # 2. Smoothness (mean squared velocity)
        metrics['smoothness'] = float(np.mean(np.linalg.norm(diffs, axis=1) ** 2))
        
        # 3. Acceleration (mean squared second-order derivative)
        if len(path) >= 3:
            accel = path[2:] - 2 * path[1:-1] + path[:-2]
            metrics['accel'] = float(np.mean(np.linalg.norm(accel, axis=1) ** 2))
        else:
            metrics['accel'] = 0.0
        
        # 4. Max step violation
        step_lengths = np.linalg.norm(diffs, axis=1)
        metrics['max_step'] = float(np.max(step_lengths))
        
        # 5. NDF-based analysis
        with torch.no_grad():
            path_torch = torch.from_numpy(path).float().to(self.device)
            features = self.joints_to_features(path_torch)
            ndf_values = self.model(features).squeeze().cpu().numpy()
        
        violations = np.maximum(0, safety_margin - ndf_values)
        metrics['ndf_violations'] = int(np.sum(violations > 1e-6))
        metrics['max_violation'] = float(np.max(violations))
        metrics['avg_ndf'] = float(np.mean(ndf_values))
        metrics['min_clearance'] = float(np.min(ndf_values))
        
        # 6. Joint limit violations
        lower_viol = np.maximum(0, self.lower - path)
        upper_viol = np.maximum(0, path - self.upper)
        metrics['joint_limit_violations'] = int(np.sum((lower_viol + upper_viol) > 1e-6))
        
        # 7. TRUE COLLISION CHECK (using MuJoCo - ground truth)
        def is_in_collision(q):
            q_clipped = np.clip(q, self.lower, self.upper)
            return not self.is_collision_free(q_clipped)
        
        is_collision_free, _ = check_path_collision_simple(
            path, is_in_collision, interp_steps=0, verbose=False
        )
        metrics['collision_free'] = is_collision_free
        metrics['success_rate'] = 1.0 if is_collision_free else 0.0
        
        return metrics

    def optimize_adam(
        self,
        initial_path: np.ndarray,
        num_waypoints: int = 30,
        safety_margin: float = 0.010,
        max_iter: int = 200,
        num_trials: int = 3,
        collision_weight: float = 500.0,
        smooth_weight: float = 1.0,
        lr: float = 0.01,
        verbose: bool = True
    ) -> Tuple[bool, np.ndarray, Dict]:
        """Adam optimizer with weighted soft constraints (no hard constraints)."""
        if verbose:
            print(f"\n{'='*60}")
            print("NDF TRAJECTORY OPTIMIZATION (Franka 7D - Adam)")
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
                
                if len(q_full) >= 3:
                    accel = q_full[2:] - 2 * q_full[1:-1] + q_full[:-2]
                    smooth_loss = torch.sum(accel ** 2)
                else:
                    smooth_loss = torch.tensor(0.0, device=self.device)
                
                features = self.joints_to_features(q_full)
                ndf_vals = self.model(features).squeeze()
                self.cnt_check += len(q_full)
                collision_penalty = torch.sum(torch.clamp(safety_margin - ndf_vals, min=0))
                
                lower_torch = torch.from_numpy(self.lower).float().to(self.device)
                upper_torch = torch.from_numpy(self.upper).float().to(self.device)
                joint_penalty = torch.sum(
                    torch.clamp(lower_torch - q_full, min=0)**2 +
                    torch.clamp(q_full - upper_torch, min=0)**2
                )
                
                loss = smooth_weight * smooth_loss + collision_weight * collision_penalty + 1000.0 * joint_penalty
                loss.backward()
                optimizer.step()
                
                if verbose and step % 50 == 0:
                    print(f"  Trial {trial} Step {step}: Loss={loss.item():.4f}")
            
            final_loss = loss.item()
            if final_loss < best_cost:
                best_cost = final_loss
                best_solution = torch.cat([start_tensor, x.detach(), goal_tensor], dim=0)
            
            if verbose:
                print(f"  Trial {trial}: cost={final_loss:.6f}")
        
        end_time = time.time()
        
        optimized_path = best_solution.cpu().numpy()
        optimized_path = np.clip(optimized_path, self.lower, self.upper)
        
        metrics = self.evaluate_path(optimized_path, safety_margin)
        metrics['success'] = True
        metrics['time'] = end_time - start_time
        metrics['cnt_check'] = self.cnt_check
        metrics['optimizer_cost'] = best_cost
        
        if verbose:
            print(f"Optimization complete! Time: {metrics['time']:.2f}s\n")
        
        return True, optimized_path, metrics



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
    print(f"\n  === TRUE COLLISION CHECK (MuJoCo Ground Truth) ===")
    print(f"  Success Rate:       {'✓ YES' if metrics_before['collision_free'] else '✗ NO'} → {'✓ YES' if metrics_after['collision_free'] else '✗ NO'}")
    print(f"  Collision-Free:     {metrics_before['collision_free']} → {metrics_after['collision_free']}")
    print(f"\n  Optimization Time:  {metrics_after['time']:.2f}s")
    print(f"  Collision Checks:   {metrics_after['cnt_check']}")
    print(f"  Optimizer Cost:     {metrics_after['optimizer_cost']:.6f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Franka 7D NDF-based trajectory optimizer")
    parser.add_argument("--path", type=str, default="output/rrt_path.npy", help="Input RRT path")
    parser.add_argument("--output", type=str, default="output/optimized_path.npy", help="Output path")
    parser.add_argument("--scene", type=str, default="scenes/bookshelf.xml", help="Scene XML")
    parser.add_argument("--model", type=str, default="output/bookshelf/model_final.pt", help="NDF model path")
    parser.add_argument("--waypoints", type=int, default=30, help="Number of waypoints")
    parser.add_argument("--safety-margin", type=float, default=0.010, help="Safety margin for NDF")
    parser.add_argument("--max-iter", type=int, default=200, help="Max iterations per trial")
    parser.add_argument("--num-trials", type=int, default=3, help="Number of optimization trials")
    parser.add_argument("--optimizer", choices=["slsqp", "trust-constr"], default="slsqp", help="Optimizer type")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Error: Path file not found: {args.path}")
        sys.exit(1)

    initial_path = np.load(args.path)
    print(f"Loaded initial path: {args.path}")
    print(f"  Shape: {initial_path.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    opt = NDFOptimizerFranka(
        args.model,
        args.scene,
        device=device
    )


    print("\nEvaluating initial path...")
    metrics_before = opt.evaluate_path(initial_path, args.safety_margin)

    success, optimized_path, metrics_after = opt.optimize(
        initial_path,
        num_waypoints=args.waypoints,
        safety_margin=args.safety_margin,
        max_iter=args.max_iter,
        num_trials=args.num_trials,
        optimizer_method=args.optimizer.upper(),
        verbose=True
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, optimized_path)
    print(f"Saved optimized path to: {args.output}")

    print_metrics_comparison(metrics_before, metrics_after)
