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
        self.p_hash_collision = None
        self.p_hash_limit = None
        self.p_hash_cost = None
        self.p_hash_speed = None
        self.var_p_collision = None
        self.var_p_limit = None
        self.var_p_cost = None
        self.var_p_speed = None
        self.collision_cost = None
        self.joint_limit_cost = None
        self.speed_cost = None
        self.obj = None
        
        # Hyperparameters for grad recomputation
        self.init_path_cache = None
        self.safety_margin_cache = None
        self.interp_steps_cache = None
        self.max_speed_cache = None
        self.smoothness_type_cache = None
        self.smoothness_weight_cache = None

    def is_collision_free(self, q: np.ndarray) -> bool:
        """Check if configuration q is collision-free using MuJoCo (XOR logic)."""
        self.mj_data.qpos[:self.nq] = q
        # Set fingers to open (0.04)
        if self.mj_model.nq > self.nq:
            self.mj_data.qpos[self.nq:self.mj_model.nq] = 0.04
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            b1, b2 = self.mj_model.geom_bodyid[c.geom1], self.mj_model.geom_bodyid[c.geom2]
            # XOR: ignore self-collision, only robot vs environment
            if self.is_robot_body[b1] ^ self.is_robot_body[b2]:
                return False
        return True

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

    def con_collision_free(self, p: np.ndarray, init_path: np.ndarray, safety_margin: float, interp_steps: int = 5) -> float:
        """Collision constraint: must be >= 0 for feasibility."""
        p_hash = hash(p.tobytes())
        if p_hash == self.p_hash_collision:
            return float(self.collision_cost.item())

        self.var_p_collision = self.pre_process(p, init_path)
        self.p_hash_collision = p_hash
        
        q_interp = self.get_interpolated_path(self.var_p_collision, interp_steps)
        self.cnt_check += len(q_interp)
        
        # Model handles internal FK preprocessing for G-NDF
        ndf_values = self.model(q_interp).squeeze()
        
        self.collision_cost = -torch.sum(torch.clamp(safety_margin - ndf_values, min=0))
        return float(self.collision_cost.item())

    def grad_con_collision_free(self, p: np.ndarray) -> np.ndarray:
        """Gradient of collision constraint."""
        p_hash = hash(p.tobytes())
        if p_hash != self.p_hash_collision:
            self.con_collision_free(p, self.init_path_cache, self.safety_margin_cache, self.interp_steps_cache)
        
        self.var_p_collision.grad = None
        self.collision_cost.backward()
        if self.var_p_collision.grad is None:
            return np.zeros(len(p), dtype=np.float64)
        return self.var_p_collision.grad[1:-1].cpu().numpy().reshape(-1).astype(np.float64)

    def con_joint_limit(self, p: np.ndarray, init_path: np.ndarray) -> float:
        """Joint limit constraint: must be >= 0 for feasibility."""
        p_hash = hash(p.tobytes())
        if p_hash == self.p_hash_limit:
            return float(self.joint_limit_cost.item())

        self.var_p_limit = self.pre_process(p, init_path)
        self.p_hash_limit = p_hash
        
        lower_torch = torch.from_numpy(self.lower).float().to(self.device)
        upper_torch = torch.from_numpy(self.upper).float().to(self.device)
        
        self.joint_limit_cost = -torch.sum(
            torch.clamp(lower_torch - self.var_p_limit, min=0) + 
            torch.clamp(self.var_p_limit - upper_torch, min=0)
        )
        return float(self.joint_limit_cost.item())

    def grad_con_joint_limit(self, p: np.ndarray) -> np.ndarray:
        """Gradient of joint limit constraint."""
        p_hash = hash(p.tobytes())
        if p_hash != self.p_hash_limit:
            self.con_joint_limit(p, self.init_path_cache)
        self.var_p_limit.grad = None
        self.joint_limit_cost.backward()
        if self.var_p_limit.grad is None:
            return np.zeros(len(p), dtype=np.float64)
        return self.var_p_limit.grad[1:-1].cpu().numpy().reshape(-1).astype(np.float64)

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
                accel = self.var_p_cost[2:] - 2 * self.var_p_cost[1:-1] + self.var_p_cost[:-2]
                self.obj = smoothness_weight * torch.sum(accel ** 2)
            else:
                # Workspace smoothness (movement of control points)
                cp = self.cp_manager.get_control_points(self.var_p_cost) # (N, CP, 3)
                diffs = cp[1:] - cp[:-1]
                self.obj = smoothness_weight * torch.sum(diffs ** 2)
        
        return float(self.obj.item())

    def grad_cost(self, p: np.ndarray) -> np.ndarray:
        """Gradient of objective function."""
        p_hash = hash(p.tobytes())
        if p_hash != self.p_hash_cost:
            self.cost(p, self.init_path_cache, self.smoothness_type_cache, self.smoothness_weight_cache)
        self.var_p_cost.grad = None
        self.obj.backward()
        if self.var_p_cost.grad is None:
            return np.zeros(len(p), dtype=np.float64)
        return self.var_p_cost.grad[1:-1].cpu().numpy().reshape(-1).astype(np.float64)

    def con_max_speed(self, p: np.ndarray, init_path: np.ndarray, max_speed: float) -> float:
        """Constraint: max distance between waypoints in joint space."""
        p_hash = hash(p.tobytes())
        if p_hash == self.p_hash_speed:
            return float(self.speed_cost.item())
            
        self.var_p_speed = self.pre_process(p, init_path)
        self.p_hash_speed = p_hash
        
        diffs = self.var_p_speed[1:] - self.var_p_speed[:-1]
        dist_sq = torch.sum(diffs ** 2, dim=1)
        self.speed_cost = -torch.sum(torch.clamp(dist_sq - max_speed**2, min=0))
        return float(self.speed_cost.item())

    def grad_con_max_speed(self, p: np.ndarray) -> np.ndarray:
        p_hash = hash(p.tobytes())
        if p_hash != self.p_hash_speed:
            self.con_max_speed(p, self.init_path_cache, self.max_speed_cache)
        self.var_p_speed.grad = None
        self.speed_cost.backward()
        if self.var_p_speed.grad is None:
            return np.zeros(len(p), dtype=np.float64)
        return self.var_p_speed.grad[1:-1].cpu().numpy().reshape(-1).astype(np.float64)

    def optimize(
        self,
        initial_path: np.ndarray,
        num_waypoints: int = 30,
        safety_margin: float = 0.010,
        max_iter: int = 200,
        num_trials: int = 3,
        optimizer_method: str = "SLSQP",
        smoothness_type: str = "joint",
        smoothness_weight: float = 1.0,
        max_speed: float = 0.3,
        interp_steps: int = 5,
        verbose: bool = True
    ) -> Tuple[bool, np.ndarray, Dict]:
        """Optimize trajectory using constrained optimization."""
        if verbose:
            print(f"\n{'='*60}")
            print(f"NDF TRAJECTORY OPTIMIZATION (Franka 7D - {optimizer_method})")
            print(f"{'='*60}")
            print(f"  Target waypoints: {num_waypoints}, Smoothness: {smoothness_type}")
            print(f"  Safety margin: {safety_margin}, Max speed: {max_speed}")
            print(f"  Interp steps: {interp_steps}, Num trials: {num_trials}")

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
                init_path = np.random.uniform(self.lower, self.upper, (num_waypoints, self.nq))
                init_path[0] = initial_path[0]
                init_path[-1] = initial_path[-1]
            
            x0 = init_path[1:-1].reshape(-1)
            
            # Cache for gradient recomputation
            self.init_path_cache = init_path
            self.safety_margin_cache = safety_margin
            self.interp_steps_cache = interp_steps
            self.max_speed_cache = max_speed
            self.smoothness_type_cache = smoothness_type
            self.smoothness_weight_cache = smoothness_weight
            
            # Partial function wrappers
            cost_fn = lambda p: self.cost(p, init_path, smoothness_type, smoothness_weight)
            grad_cost_fn = lambda p: self.grad_cost(p)
            collision_fn = lambda p: self.con_collision_free(p, init_path, safety_margin, interp_steps)
            grad_collision_fn = lambda p: self.grad_con_collision_free(p)
            limit_fn = lambda p: self.con_joint_limit(p, init_path)
            grad_limit_fn = lambda p: self.grad_con_joint_limit(p)
            speed_fn = lambda p: self.con_max_speed(p, init_path, max_speed)
            grad_speed_fn = lambda p: self.grad_con_max_speed(p)
            
            # Use hard bounds for joint limits instead of constraints for better performance
            bounds = []
            for _ in range(num_waypoints - 2):
                for j in range(self.nq):
                    bounds.append((self.lower[j], self.upper[j]))
            
            result = minimize(
                cost_fn,
                x0,
                jac=grad_cost_fn,
                method=optimizer_method,
                bounds=bounds,
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

    def evaluate_path(self, path: np.ndarray, safety_margin: float, interp_steps: int = 10) -> Dict:
        
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
        
        # 5. NDF-based analysis (interpolated)
        with torch.no_grad():
            path_torch = torch.from_numpy(path).float().to(self.device)
            q_interp = self.get_interpolated_path(path_torch, interp_steps)
            ndf_values = self.model(q_interp).squeeze().cpu().numpy()
        
        violations = np.maximum(0, safety_margin - ndf_values)
        metrics['collision_violations'] = int(np.sum(violations > 1e-6))
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
            path, is_in_collision, interp_steps=interp_steps, verbose=False
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
        collision_weight: float = 1000.0,
        smooth_weight: float = 1.0,
        smoothness_type: str = "joint",
        max_speed: float = 0.3,
        interp_steps: int = 5,
        lr: float = 0.01,
        verbose: bool = True
    ) -> Tuple[bool, np.ndarray, Dict]:
        """Adam optimizer with weighted soft constraints."""
        if verbose:
            print(f"\n{'='*60}")
            print("NDF TRAJECTORY OPTIMIZATION (Franka 7D - Adam)")
            print(f"{'='*60}")
            print(f"  Target waypoints: {num_waypoints}, Smoothness: {smoothness_type}")
            print(f"  Weights: smooth={smooth_weight}, collision={collision_weight}")
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
                
                # Smoothness
                if len(q_full) < 3:
                    smooth_loss = torch.tensor(0.0, device=self.device)
                else:
                    if smoothness_type == "joint":
                        accel = q_full[2:] - 2 * q_full[1:-1] + q_full[:-2]
                        smooth_loss = torch.sum(accel ** 2)
                    else:
                        cp = self.cp_manager.get_control_points(q_full)
                        diffs = cp[1:] - cp[:-1]
                        smooth_loss = torch.sum(diffs ** 2)
                
                # NDF (interpolated)
                q_interp = self.get_interpolated_path(q_full, interp_steps)
                ndf_vals = self.model(q_interp).squeeze()
                self.cnt_check += len(q_interp)
                collision_penalty = torch.sum(torch.clamp(safety_margin - ndf_vals, min=0))
                
                # Joint limits
                lower_torch = torch.from_numpy(self.lower).float().to(self.device)
                upper_torch = torch.from_numpy(self.upper).float().to(self.device)
                joint_penalty = torch.sum(
                    torch.clamp(lower_torch - q_full, min=0)**2 +
                    torch.clamp(q_full - upper_torch, min=0)**2
                )
                
                # Max speed
                diff_q = q_full[1:] - q_full[:-1]
                dist_sq = torch.sum(diff_q ** 2, dim=1)
                speed_penalty = torch.sum(torch.clamp(dist_sq - max_speed**2, min=0))
                
                loss = smooth_weight * smooth_loss + collision_weight * collision_penalty + \
                       1000.0 * joint_penalty + 1000.0 * speed_penalty
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
        
        metrics = self.evaluate_path(optimized_path, safety_margin, interp_steps)
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
    parser.add_argument("--path", type=str, default="output/traj/rrt.npy", help="Input RRT path")
    parser.add_argument("--output", type=str, default="output/traj/optim.npy", help="Output path")
    parser.add_argument("--scene", type=str, default="scenes/bookshelf.xml", help="Scene XML")
    parser.add_argument("--model", type=str, default="output/bookshelf/model_final.pt", help="NDF model path")
    parser.add_argument("--waypoints", type=int, default=30, help="Number of waypoints")
    parser.add_argument("--safety-margin", type=float, default=0.010, help="Safety margin for NDF")
    parser.add_argument("--max-iter", type=int, default=200, help="Max iterations per trial")
    parser.add_argument("--num-trials", type=int, default=3, help="Number of optimization trials")
    parser.add_argument("--optimizer", choices=["slsqp", "adam"], default="slsqp", help="Optimizer type")
    parser.add_argument("--smoothness-type", choices=["joint", "workspace"], default="joint", help="Smoothness type")
    parser.add_argument("--smoothness-weight", type=float, default=1.0)
    parser.add_argument("--max-speed", type=float, default=0.3)
    parser.add_argument("--interp", type=int, default=5, help="Interpolation steps for collision checking")
    parser.add_argument("--collision-weight", type=float, default=1000.0, help="For Adam only")
    parser.add_argument("--lr", type=float, default=0.01, help="For Adam only")
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
    metrics_before = opt.evaluate_path(initial_path, args.safety_margin, interp_steps=args.interp)

    if args.optimizer == "slsqp":
        success, optimized_path, metrics_after = opt.optimize(
            initial_path,
            num_waypoints=args.waypoints,
            safety_margin=args.safety_margin,
            max_iter=args.max_iter,
            num_trials=args.num_trials,
            optimizer_method=args.optimizer.upper(),
            smoothness_type=args.smoothness_type,
            smoothness_weight=args.smoothness_weight,
            max_speed=args.max_speed,
            interp_steps=args.interp,
            verbose=True
        )
    else:
        success, optimized_path, metrics_after = opt.optimize_adam(
            initial_path,
            num_waypoints=args.waypoints,
            safety_margin=args.safety_margin,
            max_iter=args.max_iter,
            num_trials=args.num_trials,
            collision_weight=args.collision_weight,
            smooth_weight=args.smoothness_weight,
            smoothness_type=args.smoothness_type,
            max_speed=args.max_speed,
            interp_steps=args.interp,
            lr=args.lr,
            verbose=True
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, optimized_path)
    print(f"Saved optimized path to: {args.output}")

    print_metrics_comparison(metrics_before, metrics_after)
