#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import mujoco
import imageio
from typing import Tuple, Optional

from utils import load_model
from path_checking import verify_trajectory


class NDFOptimizer:
    """NDF-based trajectory optimizer using gradient descent."""
    
    def __init__(
        self,
        model_path: str,
        scene_xml: str,
        device: str = "cuda",
        nq: int = 7,
        n_layers: int = 20,
        n_hidden: int = 512,
        frequencies: list = [1, 2, 4, 8]
    ):
        self.device = device
        self.nq = nq
        self.scene_xml = scene_xml
        
        # Load NDF model
        print(f"Loading NDF model from {model_path}...")
        self.model = load_model(
            model_path, 
            device, 
            n_dim=nq,
            n_layers=n_layers,
            n_hidden=n_hidden,
            frequencies=frequencies
        )
        self.model.eval()
        
        # Load MuJoCo model for joint limits and collision checking
        abs_xml = os.path.abspath(scene_xml)
        self.mj_model = mujoco.MjModel.from_xml_path(abs_xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.joint_lower = torch.from_numpy(self.mj_model.jnt_range[:nq, 0]).float().to(device)
        self.joint_upper = torch.from_numpy(self.mj_model.jnt_range[:nq, 1]).float().to(device)
        
        # Identify robot bodies for collision detection
        self.robot_body_ids = set()
        base_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if base_id == -1:
            base_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "fr3_link0")
        
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
        
        print(f"  Device: {device}")
        print(f"  Joint limits: [{self.joint_lower.cpu().numpy()}, {self.joint_upper.cpu().numpy()}]")
    
    def resample_path(self, path: np.ndarray, num_waypoints: int) -> torch.Tensor:
        """Resample path to fixed number of waypoints using linear interpolation.
        
        Args:
            path: (N, nq) numpy array
            num_waypoints: Target number of waypoints T
            
        Returns:
            (T, nq) torch tensor
        """
        N = len(path)
        if N == num_waypoints:
            return torch.from_numpy(path).float().to(self.device)
        
        # Create interpolation indices
        old_indices = np.arange(N)
        new_indices = np.linspace(0, N - 1, num_waypoints)
        
        # Interpolate each joint
        resampled = np.zeros((num_waypoints, self.nq))
        for j in range(self.nq):
            resampled[:, j] = np.interp(new_indices, old_indices, path[:, j])
        
        return torch.from_numpy(resampled).float().to(self.device)
    
    def smoothness_cost(self, q_full: torch.Tensor) -> torch.Tensor:
        """Compute joint-space acceleration penalty (sum of squared second derivatives).
        
        Args:
            q_full: (T, nq) trajectory
            
        Returns:
            Scalar smoothness cost
        """
        # Compute second derivative (acceleration)
        # q''[t] = q[t+1] - 2*q[t] + q[t-1]
        T = q_full.shape[0]
        if T < 3:
            return torch.tensor(0.0, device=self.device)
        
        q_prev = q_full[:-2]   # t-1
        q_curr = q_full[1:-1]  # t
        q_next = q_full[2:]    # t+1
        
        accel = q_next - 2 * q_curr + q_prev
        return torch.sum(accel ** 2)
    
    def ndf_collision_penalty(
        self, 
        q_full: torch.Tensor, 
        threshold: float = 0.0,
        power: float = 2.0
    ) -> torch.Tensor:
        """Compute NDF-based collision penalty.
        
        Args:
            q_full: (T, nq) trajectory
            threshold: NDF value threshold (points with f < threshold are penalized)
            power: Exponent for penalty (default: 2.0 for quadratic)
            
        Returns:
            Scalar collision penalty
        """
        # Get NDF predictions for all waypoints
        with torch.set_grad_enabled(True):
            ndf_values = self.model(q_full).squeeze()  # (T,)
        
        # Soft barrier: phi(f) = max(0, threshold - f)^power
        violations = torch.clamp(threshold - ndf_values, min=0.0)
        penalty = torch.sum(violations ** power)
        
        return penalty
    
    def check_collision(self, q: np.ndarray) -> bool:
        """Check if configuration is in collision."""
        self.mj_data.qpos[:self.nq] = q
        if self.mj_model.nq > self.nq:
            self.mj_data.qpos[self.nq:] = 0.04
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            b1, b2 = self.mj_model.geom_bodyid[c.geom1], self.mj_model.geom_bodyid[c.geom2]
            if self.is_robot_body[b1] or self.is_robot_body[b2]:
                return True
        return False
    
    def local_repair(self, path: np.ndarray, window: int = 3, max_iter: int = 100, verbose: bool = True) -> np.ndarray:
        """Local repair: fix collisions in ±window around collision points."""
        collision_indices = [i for i, q in enumerate(path) if self.check_collision(q)]
        
        if len(collision_indices) == 0:
            if verbose:
                print("No collisions, skipping repair")
            return path
        
        if verbose:
            print(f"\n{'='*60}")
            print("LOCAL REPAIR STAGE")
            print(f"{'='*60}")
            print(f"  Collisions at: {collision_indices}")
            print(f"  Window: ±{window}")
        
        repaired = path.copy()
        
        for coll_idx in collision_indices:
            start_idx = max(1, coll_idx - window)
            end_idx = min(len(path) - 1, coll_idx + window + 1)
            
            if verbose:
                print(f"  Repairing waypoint {coll_idx} (range [{start_idx}:{end_idx}])")
            
            local_seg = torch.from_numpy(repaired[start_idx:end_idx]).float().to(self.device)
            local_seg.requires_grad_(True)
            opt = torch.optim.Adam([local_seg], lr=0.01)
            
            for it in range(max_iter):
                opt.zero_grad()
                smooth = self.smoothness_cost(local_seg)
                smooth.backward()
                opt.step()
                
                with torch.no_grad():
                    local_seg.data = torch.clamp(local_seg.data, self.joint_lower, self.joint_upper)
                
                mid_idx = coll_idx - start_idx
                if 0 <= mid_idx < len(local_seg):
                    q_mid = local_seg[mid_idx].detach().cpu().numpy()
                    if not self.check_collision(q_mid):
                        if verbose:
                            print(f"    Fixed at iter {it}")
                        break
            
            repaired[start_idx:end_idx] = local_seg.detach().cpu().numpy()
        
        if verbose:
            print(f"{'='*60}\n")
        
        return repaired
    
    def render_path(self, path: np.ndarray, filename: str, azimuth: float, elevation: float):
        """Render trajectory as GIF."""
        renderer = mujoco.Renderer(self.mj_model, height=480, width=640)
        frames = []
        
        full_path = []
        for i in range(len(path) - 1):
            steps = max(1, int(np.linalg.norm(path[i+1] - path[i]) * 40))
            for s in range(steps):
                full_path.append(path[i] + (path[i+1] - path[i]) * (s / steps))
        full_path.append(path[-1])
        
        for q in full_path:
            self.mj_data.qpos[:self.nq] = q
            mujoco.mj_forward(self.mj_model, self.mj_data)
            
            cam = mujoco.MjvCamera()
            cam.azimuth = azimuth
            cam.elevation = elevation
            cam.distance = 2.0
            cam.lookat = np.array([0.6, 0, 0.7])
            
            renderer.update_scene(self.mj_data, camera=cam)
            frames.append(renderer.render())
        
        imageio.mimsave(filename, frames, fps=20, loop=0)
    
    def optimize(
        self,
        initial_path: np.ndarray,
        num_waypoints: int = 50,
        smooth_weight: float = 1.0,
        collision_weight: float = 10.0,
        epsilon: float = 0.0,
        max_iter: int = 500,
        optimizer_type: str = "lbfgs",
        lr: float = 0.01,
        tolerance: float = 1e-6,
        verbose: bool = True
    ) -> Tuple[bool, np.ndarray]:
        """Optimize trajectory using NDF gradients.
        
        Args:
            initial_path: (N, nq) initial RRT path
            num_waypoints: Number of waypoints to optimize
            smooth_weight: Weight for smoothness cost (alpha)
            collision_weight: Weight for NDF penalty (beta)
            ndf_threshold: NDF threshold for collision penalty
            max_iter: Maximum optimization iterations
            optimizer_type: 'lbfgs' or 'adam'
            lr: Learning rate (for Adam)
            tolerance: Convergence tolerance
            verbose: Print progress
            
        Returns:
            (success, optimized_path) tuple
        """
        if verbose:
            print(f"\n{'='*60}")
            print("NDF TRAJECTORY OPTIMIZATION")
            print(f"{'='*60}")
            print(f"  Initial waypoints: {len(initial_path)}")
            print(f"  Target waypoints: {num_waypoints}")
            print(f"  Smooth weight: {smooth_weight}")
            print(f"  Collision weight: {collision_weight}")
            print(f"  Optimizer: {optimizer_type.upper()}")
        
        # Resample path
        q_resampled = self.resample_path(initial_path, num_waypoints)
        
        # Extract fixed endpoints
        start = q_resampled[0].clone()
        goal = q_resampled[-1].clone()
        
        # Initialize optimization variables (only middle waypoints)
        x = q_resampled[1:-1].clone().detach().requires_grad_(True)
        
        # Closure for LBFGS
        def closure():
            if x.grad is not None:
                x.grad.zero_()
            
            # Reconstruct full trajectory
            q_full = torch.zeros(num_waypoints, self.nq, device=self.device)
            q_full[0] = start
            q_full[-1] = goal
            q_full[1:-1] = x
            
            # Compute costs
            smooth_cost = self.smoothness_cost(q_full)
            collision_cost = self.ndf_collision_penalty(q_full, threshold=epsilon)
            
            total_cost = smooth_weight * smooth_cost + collision_weight * collision_cost
            
            # Backward
            total_cost.backward()
            
            # Clip gradients to joint limits
            with torch.no_grad():
                x.data = torch.clamp(x.data, self.joint_lower, self.joint_upper)
            
            return total_cost
        
        # Create optimizer
        if optimizer_type.lower() == "lbfgs":
            optimizer = torch.optim.LBFGS(
                [x], 
                lr=1.0,
                max_iter=20,
                tolerance_grad=tolerance,
                tolerance_change=tolerance,
                line_search_fn="strong_wolfe"
            )
        elif optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam([x], lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Optimization loop
        prev_loss = float('inf')
        for iteration in range(max_iter):
            if optimizer_type.lower() == "lbfgs":
                loss = optimizer.step(closure)
            else:
                optimizer.zero_grad()
                loss = closure()
                optimizer.step()
            
            current_loss = loss.item()
            
            if verbose and iteration % 10 == 0:
                print(f"  Iter {iteration:3d}: Loss = {current_loss:.6f}")
            
            prev_loss = current_loss
        
        # Reconstruct final trajectory
        with torch.no_grad():
            q_final = torch.zeros(num_waypoints, self.nq, device=self.device)
            q_final[0] = start
            q_final[-1] = goal
            q_final[1:-1] = x
        
        optimized_path = q_final.cpu().numpy()
        
        if verbose:
            print(f"\nOptimization complete!")
            print(f"  Final loss: {current_loss:.6f}")
            print(f"{'='*60}\n")
        
        return True, optimized_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NDF-based trajectory optimization")
    parser.add_argument("--path", type=str, default="output/traj/rrt_path.npy", help="Input RRT path")
    parser.add_argument("--output", type=str, default="output/traj/optimized_path.npy", help="Output path")
    parser.add_argument("--scene", type=str, default="scenes/bookshelf.xml", help="MuJoCo scene XML")
    parser.add_argument("--model", type=str, default="output/bookshelf/model_final.pt", help="NDF model")
    parser.add_argument("--waypoints", type=int, default=20, help="Number of waypoints")
    parser.add_argument("--smooth-weight", type=float, default=1.0, help="Smoothness weight")
    parser.add_argument("--collision-weight", type=float, default=100.0, help="Collision weight")
    parser.add_argument("--epsilon", type=float, default=0.02, help="safety bias")
    parser.add_argument("--max-iter", type=int, default=500, help="Max iterations")
    parser.add_argument("--optimizer", choices=["lbfgs", "adam"], default="lbfgs", help="Optimizer")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (Adam)")
    parser.add_argument("--front", type=str, default="output/visu/opt_front.gif", help="Front view")
    parser.add_argument("--side", type=str, default="output/visu/opt_side.gif", help="Side view")
    parser.add_argument("--n-layers", type=int, default=20, help="NDF layers")
    parser.add_argument("--n-hidden", type=int, default=512, help="NDF hidden units")
    parser.add_argument("--frequencies", type=int, nargs='+', default=[1, 2, 4, 8], help="NDF frequencies")
    args = parser.parse_args()
    
    # Load initial path
    if not os.path.exists(args.path):
        print(f"Error: Path file not found: {args.path}")
        exit(1)
    
    initial_path = np.load(args.path)
    print(f"Loaded initial path: {args.path}")
    print(f"  Shape: {initial_path.shape}")
    
    # Create optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = NDFOptimizer(
        args.model,
        args.scene,
        device=device,
        nq=7,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        frequencies=args.frequencies
    )
    
    # Optimize
    success, optimized_path = optimizer.optimize(
        initial_path,
        num_waypoints=args.waypoints,
        smooth_weight=args.smooth_weight,
        collision_weight=args.collision_weight,
        epsilon=args.epsilon,
        max_iter=args.max_iter,
        optimizer_type=args.optimizer,
        lr=args.lr,
        verbose=True
    )
    
    if success:
        # Check for collisions and auto-repair if needed
        collision_indices = [i for i, q in enumerate(optimized_path) if optimizer.check_collision(q)]
        if len(collision_indices) > 0:
            print(f"\nDetected {len(collision_indices)} collision(s), running auto-repair...")
            optimized_path = optimizer.local_repair(optimized_path, window=3, max_iter=100, verbose=True)
        
        # Save optimized path
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        np.save(args.output, optimized_path)
        print(f"Saved optimized path to: {args.output}")
        
        # Verify collision-free
        print(f"\nVerifying optimized trajectory...")
        verify_trajectory(args.output, args.scene)
        
        # Render visualizations
        print(f"\nRendering visualizations...")
        os.makedirs(os.path.dirname(args.front) or ".", exist_ok=True)
        optimizer.render_path(optimized_path, args.front, azimuth=0, elevation=-20)
        optimizer.render_path(optimized_path, args.side, azimuth=30, elevation=-30)
        print(f"  Front: {args.front}")
        print(f"  Side: {args.side}")
        
        # Compute metrics
        def compute_smoothness(path):
            if len(path) < 3:
                return 0.0
            accel = path[2:] - 2*path[1:-1] + path[:-2]
            return np.sum(accel**2)
        
        def compute_path_length(path):
            return np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1))
        
        smooth_before = compute_smoothness(initial_path)
        smooth_after = compute_smoothness(optimized_path)
        length_before = compute_path_length(initial_path)
        length_after = compute_path_length(optimized_path)
        
        print(f"\nMETRICS:")
        print(f"  Smoothness: {smooth_before:.4f} -> {smooth_after:.4f} ({(smooth_after/smooth_before-1)*100:+.1f}%)")
        print(f"  Path length: {length_before:.4f} -> {length_after:.4f} ({(length_after/length_before-1)*100:+.1f}%)")
    else:
        print("Optimization failed!")
        exit(1)
