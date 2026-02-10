#!/usr/bin/env python3
import argparse
import numpy as np
import os
import torch
import robot_tools

def densify_path(path: np.ndarray, n_interp: int) -> np.ndarray:
    """
    Insert n_interp points between each pair of waypoints.
    If path has N waypoints, output has N + (N-1)*n_interp points.
    """
    if n_interp <= 0 or len(path) < 2:
        return np.asarray(path)

    out = [path[0]]
    for i in range(len(path) - 1):
        q1, q2 = path[i], path[i + 1]
        for s in range(1, n_interp + 1):
            alpha = s / (n_interp + 1)
            out.append(q1 + (q2 - q1) * alpha)
        out.append(q2)
    return np.asarray(out)

def resample_path(path: np.ndarray, n_points: int) -> np.ndarray:
    """
    Resample path to be equidistant with exactly n_points.
    """
    if len(path) < 2:
        return np.tile(path[0], (n_points, 1))
        
    diffs = np.linalg.norm(path[1:] - path[:-1], axis=1)
    cum_dist = np.concatenate(([0], np.cumsum(diffs)))
    total_dist = cum_dist[-1]
    
    if total_dist < 1e-8:
        return np.tile(path[0], (n_points, 1))
        
    target_dist = np.linspace(0, total_dist, n_points)
    new_path = np.zeros((n_points, path.shape[1]))
    for i in range(path.shape[1]):
        new_path[:, i] = np.interp(target_dist, cum_dist, path[:, i])
    return new_path

class TrajectoryEvaluator:
    """
    Unified evaluator for 2D trajectories.
    Aligned with 7D Franka evaluator pattern but native to 2D robot_tools.
    """
    def __init__(self, world_yaml, device="cuda"):
        self.problem = robot_tools.PlanningProblem(world_yaml)
        self.device = device

    def is_collision_free(self, q: np.ndarray) -> bool:
        self.problem.robot.set_link_angles(q)
        fp = self.problem.robot.get_current_polygon()
        return not any(fp.intersect(o) for o in self.problem.workspace.obstacles)

    def evaluate(self, path, solve_time=0.0):
        path = np.asarray(path)
        
        # 1. Path Length (on original path)
        diffs = path[1:] - path[:-1]
        path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
        
        # 2. Collision Success & 3. Min Clearance (on DENSE path for accuracy)
        target_points = 512
        dense = resample_path(path, target_points)
        
        success = True
        min_clearance = 1e6
        for q in dense:
            self.problem.robot.set_link_angles(q)
            shapes = self.problem.robot.get_collision_shapes()
            
            # Use all shapes for clearance check
            step_min_clr = 1e6
            for shape in shapes:
                clr = self.problem.workspace.get_clearance(shape)
                step_min_clr = min(step_min_clr, clr)
            
            min_clearance = min(min_clearance, step_min_clr)
            if success and step_min_clr <= 0:
                success = False
        
        # 4. Smoothness (on ORIGINAL path to avoid interpolation artifacts)
        if len(path) > 2:
            # Average spacing on original path
            h = path_length / (len(path) - 1)
            # (q_{i+1} - 2q_i + q_{i-1}) / h^2
            accel_norms = np.linalg.norm((path[2:] - 2*path[1:-1] + path[:-2]) / (h**2), axis=1)
            sm_mean = float(np.mean(accel_norms))
            sm_p95 = float(np.percentile(accel_norms, 95))
        else:
            sm_mean = 0.0
            sm_p95 = 0.0

        return {
            "success": success,
            "path_length": path_length,
            "min_clearance": min_clearance,
            "smoothness": sm_mean,
            "smoothness_p95": sm_p95,
            "solve_time": solve_time
        }