#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict
import time

import robot_tools
from geometry import ControlPoint
from common.models import load_model as global_load_model

def load_2d_model(path, device, n_layers=20, n_hidden=128):
    """Load the trained NDF model."""
    return global_load_model(path, device)

class NDFOptimizer2D:
    """
    NDF-based core utilities for 2D joint space.
    This class serves as a container for the model, robot problem, 
    and kinematic managers used by specialized planners (like CasADiPlanner).
    """

    def __init__(self, model_path, world_file, device, n_layers=20, n_hidden=128):
        self.device = device
        self.nq = 2
        
        # Load the neural distance field model
        self.model = load_2d_model(model_path, self.device, n_layers=n_layers, n_hidden=n_hidden)
        self.model.eval()
        
        # Load robot and workspace definitions
        self.problem = robot_tools.PlanningProblem(world_file)
        
        # Initialize 2D-native ControlPoint manager for Forward Kinematics
        self.cp_manager = ControlPoint(
            base_position=[self.problem.robot._base_position.x, self.problem.robot._base_position.y],
            link_lengths=self.problem.robot._link_lengths.tolist(),
            device=self.device
        )
        
        # Fetch joint limits from the robot model (Aligned with 7D pattern)
        # Default is [0, 2*pi] for each joint.
        self.lower = self.problem.robot.joint_limits[:, 0]
        self.upper = self.problem.robot.joint_limits[:, 1]

    def resample_path(self, path: np.ndarray, num_waypoints: int) -> np.ndarray:
        """
        Resample a path to a fixed number of waypoints.
        NOTE: No unwrapping is performed here because we enforce [0, 2pi] 
        boxed C-space without wrapping to align with 7D Franka logic.
        """
        path = np.asarray(path)
        N = len(path)
        if N == num_waypoints:
            return path
        
        old_indices = np.arange(N)
        new_indices = np.linspace(0, N - 1, num_waypoints)
        resampled = np.zeros((num_waypoints, self.nq))
        for j in range(self.nq):
            resampled[:, j] = np.interp(new_indices, old_indices, path[:, j])
        return resampled

if __name__ == "__main__":
    # Minimal CLI for basic verification
    parser = argparse.ArgumentParser(description="2D NDF Core Utility")
    parser.add_argument("--world", type=str, default="config/block_world1.yaml")
    parser.add_argument("--model", type=str, default="output/block_world1/model_final.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing 2D NDF core on {device}...")
    
    try:
        opt = NDFOptimizer2D(args.model, args.world, device)
        print(f"Successfully loaded NDF model and 2D Robot Problem.")
        print(f"Joint Limits Lower: {opt.lower}")
        print(f"Joint Limits Upper: {opt.upper}")
    except Exception as e:
        print(f"Initialization failed: {e}")
