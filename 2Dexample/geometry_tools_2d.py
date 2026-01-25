#!/usr/bin/env python3
"""
Differentiable 2D Forward Kinematics for 2-DOF planar arm
Analogous to franka_fr3/geometry_tools.py but for 2D case
"""
import torch
import torch.nn as nn
import numpy as np


class Arm2DFK(nn.Module):
    """Differentiable Forward Kinematics for 2-DOF planar arm."""
    
    def __init__(self, base_position=[5.0, 5.0], link_lengths=[2.1, 2.1], device="cuda"):
        super().__init__()
        self.device = device
        self.base = torch.tensor(base_position, dtype=torch.float32, device=device)
        self.lengths = torch.tensor(link_lengths, dtype=torch.float32, device=device)
        
    def forward(self, q):
        """
        Compute forward kinematics for batch of configurations.
        
        Args:
            q: (B, 2) tensor of joint angles [theta1, theta2]
            
        Returns:
            positions: (B, 3, 2) tensor of [base, joint1, end_effector] positions
        """
        B = q.shape[0]
        device = q.device
        
        # Base position
        p0 = self.base.unsqueeze(0).expand(B, -1)  # (B, 2)
        
        # Joint 1 position
        x1 = p0[:, 0] + self.lengths[0] * torch.cos(q[:, 0])
        y1 = p0[:, 1] + self.lengths[0] * torch.sin(q[:, 0])
        p1 = torch.stack([x1, y1], dim=-1)  # (B, 2)
        
        # End effector position
        x2 = p1[:, 0] + self.lengths[1] * torch.cos(q[:, 0] + q[:, 1])
        y2 = p1[:, 1] + self.lengths[1] * torch.sin(q[:, 0] + q[:, 1])
        p2 = torch.stack([x2, y2], dim=-1)  # (B, 2)
        
        # Stack all positions
        positions = torch.stack([p0, p1, p2], dim=1)  # (B, 3, 2)
        
        return positions


class ControlPointManager2D(nn.Module):
    """
    Extracts control points from 2D arm configurations.
    Analogous to franka_fr3's ControlPointManager but for 2D.
    """
    
    def __init__(self, base_position=[5.0, 5.0], link_lengths=[2.1, 2.1], 
                 n_points_per_link=2, device="cuda"):
        super().__init__()
        self.device = device
        self.n_points_per_link = n_points_per_link
        self.fk = Arm2DFK(base_position, link_lengths, device)
        
    def get_control_points(self, q):
        """
        Extract control points along the arm links.
        
        Args:
            q: (B, 2) tensor of joint angles
            
        Returns:
            points: (B, n_points, 2) tensor of control point positions
                   For n_points_per_link=2: returns 4 points total
                   - 2 points on link1 (at 0.33 and 0.67 of length)
                   - 2 points on link2 (at 0.33 and 0.67 of length)
        """
        B = q.shape[0]
        
        # Get joint positions
        positions = self.fk(q)  # (B, 3, 2) - [base, joint1, ee]
        p0 = positions[:, 0, :]  # base
        p1 = positions[:, 1, :]  # joint1
        p2 = positions[:, 2, :]  # end effector
        
        control_points = []
        
        # Points on link 1 (from base to joint1)
        for i in range(1, self.n_points_per_link + 1):
            ratio = i / (self.n_points_per_link + 1)
            point = p0 + ratio * (p1 - p0)
            control_points.append(point)
        
        # Points on link 2 (from joint1 to ee)
        for i in range(1, self.n_points_per_link + 1):
            ratio = i / (self.n_points_per_link + 1)
            point = p1 + ratio * (p2 - p1)
            control_points.append(point)
        
        # Stack all points: (B, n_points, 2)
        points = torch.stack(control_points, dim=1)
        
        return points
    
    def get_n_points(self):
        """Total number of control points."""
        return 2 * self.n_points_per_link  # 2 links


def test_fk():
    """Test the FK implementation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create FK
    fk = Arm2DFK(base_position=[5.0, 5.0], link_lengths=[2.1, 2.1], device=device)
    
    # Test configuration
    q = torch.tensor([[0.0, 0.0], [np.pi/2, 0.0], [0.0, np.pi/2]], device=device)
    
    positions = fk(q)
    print("FK Test:")
    print("Input q:", q)
    print("Positions shape:", positions.shape)
    print("Positions:\n", positions)
    
    # Test gradient
    q_grad = torch.tensor([[np.pi/4, np.pi/4]], device=device, requires_grad=True)
    pos = fk(q_grad)
    loss = pos[:, 2, :].sum()  # Sum of end effector position
    loss.backward()
    print("\nGradient test:")
    print("q gradient:", q_grad.grad)
    print("✓ FK is differentiable")


def test_control_points():
    """Test control point extraction."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cp_manager = ControlPointManager2D(
        base_position=[5.0, 5.0],
        link_lengths=[2.1, 2.1],
        n_points_per_link=2,
        device=device
    )
    
    q = torch.tensor([[0.0, 0.0], [np.pi/2, 0.0]], device=device)
    points = cp_manager.get_control_points(q)
    
    print("\nControl Points Test:")
    print("Input q:", q)
    print("Control points shape:", points.shape)
    print("Control points:\n", points)
    print(f"✓ Total {cp_manager.get_n_points()} control points extracted")


if __name__ == "__main__":
    test_fk()
    test_control_points()
