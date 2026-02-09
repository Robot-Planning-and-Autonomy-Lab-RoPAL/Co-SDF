#!/usr/bin/env python3
import torch
import mujoco
import torch.nn as nn
import numpy as np
from utils import ControlPoint

class WorkspaceGridSDF(nn.Module):
    """
    Workspace SDF baseline using grid interpolation.
    
    Architecture:
    1. FK: q (7D) -> control points (9, 3)
    2. For each control point: trilinear interpolation to get distance
    3. Return minimum distance across all points
    
    Uses manual trilinear interpolation (fully differentiable, L4CasADi compatible).
    """
    def __init__(self, grid_resolution=64, workspace_bounds=None, device="cuda"):
        super().__init__()
        self.device = device
        self.cp_manager = ControlPoint(device=device)
        
        if workspace_bounds is None:
            workspace_bounds = [-1.0, 1.0, -1.0, 1.0, -0.2, 1.2]
        
        self.register_buffer('bounds', torch.tensor(workspace_bounds, device=device))
        self.grid_res = grid_resolution
        
        empty_grid = torch.zeros(1, 1, grid_resolution, grid_resolution, grid_resolution, device=device)
        self.register_buffer('sdf_grid', empty_grid)
    
    def compute_grid(self, scene_xml, samples_per_dim=64):
        """
        Pre-compute SDF grid using MuJoCo collision detection.
        This is done OFFLINE, not during optimization.
        """
        import mujoco
        import os
        
        print(f"Computing workspace SDF grid ({samples_per_dim}³ = {samples_per_dim**3} points)...")
        
        mj_model = mujoco.MjModel.from_xml_path(os.path.abspath(scene_xml))
        mj_data = mujoco.MjData(mj_model)
        
        robot_body_ids = set()
        base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if base_id >= 0:
            for i in range(mj_model.nbody):
                curr = i
                while curr > 0:
                    if curr == base_id:
                        robot_body_ids.add(i)
                        break
                    curr = mj_model.body_parentid[curr]
        
        is_robot_body = np.zeros(mj_model.nbody, dtype=bool)
        if robot_body_ids:
            is_robot_body[list(robot_body_ids)] = True
        
        x_min, x_max, y_min, y_max, z_min, z_max = self.bounds.cpu().numpy()
        x = np.linspace(x_min, x_max, samples_per_dim)
        y = np.linspace(y_min, y_max, samples_per_dim)
        z = np.linspace(z_min, z_max, samples_per_dim)
        grid = np.zeros((samples_per_dim, samples_per_dim, samples_per_dim)) # Shape will be (Z, Y, X)
        
        total_pts = samples_per_dim ** 3
        print_every = total_pts // 20
        
        for k, zk in enumerate(z): # Z first
            for j, yj in enumerate(y): # Y second
                for i, xi in enumerate(x): # X last
                    idx = k * samples_per_dim * samples_per_dim + j * samples_per_dim + i
                    if idx % print_every == 0:
                        print(f"  Progress: {idx}/{total_pts} ({100*idx/total_pts:.1f}%)")
                    
                    dist = self._query_point_distance(mj_model, mj_data, 
                                                     np.array([xi, yj, zk]), 
                                                     is_robot_body)
                    grid[k, j, i] = dist # Index as (Z, Y, X)
        
        self.sdf_grid.copy_(torch.from_numpy(grid).float().to(self.device).unsqueeze(0).unsqueeze(0))
        print(f"Grid computed! Min: {grid.min():.4f}, Max: {grid.max():.4f}")
        
        return grid
    
    def _query_point_distance(self, mj_model, mj_data, point, is_robot_body):
        """
        Query distance from a single point to obstacles in MuJoCo scene.
        Returns positive for free space, negative for penetration.
        """
        min_dist = 1.0 
        
        for geom_id in range(mj_model.ngeom):
            geom_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            
            if geom_name and ("obs_" in geom_name or "table" in geom_name or "floor" in geom_name):
                geom_pos = mj_model.geom_pos[geom_id]
                geom_size = mj_model.geom_size[geom_id]
                geom_quat = mj_model.geom_quat[geom_id]
                g_type = mj_model.geom_type[geom_id]
                
                # Transform point to local frame
                # local_pt = R.T * (point - geom_pos)
                rel_pt = point - geom_pos
                neg_quat = np.zeros(4)
                mujoco.mju_negQuat(neg_quat, geom_quat)
                local_pt = np.zeros(3)
                mujoco.mju_rotVecQuat(local_pt, rel_pt, neg_quat)

                if g_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                    dist = np.linalg.norm(local_pt) - geom_size[0]
                elif g_type == mujoco.mjtGeom.mjGEOM_BOX:
                    q = np.abs(local_pt) - geom_size
                    dist = np.linalg.norm(np.maximum(q, 0)) + min(np.max(q), 0)
                elif g_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    # Cylinder axis is always Z in local frame
                    dist_h = np.abs(local_pt[2]) - geom_size[1]
                    dist_r = np.linalg.norm(local_pt[:2]) - geom_size[0]
                    dist = np.maximum(dist_h, dist_r)
                elif g_type == mujoco.mjtGeom.mjGEOM_PLANE:
                    dist = point[2] - geom_pos[2]
                else:
                    dist = 0.5
                
                min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _trilinear_interpolate(self, grid, coords):
        """
        Manual trilinear interpolation that's fully differentiable.
        
        Args:
            grid: (1, 1, D, H, W) SDF grid
            coords: (N, 3) normalized coordinates in [-1, 1]
        
        Returns:
            values: (N,) interpolated values
        """
        _, _, D, H, W = grid.shape
        N = coords.shape[0]
        
        x = (coords[:, 0] + 1) * (W - 1) / 2
        y = (coords[:, 1] + 1) * (H - 1) / 2
        z = (coords[:, 2] + 1) * (D - 1) / 2
        
        x0 = torch.floor(x).long().clamp(0, W-2)
        y0 = torch.floor(y).long().clamp(0, H-2)
        z0 = torch.floor(z).long().clamp(0, D-2)
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1
        
        xd = (x - x0.float()).unsqueeze(1)
        yd = (y - y0.float()).unsqueeze(1)
        zd = (z - z0.float()).unsqueeze(1)
        
        c000 = grid[0, 0, z0, y0, x0].unsqueeze(1)
        c001 = grid[0, 0, z0, y0, x1].unsqueeze(1)
        c010 = grid[0, 0, z0, y1, x0].unsqueeze(1)
        c011 = grid[0, 0, z0, y1, x1].unsqueeze(1)
        c100 = grid[0, 0, z1, y0, x0].unsqueeze(1)
        c101 = grid[0, 0, z1, y0, x1].unsqueeze(1)
        c110 = grid[0, 0, z1, y1, x0].unsqueeze(1)
        c111 = grid[0, 0, z1, y1, x1].unsqueeze(1)
        
        c00 = c000 * (1 - xd) + c001 * xd
        c01 = c010 * (1 - xd) + c011 * xd
        c10 = c100 * (1 - xd) + c101 * xd
        c11 = c110 * (1 - xd) + c111 * xd
        
        c0 = c00 * (1 - yd) + c01 * yd
        c1 = c10 * (1 - yd) + c11 * yd
        
        c = c0 * (1 - zd) + c1 * zd
        
        return c.squeeze(1)
    
    def forward(self, q):
        """
        Compute SDF for robot configurations.
        
        Args:
            q: (N, 7) joint configurations
        
        Returns:
            dists: (N, 1) signed distances
        """
        control_points = self.cp_manager.get_control_points(q)  # (N, 9, 3)
        N, num_cp, _ = control_points.shape
        
        x_min, x_max, y_min, y_max, z_min, z_max = self.bounds
        
        pts = control_points.reshape(-1, 3)
        
        x_norm = 2 * (pts[:, 0] - x_min) / (x_max - x_min) - 1
        y_norm = 2 * (pts[:, 1] - y_min) / (y_max - y_min) - 1
        z_norm = 2 * (pts[:, 2] - z_min) / (z_max - z_min) - 1
        pts_norm = torch.stack([x_norm, y_norm, z_norm], dim=1)
        pts_norm = pts_norm.clamp(-1 + 1e-6, 1 - 1e-6)

        sampled = self._trilinear_interpolate(self.sdf_grid, pts_norm)  # (N*9,)
        
        sampled = sampled.reshape(N, num_cp)  # (N, 9)
        
        # Switch to hard min for accurate comparison with GT
        min_dists, _ = torch.min(sampled, dim=1)
        
        return min_dists.unsqueeze(1)  # (N, 1)
