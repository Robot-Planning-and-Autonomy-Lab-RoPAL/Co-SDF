import torch
import numpy as np

class FrankaFK(torch.nn.Module):
    """Differentiable Forward Kinematics for Franka FR3 robot."""
    
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        
        # Link offsets and rotations from fr3.xml
        # Values simplified/extracted for standard FK chain
        # format: (pos_x, pos_y, pos_z, q_w, q_x, q_y, q_z)
        self.offsets = [
            (0, 0, 0.333, 1, 0, 0, 0),        # Base to Link 1 (Joint 1)
            (0, 0, 0, 0.7071, -0.7071, 0, 0), # Link 1 to Link 2 (Joint 2) -90 X
            (0, -0.316, 0, 0.7071, 0.7071, 0, 0), # Link 2 to Link 3 (Joint 3) +90 X
            (0.0825, 0, 0, 0.7071, 0.7071, 0, 0), # Link 3 to Link 4 (Joint 4) +90 X
            (-0.0825, 0.384, 0, 0.7071, -0.7071, 0, 0), # Link 4 to Link 5 (Joint 5) -90 X
            (0, 0, 0, 0.7071, 0.7071, 0, 0),  # Link 5 to Link 6 (Joint 6) +90 X
            (0.088, 0, 0, 0.7071, 0.7071, 0, 0),  # Link 6 to Link 7 (Joint 7) +90 X
            (0, 0, 0.107, 1, 0, 0, 0)         # Link 7 to Hand
        ]
        
        self.offsets = torch.tensor(self.offsets, device=device).float()
        
    def _quat_to_matrix(self, quat):
        """Convert batch of quaternions (w, x, y, z) to rotation matrices."""
        w, x, y, z = quat.unbind(-1)
        
        tx = 2.0 * x
        ty = 2.0 * y
        tz = 2.0 * z
        twx = tx * w
        twy = ty * w
        twz = tz * w
        txx = tx * x
        txy = ty * x
        txz = tz * x
        tyy = ty * y
        tyz = tz * y
        tzz = tz * z
        
        res = torch.stack([
            1.0 - (tyy + tzz), txy - twz, txz + twy,
            txy + twz, 1.0 - (txx + tzz), tyz - twx,
            txz - twy, tyz + twx, 1.0 - (txx + tyy)
        ], dim=-1).reshape(-1, 3, 3)
        return res

    def _get_joint_transform(self, axis, angle):
        """Get rotation matrix for a joint rotation around an axis."""
        # Franka FR3 joints are all around Z axis in their local frame
        # (after the Mujoco pre-rotations are applied)
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        zero = torch.zeros_like(angle)
        one = torch.ones_like(angle)
        
        # Batch rotation around Z
        R = torch.stack([
            cos, -sin, zero,
            sin,  cos, zero,
            zero, zero, one
        ], dim=-1).reshape(-1, 3, 3)
        return R

    def forward(self, q):
        """Compute forward kinematics for batch of q (B, 7).
        Returns: Tuple of joint positions (B, 8, 3) 
        and full transform matrices (B, 8, 4, 4)
        """
        B = q.shape[0]
        device = q.device
        
        transforms = []
        # Identity for base
        current_T = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
        
        joint_positions = []
        
        for i in range(7):
            # 1. Apply static offset ( Mujoco position and quat)
            offset_pos = self.offsets[i, :3]
            offset_quat = self.offsets[i, 3:]
            
            # Static translation matrix
            T_static = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
            T_static[:, :3, 3] = offset_pos
            T_static[:, :3, :3] = self._quat_to_matrix(offset_quat.unsqueeze(0)).repeat(B, 1, 1)
            
            current_T = torch.bmm(current_T, T_static)
            
            # 2. Apply joint rotation (Z-axis)
            R_joint = self._get_joint_transform(None, q[:, i])
            T_joint = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
            T_joint[:, :3, :3] = R_joint
            
            current_T = torch.bmm(current_T, T_joint)
            
            transforms.append(current_T)
            joint_positions.append(current_T[:, :3, 3])
            
        # Add end-effector offset
        offset_pos = self.offsets[7, :3]
        T_ee = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
        T_ee[:, :3, 3] = offset_pos
        current_T = torch.bmm(current_T, T_ee)
        transforms.append(current_T)
        joint_positions.append(current_T[:, :3, 3])
        
        return torch.stack(joint_positions, dim=1), torch.stack(transforms, dim=1)

class ControlPointManager:
    """Manages control points along the robot links."""
    
    def __init__(self, device="cuda", n_points_per_link=2):
        self.fk = FrankaFK(device=device)
        self.device = device
        self.n_points_per_link = n_points_per_link
        
        # Define relative control points for each link
        self.local_points = []
        
        # Link 1 (Joint 1 to 2)
        p1 = torch.zeros(n_points_per_link, 3)
        p1[:, 2] = torch.linspace(0, 0.333, n_points_per_link) # Along Z
        self.local_points.append(p1)
        
        # Link 2 to 3 segment
        p2 = torch.zeros(n_points_per_link, 3)
        p2[:, 1] = torch.linspace(0, -0.316, n_points_per_link) 
        self.local_points.append(p2)
        
        # Link 3 to 4 segment
        p3 = torch.zeros(n_points_per_link, 3)
        p3[:, 0] = torch.linspace(0, 0.0825, n_points_per_link)
        self.local_points.append(p3)
        
        # Link 4 to 5 segment
        p4 = torch.zeros(n_points_per_link, 3)
        p4[:, 1] = torch.linspace(0, 0.384, n_points_per_link)
        self.local_points.append(p4)
        
        # Link 5 to 6 - offset 0
        p5 = torch.zeros(n_points_per_link, 3) 
        self.local_points.append(p5)
        
        # Link 6 to 7
        p6 = torch.zeros(n_points_per_link, 3)
        p6[:, 0] = torch.linspace(0, 0.088, n_points_per_link)
        self.local_points.append(p6)
        
        # Link 7 to EE
        p7 = torch.zeros(n_points_per_link, 3)
        p7[:, 2] = torch.linspace(0, 0.107, n_points_per_link)
        self.local_points.append(p7)
        
        # Move all to device
        self.local_points = [p.to(device) for p in self.local_points]
        
    def get_jacobian_determinant(self, q):
        """Compute sqrt(det(J^T J)) for the control points to use in rejection sampling."""
        # q: (B, 7)
        q = q.clone().detach().requires_grad_(True)
        points = self.get_control_points(q) # (B, 14, 3)
        flat_points = points.reshape(q.shape[0], -1) # (B, 42)
        
        B = q.shape[0]
        D_out = flat_points.shape[1] # 42
        D_in = q.shape[1] # 7
        
        # Compute Jacobian J = d(flat_points)/dq
        jac = torch.zeros(B, D_out, D_in, device=self.device)
        
        for i in range(D_out):
            if q.grad is not None:
                q.grad.zero_()
            
            grad_outputs = torch.zeros_like(flat_points)
            grad_outputs[:, i] = 1.0
            
            flat_points.backward(grad_outputs, retain_graph=True)
            jac[:, i, :] = q.grad
            
        # Volume element: sqrt(det(J^T J))
        jtj = torch.matmul(jac.transpose(1, 2), jac)
        eps = 1e-4 * torch.eye(D_in, device=self.device).unsqueeze(0)
        det = torch.linalg.det(jtj + eps)
        
        return torch.sqrt(torch.abs(det))

    def get_control_points(self, q):
        """Compute workspace coordinates of all control points for batch q."""
        joint_pos, transforms = self.fk(q) # transforms: (B, 8, 4, 4)
        
        all_points = []
        for i, lp in enumerate(self.local_points):
            # lp: (N, 3)
            # transform for link i: transforms[:, i]
            T = transforms[:, i] # (B, 4, 4)
            
            # Batch transform points: x' = R*x + t
            # T[:, :3, :3] is (B, 3, 3), lp is (N, 3)
            # Use broadcasting or matmul
            R = T[:, :3, :3] # (B, 3, 3)
            t = T[:, :3, 3].unsqueeze(1) # (B, 1, 3)
            
            # (B, N, 3) = (B, N, 3) @ (B, 3, 3) ? No.
            # R @ lp^T -> (B, 3, 3) @ (3, N) -> (B, 3, N)
            pts_world = torch.matmul(R, lp.t()).transpose(1, 2) + t
            all_points.append(pts_world)
            
        return torch.cat(all_points, dim=1) # (B, TotalPoints, 3)
