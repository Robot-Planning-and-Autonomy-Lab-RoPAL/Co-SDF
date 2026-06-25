#!/usr/bin/env python3
import math
import numpy as np
import scipy.ndimage as ndi
from scipy.interpolate import RegularGridInterpolator

def _point_to_seg_dist(px, py, ax, ay, bx, by):
	vx, vy = bx-ax, by-ay
	wx, wy = px-ax, py-ay
	vv = vx*vx + vy*vy
	if vv == 0.0:
		return math.hypot(px-ax, py-ay)
	t = (wx*vx + wy*vy) / vv
	t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
	cx, cy = ax + t*vx, ay + t*vy
	return math.hypot(px-cx, py-cy)

def _orient(ax,ay,bx,by,cx,cy):
	return (bx-ax)*(cy-ay) - (by-ay)*(cx-ax)

def _on_seg(ax,ay,px,py,bx,by):
	return (min(ax,bx)-1e-12 <= px <= max(ax,bx)+1e-12 and
			min(ay,by)-1e-12 <= py <= max(ay,by)+1e-12)
			
def _seg_intersect(a,b,c,d):
	ax,ay=a; bx,by=b; cx,cy=c; dx,dy=d
	o1=_orient(ax,ay,bx,by,cx,cy); o2=_orient(ax,ay,bx,by,dx,dy)
	o3=_orient(cx,cy,dx,dy,ax,ay); o4=_orient(cx,cy,dx,dy,bx,by)
	if (o1*o2<0) and (o3*o4<0): return True
	if abs(o1)<1e-12 and _on_seg(ax,ay,cx,cy,bx,by): return True
	if abs(o2)<1e-12 and _on_seg(ax,ay,dx,dy,bx,by): return True
	if abs(o3)<1e-12 and _on_seg(cx,cy,ax,ay,dx,dy): return True
	if abs(o4)<1e-12 and _on_seg(cx,cy,bx,by,dx,dy): return True
	return False

def _drop_periodic_duplicates(theta, grid):
	
	t1, t2 = theta; g = grid
	if np.isclose((t1[-1]-t1[0]) % (2*np.pi), 0.0) and len(t1) > 1:
		t1 = t1[:-1]; g = g[:-1, :]
	if np.isclose((t2[-1]-t2[0]) % (2*np.pi), 0.0) and len(t2) > 1:
		t2 = t2[:-1]; g = g[:, :-1]
	return (t1, t2), g

def cobs_sdf_on_torus(theta, c_space):

	(t1, t2), obs0 = _drop_periodic_duplicates(theta, c_space)
	obs = (obs0 > 0)
	dth1 = float(t1[1]-t1[0]); dth2 = float(t2[1]-t2[0])
	H, W  = obs.shape
	
	obs_t   = np.tile(obs, (3, 3))
	d_out_t = ndi.distance_transform_edt(~obs_t, sampling=(dth1, dth2))
	d_in_t  = ndi.distance_transform_edt( obs_t, sampling=(dth1, dth2))
	sdf_t   = d_out_t - d_in_t
	
	sdf = sdf_t[H:2*H, W:2*W]
	
	return (t1, t2), sdf

def wrap_samples_to_period(samples, theta):

	t1, t2 = theta
	out = samples.copy()
	out[:,0] = (out[:,0]-t1[0]) % (t1[-1]-t1[0]) + t1[0]
	out[:,1] = (out[:,1]-t2[0]) % (t2[-1]-t2[0]) + t2[0]
	return out

def cspace_sdf_at_samples(theta, sdf_grid, samples_wrapped):

	t1, t2 = theta
	interp = RegularGridInterpolator((t1, t2), sdf_grid,
									bounds_error=False, fill_value=None)
	return interp(samples_wrapped)

# ── Analytical (non-grid) C-space SDF ────────────────────────────────

def analytical_cspace_sdf(samples, arm, workspace):
	"""Compute exact C-space SDF at arbitrary (θ₁, θ₂) samples.

	For each configuration the robot's FK is evaluated, collision shapes
	are built, and the minimum signed clearance to *all* obstacles is
	returned.  Positive → free, negative → in collision.

	Parameters
	----------
	samples : ndarray, shape (N, 2)
		Joint-angle pairs.
	arm : robot_tools.RobotArm2D
		The 2-link arm (mutated in-place via ``set_link_angles``).
	workspace : robot_tools.Workspace2D
		Workspace containing the obstacles.

	Returns
	-------
	sdf : ndarray, shape (N,)
		Signed distance at each sample.
	"""
	N = samples.shape[0]
	sdf = np.empty(N)
	for i in range(N):
		arm.set_link_angles(samples[i])
		shapes = arm.get_collision_shapes()
		min_d = float('inf')
		for s in shapes:
			d = workspace.get_clearance(s)
			if d < min_d:
				min_d = d
		sdf[i] = min_d
	return sdf


def analytical_cspace_sdf_grid(theta, arm, workspace):
	"""Build a full analytical-SDF grid for visualisation.

	Parameters
	----------
	theta : (array, array)
		``(theta1, theta2)`` axis tick arrays.
	arm : robot_tools.RobotArm2D
	workspace : robot_tools.Workspace2D

	Returns
	-------
	theta : (array, array)
	sdf   : ndarray, shape (len(theta1), len(theta2))
	"""
	t1, t2 = theta
	grid = np.meshgrid(t1, t2, indexing='ij')       # (H, W) each
	samples = np.column_stack([grid[0].ravel(),
							   grid[1].ravel()])     # (H*W, 2)
	sdf_flat = analytical_cspace_sdf(samples, arm, workspace)
	return (t1, t2), sdf_flat.reshape(len(t1), len(t2))