#!/usr/bin/env python3

import numpy as np
import scipy.ndimage as ndi
from scipy.interpolate import RegularGridInterpolator

# =========================================================
# C-space (θ1, θ2) → C-obs 的 Signed Distance Field (SDF)
# =========================================================
# - 每个关节角度在配置空间里是一个周期变量，定义在 [0, 2π) 上。
# - 配置空间是环面 T² = S¹ × S¹，两个角度 θ1, θ2 都是模 2π 周期。
# - 我们要计算：某个配置 q = (θ1, θ2) 到最近 C-obs 边界的角度距离
#     φ(q) > 0 → 自由区，数值表示与障碍的“角度裕度”
#     φ(q) < 0 → 障碍区，数值表示“穿透深度”（弧度）
# =========================================================

# ---------- 基础几何 ----------
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

# ---------- 线段到多边形的最近距离（碰撞=0） ----------
def _drop_periodic_duplicates(theta, grid):
	"""去掉 linspace(..., endpoint=True) 带来的 2π 重复行/列"""
	t1, t2 = theta; g = grid
	if np.isclose((t1[-1]-t1[0]) % (2*np.pi), 0.0) and len(t1) > 1:
		t1 = t1[:-1]; g = g[:-1, :]
	if np.isclose((t2[-1]-t2[0]) % (2*np.pi), 0.0) and len(t2) > 1:
		t2 = t2[:-1]; g = g[:, :-1]
	return (t1, t2), g

def cobs_sdf_on_torus(theta, c_space):
	"""
	输入：
		theta   - (θ1 网格, θ2 网格)，都在 [0, 2π) 上
		c_space - 二维数组，c_space>0 表示 C-obs，占据的配置
	输出：
		(θ1, θ2), sdf - 同样大小的 SDF 网格，单位=弧度
	方法：
		1. 把 C-obs 二值化：obs=True 表示障碍，False 表示自由
		2. 在环面上做欧式距离变换：
				- d_out: 点到自由区的最短角度距离
				- d_in : 点到障碍区的最短角度距离
				- sdf   = d_out - d_in
		3. 为了处理周期边界，在 θ1, θ2 上做 3×3 平铺，再取中心块
	"""
	(t1, t2), obs0 = _drop_periodic_duplicates(theta, c_space)
	obs = (obs0 > 0)
	dth1 = float(t1[1]-t1[0]); dth2 = float(t2[1]-t2[0])
	H, W  = obs.shape
	
	# 3x3 平铺实现周期边界
	obs_t   = np.tile(obs, (3, 3))
	d_out_t = ndi.distance_transform_edt(~obs_t, sampling=(dth1, dth2))
	d_in_t  = ndi.distance_transform_edt( obs_t, sampling=(dth1, dth2))
	sdf_t   = d_out_t - d_in_t
	
	# 取中心块
	sdf = sdf_t[H:2*H, W:2*W]
	return (t1, t2), sdf

def wrap_samples_to_period(samples, theta):
	"""
	把 (θ1, θ2) 的样本点 wrap 回 [tmin, tmax) 区间。
	例如如果 tmin=0, tmax=2π，那么所有角度都归一化到 [0, 2π)。
	"""
	t1, t2 = theta
	out = samples.copy()
	out[:,0] = (out[:,0]-t1[0]) % (t1[-1]-t1[0]) + t1[0]
	out[:,1] = (out[:,1]-t2[0]) % (t2[-1]-t2[0]) + t2[0]
	return out

def cspace_sdf_at_samples(theta, sdf_grid, samples_wrapped):
	"""在样本点上双线性插值读取 C-space SDF（弧度）"""
	t1, t2 = theta
	interp = RegularGridInterpolator((t1, t2), sdf_grid,
									bounds_error=False, fill_value=None)
	return interp(samples_wrapped)