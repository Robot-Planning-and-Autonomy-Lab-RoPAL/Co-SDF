#!/usr/bin/env python3
import os
import time
import argparse
import robot_tools
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from skimage import measure
from geometry_tools_2d import ControlPointManager2D

_GLOBAL_PROBLEM = None

def _init_oracle(world_path):
	global _GLOBAL_PROBLEM
	_GLOBAL_PROBLEM = robot_tools.PlanningProblem(world_path)
	
def _oracle_worker(q):
	p = _GLOBAL_PROBLEM
	p.robot.set_link_angles(q)
	fp = p.robot.get_current_polygon()
	for o in p.workspace.obstacles:
		if fp.intersect(o):
			return -1
	return 1

def extract_boundary_edges(c_space, theta):
	contours = measure.find_contours(c_space, 0.5)
	t1, t2 = theta
	dth1 = (t1[-1] - t1[0]) / (len(t1) - 1)
	dth2 = (t2[-1] - t2[0]) / (len(t2) - 1)
	edges = []
	for contour in contours:
		pts = np.zeros_like(contour)
		pts[:, 0] = t1[0] + contour[:, 0] * dth1
		pts[:, 1] = t2[0] + contour[:, 1] * dth2
		for i in range(len(pts) - 1):
			edges.append((pts[i], pts[i+1]))
	return edges

def sample_boundary_uniform(edges, n_pts):
	if not edges or n_pts <= 0: return np.zeros((0, 2))
	lengths = np.array([np.linalg.norm(p2 - p1) for p1, p2 in edges])
	total_len = np.sum(lengths)
	if total_len == 0: return np.zeros((0, 2))
	probs = lengths / total_len
	edge_indices = np.random.choice(len(edges), size=n_pts, p=probs)
	samples = []
	for idx in edge_indices:
		p1, p2 = edges[idx]
		t = np.random.rand()
		samples.append(p1 + t * (p2 - p1))
	return np.array(samples)

def generate_gndf_dataset(n_pts, bounds, world_file, near_ratio=0.5, sigma=0.01, 
                         c_space=None, theta=None, n_points_per_link=2):
	"""
	Consolidated Sampling and Augmentation.
	1) Boundary & AABB Sampling (q space)
	2) Compute Link xyz features (Workspace space) and concatenate with q
	"""
	dim = len(bounds)
	lo = np.array([b[0] for b in bounds], dtype=float)
	hi = np.array([b[1] for b in bounds], dtype=float)
	span = hi - lo
	
	n_workers = os.cpu_count() or 4
	with Pool(processes=n_workers, initializer=_init_oracle, initargs=(world_file,)) as pool:
		def _parallel_label(points, desc):
			if len(points) == 0: return np.array([], dtype=int)
			it = pool.imap(_oracle_worker, points, chunksize=1024)
			return np.fromiter(tqdm(it, total=len(points), desc=desc, ncols=80, leave=False), dtype=int, count=len(points))

		# 1. Boundary Sampling
		n_boundary_target = int(n_pts * near_ratio)
		X_boundary = np.zeros((0, dim))
		if n_boundary_target > 1 and c_space is not None and theta is not None:
			n_needed_pairs = n_boundary_target // 2
			edges = extract_boundary_edges(c_space, theta)
			if edges:
				vq1, vq2 = [], []
				for _ in range(10): # attempts
					if len(vq1) >= n_needed_pairs: break
					batch_n = n_needed_pairs - len(vq1)
					X_base = sample_boundary_uniform(edges, batch_n * 2)
					noise = np.random.randn(len(X_base), dim) * (sigma * span)
					Q1, Q2 = np.clip(X_base + noise, lo, hi), np.clip(X_base - noise, lo, hi)
					L1, L2 = _parallel_label(Q1, "Crossing Q1"), _parallel_label(Q2, "Crossing Q2")
					mask = (L1 != L2)
					vq1.extend(Q1[mask]); vq2.extend(Q2[mask])
				X_boundary = np.vstack([np.array(vq1[:n_needed_pairs]), np.array(vq2[:n_needed_pairs])])
		
		# 2. AABB Sampling
		n_aabb_needed = max(0, n_pts - len(X_boundary))
		X_aabb = lo + np.random.rand(n_aabb_needed, dim) * span
		L_aabb = _parallel_label(X_aabb, "AABB Points")
		X_in_aabb, X_out_aabb = X_aabb[L_aabb == -1], X_aabb[L_aabb == 1]
		
		# Combine and Re-label boundary to be safe
		X_all = np.vstack([X_boundary, X_aabb])
		L_all = _parallel_label(X_all, "Final Labelling")
		X_in, X_out = X_all[L_all == -1], X_all[L_all == 1]

	# 3. Augmentation (Workspace xyz)
	device = "cuda" if torch.cuda.is_available() else "cpu"
	# Concatenate [q, FK control points]
	cp_manager = ControlPointManager2D(n_points_per_link=n_points_per_link, device=device)

	def augment(q_np):
		if len(q_np) == 0: return np.zeros((0, 2 + n_points_per_link*2*2))
		q_t = torch.from_numpy(q_np).float().to(device)
		with torch.no_grad():
			pts = cp_manager.get_control_points(q_t) # (N, CP, 2)
			pts_flat = pts.reshape(q_np.shape[0], -1)
		return torch.cat([q_t, pts_flat], dim=-1).cpu().numpy()

	X_train_in_gndf = augment(X_in)
	X_train_out_gndf = augment(X_out)
	
	return X_train_in_gndf, X_train_out_gndf


def plot_cspace(theta, c_space, outdir, prefix):
	"""Save a quick visualization of the configuration space grid."""
	t1, t2 = theta
	plt.figure(figsize=(6, 6))
	im = plt.imshow(
		c_space.T,
		extent=[t1[0], t1[-1], t2[0], t2[-1]],
		origin='lower',
		aspect='auto',
		cmap='RdBu_r'
	)
	plt.colorbar(im, fraction=0.046, pad=0.04)
	plt.xlabel(r"$\theta_1$")
	plt.ylabel(r"$\theta_2$")
	plt.title("C-space occupancy")
	plt.tight_layout()
	os.makedirs(outdir, exist_ok=True)
	out_path = os.path.join(outdir, f"{prefix}_cspace.png")
	plt.savefig(out_path, dpi=200)
	plt.close()
	print(f"Saved c-space visualization to {out_path}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--world", default='config/block_world5.yaml', help="path to the world file")
	parser.add_argument("--n-train", type=int, default=5000, help="number of training samples to generate")
	parser.add_argument("--near-ratio", type=float, default=0.6, help="ratio of near-boundary samples")
	parser.add_argument("--sigma", type=float, default=0.01, help="stddev for boundary sampling noise")
	parser.add_argument("--n-points-per-link", type=int, default=1, help="control points per link for workspace augmentation")
	parser.add_argument("--outdir", type=str, default="inputs/block_world5", help="output directory")
	parser.add_argument("-visu", action="store_true",  default=True, help="generate visualization of sampled configurations")
	args = parser.parse_args()
	
	os.makedirs(args.outdir, exist_ok=True)
	prefix = Path(args.world).stem
	
	arm_problem = robot_tools.PlanningProblem(args.world)
	theta, c_space = arm_problem.construct_config_space(nx=101)
	bounds = [(theta[0][0], theta[0][-1]), (theta[1][0], theta[1][-1])]
	
	t0 = time.time()
	X_in, X_out = generate_gndf_dataset(
		args.n_train, bounds, args.world, args.near_ratio, args.sigma, 
		c_space, theta, args.n_points_per_link
	)

	# Optional visualization

	np.save(os.path.join(args.outdir, f"{prefix}_Xtrain_in.npy"), X_in)
	np.save(os.path.join(args.outdir, f"{prefix}_Xtrain_out.npy"), X_out)
	print(f"DONE. Saved G-NDF data to {args.outdir} in {time.time()-t0:.2f}s")
	print(f"Shape: {X_in.shape}")
	print(f"Shape: {X_out.shape}")

	if args.visu:

		plt.figure(figsize=(6,5))
		plt.imshow(c_space.T, origin="lower",
					extent=[theta[0][0], theta[0][-1], theta[1][0], theta[1][-1]],
					cmap="gray_r")
		if len(X_in):  
			plt.scatter(X_in[:,0],  X_in[:,1],  c='red',  s=5, label="Collision")
		if len(X_out): 
			plt.scatter(X_out[:,0], X_out[:,1], c='blue', s=5, label="Free")
		
		out_path = os.path.join(args.outdir, f"{prefix}_cspace_{args.n_train}_samples.png")
		plt.xlabel(r"$\theta_1$")
		plt.ylabel(r"$\theta_2$")
		plt.title("Configuration Space with Sampled Points")
		plt.legend(loc='lower right', frameon=True, framealpha=0.9)

		pi = np.pi
		plt.xticks([0, pi/2, pi, 3*pi/2, 2*pi], [r'-$\pi$', r'$-\pi/2$', r'$0$', r'-$\pi/2$', r'$\pi$'])
		plt.yticks([0, pi/2, pi, 3*pi/2, 2*pi], [r'-$\pi$', r'$-\pi/2$', r'$0$', r'-$\pi/2$', r'$\pi$'])

		plt.savefig(out_path, dpi=300, bbox_inches='tight')
		plt.close()
		print(f"Figure saved to {out_path}")