#!/usr/bin/env python3
import os
import time
import argparse
import robot_tools
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from skimage import measure

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
	"""
	Extract boundary edges from the C-space grid using marching squares (find_contours).
	Returns a list of segments: [(p1, p2), ...] where p1, p2 are np.array([theta1, theta2])
	"""
	# Find contours at level 0.5 (between 0=free and 1+=obs)
	# c_space is usually (nx, ny)
	contours = measure.find_contours(c_space, 0.5)
	
	t1, t2 = theta
	# Grid spacing
	dth1 = (t1[-1] - t1[0]) / (len(t1) - 1)
	dth2 = (t2[-1] - t2[0]) / (len(t2) - 1)
	
	edges = []
	for contour in contours:
		# contour is (N, 2) in grid coordinates (row, col) which corresponds to (i, j)
		# Convert to angle coordinates
		pts = np.zeros_like(contour)
		pts[:, 0] = t1[0] + contour[:, 0] * dth1
		pts[:, 1] = t2[0] + contour[:, 1] * dth2
		
		for i in range(len(pts) - 1):
			edges.append((pts[i], pts[i+1]))
			
	return edges

def sample_boundary_uniform(edges, n_pts):
	"""
	Sample n_pts points uniformly on boundary edges, weighted by edge length.
	"""
	if not edges or n_pts <= 0:
		return np.zeros((0, 2))
	
	lengths = np.array([np.linalg.norm(p2 - p1) for p1, p2 in edges])
	total_len = np.sum(lengths)
	if total_len == 0:
		return np.zeros((0, 2))
		
	probs = lengths / total_len
	
	# Select edges based on length probability
	edge_indices = np.random.choice(len(edges), size=n_pts, p=probs)
	
	samples = []
	for idx in edge_indices:
		p1, p2 = edges[idx]
		t = np.random.rand()
		samples.append(p1 + t * (p2 - p1))
		
	return np.array(samples)

def extract_train_point_cloud(n_pts, bounds, oracle, near_ratio=0.5, sigma=0.01, c_space=None, theta=None, pool=None):
	"""
	Generate C-space training dataset.
	1) Boundary Sampling: Uniformly on edges extracted from c_space grid, with optional sigma-noise.
	2) AABB Sampling: Uniformly within the joint limits.

	Args:
		n_pts (int): Total number of samples
		bounds (list): [(min, max), ...]
		oracle (callable): Collision check function
		near_ratio (float): Ratio of points on the boundary (0~1)
		sigma (float): Gaussian noise std applied to boundary points (relative to 2pi if bounds are angles)
		c_space (ndarray): C-space occupancy grid for boundary extraction
		theta (list): [theta1, theta2] grid vectors
		pool (Pool): multiprocessing pool
	"""
	dim = len(bounds)
	lo = np.array([b[0] for b in bounds], dtype=float)
	hi = np.array([b[1] for b in bounds], dtype=float)
	span = hi - lo
	
	def _parallel_label(points, pool, desc):
		if len(points) == 0: return np.array([], dtype=int)
		it = pool.imap(_oracle_worker, points, chunksize=1024)
		return np.fromiter(tqdm(it, total=len(points), desc=desc, ncols=80, leave=False), dtype=int, count=len(points))

	# 1) Boundary Sampling (Symmetric Crossing Pairs)
	n_boundary_target = int(n_pts * near_ratio)
	X_boundary = np.zeros((0, dim))
	
	if n_boundary_target > 0 and c_space is not None and theta is not None:
		n_needed_pairs = n_boundary_target // 2
		print(f"Sampling crossing pairs from boundary edges (Target: {n_needed_pairs*2} pts)...")
		edges = extract_boundary_edges(c_space, theta)
		
		if edges:
			valid_q1 = []
			valid_q2 = []
			pbar = tqdm(total=n_needed_pairs, desc="Found Crossing Pairs", ncols=80, leave=False)
			
			attempts = 0
			while len(valid_q1) < n_needed_pairs and attempts < 10:
				batch_n = n_needed_pairs - len(valid_q1)
				X_base = sample_boundary_uniform(edges, batch_n * 2)
				noise = np.random.randn(len(X_base), dim) * (sigma * span)
				Q1 = np.clip(X_base + noise, lo, hi)
				Q2 = np.clip(X_base - noise, lo, hi)
				
				L1 = _parallel_label(Q1, pool, "Labeling Q1")
				L2 = _parallel_label(Q2, pool, "Labeling Q2")
				
				mask = (L1 != L2)
				valid_q1.extend(Q1[mask])
				valid_q2.extend(Q2[mask])
				
				pbar.n = min(len(valid_q1), n_needed_pairs)
				pbar.refresh()
				attempts += 1
				
			pbar.close()
			X_boundary = np.vstack([np.array(valid_q1[:n_needed_pairs]), np.array(valid_q2[:n_needed_pairs])])
			print(f"Final crossing-boundary points: {len(X_boundary)}")
	
	# 2) AABB Sampling (Uniform)
	n_aabb_needed = n_pts - len(X_boundary)
	if n_aabb_needed > 0:
		print(f"Sampling {n_aabb_needed} points from AABB...")
		u = np.random.rand(n_aabb_needed, dim)
		X_aabb = lo + u * (hi - lo)
		
		L_aabb = _parallel_label(X_aabb, pool, "Labeling AABB")
		X_in_aabb = X_aabb[L_aabb == -1]
		X_out_aabb = X_aabb[L_aabb == 1]
	else:
		X_in_aabb = np.zeros((0, dim))
		X_out_aabb = np.zeros((0, dim))

	# Collect all labeled points
	# Since X_boundary already contains both in/out points from crossing pairs:
	# We need to label X_boundary one more time or just use the L1, L2 results
	# but for simplicity let's just label the final sets
	
	if len(X_boundary) > 0:
		it = pool.imap(_oracle_worker, X_boundary, chunksize=1024)
		L_bound = np.fromiter(tqdm(it, total=len(X_boundary), desc="Final Boundary Labels", ncols=80, leave=False), dtype=int, count=len(X_boundary))
		X_in_bound = X_boundary[L_bound == -1]
		X_out_bound = X_boundary[L_bound == 1]
	else:
		X_in_bound = np.zeros((0, dim))
		X_out_bound = np.zeros((0, dim))

	X_in = np.vstack([X_in_bound, X_in_aabb]) if len(X_in_bound) and len(X_in_aabb) else (X_in_bound if len(X_in_bound) else X_in_aabb)
	X_out = np.vstack([X_out_bound, X_out_aabb]) if len(X_out_bound) and len(X_out_aabb) else (X_out_bound if len(X_out_bound) else X_out_aabb)
	
	print(f"Final Dataset: {len(X_in)} Collision, {len(X_out)} Free. Total: {len(X_in)+len(X_out)}")
	return X_in, X_out

if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog="C-space Dataset Generator", description="Generate a dataset to train a neural distance field from C-space sampling")
	parser.add_argument('-w', '--world', default='config/block_world4.yaml', help='World definition (obstacles)')
	parser.add_argument("-n", "--n-train", type=int, default=2000, help="number of samples in the training set")
	parser.add_argument("-visu", action="store_true",  default=True, help="generate visualization of sampled configurations")
	parser.add_argument("-r", "--near-ratio", type=float, default=0.5, help="ratio of samples drawn near the free/obs boundary")
	parser.add_argument("-s", "--sigma", type=float, default=0.01, help="noise scale for jitter on boundary sampling")
	parser.add_argument("-rmin", "--min-dist", type=float, default=0.0, help="minimum distance from boundary to keep (optional)")
	parser.add_argument("-bal", "--balance", action="store_true", help="balance dataset: equal free/obs samples")
	args = parser.parse_args()
	
	outdir = getattr(args, "outdir", "inputs/gaussian_aabb")
	os.makedirs(outdir, exist_ok=True)
	
	if hasattr(args, "prefix") and args.prefix is not None:
		prefix = args.prefix
	else:
		prefix = Path(args.world).stem
		
	arm_problem = robot_tools.PlanningProblem(args.world)
	theta, c_space = arm_problem.construct_config_space(nx=101)
		
	# Angle range
	bounds = [(theta[0][0], theta[0][-1]), (theta[1][0], theta[1][-1])]
	
	# uniform sampling with multiprocessing
	n_workers = os.cpu_count() or 4
	with Pool(processes=n_workers, initializer=_init_oracle, initargs=(args.world,)) as pool:
		t0 = time.time()
		X_in, X_out = extract_train_point_cloud(
			n_pts=args.n_train,
			bounds=bounds,
			oracle=None,
			near_ratio=args.near_ratio,
			sigma=args.sigma,
			c_space=c_space,
			theta=theta,
			pool=pool
		)
		t1 = time.time()
		
	print(f"Random sampling takes {t1 - t0:.3f} second with {n_workers} workers")
	
	np.save(os.path.join(outdir, f"{prefix}_Xtrain_in.npy"),  X_in)      # C_obs
	np.save(os.path.join(outdir, f"{prefix}_Xtrain_out.npy"), X_out)     # C_free
	np.save(os.path.join(outdir, f"{prefix}_cspace.npy"),     c_space)   # Occupancy
	np.save(os.path.join(outdir, f"{prefix}_theta0.npy"),     theta[0])  # theta 1
	np.save(os.path.join(outdir, f"{prefix}_theta1.npy"),     theta[1])  # theta 2
	
	if args.visu:

		plt.figure(figsize=(6,5))
		plt.imshow(c_space.T, origin="lower",
					extent=[theta[0][0], theta[0][-1], theta[1][0], theta[1][-1]],
					cmap="gray_r")
		if len(X_in):  
			plt.scatter(X_in[:,0],  X_in[:,1],  c='red',  s=5, label="Collision")
		if len(X_out): 
			plt.scatter(X_out[:,0], X_out[:,1], c='blue', s=5, label="Free")
		
		out_path = os.path.join(outdir, f"{prefix}_cspace_{args.n_train}_samples.png")
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