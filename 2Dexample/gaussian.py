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

def extract_train_point_cloud(n_pts, bounds, oracle, near_ratio=0.5, sigma=0.05, pool=None):

	"""
	Generate C-space training dataset using Gaussian sampling.
	Args:
		n_pts (int): Total number of samples
		bounds (list): [(min, max), ...]
		oracle: (unused, uses global _GLOBAL_PROBLEM)
		near_ratio (float): Boundary sample ratio (0.0–1.0)
		sigma (float): Gaussian noise std (relative to axis length)
		pool: multiprocessing pool
	"""

	dim = len(bounds)
	lo = np.array([b[0] for b in bounds], float)
	hi = np.array([b[1] for b in bounds], float)
	span = hi - lo

	# Uniform sampling
	def sample_uniform(m):
		if m <= 0: return np.zeros((0, dim))
		return lo + np.random.rand(m, dim) * span

	# Parallel labeling
	def parallel_label(points, desc):
		if len(points) == 0: return np.array([], dtype=int)
		# chunksize affects performance, 1024 is reasonable
		it = pool.imap(_oracle_worker, points, chunksize=1024)
		return np.fromiter(tqdm(it, total=len(points), desc=desc, ncols=80, leave=False), dtype=int, count=len(points))

	X_in_all = []
	X_out_all = []
	
	# Target counts
	n_boundary = int(n_pts * near_ratio)
	n_uniform  = n_pts - n_boundary
	
	# Boundary Sampling
	if n_boundary > 0:
		print(f"Sampling {n_boundary} boundary points (Gaussian pairs)...")

		batch_size = n_pts * 2 
		
		pbar = tqdm(total=n_boundary, desc="Found Boundary Pts", ncols=80)
		
		while (len(X_in_all) + len(X_out_all)) < n_boundary:
			# Generate pairs
			q1 = sample_uniform(batch_size)
			noise = np.random.randn(batch_size, dim) * (sigma * span)
			q2 = np.clip(q1 + noise, lo, hi)
			
			# Label
			l1 = parallel_label(q1, "Label Batch Q1")
			l2 = parallel_label(q2, "Label Batch Q2")
			
			# Find crossing pairs: (Free, Obs) or (Obs, Free)
			# l=1 (free), l=-1 (obs)
			mask = (l1 != l2)
			
			# Extract
			# Case 1: q1 obs (-1), q2 free (1)
			mask1 = mask & (l1 == -1)
			if np.any(mask1):
				X_in_all.extend(q1[mask1])
				X_out_all.extend(q2[mask1])
				
			# Case 2: q1 free (1), q2 obs (-1)
			mask2 = mask & (l1 == 1)
			if np.any(mask2):
				X_out_all.extend(q1[mask2])
				X_in_all.extend(q2[mask2])
				
			current_count = len(X_in_all) + len(X_out_all)
			pbar.n = min(current_count, n_boundary)
			pbar.refresh()
			
			if current_count >= n_boundary:
				break
				
		pbar.close()
	
	# Convert to arrays
	X_in_bound = np.array(X_in_all)
	X_out_bound = np.array(X_out_all)

	# Uniform Sampling
	cur_total = len(X_in_bound) + len(X_out_bound)
	n_uniform_needed = max(n_pts - cur_total, 0)
	
	if n_uniform_needed > 0:
		print(f"Sampling {n_uniform_needed} uniform points...")

		U = sample_uniform(n_uniform_needed)
		lU = parallel_label(U, "Label Uniform")
		
		U_in = U[lU == -1]
		U_out = U[lU == 1]
		
		# Combine
		if len(X_in_bound) > 0: X_in = np.vstack([X_in_bound, U_in]) if len(U_in) else X_in_bound
		else: X_in = U_in
		
		if len(X_out_bound) > 0: X_out = np.vstack([X_out_bound, U_out]) if len(U_out) else X_out_bound
		else: X_out = U_out
	else:
		X_in = X_in_bound
		X_out = X_out_bound

	# Final Shuffle & Trim
	if len(X_in) == 0: X_in = np.zeros((0, dim))
	if len(X_out) == 0: X_out = np.zeros((0, dim))
	
	total_have = len(X_in) + len(X_out)
	if total_have > n_pts:
		# Randomly remove
		keep_idx = np.random.permutation(total_have)[:n_pts]
		frac = n_pts / total_have
		n_in_keep = int(len(X_in) * frac)
		n_out_keep = n_pts - n_in_keep
		
		if len(X_in) > 0: X_in = X_in[np.random.permutation(len(X_in))[:n_in_keep]]
		if len(X_out) > 0: X_out = X_out[np.random.permutation(len(X_out))[:n_out_keep]]
	
	print(f"Final Dataset: {len(X_in)} Inj, {len(X_out)} Free. Total: {len(X_in)+len(X_out)}")
	return X_in, X_out

if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog="C-space Dataset Generator", description="Generate a dataset to train a neural distance field from C-space sampling")
	parser.add_argument('-w', '--world', default='config/block_world4.yaml', help='World definition (obstacles)')
	parser.add_argument("-n", "--n-train", type=int, default=2000, help="number of samples in the training set")
	parser.add_argument("-nt", "--n-test",  type=int, default=1000, help="number of samples in the test set")
	parser.add_argument("-nti", "--n-test-boundary", type=int, default=500, help="number of points in the set dataset that are sampled _on_ the surface.")
	parser.add_argument("-visu", action="store_true",  default=True, help="generate visualization of sampled configurations")
	parser.add_argument("-r", "--near-ratio", type=float, default=0.7, help="ratio of samples drawn near the free/obs boundary")
	parser.add_argument("-s", "--sigma", type=float, default=0.05, help="noise scale for near-boundary sampling")
	parser.add_argument("-rmin", "--min-dist", type=float, default=0.0, help="minimum distance from boundary to keep (optional)")
	parser.add_argument("-bal", "--balance", action="store_true", help="balance dataset: equal free/obs samples")
	args = parser.parse_args()
	
	outdir = getattr(args, "outdir", "inputs/gaussian")
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
			pool=pool
		)
		t1 = time.time()
		
	print(f"Random sampling takes {t1 - t0:.3f} second with {n_workers} workers")
	
	np.save(os.path.join(outdir, f"{prefix}_Xtrain_in.npy"),  X_in)      # C_obs
	np.save(os.path.join(outdir, f"{prefix}_Xtrain_out.npy"), X_out)     # C_free
	np.save(os.path.join(outdir, f"{prefix}_cspace.npy"),     c_space)   # Occupancy
	np.save(os.path.join(outdir, f"{prefix}_theta0.npy"),     theta[0])  # theta 1
	np.save(os.path.join(outdir, f"{prefix}_theta1.npy"),     theta[1])  # theta 2
	
	print("\nGenerate test set")
	with Pool(processes=n_workers, initializer=_init_oracle, initargs=(args.world,)) as pool:
		Xtest_in, Xtest_out = extract_train_point_cloud(
			n_pts=args.n_test,
			bounds=bounds,
			oracle=None,
			near_ratio=args.near_ratio,
			sigma=args.sigma,
			pool=pool
		)
	Xtest = np.vstack([Xtest_in, Xtest_out])
	Ytest = np.concatenate([-np.ones(len(Xtest_in)), np.ones(len(Xtest_out))])
	
	np.save(os.path.join(outdir, f"{prefix}_Xtest.npy"), Xtest)
	np.save(os.path.join(outdir, f"{prefix}_Ytest.npy"), Ytest)
	print(f"Saved test set with {len(Xtest)} samples ({len(Xtest_in)} collision, {len(Xtest_out)} free)")
		
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