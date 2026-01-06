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

def sample_points_from_edges(edges, n_pts):
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
	edge_indices = np.random.choice(len(edges), size=n_pts, p=probs)
	
	samples = []
	for idx in edge_indices:
		p1, p2 = edges[idx]
		t = np.random.rand()
		samples.append(p1 + t * (p2 - p1))
		
	return np.array(samples)

def parallel_label(points, pool, desc):
    if len(points) == 0: return np.array([], dtype=int)
    it = pool.imap(_oracle_worker, points, chunksize=1024)
    return np.fromiter(tqdm(it, total=len(points), desc=desc, ncols=80, leave=False), dtype=int, count=len(points))

if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog="C-space Dataset Generator", description="Generate a dataset to train a neural distance field from C-space sampling")
	parser.add_argument('-w', '--world', default='config/block_world4.yaml', help='World definition (obstacles)')
	parser.add_argument("-n", "--n-train", type=int, default=2000, help="number of samples in the training set")
	parser.add_argument("-mode", "--mode", default="signed", choices=["signed", "unsigned"], help="which type of dataset to generate")
	parser.add_argument("-visu", action="store_true",  default=True, help="generate visualization of sampled configurations")
	parser.add_argument("-nti", "--n-test-boundary", type=int, default=500, help="number of points on boundary for test set")
	args = parser.parse_args()
	
	outdir = getattr(args, "outdir", "inputs/uniform_aabb")
	os.makedirs(outdir, exist_ok=True)
	
	prefix = Path(args.world).stem
		
	arm_problem = robot_tools.PlanningProblem(args.world)
	theta, c_space = arm_problem.construct_config_space(nx=101)
	edges = extract_boundary_edges(c_space, theta)
		
	bounds = [(theta[0][0], theta[0][-1]), (theta[1][0], theta[1][-1])]
	lo = np.array([b[0] for b in bounds], dtype=float)
	hi = np.array([b[1] for b in bounds], dtype=float)
	
	n_workers = os.cpu_count() or 4
	with Pool(processes=n_workers, initializer=_init_oracle, initargs=(args.world,)) as pool:
		t0 = time.time()
		
		X_in, X_out, X_on = None, None, None
		
		if args.mode == "unsigned":
			print("Generating unsigned dataset...")
			X_on = sample_points_from_edges(edges, args.n_train)
			X_out = lo + np.random.rand(args.n_train, 2) * (hi - lo)
			print(f" | Generated {X_on.shape[0]} (on), {X_out.shape[0]} (outside)")
			
			np.save(os.path.join(outdir, f"{prefix}_Xtrain_on.npy"), X_on)
			np.save(os.path.join(outdir, f"{prefix}_Xtrain_out.npy"), X_out)
			
		elif args.mode == "signed":
			print("Generating signed dataset...")
			# Sample a small portion exactly on the boundary and label as "inside" (SDF <= 0)
			X_on = sample_points_from_edges(edges, args.n_train // 20)
			
			mult = 5
			n_found_in, n_found_out = 0, 0
			X_in_list, X_out_list = [], []
			
			print(" | Sampling from AABB and filtering...")
			while (len(X_in_list) < args.n_train or len(X_out_list) < args.n_train):
				X_random = lo + np.random.rand(mult * args.n_train, 2) * (hi - lo)
				labels = parallel_label(X_random, pool, f"Filtering Batch (mult={mult})")
				
				X_in_batch = X_random[labels == -1]
				X_out_batch = X_random[labels == 1]
				
				X_in_list.extend(X_in_batch)
				X_out_list.extend(X_out_batch)
				
				print(f" | Progress: In {len(X_in_list)}/{args.n_train}, Out {len(X_out_list)}/{args.n_train}")
				if len(X_in_list) >= args.n_train and len(X_out_list) >= args.n_train:
					break
				mult *= 2 # Increase batch size if not enough found
				if mult > 100: break # Safety break
			
			X_in = np.array(X_in_list[:args.n_train])
			X_out = np.array(X_out_list[:args.n_train])
			
			# Combine boundary samples into the "in" set as done in polyline script
			if len(X_on) > 0:
				X_in = np.concatenate((X_on, X_in))[:X_out.shape[0], :]
			
			np.random.shuffle(X_in)
			print(f" | Final Dataset: {X_in.shape[0]} Collision, {X_out.shape[0]} Free")
			
			np.save(os.path.join(outdir, f"{prefix}_Xtrain_in.npy"), X_in)
			np.save(os.path.join(outdir, f"{prefix}_Xtrain_out.npy"), X_out)

		t1 = time.time()
		print(f"Sampling takes {t1 - t0:.3f} seconds with {n_workers} workers")
	
	# Save grid info
	np.save(os.path.join(outdir, f"{prefix}_cspace.npy"),     c_space)
	np.save(os.path.join(outdir, f"{prefix}_theta0.npy"),     theta[0])
	np.save(os.path.join(outdir, f"{prefix}_theta1.npy"),     theta[1])
	
	if args.visu:
		plt.figure(figsize=(6,5))
		plt.imshow(c_space.T, origin="lower",
					extent=[theta[0][0], theta[0][-1], theta[1][0], theta[1][-1]],
					cmap="gray_r", alpha=0.3)
		
		# For visualization, just use whatever we have
		if X_in is not None and len(X_in):
			plt.scatter(X_in[:,0], X_in[:,1], c='red', s=2, label="Collision", alpha=0.6)
		if X_out is not None and len(X_out):
			plt.scatter(X_out[:,0], X_out[:,1], c='blue', s=2, label="Free", alpha=0.6)
		if X_on is not None and len(X_on):
			plt.scatter(X_on[:,0], X_on[:,1], c='green', s=2, label="Boundary", alpha=0.6)
		
		plt.xlabel(r"$\theta_1$")
		plt.ylabel(r"$\theta_2$")
		plt.title(f"Uniform AABB Sampling ({args.mode})")
		plt.legend(loc='lower right')
		
		pi = np.pi
		plt.xticks([0, pi/2, pi, 3*pi/2, 2*pi], [r'0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
		plt.yticks([0, pi/2, pi, 3*pi/2, 2*pi], [r'0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
		
		out_path = os.path.join(outdir, f"{prefix}_cspace_{args.n_train}_samples.png")
		plt.savefig(out_path, dpi=300, bbox_inches='tight')
		plt.close()
		print(f"Figure saved to {out_path}")