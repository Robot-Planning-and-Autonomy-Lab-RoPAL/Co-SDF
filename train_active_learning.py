#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from types import SimpleNamespace
from torch.utils.data import TensorDataset, DataLoader

# Import project modules
from common.models import select_model, count_parameters
from common.training import Trainer
from common.utils import get_device
from common.callback import LoggerCB, CheckpointCB, UpdateHkrRegulCB

# Import Oracle logic
import robot_tools
from extract_dataset_pointcloud import _init_oracle, _oracle_worker, extract_train_point_cloud
import extract_dataset_pointcloud # to access global if needed

# --- Data Management Utils ---
def update_loaders(X_in, X_out, batch_size, device):
	"""Create DataLoaders from tensors"""
	if len(X_in) > 0:
		t_in = torch.Tensor(X_in).to(device)
		l_in = DataLoader(TensorDataset(t_in), batch_size=batch_size, shuffle=True)
	else:
		l_in = []
		
	if len(X_out) > 0:
		t_out = torch.Tensor(X_out).to(device)
		l_out = DataLoader(TensorDataset(t_out), batch_size=batch_size, shuffle=True)
	else:
		l_out = []
		
	return l_in, l_out

# --- Model Definition (with Embedding) ---
class PeriodicEmbedding(nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self, x):
		return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

def create_periodic_model(args, device):
	embedding = PeriodicEmbedding()
	# Hardcoded dim=2 -> 4
	model_core = select_model(args.model, 4, args.n_layers, args.n_hidden)
	model = nn.Sequential(embedding, model_core).to(device)
	model.id = model_core.id
	model.meta = model_core.meta
	return model

# --- Active Learning Loop ---
def run_active_learning(args):
	device = get_device()
	print("DEVICE:", device)
	
	# 1. Initialize Oracle
	print("Initializing Oracle...")
	_init_oracle(args.world)
	# Also construct C-space for bounds (needed for sampling)
	arm_problem = robot_tools.PlanningProblem(args.world)
	theta, _ = arm_problem.construct_config_space(nx=10) # coarse
	bounds = [(theta[0][0], theta[0][-1]), (theta[1][0], theta[1][-1])]
	lo = np.array([b[0] for b in bounds])
	hi = np.array([b[1] for b in bounds])
	
	# 2. Initial Dataset (Seed)
	print(f"Generating Seed Dataset ({args.n_seed} samples)...")
	# Use the smart gaussian sampler we made earlier to start strong, or just uniform?
	# User asked for Active Learning, so we can start with something simple roughly.
	# But let's use our good sampler to kickstart.
	# We can reuse the function from extract_dataset_pointcloud (assuming we can run it in this process)
	# We need a pool for it? Or run single threaded.
	# To simplify, let's implement a simple uniform seed + current gaussian logic if possible.
	# Actually, multiprocessing within this script might be tricky with torch on GPU.
	# Let's run single-threaded oracle for the loop to avoid context switching overhead for small batches.
	
	# If pool is None, extract_train_point_cloud might fail if it strictly expects a pool map.
	# We'll create a dummy pool wrapper for single thread.
	class DummyPool:
		def imap(self, func, iterable, chunksize=1):
			return map(func, iterable)
	
	dummy_pool = DummyPool()
	
	X_in, X_out = extract_train_point_cloud(
		n_pts=args.n_seed, 
		bounds=bounds, 
		oracle=None, 
		near_ratio=0.5, 
		sigma=0.05, 
		pool=dummy_pool
	)
	
	print(f"Seed Data: {len(X_in)} Obs, {len(X_out)} Free")
	
	# 3. Initialize Model
	model = create_periodic_model(args, device)
	
	# 4. Config & Trainer Setup
	config = SimpleNamespace(
		signed = True,
		device = device,
		n_epochs = args.epochs_per_iter, 
		checkpoint_freq = 0, # Don't save every epoch
		batch_size = args.batch_size,
		test_batch_size = 1000,
		loss_margin = 0.01,
		loss_regul = 100.,
		optimizer = "adam",
		learning_rate = 1e-3,
		output_folder = args.output
	)
	os.makedirs(config.output_folder, exist_ok=True)
	
	# Active Learning Iterations
	for iteration in range(args.n_iters):
		print(f"\n=== Active Learning Iteration {iteration+1}/{args.n_iters} ===")
		
		# Update Loaders
		loader_in, loader_out = update_loaders(X_in, X_out, args.batch_size, device)
		trainer = Trainer((loader_in, loader_out), None, config) # No test set for now
		
		# We add a callback to adjust lambda?
		trainer.add_callbacks(UpdateHkrRegulCB({1:1., 5:10., 10:100.}))
		
		# Train (Fine-tune)
		print("Training model...")
		trainer.train_lip(model)
		
		# Query / Sampling Step
		if iteration < args.n_iters - 1:
			n_query = args.n_query
			print(f"Querying {n_query} candidates...")
			
			# Generate huge pool of random points
			candidates = lo + np.random.rand(n_query * 10, 2) * (hi - lo)
			cand_tensor = torch.Tensor(candidates).to(device)
			
			# Predict distances
			with torch.no_grad():
				dists = model(cand_tensor).cpu().numpy().flatten()
				
			# Strategy: Uncertainty Sampling (Close to boundary => |f(x)| is small)
			# Sort by absolute distance
			abs_dists = np.abs(dists)
			unc_idx = np.argsort(abs_dists)[:n_query]
			
			queries = candidates[unc_idx]
			
			print(f"Labeling {len(queries)} hard points (Min dist: {abs_dists[unc_idx[0]]:.5f})...")
			
			# Oracle Labeling
			new_labels = []
			for q in queries:
				new_labels.append(_oracle_worker(q)) # 1 or -1
			new_labels = np.array(new_labels)
			
			# Add to dataset
			new_in = queries[new_labels == -1]
			new_out = queries[new_labels == 1]
			
			if len(new_in) > 0: X_in = np.vstack([X_in, new_in])
			if len(new_out)> 0: X_out= np.vstack([X_out,new_out])
			
			print(f"Added {len(new_in)} Obs, {len(new_out)} Free. Total: {len(X_in)+len(X_out)}")
			
	# Save Final Model
	print("Saving final model...")
	from common.models import save_model
	save_model(model, os.path.join(config.output_folder, "model_active.pt"))
	
	# Visualize Final
	import matplotlib.pyplot as plt
	
	plt.figure(figsize=(6,5))
	if len(X_in): plt.scatter(X_in[:,0], X_in[:,1], c='red', s=1, alpha=0.5, label='Obs')
	if len(X_out): plt.scatter(X_out[:,0], X_out[:,1], c='blue', s=1, alpha=0.5, label='Free')
	plt.title(f"Active Learning Final Dataset ({len(X_in)+len(X_out)} pts)")
	plt.legend()
	plt.savefig(os.path.join(config.output_folder, "active_dataset.png"))
	print("Done.")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--world", default="config/block_world.yaml")
	parser.add_argument("--output", default="output/active_learning")
	parser.add_argument("--model", default="sll")
	parser.add_argument("--n-layers", type=int, default=20)
	parser.add_argument("--n-hidden", type=int, default=128)
	parser.add_argument("--n-seed", type=int, default=500, help="Initial random samples")
	parser.add_argument("--n-iters", type=int, default=5, help="Number of active learning loops")
	parser.add_argument("--epochs-per-iter", type=int, default=10, help="Epochs per loop")
	parser.add_argument("--n-query", type=int, default=300, help="Samples to add per loop")
	parser.add_argument("--batch-size", type=int, default=100)
	
	args = parser.parse_args()
	run_active_learning(args)
