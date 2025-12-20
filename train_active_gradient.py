#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import scipy.ndimage as ndi
from types import SimpleNamespace
from torch.utils.data import TensorDataset, DataLoader
from scipy.interpolate import RegularGridInterpolator

# Project imports
from common.models import select_model, count_parameters
from common.utils import get_device
from common.training import Trainer # Will use custom loop instead
import robot_tools
import cspace_sdf

# --- Oracle with Gradient ---
class GradientOracle:
	def __init__(self, world_file, res=500):
		print("Building High-Res C-space Oracle (Ground Truth)...")
		self.problem = robot_tools.PlanningProblem(world_file)
		# Build dense grid
		theta, c_space = self.problem.construct_config_space(nx=res)
		
		# Compute SDF
		self.axes, self.sdf_grid = cspace_sdf.cobs_sdf_on_torus(theta, c_space)
		
		# Compute Gradient on Grid (Central Difference)
		# dSDF/dtheta
		dth0 = self.axes[0][1] - self.axes[0][0]
		dth1 = self.axes[1][1] - self.axes[1][0]
		grad_0, grad_1 = np.gradient(self.sdf_grid, dth0, dth1)
		
		# Interpolators
		self.interp_sdf = RegularGridInterpolator(self.axes, self.sdf_grid, bounds_error=False, fill_value=None)
		self.interp_g0  = RegularGridInterpolator(self.axes, grad_0, bounds_error=False, fill_value=None)
		self.interp_g1  = RegularGridInterpolator(self.axes, grad_1, bounds_error=False, fill_value=None)
		
		# Bounds for wrapping
		self.t0_range = (self.axes[0][0], self.axes[0][-1])
		self.t1_range = (self.axes[1][0], self.axes[1][-1])

	def query_batch(self, points):
		# wrap points
		pts_wrapped = points.copy()
		# Simple wrap for [0, 2pi] assumes grid covers it.
		# But cspace_sdf handles drop periodic. Let's trust logic or implement wrap
		w = pts_wrapped[:,0] - self.t0_range[0]
		span0 = self.t0_range[1] - self.t0_range[0] + (self.axes[0][1] - self.axes[0][0]) # add one step back?
		# Actually cspace_sdf.cobs_sdf_on_torus returns axes of the truncated grid?
		# No, it returns observed axes. 
		# Let's just use cspace_sdf.wrap_samples_to_period logic.
		pts_wrapped = cspace_sdf.wrap_samples_to_period(pts_wrapped, self.axes)
		
		dists = self.interp_sdf(pts_wrapped)
		g0 = self.interp_g0(pts_wrapped)
		g1 = self.interp_g1(pts_wrapped)
		grads = np.stack([g0, g1], axis=1)
		
		return dists, grads

# --- Model ---
class PeriodicEmbedding(nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self, x):
		return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

def create_model(device):
	embedding = PeriodicEmbedding()
	model_core = select_model("sll", 4, 20, 128)
	model = nn.Sequential(embedding, model_core).to(device)
	return model

# --- Custom Training Loop with Gradient Loss ---
def train_with_gradient(model, loader, optimizer, epochs, grad_weight=1.0):
	model.train()
	loss_log = []
	for ep in range(epochs):
		ep_loss = 0.
		for pts, gt_dist, gt_grad in loader:
			pts.requires_grad = True
			optimizer.zero_grad()
			
			# Predict
			pred_dist = model(pts)
			
			# Gradient of Network
			# d(pred_dist)/d(pts)
			# We need to sum outputs to scalar to call grad, or use autograd.grad with ones
			net_grad = torch.autograd.grad(
				outputs=pred_dist, 
				inputs=pts, 
				grad_outputs=torch.ones_like(pred_dist),
				create_graph=True
			)[0]
			
			# Losses
			# 1. Regression (Dist)
			loss_dist = F.mse_loss(pred_dist.flatten(), gt_dist.flatten())
			
			# 2. Gradient Direction (Cosine? MSE?)
			# MSE is stronger as it enforces norm=1 (for SDF) and direction.
			loss_grad = F.mse_loss(net_grad, gt_grad)
			
			total_loss = loss_dist + grad_weight * loss_grad
			
			total_loss.backward()
			optimizer.step()
			ep_loss += total_loss.item()
		loss_log.append(ep_loss / len(loader))
	return loss_log

# --- Main Active Loop ---
def run_active_gradient(args):
	device = get_device()
	
	# 1. Init Oracle
	oracle = GradientOracle(args.world)
	
	# 2. Init Dataset (Seed) - Uniform
	print(f"Generating Seed ({args.n_seed})...")
	lo = np.array([oracle.t0_range[0], oracle.t1_range[0]])
	hi = np.array([oracle.t0_range[1], oracle.t1_range[1]])
	
	X_seed = lo + np.random.rand(args.n_seed, 2) * (hi - lo)
	Y_dist, Y_grad = oracle.query_batch(X_seed)
	
	# Store as tensors
	data_pts = torch.Tensor(X_seed).to(device)
	data_dist = torch.Tensor(Y_dist).float().to(device)
	data_grad = torch.Tensor(Y_grad).float().to(device) # (N, 2)
	
	# 3. Model
	model = create_model(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	
	# 4. Loop
	for it in range(args.n_iters):
		print(f"\n--- Iteration {it+1}/{args.n_iters} (Data: {len(data_pts)}) ---")
		
		# Train
		loader = DataLoader(TensorDataset(data_pts, data_dist, data_grad), batch_size=args.batch_size, shuffle=True)
		losses = train_with_gradient(model, loader, optimizer, args.epochs, grad_weight=0.5)
		print(f"Train Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
		
		# Query
		if it < args.n_iters - 1:
			# Generate candidates
			cand = lo + np.random.rand(args.n_query * 10, 2) * (hi - lo)
			cand_t = torch.Tensor(cand).to(device)
			
			with torch.no_grad():
				pred = model(cand_t).cpu().numpy().flatten()
				
			# Uncertainty: closer to 0
			unc = np.abs(pred)
			idx = np.argsort(unc)[:args.n_query]
			
			new_pts = cand[idx]
			new_dist, new_grad = oracle.query_batch(new_pts)
			
			print(f"Adding {len(new_pts)} points (Min Dist Est: {unc[idx[0]]:.5f})")
			
			# Append
			data_pts = torch.cat([data_pts, torch.Tensor(new_pts).to(device)])
			data_dist = torch.cat([data_dist, torch.Tensor(new_dist).float().to(device)])
			data_grad = torch.cat([data_grad, torch.Tensor(new_grad).float().to(device)])
			
	# Save
	torch.save(model.state_dict(), args.output + "_model.pt")
	
	# Render
	import render_model
	render_model.render(model, device, args.output + ".png")
	print("Done.")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--world", default="config/block_world.yaml")
	parser.add_argument("--output", default="output/active_grad_result")
	parser.add_argument("--n-seed", type=int, default=100, help="Start super small to show gradient power")
	parser.add_argument("--n-iters", type=int, default=5)
	parser.add_argument("--n-query", type=int, default=100)
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--batch-size", type=int, default=64)
	args = parser.parse_args()
	
	run_active_gradient(args)
