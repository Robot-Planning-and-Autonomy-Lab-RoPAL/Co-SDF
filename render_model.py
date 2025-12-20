#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import mouette as M
import argparse
from types import SimpleNamespace
from common.models import select_model
from common.utils import get_device

# Re-define Embedding needed to load model
class PeriodicEmbedding(nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self, x):
		return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

def load_periodic_model(path, device):
	# We need to reconstruct the structure
	# Loading state dictionary requires matching keys
	checkpoint = torch.load(path, map_location=device)
	
	# Metadata might be inside or we infer
	# Assuming SLL, 20 layers, 128 hidden from our defaults
	# Or read from checkpoint if saved with meta
	
	embedding = PeriodicEmbedding()
	# Check if meta exists
	if "meta" in checkpoint:
		meta = checkpoint["meta"] # [dim_in, dim_hidden, n_layers]
		print(f"Loading model with meta: {meta}")
		# Note: stored meta[0] is likely 4 (after embedding)
		model_core = select_model("sll", meta[0], meta[2], meta[1]) 
	else:
		print("Warning: No meta found, using default SLL 20x128")
		model_core = select_model("sll", 4, 20, 128)
		
	model = nn.Sequential(embedding, model_core).to(device)
	
	# Load weights
	# The saved model from train_active_learning used save_model which saves dict
	if "state_dict" in checkpoint:
		model.load_state_dict(checkpoint["state_dict"])
	else:
		model.load_state_dict(checkpoint) # raw
		
	return model

def render(model, device, output_path):
	model.eval()
	
	# Grid
	res = 500
	x = np.linspace(0, 2*np.pi, res)
	y = np.linspace(0, 2*np.pi, res)
	xv, yv = np.meshgrid(x, y)
	
	# Flatten for batch prediction
	pts = np.stack([xv.flatten(), yv.flatten()], axis=1)
	pts_t = torch.Tensor(pts).to(device)
	
	# Predict in chunks to save VRAM
	vals = []
	batch_size = 10000
	with torch.no_grad():
		for i in range(0, len(pts_t), batch_size):
			batch = pts_t[i:i+batch_size]
			v = model(batch)
			vals.append(v.cpu().numpy())
	
	vals = np.concatenate(vals).reshape(res, res)
	
	# Plot
	plt.figure(figsize=(6,6))
	# Plot SDF
	plt.imshow(vals, origin="lower", extent=[0, 2*np.pi, 0, 2*np.pi], cmap="RdBu", vmin=-1, vmax=1)
	plt.colorbar(label="Predicted Distance")
	
	# Contour 0
	plt.contour(xv, yv, vals, levels=[0], colors='black', linewidths=2)
	
	plt.title("Active Learning Model Prediction")
	plt.xlabel(r"$\theta_1$")
	plt.ylabel(r"$\theta_2$")
	plt.tight_layout()
	plt.savefig(output_path, dpi=300)
	print(f"Saved to {output_path}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("model_path")
	parser.add_argument("output_path")
	args = parser.parse_args()
	
	device = get_device()
	model = load_periodic_model(args.model_path, device)
	render(model, device, args.output_path)
