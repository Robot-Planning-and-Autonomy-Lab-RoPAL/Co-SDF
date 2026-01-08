import os
import torch
import torch.nn as nn
import numpy as np
import mouette as M

from common.models import select_model
from common.visualize import reconstruct_surface_marching_cubes
from utils import PeriodicEmbedding

def generate_visualization():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    DIM = 3
    embedding = PeriodicEmbedding()
    model_core = select_model("sll", DIM * 2, 20, 128)
    model = nn.Sequential(embedding, model_core).to(device)
    
    model_path = "output/ndf/model_final.pt"
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
        print("Model weights loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Use -pi to pi domain
    plot_domain = M.geometry.AABB(
        [-np.pi, -np.pi, -np.pi],
        [np.pi, np.pi, np.pi]
    )

    print("Starting reconstruction in [-pi, pi]...")
    try:
        iso_surfaces = reconstruct_surface_marching_cubes(
            model, 
            plot_domain, 
            device, 
            iso=0, 
            res=100, 
            batch_size=5000)
        
        for (n, off), mesh in iso_surfaces.items():
            out_file = "output/ndf/final_3d_viz.obj"
            M.mesh.save(mesh, out_file)
            print(f"Saved 3D visualization to {out_file}")
            
    except Exception as e:
        print(f"Reconstruction failed: {e}")

if __name__ == "__main__":
    generate_visualization()
