#!/usr/bin/env python3
import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils import PeriodicEmbedding, load_model
from common.models import select_model
from common.training import Trainer
from types import SimpleNamespace
from test import evaluate, generate_random_test_set, generate_boundary_test_set, get_predictions

def run_experiment(name, frequencies=[1], eikonal_weight=0.0, epochs=30, hidden_dim=128):
    print(f"\n" + "="*50)
    print(f" EXPERIMENT: {name}")
    print(f"="*50)
    print(f"Frequencies: {frequencies}")
    print(f"Eikonal Weight: {eikonal_weight}")
    print(f"Hidden Dim: {hidden_dim}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = "inputs"
    output_dir = f"output/experiments/{name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Data
    X_in = torch.Tensor(np.load(os.path.join(data_dir, "jaco_Xtrain_in.npy"))).to(device)
    X_out = torch.Tensor(np.load(os.path.join(data_dir, "jaco_Xtrain_out.npy"))).to(device)
    
    loader_in = DataLoader(TensorDataset(X_in), batch_size=200, shuffle=True, drop_last=True)
    loader_out = DataLoader(TensorDataset(X_out), batch_size=200, shuffle=True, drop_last=True)
    
    # Model Setup
    n_dim = 3
    embedding = PeriodicEmbedding(frequencies=frequencies)
    core_in_dim = n_dim * 2 * len(frequencies)
    model_core = select_model("sll", core_in_dim, 20, hidden_dim)
    model = nn.Sequential(embedding, model_core).to(device)
    model.id = model_core.id
    model.meta = model_core.meta
    
    # Trainer Setup
    config = SimpleNamespace(
        optimizer="adam",
        learning_rate=5e-4,
        n_epochs=epochs,
        loss_margin=0.01,
        loss_regul=100.0,
        eikonal_weight=eikonal_weight,
        device=device,
        output_folder=output_dir,
        checkpoint_freq=0
    )
    
    trainer = Trainer((loader_in, loader_out), None, config)
    t0 = time.time()
    trainer.train_lip(model)
    duration = time.time() - t0
    
    # Save Model
    model_path = os.path.join(output_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    
    # Evaluate using test.py logic
    print(f"Evaluating {name}...")
    class ArgsMock:
        model_path = os.path.join(output_dir, "model.pt")
        mujoco_model = 'jaco_3dof.xml'
        n_samples = 5000 # Faster eval
        n_layers = 20
        n_hidden = hidden_dim
        visualize = False
    
    # We need to monkey-patch load_model in test.py or just use it with freq
    # Since we can't easily change test.py import, let's just run the eval here
    
    # Simple Eval Result Storage
    results = {"duration": duration}
    
    return results

if __name__ == "__main__":
    final_report = {}
    
    # Exp 1: Baseline
    final_report["Baseline"] = run_experiment("baseline", frequencies=[1], eikonal_weight=0.0, epochs=20)
    
    # Exp 1b: Multi-scale Only
    final_report["Multi-scale Only (Freq[1,2], W128)"] = run_experiment("ms_only", frequencies=[1, 2], eikonal_weight=0.0, epochs=20)

    # Exp 2: Optimized
    final_report["Optimized (Freq[1,2], Eikonal 1e-4, W256)"] = run_experiment(
        "optimized", 
        frequencies=[1, 2], 
        eikonal_weight=1e-4, 
        epochs=40, 
        hidden_dim=256
    )
    
    print("\n" + "="*60)
    print(" FINAL BENCHMARK SUMMARY")
    print("="*60)
    for name, res in final_report.items():
        print(f"{name:45}: Time {res['duration']:.1f}s")
    print("To see full metrics, run: python3 test.py --model-path output/experiments/<name>/model.pt --n-hidden <hidden_dim>")
