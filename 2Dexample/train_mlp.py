#!/usr/bin/env python3
import os
from types import SimpleNamespace
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import mouette as M
from common.models import select_model, save_model, count_parameters
from common.training import Trainer
from common.utils import get_device
from common.callback import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLP/DeepSDF Eikonal Trainer")
    parser.add_argument("--world", type=str, default="block_world2", help="name of the world")
    parser.add_argument("--n-hidden", type=int, default=256, help="size of the layers")
    parser.add_argument("--n-layers", type=int, default=8, help="number of layers")
    parser.add_argument("--model", type=str, default="gndf_mlp", help="model type (gndf_mlp, mlp, mlps)")
    parser.add_argument("--inner-model", type=str, default="mlps", help="inner model type for gndf_mlp")
    
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=200, help="Train batch size")
    parser.add_argument("--test-batch-size", type=int, default=5000, help="Test batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Adam's learning rate")
    parser.add_argument("--eikonal-weight", type=float, default=0.1, help="Weight for Eikonal loss")
    parser.add_argument("--eikonal-normalize-scale", type=str, default="auto", help="Scale factor to normalize gradient norm (default 'auto')")
    parser.add_argument("--checkpoint-freq", type=int, default=10, help="save frequency")
    parser.add_argument("--n-points-per-link", type=int, default=1)
    
    args = parser.parse_args()
    
    data = f"inputs/{args.world}"
    output_name = f"{args.world}_mlp"
    device = get_device()
    
    config = SimpleNamespace(
        device = device,
        n_epochs = args.epochs,
        checkpoint_freq = args.checkpoint_freq,
        batch_size = args.batch_size,
        test_loader_batch_size = args.test_batch_size,
        test_batch_size = args.test_batch_size,
        eikonal_weight = args.eikonal_weight,
        eikonal_normalize_scale = args.eikonal_normalize_scale,
        optimizer = "adam",
        learning_rate = args.learning_rate,
        output_folder = os.path.join("output", output_name)
    )
    os.makedirs(config.output_folder, exist_ok=True)
    
    X_in = np.load(os.path.join(data, f"{args.world}_Xtrain_in.npy"))
    X_out = np.load(os.path.join(data, f"{args.world}_Xtrain_out.npy"))
    
    X_all = np.vstack([X_in, X_out])
    Y_all = np.vstack([-np.ones((X_in.shape[0], 1)), np.ones((X_out.shape[0], 1))])
    
    X_all = torch.Tensor(X_all).to(config.device)
    Y_all = torch.Tensor(Y_all).to(config.device)
    
    train_loader = DataLoader(TensorDataset(X_all, Y_all), batch_size=config.batch_size, shuffle=True)
    
    X_test_path = os.path.join(data, f"{args.world}_Xtest.npy")
    Y_test_path = os.path.join(data,f"{args.world}_Ytest.npy")
    test_loader = None
    if os.path.exists(X_test_path) and os.path.exists(Y_test_path):
        X_test = torch.Tensor(np.load(X_test_path)).to(config.device)
        Y_test = torch.Tensor(np.load(Y_test_path)).reshape((-1, 1)).to(config.device)
        test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=config.test_batch_size)
        
    DIM = X_all.shape[1]
    
    model = select_model(args.model, DIM, args.n_layers, args.n_hidden, 
                        device=config.device, 
                        n_points_per_link=args.n_points_per_link,
                        model_type=args.inner_model).to(config.device)
    
    if getattr(config, "eikonal_weight", 0) > 0:
        if config.eikonal_normalize_scale == "auto":
            preprocess = None
            if isinstance(model, nn.Sequential):
                preprocess = model[0]
            elif hasattr(model, "preprocess"):
                preprocess = model.preprocess
            
            if preprocess is not None and hasattr(preprocess, "xyz_std"):
                config.eikonal_normalize_scale = preprocess.xyz_std.item()
                print(f"AUTO-EIKONAL: Extracted scale {config.eikonal_normalize_scale:.4f} from model.")
            else:
                config.eikonal_normalize_scale = 1.0
                print("AUTO-EIKONAL: Warning: Could not extract xyz_std. Defaulting to 1.0.")
        else:
            try:
                config.eikonal_normalize_scale = float(config.eikonal_normalize_scale)
            except ValueError:
                config.eikonal_normalize_scale = 1.0
    
    callbacks = []
    callbacks.append(LoggerCB(os.path.join(config.output_folder, "log.csv")))
    if config.checkpoint_freq > 0:
        callbacks.append(CheckpointCB(set(range(config.checkpoint_freq, config.n_epochs+1, config.checkpoint_freq))))
    
    plot_domain = M.geometry.AABB([0.0, 0.0], [2*np.pi, 2*np.pi])
    callbacks.append(Render2DCB(config.output_folder, config.checkpoint_freq, plot_domain, res=500))
    
    trainer = Trainer(train_loader, test_loader, config)
    trainer.add_callbacks(*callbacks)
    
    trainer.train_full_info(model)
    
    path = os.path.join(config.output_folder, "model_final.pt")
    save_model(model, path)
    print(f"Training complete. Model saved to {path}")
