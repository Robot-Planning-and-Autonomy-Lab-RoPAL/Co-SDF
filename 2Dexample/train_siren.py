#!/usr/bin/env python3

import os
from types import SimpleNamespace
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import mouette as M

from common.models import *
from common.visualize import point_cloud_from_arrays
from common.training import Trainer
from common.utils import get_device, get_BB
from common.callback import *

"""
SIREN Training with Eikonal Loss
Uses MSE on occupancy labels (-1, 1) + Eikonal loss (|grad| = 1).
This matches the standard methodology for SIREN-based SDFs.
"""

if __name__ == "__main__":
    
    #### Commandline ####
    parser = argparse.ArgumentParser(
        prog="SIREN Eikonal Trainer",
        description="Trains a SIREN-based G-NDF using MSE + Eikonal loss."
    )
    
    # dataset parameters
    parser.add_argument("--dataset", type=str, default="block_world20", help="prefix of dataset files")
    parser.add_argument("--data", type=str, default="inputs/block_world20", help="directory where npy files are stored")    
    parser.add_argument("--output-name", type=str, default="block_world20_siren", help="custom output folder name")
    parser.add_argument("--n-hidden", type=int, default=256, help="size of the layers")
    parser.add_argument("--n-layers", type=int, default=5, help="number of layers")
    parser.add_argument("--model", type=str, default="gndf_siren", help="model type")
    
    # optimization parameters
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=200, help="Train batch size")
    parser.add_argument("--test-batch-size", type=int, default=5000, help="Test batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam's learning rate")
    
    # SIREN specific
    parser.add_argument("--eikonal-weight", type=float, default=0.1, help="Weight for Eikonal loss (|grad|=1)")
    
    # misc
    parser.add_argument("--checkpoint-freq", type=int, default=10, help="save frequency")
    parser.add_argument("--n-points-per-link", type=int, default=1)
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    
    args = parser.parse_args()
    
    device = "cpu" if args.cpu else get_device()
    
    #### Config ####
    config = SimpleNamespace(
        device = device,
        n_epochs = args.epochs,
        checkpoint_freq = args.checkpoint_freq,
        batch_size = args.batch_size,
        test_batch_size = args.test_batch_size,
        eikonal_weight = args.eikonal_weight,
        optimizer = "adam",
        learning_rate = args.learning_rate,
        output_folder = os.path.join("output", args.output_name)
    )
    os.makedirs(config.output_folder, exist_ok=True)
    print("DEVICE:", config.device)
    
    #### Load train set ####
    # For Eikonal training, we combine IN and OUT points with -1/1 targets
    X_in = np.load(os.path.join(args.data, f"{args.dataset}_Xtrain_in.npy"))
    X_out = np.load(os.path.join(args.data, f"{args.dataset}_Xtrain_out.npy"))
    
    X_all = np.vstack([X_in, X_out])
    Y_all = np.vstack([-np.ones((X_in.shape[0], 1)), np.ones((X_out.shape[0], 1))])
    
    X_all = torch.Tensor(X_all).to(config.device)
    Y_all = torch.Tensor(Y_all).to(config.device)
    
    train_loader = DataLoader(TensorDataset(X_all, Y_all), batch_size=config.batch_size, shuffle=True)
    print(f"Loaded dataset: {X_all.shape[0]} points (In: {X_in.shape[0]}, Out: {X_out.shape[0]})")
    
    #### Load test set ####
    X_test_path = os.path.join(args.data, f"{args.dataset}_Xtest.npy")
    Y_test_path = os.path.join(args.data,f"{args.dataset}_Ytest.npy")
    test_loader = None
    if os.path.exists(X_test_path) and os.path.exists(Y_test_path):
        X_test = torch.Tensor(np.load(X_test_path)).to(config.device)
        Y_test = torch.Tensor(np.load(Y_test_path)).reshape((-1, 1)).to(config.device)
        test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=config.test_batch_size)
        
    DIM = X_all.shape[1]
    
    #### Create model ####
    print(f"Using G-NDF-SIREN (Arch: {args.model})")
    model = select_model(args.model, DIM, args.n_layers, args.n_hidden, 
                    device=config.device, 
                    n_points_per_link=args.n_points_per_link).to(config.device)
    
    print("PARAMETERS:", count_parameters(model))
    
    #### Building and calling the trainer ####
    callbacks = []
    callbacks.append(LoggerCB(os.path.join(config.output_folder, "log.csv")))
    if config.checkpoint_freq > 0:
        callbacks.append(CheckpointCB(set(range(config.checkpoint_freq, config.n_epochs+1, config.checkpoint_freq))))
    
    # Rendering
    plot_domain = M.geometry.AABB([0.0, 0.0], [2*np.pi, 2*np.pi])
    callbacks.append(Render2DCB(config.output_folder, config.checkpoint_freq, plot_domain, res=500))
    
    trainer = Trainer(train_loader, test_loader, config)
    trainer.add_callbacks(*callbacks)
    
    print(f"STARTING SIREN EIKONAL TRAINING (Weight: {args.eikonal_weight})...")
    # train_full_info implements MSE loss on labels + Eikonal on gradients
    trainer.train_full_info(model)
    
    #### Save final model ####
    path = os.path.join(config.output_folder, "model_final.pt")
    save_model(model, path)
    print(f"Training complete. Model saved to {path}")
