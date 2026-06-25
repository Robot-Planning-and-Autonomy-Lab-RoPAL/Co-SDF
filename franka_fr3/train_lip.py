#!/usr/bin/env python3

import os
from types import SimpleNamespace
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from common.models import select_model, count_parameters, save_model
from common.training import Trainer
from common.utils import get_device
from common.callback import LoggerCB, CheckpointCB, UpdateHkrRegulCB


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Training of a 1-Lipschitz architecture",
        description="This scripts runs the training optimization of a 1-Lipschitz neural network on some precomputed point cloud dataset."
    )
    parser.add_argument("--world", type=str, default="obs30", help="sence name")
    parser.add_argument("--unsigned", action="store_true", help="flag for training an unsigned distance field instead of a signed one")
    
    # model parameters
    parser.add_argument("--model", choices=["sll", "gndf"], default="gndf", help="Which Lipschitz architecture to consider. 'SLL' is the one used in the paper. 'Ortho' is the Bjorck orthonormalization-based architecture of Anil et al. (2019). 'GNDF' uses FK-based geometric features.")
    parser.add_argument("--n-layers", type=int, default=20, help="number of layers in the network")
    parser.add_argument("--n-hidden", type=int, default=512, help="size of the layers")
    
    # optimization parameters
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=200, help="Train batch size")
    parser.add_argument("--test-batch-size", type = int, default = 5000, help="Test batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Adam's learning rate")
    parser.add_argument("--loss-margin", type=float, default=1e-2, help="margin m in the hKR loss")
    parser.add_argument("--loss-lambda", type=float, default=100., help="lambda in the hKR loss")
    parser.add_argument("--eikonal-weight", type=float, default=0.1, help="Weight for Eikonal regularization")
    parser.add_argument("--eikonal-target", type=float, default=1.0, help="Target gradient norm for Eikonal loss")
    parser.add_argument("--eikonal-normalize-scale", type=str, default="auto", help="Scale factor to normalize gradient norm (default 'auto' extracts from model)")
    parser.add_argument("--n-points-per-link", type=int, default=1, help="Number of control points per link for G-NDF")
    
    parser.add_argument("--checkpoint-freq", type=int, default=10, help="Number of epochs between each model save")
    parser.add_argument("--cpu", action="store_true", help="force training on CPU")
    parser.add_argument("--resume", type=str, default=None, help="path to a model checkpoint to resume from (e.g. output/ndf/model_final.pt)")

    args = parser.parse_args()

    dataset_prefix = "fr3"
    data_dir = os.path.join("inputs", args.world)
    output_dir = os.path.join("output", args.world)

    config = SimpleNamespace(
        signed = not args.unsigned,
        device = get_device(),
        n_epochs = args.epochs,
        checkpoint_freq = args.checkpoint_freq,
        batch_size = args.batch_size,
        test_batch_size = args.test_batch_size,
        loss_margin = args.loss_margin,
        loss_regul = args.loss_lambda,
        optimizer = "adam",
        learning_rate = args.learning_rate,
        eikonal_weight = args.eikonal_weight,
        eikonal_target = args.eikonal_target,
        eikonal_normalize_scale = args.eikonal_normalize_scale,
        output_folder = output_dir
    )

    os.makedirs(config.output_folder, exist_ok=True)
    print("DEVICE:", config.device)

    DIM : int = None
    if config.signed:
        X_train_in = np.load(os.path.join(data_dir, f"{dataset_prefix}_Xtrain_in.npy"))
        X_train_in = torch.Tensor(X_train_in).to(config.device)
        loader_in = DataLoader(TensorDataset(X_train_in), batch_size=config.batch_size, shuffle=True)
        
        X_train_out = np.load(os.path.join(data_dir, f"{dataset_prefix}_Xtrain_out.npy"))
        X_train_out = torch.Tensor(X_train_out).to(config.device)
        loader_out = DataLoader(TensorDataset(X_train_out), batch_size=config.batch_size, shuffle=True)
        
        train_loaders = (loader_in, loader_out)

        print(f"Successfully loaded train set:\n", 
              f"Inside: {X_train_in.shape}\n", 
              f"Outside: {X_train_out.shape}")
        
        DIM = X_train_in.shape[1]
    else:
        try:
            X_train_on = np.load(os.path.join(data_dir, f"{dataset_prefix}_Xtrain_on.npy"))
        except FileNotFoundError:
            print("Warning: _Xtrain_on.npy not found, trying fallback...")
            X_train_on = np.load(os.path.join(data_dir, f"{dataset_prefix}_Xtrain_in.npy"))
            
        X_train_out = np.load(os.path.join(data_dir, f"{dataset_prefix}_Xtrain_out.npy"))
        
        X_train_on = torch.from_numpy(X_train_on).float().to(config.device)
        X_train_out = torch.from_numpy(X_train_out).float().to(config.device)
        
        loader_on = DataLoader(TensorDataset(X_train_on), batch_size=config.batch_size, shuffle=True, drop_last=False)
        loader_out = DataLoader(TensorDataset(X_train_out), batch_size=config.batch_size, shuffle=True, drop_last=False)
        train_loaders = (loader_on, loader_out)
        
        print(f"Succesfully loaded train set (unsigned):\n", 
              f"On: {X_train_on.shape}\n", 
              f"Outside: {X_train_out.shape}")
        
        DIM = X_train_out.shape[1]

    #### Load test set ####
    try:
        X_test = np.load(os.path.join(data_dir, f"{dataset_prefix}_Xtest.npy"))
        Y_test = np.load(os.path.join(data_dir, f"{dataset_prefix}_Ytest.npy"))
        X_test = torch.Tensor(X_test).to(config.device)
        Y_test = torch.Tensor(Y_test).to(config.device).unsqueeze(-1)
        test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=config.test_batch_size)
        print(f"Succesfully loaded test set: {X_test.shape}")
    except FileNotFoundError:
        print("Warning: test set not found, proceeding without testing.")
        test_loader = None

    if args.model.lower() == "gndf":
        print("Using G-NDF (FK-based Geometric Feature NDF)")
        model = select_model(
            args.model, DIM, args.n_layers, args.n_hidden, 
            device=config.device,
            n_points_per_link=args.n_points_per_link
        ).to(config.device)
    else:
        print(f"Using {args.model.upper()} model")
        model = select_model(args.model, DIM, args.n_layers, args.n_hidden).to(config.device)

    if args.resume:
        try:
            checkpoint = torch.load(args.resume, map_location=config.device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint) # raw state dict support
            print(f"Resumed from {args.resume}")
        except Exception as e:
            print(f"Error loading checkpoint {args.resume}: {e}")

    if config.eikonal_weight > 0:
        if args.eikonal_normalize_scale == "auto":
            preprocess = None
            if isinstance(model, nn.Sequential):
                preprocess = model[0]
            elif hasattr(model, "preprocess"):
                preprocess = model.preprocess
            
            if preprocess is not None and hasattr(preprocess, "xyz_std"):
                config.eikonal_normalize_scale = preprocess.xyz_std.item()
                print(f"AUTO-EIKONAL: Extracted scale {config.eikonal_normalize_scale:.4f} from model statistics.")
            else:
                config.eikonal_normalize_scale = 1.0
                print("AUTO-EIKONAL: Warning: Could not extract xyz_std from model. Defaulting to 1.0.")
        else:
            try:
                config.eikonal_normalize_scale = float(args.eikonal_normalize_scale)
            except ValueError:
                config.eikonal_normalize_scale = 1.0
                print(f"AUTO-EIKONAL: Warning: Invalid scale '{args.eikonal_normalize_scale}'. Defaulting to 1.0.")

    print("PARAMETERS:", count_parameters(model))

    callbacks = []
    callbacks.append(LoggerCB(os.path.join(config.output_folder, "log.csv")))
    
    if config.checkpoint_freq > 0:
        callbacks.append(CheckpointCB([x for x in range(0, config.n_epochs, config.checkpoint_freq) if x > 0]))
    
    callbacks.append(UpdateHkrRegulCB({1 : 1., 5 : 10., 10: 100., 30: config.loss_regul}))
	
    trainer = Trainer(train_loaders, test_loader, config)
    trainer.add_callbacks(*callbacks)
    
    if config.signed:
        trainer.train_lip(model)
    else:
        trainer.train_lip_unsigned(model)
			
    path = os.path.join(config.output_folder, "model_final.pt")
    save_model(model, path)
    print(f"Training finished. Model saved to {path}")