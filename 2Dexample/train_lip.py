#!/usr/bin/env python3
import os
from types import SimpleNamespace
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from common.models import count_parameters, save_model, select_model
from common.training import Trainer
from common.utils import get_device
from common.callback import CheckpointCB, LoggerCB, Render2DCB, UpdateHkrRegulCB

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(prog="Training of a 1-Lipschitz architecture")
    parser.add_argument("--world", type=str, default="block_world1", help="name of the world")
    parser.add_argument("--n-hidden", type=int, default=256, help="size of the layers")
    parser.add_argument("--n-layers", type=int, default=10, help="number of layers in the network")
    parser.add_argument("--model", type=str, default="cosdf", help="which model to train (SLL or COSDF)")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=200, help="Train batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Adam's learning rate")
    parser.add_argument("--loss-margin", type=float, default=1e-2, help="margin m in the hKR loss")
    parser.add_argument("--loss-lambda", type=float, default=100.0, help="lambda in the hKR loss")
    parser.add_argument("--eikonal-weight", type=float, default=0.1, help="Weight of Eikonal loss (gradient norm ≈ 1 on θ)")
    parser.add_argument("--eikonal-normalize-scale", type=str, default="auto", help="Scale factor to normalize gradient norm (default 'auto' extracts from model)")
    parser.add_argument("--checkpoint-freq", type=int, default=10, help="Number of epochs between each model save")
    
    args = parser.parse_args()
    
    data = f"inputs/{args.world}"
    device = get_device()
    print("Using device:", device)

    config = SimpleNamespace(
        device = device,
        n_epochs = args.epochs,
        checkpoint_freq = args.checkpoint_freq,
        batch_size = args.batch_size,
        loss_margin = args.loss_margin,
        loss_regul = args.loss_lambda,
        eikonal_weight = args.eikonal_weight,
        eikonal_normalize_scale = args.eikonal_normalize_scale,
        learning_rate = args.learning_rate,
        output_folder = os.path.join("output", args.world)
    )
    os.makedirs(config.output_folder, exist_ok=True)
    
    DIM : int = None
    
    X_train_in = np.load(os.path.join(data, f"{args.world}_Xtrain_in.npy"))
    X_train_in = torch.Tensor(X_train_in).to(config.device)
    loader_in = DataLoader(TensorDataset(X_train_in), batch_size=config.batch_size, shuffle=True)

    X_train_out = np.load(os.path.join(data, f"{args.world}_Xtrain_out.npy"))
    X_train_out = torch.Tensor(X_train_out).to(config.device)
    loader_out = DataLoader(TensorDataset(X_train_out), batch_size=config.batch_size, shuffle=True)

    print("Succesfully loaded train set:\n",
          f"Inside: {X_train_in.shape}\n",
          f"Outside: {X_train_out.shape}")
        
    DIM = X_train_out.shape[1]
    
    if args.model.lower() == "cosdf":
        print("Using COSDF (configuration-space obstacle signed distance field)")
        if DIM % 4 != 0:
            raise ValueError(f"Invalid Co-SDF input dimension: {DIM}")
        model = select_model(args.model, DIM, args.n_layers, args.n_hidden, 
                        device=config.device, 
                        n_points_per_link=DIM // 4).to(config.device)
    else:
        class PeriodicEmbedding(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x):
                return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

        embedding = PeriodicEmbedding()
        model_core = select_model(args.model, DIM * 2, args.n_layers, args.n_hidden)
        model = nn.Sequential(embedding, model_core).to(config.device)
        model.id = model_core.id
        model.meta = model_core.meta
    
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
    
    print("PARAMETERS:", count_parameters(model))
    
    callbacks = []
    callbacks.append(LoggerCB(os.path.join(config.output_folder, "log.csv")))
    if config.checkpoint_freq > 0:
        checkpoint_epochs = set(x for x in range(config.checkpoint_freq, config.n_epochs + 1, config.checkpoint_freq))
        callbacks.append(CheckpointCB(checkpoint_epochs))
    
    if args.model.lower() == "cosdf":
        plot_domain = ([0.0, 0.0], [2*np.pi, 2*np.pi])
        callbacks.append(Render2DCB(config.output_folder, config.checkpoint_freq, plot_domain, res=500))
    
    callbacks.append(UpdateHkrRegulCB({1 : 1., 5 : 10., 10: 100., 30: config.loss_regul}))
    
    trainer = Trainer((loader_in, loader_out), config)
    trainer.add_callbacks(*callbacks)
    trainer.train_lip(model)
        
    path = os.path.join(config.output_folder, "model_final.pt")
    save_model(model, path)
