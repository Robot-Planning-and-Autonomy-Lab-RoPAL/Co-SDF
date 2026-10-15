#!/usr/bin/env python3

import os
from types import SimpleNamespace
import argparse
import warnings
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from common.models import count_parameters, save_model
from common.models.cosdf import COSDF
from common.training import Trainer
from common.utils import get_device
from common.callback import LoggerCB, CheckpointCB, UpdateHkrRegulCB


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="Training of a 1-Lipschitz architecture")
    parser.add_argument("--world", type=str, default="bookshelf")
    parser.add_argument("--n-layers", type=int, default=20)
    parser.add_argument("--n-hidden", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--test-batch-size", type = int, default = 5000)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--loss-margin", type=float, default=1e-2)
    parser.add_argument("--loss-lambda", type=float, default=100.)
    parser.add_argument("--eikonal-weight", type=float, default=0.1)
    parser.add_argument("--eikonal-target", type=float, default=1.0)
    parser.add_argument("--eikonal-normalize-scale", type=str, default="auto")
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)

    warnings.filterwarnings(
        "ignore",
        message=r"Attempting to run cuBLAS, but there was no current CUDA context! Attempting to set the primary context\.\.\.",
        category=UserWarning,
        module=r"torch\.autograd\.graph",
    )
    
    args = parser.parse_args()

    dataset_prefix = "fr3"
    data_dir = os.path.join("inputs", args.world)
    output_dir = os.path.join("output", args.world)

    config = SimpleNamespace(
        device = get_device(),
        n_epochs = args.epochs,
        batch_size = args.batch_size,
        test_batch_size = args.test_batch_size,
        loss_margin = args.loss_margin,
        loss_regul = args.loss_lambda,
        learning_rate = args.learning_rate,
        eikonal_weight = args.eikonal_weight,
        eikonal_target = args.eikonal_target,
        eikonal_normalize_scale = args.eikonal_normalize_scale,
        output_folder = output_dir
    )

    os.makedirs(config.output_folder, exist_ok=True)
    print("DEVICE:", config.device)

    X_train_in = np.load(os.path.join(data_dir, f"{dataset_prefix}_Xtrain_in.npy"))
    X_train_in = torch.tensor(X_train_in, dtype=torch.float32, device=config.device)
    loader_in = DataLoader(TensorDataset(X_train_in), batch_size=config.batch_size, shuffle=True)

    X_train_out = np.load(os.path.join(data_dir, f"{dataset_prefix}_Xtrain_out.npy"))
    X_train_out = torch.tensor(X_train_out, dtype=torch.float32, device=config.device)
    loader_out = DataLoader(TensorDataset(X_train_out), batch_size=config.batch_size, shuffle=True)

    train_loaders = (loader_in, loader_out)
    print(
        "Successfully loaded train set:\n",
        f"Inside: {X_train_in.shape}\n",
        f"Outside: {X_train_out.shape}",
    )

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

    model = COSDF(
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        device=config.device,
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=config.device)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Resumed from {args.resume}")

    if config.eikonal_weight > 0:
        if args.eikonal_normalize_scale == "auto":
            config.eikonal_normalize_scale = model[0].xyz_std.item()
        else:
            try:
                config.eikonal_normalize_scale = float(args.eikonal_normalize_scale)
            except ValueError:
                config.eikonal_normalize_scale = 1.0
                print(f"AUTO-EIKONAL: Warning: Invalid scale '{args.eikonal_normalize_scale}'. Defaulting to 1.0.")

    print("PARAMETERS:", count_parameters(model))

    callbacks = []
    callbacks.append(LoggerCB(os.path.join(config.output_folder, "log.csv")))
    
    if args.checkpoint_freq > 0:
        callbacks.append(CheckpointCB([x for x in range(0, config.n_epochs, args.checkpoint_freq) if x > 0]))
    
    callbacks.append(UpdateHkrRegulCB({1 : 1., 5 : 10., 10: 100., 30: config.loss_regul}))
	
    trainer = Trainer(train_loaders, test_loader, config)
    trainer.add_callbacks(*callbacks)
    
    trainer.train_lip(model)
			
    path = os.path.join(config.output_folder, "model_final.pt")
    save_model(model, path)
    print(f"Training finished. Model saved to {path}")
