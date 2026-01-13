#!/usr/bin/env python3

import os
from types import SimpleNamespace
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import mouette as M

from common.models import select_model, count_parameters, save_model
from common.visualize import point_cloud_from_arrays
from common.training import Trainer
from common.utils import get_device, get_BB
from common.callback import LoggerCB, CheckpointCB, UpdateHkrRegulCB, Render2DCB, MarchingCubeCB

from utils import PeriodicEmbedding


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Training of a 1-Lipschitz architecture",
        description="This scripts runs the training optimization of a 1-Lipschitz neural network on some precomputed point cloud dataset."
    )
    parser.add_argument("--dataset", type=str, default="jaco", help="prefix of dataset files, e.g. 'block_world'")
    parser.add_argument("--data", type=str, default="inputs")
    parser.add_argument("-o", "--output-name", type=str, default="ndf")
    parser.add_argument("--unsigned", action="store_true", help="flag for training an unsigned distance field instead of a signed one")
    
    # model parameters
    parser.add_argument("-model","--model", choices=["ortho", "sll"], default="sll", help="Which Lipschitz architecture to consider. 'SLL' is the one used in the paper. 'Ortho' is the Bjorck orthonormalization-based architecture of Anil et al. (2019)")
    parser.add_argument("-n-layers", "--n-layers", type=int, default=20, help="number of layers in the network")
    parser.add_argument("-n-hidden", "--n-hidden", type=int, default=256, help="size of the layers")
    
    # optimization parameters
    parser.add_argument("-ne", "--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument('-bs',"--batch-size", type=int, default=200, help="Train batch size")
    parser.add_argument("-tbs", "--test-batch-size", type = int, default = 5000, help="Test batch size")
    parser.add_argument("-lr", "--learning-rate", type=float, default=5e-4, help="Adam's learning rate")
    parser.add_argument("-lm", "--loss-margin", type=float, default=1e-2, help="margin m in the hKR loss")
    parser.add_argument("-lmbd", "--loss-lambda", type=float, default=100., help="lambda in the hKR loss")
    parser.add_argument("--eikonal-weight", type=float, default=1e-4, help="Weight for Eikonal regularization")
    parser.add_argument("--frequencies", type=int, nargs='+', default=[1, 2], help="Frequencies for PeriodicEmbedding")
    
    parser.add_argument("-cp", "--checkpoint-freq", type=int, default=10, help="Number of epochs between each model save")
    parser.add_argument("-cpu", action="store_true", help="force training on CPU")
    parser.add_argument("--resume", type=str, default=None, help="path to a model checkpoint to resume from (e.g. output/ndf/model_final.pt)")

    args = parser.parse_args()

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
        output_folder = os.path.join("output", args.output_name if len(args.output_name)>0 else args.dataset)
    )

    os.makedirs(config.output_folder, exist_ok=True)
    print("DEVICE:", config.device)

    DIM : int = None
    #### Load train set ####
    if config.signed: # SIGNED MODE
        X_train_in = np.load(os.path.join(args.data, f"{args.dataset}_Xtrain_in.npy"))
        X_train_out = np.load(os.path.join(args.data, f"{args.dataset}_Xtrain_out.npy"))
        
        # Convert to torch tensors first
        X_train_in = torch.Tensor(X_train_in).to(config.device)
        X_train_out = torch.Tensor(X_train_out).to(config.device)
        
        # Then create DataLoaders
        loader_in = DataLoader(TensorDataset(X_train_in), batch_size=config.batch_size, shuffle=True, drop_last=True)
        loader_out = DataLoader(TensorDataset(X_train_out), batch_size=config.batch_size, shuffle=True, drop_last=True)
        train_loaders = (loader_in, loader_out)

        print(f"Succesfully loaded train set:\n", 
              f"Inside: {X_train_in.shape}\n", 
              f"Outside: {X_train_out.shape}")
        
        DIM = X_train_in.shape[1]
    else: # UNSIGNED MODE
        # Assuming Xtrain_on is the boundary points
        try:
            X_train_on = np.load(os.path.join(args.data, f"{args.dataset}_Xtrain_on.npy"))
        except FileNotFoundError:
            # Fallback if _on file is missing, maybe it's named differently or we use _in as boundary
            print("Warning: _Xtrain_on.npy not found, trying fallback...")
            X_train_on = np.load(os.path.join(args.data, f"{args.dataset}_Xtrain_in.npy"))
            
        X_train_out = np.load(os.path.join(args.data, f"{args.dataset}_Xtrain_out.npy"))
        
        X_train_on = torch.Tensor(X_train_on).to(config.device)
        X_train_out = torch.Tensor(X_train_out).to(config.device)
        
        loader_on = DataLoader(TensorDataset(X_train_on), batch_size=config.batch_size, shuffle=True, drop_last=True)
        loader_out = DataLoader(TensorDataset(X_train_out), batch_size=config.batch_size, shuffle=True, drop_last=True)
        train_loaders = (loader_on, loader_out)
        
        print(f"Succesfully loaded train set (unsigned):\n", 
              f"On: {X_train_on.shape}\n", 
              f"Outside: {X_train_out.shape}")
        
        DIM = X_train_out.shape[1]

    #### Load test set ####
    try:
        X_test = np.load(os.path.join(args.data, f"{args.dataset}_Xtest.npy"))
        Y_test = np.load(os.path.join(args.data, f"{args.dataset}_Ytest.npy"))
        X_test = torch.Tensor(X_test).to(config.device)
        Y_test = torch.Tensor(Y_test).to(config.device).unsqueeze(-1)
        test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=config.test_batch_size)
        print(f"Succesfully loaded test set: {X_test.shape}")
    except FileNotFoundError:
        print("Warning: test set not found, proceeding without testing.")
        test_loader = None

    if len(args.frequencies) == 0 or (len(args.frequencies) == 1 and args.frequencies[0] == 0):
        print("Using RAW coordinates (no PeriodicEmbedding)")
        model = select_model(args.model, DIM, args.n_layers, args.n_hidden).to(config.device)
    else:
        embedding = PeriodicEmbedding(frequencies=args.frequencies)
        model_core = select_model(args.model, DIM * 2 * len(args.frequencies), args.n_layers, args.n_hidden)
        model = nn.Sequential(embedding, model_core).to(config.device)
        model.id = model_core.id
        model.meta = model_core.meta

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

    print("PARAMETERS:", count_parameters(model))

    #### Export point cloud for visualization ####
    if config.signed:
        pc = point_cloud_from_arrays(
            (X_train_in.detach().cpu(), -1.),
            (X_train_out.detach().cpu(), 1.)
        )
    else:
        pc = point_cloud_from_arrays(
            (X_train_on.detach().cpu(), -1.),
            (X_train_out.detach().cpu(), 1.)
        )
    M.mesh.save(pc, os.path.join(config.output_folder, f"pc_0.geogram_ascii"))
    
    #### Building and calling the trainer ####
    callbacks = []
    callbacks.append(LoggerCB(os.path.join(config.output_folder, "log.csv")))
    
    if config.checkpoint_freq > 0:
        callbacks.append(CheckpointCB([x for x in range(0, config.n_epochs, config.checkpoint_freq) if x > 0]))
    
    plot_points = X_train_in if config.signed else X_train_on
    plot_domain = get_BB(plot_points, DIM, pad=0.5)
    
    # Override plot domain for angular space if needed, or stick to BB
    # plot_domain = M.geometry.AABB([0.0]*DIM, [2*np.pi]*DIM)

    if DIM == 2:
        callbacks.append(Render2DCB(config.output_folder, config.checkpoint_freq, plot_domain, res=500))
    else:
        callbacks.append(MarchingCubeCB(config.output_folder, config.checkpoint_freq, plot_domain, res=100, iso=0))
        
    callbacks.append(UpdateHkrRegulCB({1 : 1., 5 : 10., 10: 100., 30: config.loss_regul}))
	
    trainer = Trainer(train_loaders, test_loader, config)
    trainer.add_callbacks(*callbacks)
    
    if config.signed:
        trainer.train_lip(model)
    else:
        trainer.train_lip_unsigned(model)
			
    #### Save final model
    path = os.path.join(config.output_folder, "model_final.pt")
    save_model(model, path)
    print(f"Training finished. Model saved to {path}")
