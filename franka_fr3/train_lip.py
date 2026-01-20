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

from utils import PeriodicEmbedding


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Training of a 1-Lipschitz architecture",
        description="This scripts runs the training optimization of a 1-Lipschitz neural network on some precomputed point cloud dataset."
    )
    parser.add_argument("--dataset", type=str, default="fr3", help="prefix of dataset files, e.g. 'block_world'")
    parser.add_argument("--data", type=str, default="inputs/bookshelf")
    parser.add_argument("-o", "--output-name", type=str, default="bookshelf")
    parser.add_argument("--unsigned", action="store_true", help="flag for training an unsigned distance field instead of a signed one")
    
    # model parameters
    parser.add_argument("-model","--model", choices=["ortho", "sll"], default="sll", help="Which Lipschitz architecture to consider. 'SLL' is the one used in the paper. 'Ortho' is the Bjorck orthonormalization-based architecture of Anil et al. (2019)")
    parser.add_argument("-n-layers", "--n-layers", type=int, default=20, help="number of layers in the network")
    parser.add_argument("-n-hidden", "--n-hidden", type=int, default=512, help="size of the layers")
    
    # optimization parameters
    parser.add_argument("-ne", "--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument('-bs',"--batch-size", type=int, default=200, help="Train batch size")
    parser.add_argument("-tbs", "--test-batch-size", type = int, default = 5000, help="Test batch size")
    parser.add_argument("-lr", "--learning-rate", type=float, default=5e-4, help="Adam's learning rate")
    parser.add_argument("-lm", "--loss-margin", type=float, default=1e-2, help="margin m in the hKR loss")
    parser.add_argument("-lmbd", "--loss-lambda", type=float, default=100., help="lambda in the hKR loss")
    parser.add_argument("--eikonal-weight", type=float, default=1e-2, help="Weight for Eikonal regularization")
    parser.add_argument("--eikonal-target", type=float, default=1.0, help="Target gradient norm for Eikonal loss")
    parser.add_argument("--eikonal-normalize-scale", type=float, default=1.0, help="Scale factor to normalize gradient norm (default 1.0)")
    parser.add_argument("--frequencies", type=int, nargs='+', default=[1, 2, 4, 8], help="Frequencies for PeriodicEmbedding")
    
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
        eikonal_target = args.eikonal_target,
        eikonal_normalize_scale = args.eikonal_normalize_scale,
        output_folder = os.path.join("output", args.output_name if len(args.output_name)>0 else args.dataset)
    )

    os.makedirs(config.output_folder, exist_ok=True)
    print("DEVICE:", config.device)

    DIM : int = None
    #### Load train set ####
    if config.signed: # SIGNED MODE
        X_train_in = np.load(os.path.join(args.data, f"{args.dataset}_Xtrain_in.npy"))
        X_train_out = np.load(os.path.join(args.data, f"{args.dataset}_Xtrain_out.npy"))
        
        # Load weights if they exist, otherwise default to 1.0
        try:
            W_train_in = np.load(os.path.join(args.data, f"{args.dataset}_Wtrain_in.npy"))
            W_train_out = np.load(os.path.join(args.data, f"{args.dataset}_Wtrain_out.npy"))
            print("Successfully loaded importance weights.")
        except FileNotFoundError:
            print("Warning: Importance weights not found, using uniform weights.")
            W_train_in = np.ones(len(X_train_in), dtype=np.float32)
            W_train_out = np.ones(len(X_train_out), dtype=np.float32)

        # Convert to torch tensors
        device = config.device
        X_train_in = torch.from_numpy(X_train_in).float().to(device)
        X_train_out = torch.from_numpy(X_train_out).float().to(device)
        W_train_in = torch.from_numpy(W_train_in).float().to(device)
        W_train_out = torch.from_numpy(W_train_out).float().to(device)
        
        # Then create DataLoaders with weights
        loader_in = DataLoader(TensorDataset(X_train_in, W_train_in), batch_size=config.batch_size, shuffle=True, drop_last=False)
        loader_out = DataLoader(TensorDataset(X_train_out, W_train_out), batch_size=config.batch_size, shuffle=True, drop_last=False)
        train_loaders = (loader_in, loader_out)

        print(f"Successfully loaded train set:\n", 
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

    #### Building and calling the trainer ####
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
			
    #### Save final model
    path = os.path.join(config.output_folder, "model_final.pt")
    save_model(model, path)
    print(f"Training finished. Model saved to {path}")
