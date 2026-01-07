#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from types import SimpleNamespace

# Import from common and local utils
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '2Dexample'))
from common.models import select_model
from common.training import Trainer
from common.utils import get_device
from common.callback import LoggerCB, CheckpointCB, UpdateHkrRegulCB

from utils import PeriodicEmbedding, count_parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="jaco")
    parser.add_argument("--data", type=str, default="inputs/jaco_ndf")
    parser.add_argument("-o", "--output-name", type=str, default="jaco_ndf")
    parser.add_argument("-model", "--model", choices=["ortho", "sll"], default="sll")
    parser.add_argument("-n-layers", "--n-layers", type=int, default=20)
    parser.add_argument("-n-hidden", "--n-hidden", type=int, default=128)
    parser.add_argument("-ne", "--epochs", type=int, default=100)
    parser.add_argument("-bs", "--batch-size", type=int, default=256)
    parser.add_argument("-lr", "--learning-rate", type=float, default=5e-4)
    args = parser.parse_args()

    config = SimpleNamespace(
        signed=True,
        device=get_device(),
        n_epochs=args.epochs,
        checkpoint_freq=10,
        batch_size=args.batch_size,
        test_batch_size=5000,
        loss_margin=1e-2,
        loss_regul=100.0,
        optimizer="adam",
        learning_rate=args.learning_rate,
        output_folder=os.path.join("output", args.output_name)
    )
    os.makedirs(config.output_folder, exist_ok=True)

    # 1. Load Data
    X_train_in = np.load(os.path.join(args.data, f"{args.dataset}_Xtrain_in.npy"))
    X_train_out = np.load(os.path.join(args.data, f"{args.dataset}_Xtrain_out.npy"))
    
    X_train_in = torch.Tensor(X_train_in).to(config.device)
    X_train_out = torch.Tensor(X_train_out).to(config.device)
    
    loader_in = DataLoader(TensorDataset(X_train_in), batch_size=config.batch_size, shuffle=True, drop_last=True)
    loader_out = DataLoader(TensorDataset(X_train_out), batch_size=config.batch_size, shuffle=True, drop_last=True)
    
    print(f"Loaded: {len(X_train_in)} in, {len(X_train_out)} out")

    # 2. Build Model
    DIM = X_train_in.shape[1] # Should be 3

    embedding = PeriodicEmbedding()

    model_core = select_model(args.model, DIM * 2, args.n_layers, args.n_hidden)
    model = nn.Sequential(embedding, model_core).to(config.device)
    
    # Required for the trainer
    model.id = model_core.id
    model.meta = model_core.meta

    print("Parameters:", count_parameters(model))

    # 3. Train
    callbacks = [
        LoggerCB(os.path.join(config.output_folder, "log.csv")),
        CheckpointCB([x for x in range(0, config.n_epochs, config.checkpoint_freq) if x > 0]),
        UpdateHkrRegulCB({1: 1., 5: 10., 10: 100., 30: config.loss_regul})
    ]

    trainer = Trainer((loader_in, loader_out), None, config)
    trainer.add_callbacks(*callbacks)
    trainer.train_lip(model)

    # 4. Save
    torch.save(model.state_dict(), os.path.join(config.output_folder, "model_final.pt"))
    print(f"Model saved to {config.output_folder}")
