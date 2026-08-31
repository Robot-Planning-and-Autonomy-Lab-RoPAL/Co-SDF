import torch
import torch.nn as nn

from tqdm import tqdm
from .loss import EikonalLoss, LossHKR
from .callback import Callback

import mouette as M
import time

class Trainer(M.Logger):

    def __init__(self, train_loaders, test_loader, config):
        super().__init__("Training")
        self.config = config
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        self.optimizer = None
        self.callbacks = []
        self.metrics = dict()
    
    def get_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

    def add_callbacks(self, *args):
        for cb in args:
            assert(isinstance(cb, Callback))
            self.callbacks.append(cb)

    def evaluate_model(self, model):
        if self.test_loader is None: return
        test_loss = 0.
        testlossfun = nn.MSELoss()
        for inputs,labels in self.test_loader:
            outputs = model(inputs)
            loss = testlossfun(outputs, labels)
            test_loss += loss.item()
        self.metrics["test_loss"] = test_loss
        for cb in self.callbacks:
            cb.callOnEndTest(self, model)

    def train_lip(self, model):
        self.optimizer = self.get_optimizer(model)
        for epoch in range(self.config.n_epochs):
            self.metrics["epoch"] = epoch+1
            for cb in self.callbacks:
                cb.callOnBeginTrain(self, model)
            t0 = time.time()
            lossfun_hkr = LossHKR(self.config.loss_margin, self.config.loss_regul)
            
            eik_target = getattr(self.config, "eikonal_target", 1.0)
            eik_scale = getattr(self.config, "eikonal_normalize_scale", None)
            lossfun_eik = EikonalLoss(target_grad_norm=eik_target, normalize_scale=eik_scale)
            train_loss = dict()
            train_loss["hkr"] = 0.
            lengths = [len(loader) for loader in self.train_loaders]
            train_length = max(lengths)
            iters = [iter(loader) for loader in self.train_loaders]

            for _ in tqdm(range(train_length), total=train_length):
                batches = []
                for i in range(len(iters)):
                    try:
                        batch = next(iters[i])
                    except StopIteration:
                        iters[i] = iter(self.train_loaders[i])
                        batch = next(iters[i])
                    batches.append(batch)
                
                X_in_batch, X_out_batch = batches
                self.optimizer.zero_grad()
                
                points_in = X_in_batch[0]
                points_out = X_out_batch[0]
                
                if hasattr(self.config, "eikonal_weight") and self.config.eikonal_weight > 0:
                    points_in.requires_grad = True
                    points_out.requires_grad = True

                Y_in = model(points_in)
                Y_out = model(points_out)
                
                batch_loss_hkr = torch.sum(lossfun_hkr(-Y_in) + lossfun_hkr(Y_out))
                total_batch_loss = batch_loss_hkr
                
                if hasattr(self.config, "eikonal_weight") and self.config.eikonal_weight > 0:
                    eik_in = lossfun_eik(Y_in, points_in)
                    eik_out = lossfun_eik(Y_out, points_out)
                    eik_loss = (eik_in + eik_out) / 2
                    total_batch_loss += self.config.eikonal_weight * eik_loss
                    if "eik" not in train_loss: train_loss["eik"] = 0.
                    train_loss["eik"] += float(eik_loss.detach())

                total_batch_loss.backward()
                train_loss["hkr"] += float(batch_loss_hkr.detach())
                self.optimizer.step() 
                for cb in self.callbacks:
                    cb.callOnEndForward(self, model)
            self.metrics["train_loss"] = train_loss
            self.metrics["epoch_time"] = time.time() - t0
            self.evaluate_model(model)
            for cb in self.callbacks:
                cb.callOnEndTrain(self, model)
