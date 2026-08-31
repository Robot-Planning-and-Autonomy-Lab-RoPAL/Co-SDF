import torch
import torch.nn.functional as F
from tqdm import tqdm
from .loss import LossHKR
from .callback import Callback
import time

class Trainer:

    def __init__(self, train_loaders, config):
        self.config = config
        self.train_loaders = train_loaders
        self.optimizer = None
        self.callbacks = []
        self.metrics = dict()

    def log(self, *args):
        print(*args)

    def add_callbacks(self, *args):
        for cb in args:
            assert(isinstance(cb, Callback))
            self.callbacks.append(cb)

    def train_lip(self, model):
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        for epoch in range(self.config.n_epochs):
            self.metrics["epoch"] = epoch+1
            for cb in self.callbacks:
                cb.callOnBeginTrain(self, model)
            t0 = time.time()
            lossfun_hkr = LossHKR(self.config.loss_margin, self.config.loss_regul)
            train_loss = {"hkr": 0.}
            if self.config.eikonal_weight > 0.:
                train_loss["eik"] = 0.
            train_length = len(self.train_loaders[0])
            for (X_in, X_out) in tqdm(zip(*self.train_loaders), total=train_length):
                if X_in[0].shape != X_out[0].shape: continue
                self.optimizer.zero_grad()
                Y_in = model(X_in[0])
                Y_out = model(X_out[0])
                
                batch_loss_hkr = torch.sum(lossfun_hkr(-Y_in) + lossfun_hkr(Y_out))
                total_batch_loss = batch_loss_hkr

                if self.config.eikonal_weight > 0.:
                    X_mix = torch.cat([X_in[0], X_out[0]], dim=0)
                    X_mix.requires_grad = True
                    Y_mix = model(X_mix)
                    grad = torch.autograd.grad(Y_mix, X_mix, grad_outputs=torch.ones_like(Y_mix), create_graph=True)[0]
                    grad_q = grad[:, :2]
                    grad_norm = grad_q.norm(dim=-1)
                    
                    eik_scale = getattr(self.config, "eikonal_normalize_scale", 1.0)
                    batch_loss_eik = self.config.eikonal_weight * F.mse_loss(grad_norm / eik_scale, torch.ones_like(grad_norm))
                    total_batch_loss += batch_loss_eik
                    train_loss["eik"] += float(batch_loss_eik.detach())

                total_batch_loss.backward()
                train_loss["hkr"] += float(batch_loss_hkr.detach())
                self.optimizer.step() 
                for cb in self.callbacks:
                    cb.callOnEndForward(self, model)
            self.metrics["train_loss"] = train_loss
            self.metrics["epoch_time"] = time.time() - t0
            for cb in self.callbacks:
                cb.callOnEndTrain(self, model)
