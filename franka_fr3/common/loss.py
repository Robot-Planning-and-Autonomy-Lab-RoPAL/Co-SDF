import torch
import torch.nn.functional as F

class LossHKR:
    def __init__(self, margin=0.1, lbda=10.0):
        self.margin = margin
        self.lbda = lbda
        
    def __call__(self, signed_fx, weights=None):
        if weights is None:
            kr_loss = torch.mean(-signed_fx)
            hinge_loss = torch.mean(F.relu(self.margin - signed_fx))
        else:
            kr_loss = torch.mean(-signed_fx * weights)
            hinge_loss = torch.mean(F.relu(self.margin - signed_fx) * weights)
            
        return kr_loss + self.lbda * hinge_loss
class EikonalLoss:
    def __init__(self, target_grad_norm=1.0, normalize_scale=None):
        self.target_grad_norm = target_grad_norm
        self.normalize_scale = normalize_scale

    def __call__(self, y, x):
        if not y.requires_grad:
            grad = torch.zeros_like(x)
        else:
            grad_outputs = torch.ones_like(y)
            grad = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=grad_outputs,
                create_graph=True,
                only_inputs=True,
                allow_unused=True
            )[0]
            
            if grad is None:
                grad = torch.zeros_like(x)

        grad_norm = grad.norm(dim=-1).clamp_min(1e-8)
        
        if self.normalize_scale is not None:
            loss = torch.mean((grad_norm / self.normalize_scale - self.target_grad_norm)**2)
        else:
            loss = torch.mean((grad_norm - self.target_grad_norm)**2)
            
        return loss
