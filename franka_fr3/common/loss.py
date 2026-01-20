import torch
import torch.nn.functional as F

class LossHKR:
    """
    Hinge Kantorovich–Rubinstein loss
    RSS20-compatible version:
    - free samples:   f(x)
    - obstacle samples: -f(x)
    """
    
    def __init__(self, margin=0.1, lbda=10.0):
        self.margin = margin
        self.lbda = lbda
        
    def __call__(self, signed_fx, weights=None):
        """
        Args:
            signed_fx: signed predictions
            weights: optional importance weights for each sample
        """
        if weights is None:
            # KR term: E[- signed f(x)]
            kr_loss = torch.mean(-signed_fx)
            # Hinge term: max(0, m - signed f(x))
            hinge_loss = torch.mean(F.relu(self.margin - signed_fx))
        else:
            kr_loss = torch.mean(-signed_fx * weights)
            hinge_loss = torch.mean(F.relu(self.margin - signed_fx) * weights)
            
        return kr_loss + self.lbda * hinge_loss
    
    
    
def vector_alignment_loss(y, target):
    return (1-F.cosine_similarity(y, target, dim = 1)*2).mean()

class SALLoss:
    """SAL: Sign Agnostic Learning of Shapes from Raw Data"""
    def __init__(self, l=1., metric="l2"):
        self.l = l

        self.callfun = {
            "l2" : self.SAL_l2,
            "l0" : self.SAL_l0
        }.get(metric, self.SAL_l2)

    def __call__(self, y_pred, y_target):
        return self.callfun(y_pred, y_target)

    def SAL_l2(self,y_pred,y_target):
        return torch.mean(torch.abs(torch.abs(y_pred) - y_target)**self.l)
    
    def SAL_l0(self,y_pred,y_target):
        return torch.mean(torch.abs(torch.abs(y_pred) - 1)**self.l)
    

class SALDLoss:
    """
    SALD: Sign Agnostic Learning with Derivatives
    
    Improving SAL using losses on gradients
    """
    def __init__(self):
        pass

    def __call__(self,y_pred,y_target):
        return torch.min( torch.norm(y_pred-y_target), torch.norm(y_pred+y_target))

class EikonalLoss:
    """
    Eikonal Regularization: enforces |grad f(x)| = 1
    Used for stabilizing Distance Field learning.
    """
    def __init__(self, target_grad_norm=1.0, normalize_scale=None):
        """
        Args:
            target_grad_norm: The desired norm of the gradient (default 1.0).
            normalize_scale: If provided, scales the gradient norm before comparison.
                             Useful for compensating PeriodicEmbedding amplification.
        """
        self.target_grad_norm = target_grad_norm
        self.normalize_scale = normalize_scale

    def __call__(self, y, x):
        """
        Calculates Eikonal loss: mean((|grad_x y|/scale - target)**2)
        
        Args:
            y: model output (requires grad)
            x: model input (requires grad)
        """
        if not y.requires_grad:
            # If y doesn't require grad, its gradient with respect to x is zero
            grad = torch.zeros_like(x)
        else:
            # Calculate gradients
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
                # If y doesn't depend on x, gradient is zero
                grad = torch.zeros_like(x)
        
        # Compute loss: (|grad|/scale - target)^2 with clamping for stability
        grad_norm = grad.norm(dim=-1).clamp_min(1e-8)
        
        if self.normalize_scale is not None:
            loss = torch.mean((grad_norm / self.normalize_scale - self.target_grad_norm)**2)
        else:
            loss = torch.mean((grad_norm - self.target_grad_norm)**2)
            
        return loss