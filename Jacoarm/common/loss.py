
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
    def __init__(self):
        pass

    def __call__(self, y, x):
        """
        Calculates Eikonal loss: mean((|grad_x y| - 1)**2)
        
        Args:
            y: model output (requires grad)
            x: model input (requires grad)
        """
        # Calculate gradients
        grad = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute loss: (|grad| - 1)^2
        grad_norm = grad.norm(dim=-1)
        loss = torch.mean((grad_norm - 1)**2)
        return loss