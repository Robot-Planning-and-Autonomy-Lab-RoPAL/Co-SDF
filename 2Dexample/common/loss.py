
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
                       +f(x) for C_free
                       -f(x) for C_obs
            weights: optional importance weights for each sample
        """
        
        if weights is None:
            # KR term: E[- signed f(x)]
            kr_loss = torch.mean(-signed_fx)
            # Hinge term: max(0, m - signed f(x))
            hinge_loss = torch.mean(F.relu(self.margin - signed_fx))
        else:
            # Weighted KR term
            kr_loss = torch.mean(-signed_fx * weights)
            # Weighted Hinge term
            hinge_loss = torch.mean(F.relu(self.margin - signed_fx) * weights)
            
        return kr_loss + self.lbda * hinge_loss
