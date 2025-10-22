import torch
import torch.nn.functional as F

class LossHKR:
    """ Hinge Kantorovitch-Rubinstein loss"""

    def __init__(self, margin, lbda):
        self.margin = margin  # must be small but not too small.
        self.lbda   = lbda  # must be high.

    def __call__(self, y):
        """
        Args:
            y: vector of predictions.
        """
        return  F.relu(self.margin - y) + (1./self.lbda) * torch.mean(-y)
    
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
    
    
def eikonal_penalty(x, model):
    """
    对一批点 x 计算 Eikonal 正则项。
    要求 x 需要 requires_grad=False（或 detach），
    内部重新设定 requires_grad=True 以计算梯度。
    """
    # 创建一个可求导的 copy
    x_req = x.detach().clone().requires_grad_(True)
    y = model(x_req).sum()
    grads = torch.autograd.grad(y, x_req, create_graph=True)[0]  # shape (B, d)
    # 惩罚 ‖∇f‖ 与 1 的偏差
    return (grads.norm(dim=1) - 1.0).abs().mean()

class SDFRegressionLoss:
    """监督回归 + Eikonal 正则损失"""
    def __init__(self, lambda_eikonal=0.0):
        self.lambda_eik = lambda_eikonal
        
    def __call__(self, pred, target, x_batch, model):
        """
        pred: 模型在 x_batch 上的预测值 (shape B 或 B×1)
        target: 真实 signed distance (shape B 或 B×1)
        x_batch: 用于计算梯度的输入点（用于 eikonal）
        model: 模型本身，用于梯度计算
        """
        # 回归损失（使用 L1 或 SmoothL1）
        loss_reg = F.l1_loss(pred, target)
        # 或者换成：
        # loss_reg = F.smooth_l1_loss(pred, target)
        
        # Eikonal 正则项
        loss_eik = 0.0
        if self.lambda_eik > 0:
            loss_eik = self.lambda_eik * eikonal_penalty(x_batch, model)
            
        return loss_reg + loss_eik