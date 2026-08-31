import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
    
def forward_in_batches(model, inputs : np.ndarray, device:str, compute_grad:bool=False, batch_size=5_000):
    inputs = torch.Tensor(inputs).to(device)
    inputs = DataLoader(inputs, batch_size=batch_size)
    outputs = []
    grads = []
    for batch in tqdm(inputs, total=len(inputs)):
        batch.requires_grad = compute_grad
        v_batch = model(batch)
        if compute_grad:
            torch.sum(v_batch).backward()
            grads.append(batch.grad.detach().cpu().numpy())
        outputs.append(v_batch.detach().cpu().numpy())
    
    if compute_grad:
        return np.concatenate(outputs), np.concatenate(grads)
    else:
        return np.concatenate(outputs)
