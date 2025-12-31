import torch
import torch.nn as nn
import numpy as np
import os
from common.models import select_model

class PeriodicEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

def build_model(device):
    embedding = PeriodicEmbedding()
    model_core = select_model("sll", 4, 20, 128)
    model = nn.Sequential(embedding, model_core).to(device)
    return model

def load_and_eval(model_path, X_test, Y_test, device):
    model = build_model(device)
    data = torch.load(model_path, map_location=device)
    
    if isinstance(data, dict) and "state_dict" in data:
        state_dict = data["state_dict"]
    else:
        state_dict = data
        
    # Mapping keys if necessary
    model_keys = set(model.state_dict().keys())
    sd_keys = set(state_dict.keys())
    
    if sd_keys == model_keys:
        model.load_state_dict(state_dict)
    else:
        # Try to fix "1.1.weight" -> "1.weight" etc or vice versa
        # The observed error showed sd keys have "1.1.weight"
        # and model expects "1.weight" (if it was just DenseSDP)
        # BUT our build_model returns Sequential(Embedding, DenseSDP)
        # So it SHOULD expect "1.1.weight".
        
        # Let's use strict=False and print what's missing
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if len(missing) > 0 or len(unexpected) > 0:
            print(f"  Loaded with {len(missing)} missing and {len(unexpected)} unexpected keys.")
            # If missing everything, try to map
            if len(missing) > 0 and len(unexpected) > 0:
                print("  Attempting automatic remapping...")
                new_sd = {}
                for uk in unexpected:
                    val = state_dict[uk]
                    # Try removing "1." prefix
                    if uk.startswith("1."):
                        nk = uk[2:]
                        if nk in missing:
                            new_sd[nk] = val
                            continue
                    # Try adding "1." prefix
                    nk = "1." + uk
                    if nk in missing:
                        new_sd[nk] = val
                        continue
                
                if new_sd:
                    m2, u2 = model.load_state_dict(new_sd, strict=False)
                    print(f"  Second pass: {len(m2)} missing, {len(u2)} unexpected.")
    
    model.eval()
    with torch.no_grad():
        X_test_t = torch.Tensor(X_test).to(device)
        preds = model(X_test_t).cpu().numpy().flatten()
        
    y_pred = np.ones_like(preds)
    y_pred[preds < 0] = -1
    
    accuracy = (y_pred == Y_test).mean()
    mse = ((preds - Y_test)**2).mean()
    return accuracy, mse

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load test set
    # Try block_world first
    prefix = "block_world"
    if not os.path.exists(f"inputs/{prefix}_Xtest.npy"):
        prefix = "block_world2"
        
    X_test = np.load(f"inputs/{prefix}_Xtest.npy")
    Y_test = np.load(f"inputs/{prefix}_Ytest.npy")
    print(f"Using test set: {prefix} ({len(X_test)} samples)")
    
    al_model_path = "output/active_learning/model_active.pt"
    grad_model_path = "output/active_grad_result_model.pt"
    grad_high_model_path = "output/active_grad_high_acc_model.pt"
    std_model_path = "output/block_world_2000_periodic/model_final.pt"
    
    if os.path.exists(al_model_path):
        print(f"\nEvaluating AL model: {al_model_path}")
        acc, mse = load_and_eval(al_model_path, X_test, Y_test, device)
        print(f"AL Accuracy: {acc*100:.2f}%, MSE: {mse:.4f}")
        
    if os.path.exists(grad_model_path):
        print(f"\nEvaluating AL+Gradient model: {grad_model_path}")
        acc, mse = load_and_eval(grad_model_path, X_test, Y_test, device)
        print(f"AL+Gradient Accuracy: {acc*100:.2f}%, MSE: {mse:.4f}")

    if os.path.exists(grad_high_model_path):
        print(f"\nEvaluating AL+Gradient (High Acc) model: {grad_high_model_path}")
        acc, mse = load_and_eval(grad_high_model_path, X_test, Y_test, device)
        print(f"AL+Gradient High Accuracy: {acc*100:.2f}%, MSE: {mse:.4f}")

    if os.path.exists(std_model_path):
        print(f"\nEvaluating Standard (2000 pts) model: {std_model_path}")
        acc, mse = load_and_eval(std_model_path, X_test, Y_test, device)
        print(f"Standard Accuracy: {acc*100:.2f}%, MSE: {mse:.4f}")
