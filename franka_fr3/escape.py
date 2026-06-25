import torch
import numpy as np
import os
import time
from tqdm import tqdm
from common.models import load_model
from common.utils import get_device

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class GradientEscaper:
    def __init__(self, model_path, check_collision, q_lo, q_hi, device):
        self.device = device
        self.model = load_model(model_path, device)
        self.model.eval()
        self.check_collision = check_collision
        self.q_lo = np.array(q_lo)
        self.q_hi = np.array(q_hi)

    def run(self, start_cfg, lr=0.05, max_steps=100):
        t_start = time.time()
        q_lo_t = torch.from_numpy(self.q_lo).float().to(self.device)
        q_hi_t = torch.from_numpy(self.q_hi).float().to(self.device)
        q = torch.tensor(start_cfg, dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([q], lr=lr)
        
        trajectory = [start_cfg.copy()]
        ndf_history = []
        success = False
        joint_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.2, 1.5, 1.5], device=self.device)
        
        for step in range(max_steps):
            optimizer.zero_grad()
            ndf_val = self.model(q.unsqueeze(0)).squeeze()
            ndf_history.append(ndf_val.item())
            
            # Termination: ground-truth MuJoCo collision check
            q_numpy = q.detach().cpu().numpy()
            if not self.check_collision(q_numpy):
                success = True
                break
            
            loss = torch.nn.functional.relu(- ndf_val)
            loss.backward()
            
            with torch.no_grad():
                q.grad.mul_(joint_weights)
            optimizer.step()
            
            with torch.no_grad():
                if step >= 10:
                    recent_avg = np.mean(ndf_history[-5:])
                    prev_avg = np.mean(ndf_history[-10:-5]) if len(ndf_history) >= 10 else ndf_history[0]
                    if abs(recent_avg - prev_avg) < 1e-4:
                        q.add_(torch.randn_like(q) * 0.01)
                q.clamp_(q_lo_t, q_hi_t)
            
            trajectory.append(q.detach().cpu().numpy().copy())
            
        
        final_q_np = q.detach().cpu().numpy()
        t_escape = time.time() - t_start
        
        return {
            "success": success,
            "steps": step + 1,
            "final_q": final_q_np,
            "trajectory": np.array(trajectory),
            "ndf_history": ndf_history,
            "time": t_escape
        }

class SamplingEscaper:
    def __init__(self, model_path, check_collision, q_lo, q_hi, device):
        self.device = device
        self.model = load_model(model_path, device)
        self.model.eval()
        self.check_collision = check_collision
        self.lower = np.array(q_lo)
        self.upper = np.array(q_hi)

    def run(self, q0, step_size=0.05, max_steps=100, K=50):
        t_start = time.time()
        q_curr = torch.tensor(q0, dtype=torch.float32, device=self.device)
        
        trajectory = [q0.copy()]
        ndf_history = []
        success = False
        
        for step in range(max_steps):
            with torch.no_grad():
                # Termination: ground-truth MuJoCo collision check
                q_numpy = q_curr.cpu().numpy()
                if not self.check_collision(q_numpy):
                    success = True
                    break

                ndf_curr = self.model(q_curr.reshape(1, -1)).item()
                ndf_history.append(ndf_curr)
                
                noise = torch.randn(K, 7, device=self.device) * step_size
                candidates = torch.clamp(q_curr + noise, 
                                        torch.from_numpy(self.lower).float().to(self.device),
                                        torch.from_numpy(self.upper).float().to(self.device))
                ndf_candidates = self.model(candidates).squeeze()
                best_idx = torch.argmax(ndf_candidates)
                q_curr = candidates[best_idx]
                trajectory.append(q_curr.cpu().numpy().copy())
        
        t_escape = time.time() - t_start
        final_q_np = q_curr.cpu().numpy()
        
        return {
            "success": success,
            "steps": step + 1,
            "final_q": final_q_np,
            "trajectory": np.array(trajectory),
            "ndf_history": ndf_history,
            "time": t_escape
        }