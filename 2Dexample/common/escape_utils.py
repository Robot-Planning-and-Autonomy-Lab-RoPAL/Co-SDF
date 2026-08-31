import numpy as np
from tqdm import tqdm
import torch

def check_2d_collision(problem, q):

    problem.robot.set_link_angles(q)
    fp = problem.robot.get_current_polygon()
    for obstacle in problem.workspace.obstacles:
        if hasattr(obstacle, 'intersect'):
            if obstacle.intersect(fp): return True
        elif hasattr(obstacle, 'intersects'):
            if obstacle.intersects(fp): return True
        else:
            try:
                if fp.intersect(obstacle): return True
            except:
                pass
    return False

def sample_near_boundary_collision(problem, n_target):

    configs = []
    pbar = tqdm(total=n_target, desc="Boundary Sampling", ncols=100)
    
    while len(configs) < n_target:
        q1 = np.random.uniform(0, 2*np.pi, 2)
        c1 = check_2d_collision(problem, q1)
        
        q2 = (q1 + np.random.randn(2) * 0.5) % (2 * np.pi)
        c2 = check_2d_collision(problem, q2)
        
        if c1 != c2:
            q_coll, q_free = (q1, q2) if c1 else (q2, q1)
            for _ in range(6):
                mid = (q_coll + q_free) / 2
                if check_2d_collision(problem, mid):
                    q_coll = mid
                else:
                    q_free = mid
            
            configs.append(q_coll)
            pbar.update(1)
            
    pbar.close()
    return np.array(configs)

def sample_random_collision(problem, n_target):
    configs = []
    pbar = tqdm(total=n_target, desc="Random Collision Sampling", ncols=100)
    
    while len(configs) < n_target:
        q = np.random.uniform(0, 2*np.pi, 2)
        if check_2d_collision(problem, q):
            configs.append(q)
            pbar.update(1)
            
    pbar.close()
    return np.array(configs)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")