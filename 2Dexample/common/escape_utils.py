import numpy as np
from tqdm import tqdm
import torch

def check_2d_collision(problem, q):
    """Ground truth collision check using robot_tools."""
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
    """
    Find points barely on the collision boundary using binary search.
    """
    configs = []
    pbar = tqdm(total=n_target, desc="Boundary Sampling", ncols=100)
    
    while len(configs) < n_target:
        # 1. Start with a collision point and a free point
        q1 = np.random.uniform(0, 2*np.pi, 2)
        c1 = check_2d_collision(problem, q1)
        
        # Random step to find the other state
        q2 = (q1 + np.random.randn(2) * 0.5) % (2 * np.pi)
        c2 = check_2d_collision(problem, q2)
        
        if c1 != c2:
            q_coll, q_free = (q1, q2) if c1 else (q2, q1)
            # Binary search to find boundary configuration
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
    """
    Find random points that are in collision.
    """
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
