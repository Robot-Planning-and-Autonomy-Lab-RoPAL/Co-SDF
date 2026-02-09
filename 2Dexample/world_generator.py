#!/usr/bin/env python3
import yaml
import numpy as np
import os
import argparse
import polygon_tools as poly

from collections import deque
import robot_tools

def check_cspace_connectivity(world_dict, nx=101):
    # Save to temp file because PlanningProblem expects a file path
    temp_path = "config/temp_connectivity_check.yaml"
    os.makedirs('config', exist_ok=True)
    with open(temp_path, 'w') as f:
        yaml.dump(world_dict, f)
        
    problem = robot_tools.PlanningProblem(temp_path)
    # Using a slightly lower resolution (101x101) for faster validation during generation
    theta, c_space = problem.construct_config_space(nx)
    
    free_cells = (c_space == 0).astype(int)
    rows, cols = free_cells.shape
    visited = np.zeros_like(free_cells, dtype=bool)
    
    max_component = 0
    total_free = np.sum(free_cells)
    if total_free == 0: return 0.0
    
    for r in range(rows):
        for c in range(cols):
            if free_cells[r, c] == 1 and not visited[r, c]:
                curr_size = 0
                q = deque([(r, c)])
                visited[r, c] = True
                while q:
                    cr, cc = q.popleft()
                    curr_size += 1
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if free_cells[nr, nc] == 1 and not visited[nr, nc]:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                max_component = max(max_component, curr_size)
    
    return max_component / total_free

def generate_forest(n_obstacles, world_size=10.0, output_name=None, target_connectivity=0.60):
    """
    Generate a 2D world with All-Rectangle obstacles.
    Targeting 60-65% connectivity for high-quality benchmarking.
    """
    max_retries = 60 
    for retry in range(max_retries):
        robot_config = {
            'type': 'RobotArm2D',
            'parameters': {
                'base_position': [5.0, 5.0],
                'link_lengths': [2.1, 2.1],
                'link_width': 0.2
            }
        }
        
        workspace_config = {
            'type': 'Workspace2D',
            'parameters': {
                'limits': [[0, world_size], [0, world_size]],
                'obstacles': []
            }
        }
        
        base_pos = np.array([5.0, 5.0])
        obs_count = 0
        generated_centers = [base_pos]
        
        # Distal range for challenging but navigable layout
        r_inner = 2.4
        r_outer = 4.8
        
        # Spacing logic: Increased to hit 60-65% target
        min_spacing = 2.3 / np.log(n_obstacles + 2.5) 
        
        # Size factor for rectangles
        # Slightly smaller s_base to allow more "air" in C-space
        s_base = 1.6 / np.sqrt(n_obstacles + 1)
        
        attempts = 0
        while obs_count < n_obstacles and attempts < 400:
            r = np.sqrt(np.random.uniform(r_inner**2, r_outer**2))
            angle = np.random.uniform(0, 2*np.pi)
            center = base_pos + np.array([r * np.cos(angle), r * np.sin(angle)])
            center = np.clip(center, 0.8, 9.2)
            
            dists = [np.linalg.norm(center - c) for c in generated_centers]
            if min(dists) < min_spacing:
                attempts += 1
                continue
            
            generated_centers.append(center)
            
            # 100% RECTANGLES
            w = np.random.uniform(1.2 * s_base, 2.5 * s_base)
            h = np.random.uniform(1.2 * s_base, 2.5 * s_base)
            w, h = np.clip([w, h], 0.25, 1.2)
            workspace_config['parameters']['obstacles'].append({
                'type': 'Rectangle',
                'parameters': {
                    'xlim': [float(center[0] - w/2), float(center[0] + w/2)],
                    'ylim': [float(center[1] - h/2), float(center[1] + h/2)]
                }
            })
            
            obs_count += 1
            attempts = 0

        world_dict = {'robot': robot_config, 'workspace': workspace_config}
        
        conn = check_cspace_connectivity(world_dict)
        print(f"World {output_name} (Attempt {retry+1}): Connectivity = {conn*100:.2f}%")
        
        # Target: 60% - 66%
        if 0.59 <= conn <= 0.68:
            if output_name:
                os.makedirs('config', exist_ok=True)
                path = f"config/{output_name}.yaml"
                with open(path, 'w') as f:
                    yaml.dump(world_dict, f, sort_keys=False)
                print(f"Validated & Generated: {path} (Final Connectivity: {conn*100:.1f}%)")
            return world_dict
            
        # Adjust heuristic
        if conn < 0.59:
            min_spacing *= 1.05 
        else:
            min_spacing *= 0.95 
        
    print(f"Warning: Final connectivity {conn*100:.2f}%. Proceeding.")
    return world_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 2D Forest Worlds")
    parser.add_argument("--obs", type=int, default=10, help="Number of obstacles")
    parser.add_argument("--name", type=str, default="forest_10", help="Output filename")
    args = parser.parse_args()
    
    generate_forest(args.obs, output_name=args.name)
