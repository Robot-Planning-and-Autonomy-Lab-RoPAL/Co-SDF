#!/usr/bin/env python3
import numpy as np
import robot_tools
import os
from collections import deque

def check_connectivity(world_yaml, nx=201):
    print(f"Checking connectivity for: {world_yaml}")
    problem = robot_tools.PlanningProblem(world_yaml)
    theta, c_space = problem.construct_config_space(nx)
    
    # 0 is free, 1 is obstacle. Let's flip it for BFS
    free_cells = (c_space == 0).astype(int)
    rows, cols = free_cells.shape
    visited = np.zeros_like(free_cells, dtype=bool)
    
    components = []
    
    for r in range(rows):
        for c in range(cols):
            if free_cells[r, c] == 1 and not visited[r, c]:
                # Start BFS
                component_size = 0
                q = deque([(r, c)])
                visited[r, c] = True
                while q:
                    curr_r, curr_c = q.popleft()
                    component_size += 1
                    # 4-connectivity
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if free_cells[nr, nc] == 1 and not visited[nr, nc]:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                components.append(component_size)
    
    if not components:
        print("No free space found!")
        return 0.0
        
    components.sort(reverse=True)
    total_free = np.sum(free_cells)
    largest_ratio = components[0] / total_free
    
    print(f"Total free cells: {total_free}")
    print(f"Number of components: {len(components)}")
    print(f"Largest component size: {components[0]} ({largest_ratio*100:.2f}%)")
    
    return largest_ratio

if __name__ == "__main__":
    worlds = ["config/forest_10.yaml", "config/forest_20.yaml", "config/forest_30.yaml"]
    for w in worlds:
        if os.path.exists(w):
            check_connectivity(w)
        else:
            print(f"File not found: {w}")
