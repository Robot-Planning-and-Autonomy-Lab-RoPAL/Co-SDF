#!/usr/bin/env python3
import os
import argparse
import numpy as np
import sys
import time
from typing import List, Optional

# Add the directory containing our new .so to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'ompl_integration'))

# Also need the lib path for OMPL
os.environ['LD_LIBRARY_PATH'] = '/home/qisong/ompl/build/Release/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

try:
    import my_ompl_core
except ImportError as e:
    print(f"Error loading my_ompl_core: {e}")
    sys.exit(1)

import robot_tools

def wrap_angles(q: np.ndarray) -> np.ndarray:
    """Wrap each angle to [0, 2π)."""
    return np.mod(q, 2 * np.pi)

class OMPLPlanner2D:
    """2D configuration-space planner using OMPL for a planar 2-link arm.
    """

    def __init__(self, problem: robot_tools.PlanningProblem):
        self.problem = problem
        self.dim = 2
        self.low = [0.0, 0.0]
        self.high = [2 * np.pi, 2 * np.pi]

    def is_collision_free(self, q_list: List[float]) -> bool:
        """Return True if configuration q is collision-free. 
        Note: q_list is passed from C++ as a list of floats.
        """
        q = wrap_angles(np.array(q_list))
        self.problem.robot.set_link_angles(q)
        fp = self.problem.robot.get_current_polygon()
        # Collision check against obstacles
        if any(fp.intersect(o) for o in self.problem.workspace.obstacles):
            return False
        return True

    def plan(self, start_q: np.ndarray, goal_q: np.ndarray) -> Optional[List[np.ndarray]]:
        start_q = wrap_angles(start_q).tolist()
        goal_q = wrap_angles(goal_q).tolist()
        
        if not self.is_collision_free(start_q):
            print("Start configuration is in collision")
            return None
        if not self.is_collision_free(goal_q):
            print("Goal configuration is in collision")
            return None

        print(f"OMPL: Planning path from {start_q} to {goal_q}...")
        
        # Call the C++ OMPL planner with the Python callback
        path = my_ompl_core.plan(
            self.dim, 
            self.low, 
            self.high, 
            start_q, 
            goal_q, 
            self.is_collision_free
        )

        if path and len(path) > 0:
            print(f"OMPL: Found path with {len(path)} states.")
            return [np.array(p) for p in path]
        else:
            print("OMPL: Failed to find a path.")
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D OMPL Planner in joint space (θ1, θ2)")
    parser.add_argument("--world", type=str, default="config/block_world1.yaml", help="World YAML")
    parser.add_argument("--start", type=float, nargs=2, default=[2.442131435811337,1.7049362495591496], help="Start angles")
    parser.add_argument("--goal", type=float, nargs=2, default=[5.20711134101201,2.241547260568599], help="Goal angles")
    parser.add_argument("--out", type=str, default="output/traj/rrt.npy", help="Output path")
    args = parser.parse_args()

    problem = robot_tools.PlanningProblem(args.world)
    planner = OMPLPlanner2D(problem)

    start_q = np.array(args.start, dtype=float)
    goal_q = np.array(args.goal, dtype=float)

    t0 = time.time()
    path = planner.plan(start_q, goal_q)
    t1 = time.time()
    print(f"Planning time: {t1 - t0:.2f} seconds")

    if path:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.save(args.out, np.array(path))
        print(f"Success! Path saved to {args.out}")
    else:
        print("No path found.")
