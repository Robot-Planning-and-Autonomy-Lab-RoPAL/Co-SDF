#!/usr/bin/env python3
import os
import argparse
import numpy as np
import sys
import time
from typing import List, Optional, Tuple

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'ompl_integration'))
os.environ['LD_LIBRARY_PATH'] = '/home/qisong/ompl/build/Release/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import my_ompl_core
import robot_tools

def wrap_angles(q: np.ndarray) -> np.ndarray:
    """Wrap each angle to [0, 2π)."""
    return np.mod(q, 2 * np.pi)


class RRT:
    def __init__(self, problem: robot_tools.PlanningProblem, max_iter, step_size):
        self.problem = problem
        self.max_iter = max_iter
        self.step_size = step_size
        self.dim = 2
        self.low = [0.0, 0.0]
        self.high = [2 * np.pi, 2 * np.pi]

    def is_collision_free(self, q_list: List[float]) -> bool:
        q = wrap_angles(np.array(q_list))
        self.problem.robot.set_link_angles(q)
        fp = self.problem.robot.get_current_polygon()
        return not any(fp.intersect(o) for o in self.problem.workspace.obstacles)

    def plan(self, start_q, goal_q) -> Optional[List[np.ndarray]]:
        start_q_list = wrap_angles(start_q).tolist()
        goal_q_list = wrap_angles(goal_q).tolist()
        
        if not self.is_collision_free(start_q_list):
            print("Start is in collision")
            return None
        if not self.is_collision_free(goal_q_list):
            print("Goal is in collision")
            return None

        print(f"RRT: Planning path from {start_q_list} to {goal_q_list}...")
        
        t0 = time.time()
        path = my_ompl_core.plan(
            self.dim, 
            self.low, 
            self.high, 
            start_q_list, 
            goal_q_list, 
            self.is_collision_free
        )
        plan_time = time.time() - t0

        if path and len(path) > 0:
            print(f"RRT: Found path with {len(path)} states in {plan_time:.4f}s.")
            return [np.array(p) for p in path]
        else:
            print(f"RRT: Failed to find a path (Time: {plan_time:.4f}s).")
            return None

    def plan_with_time(self, start_q: np.ndarray, goal_q: np.ndarray) -> Tuple[Optional[List[np.ndarray]], float]:

        start_q_list = wrap_angles(start_q).tolist()
        goal_q_list = wrap_angles(goal_q).tolist()
        
        if not self.is_collision_free(start_q_list) or not self.is_collision_free(goal_q_list):
            return None, 0.0

        t0 = time.time()
        path = my_ompl_core.plan(
            self.dim, 
            self.low, 
            self.high, 
            start_q_list, 
            goal_q_list, 
            self.is_collision_free
        )
        plan_time = time.time() - t0
        
        if path and len(path) > 0:
            return [np.array(p) for p in path], plan_time
        return None, plan_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D OMPL Planner in joint space (θ1, θ2)")
    parser.add_argument("--world", type=str, default="config/block_world1.yaml", help="World YAML with obstacles and robot")
    parser.add_argument("--start", type=float, nargs=2, default=[0.2, 0.2], help="Start angles")
    parser.add_argument("--goal", type=float, nargs=2, default=[2.5, 2.0], help="Goal angles")
    parser.add_argument("--out", type=str, default="output/traj/rrt.npy", help="Output path for saved trajectory")
    parser.add_argument("--iter", type=int, default=5000, help="Ignored (OMPL uses time)")
    parser.add_argument("--size", type=float, default=0.15, help="Ignored (OMPL handles step size internally)")
    
    args = parser.parse_args()

    problem = robot_tools.PlanningProblem(args.world)
    planner = RRT(problem, max_iter=args.iter, step_size=args.size)

    start_q = np.array(args.start, dtype=float)
    goal_q = np.array(args.goal, dtype=float)

    t0 = time.time()
    path = planner.plan(start_q, goal_q)
    total_time = time.time() - t0

    if path:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.save(args.out, np.array(path))
        print(f"RRT planning complete in {total_time:.4f}s")
        print(f"Saved to: {args.out}")
    else:
        print(f"No path found after {total_time:.4f}s.")
