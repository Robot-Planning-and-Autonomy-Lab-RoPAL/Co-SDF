#!/usr/bin/env python3
import os
import argparse
import numpy as np
from typing import List, Optional

import robot_tools


def wrap_angles(q: np.ndarray) -> np.ndarray:
    """Wrap each angle to [0, 2π)."""
    return np.mod(q, 2 * np.pi)


class RRT2D:
    """2D configuration-space RRT for a planar 2-link arm.

    Plans directly in joint space (θ1, θ2) with collision checks using
    the existing `robot_tools.PlanningProblem` obstacles.
    """

    def __init__(self, problem: robot_tools.PlanningProblem, max_iter: int = 5000, step_size: float = 0.15):
        self.problem = problem
        self.max_iter = max_iter
        self.step_size = step_size

    def is_collision_free(self, q: np.ndarray) -> bool:
        """Return True if configuration q is collision-free."""
        q = wrap_angles(q)
        self.problem.robot.set_link_angles(q)
        fp = self.problem.robot.get_current_polygon()
        return not any(fp.intersect(o) for o in self.problem.workspace.obstacles)

    def check_segment(self, q1: np.ndarray, q2: np.ndarray, steps: Optional[int] = None) -> bool:
        """Check a straight-line segment in joint space for collisions."""
        q1 = wrap_angles(q1)
        q2 = wrap_angles(q2)
        if steps is None:
            steps = max(10, int(np.linalg.norm(q2 - q1) / (self.step_size / 3)))
        for i in range(1, steps + 1):
            q = wrap_angles(q1 + (q2 - q1) * (i / steps))
            if not self.is_collision_free(q):
                return False
        return True

    def step_towards(self, tree, nodes, target, step_size):
        distances = np.linalg.norm(np.array(nodes) - target, axis=1)
        nearest_idx = np.argmin(distances)
        q_near = nodes[nearest_idx]

        diff = target - q_near
        dist = np.linalg.norm(diff)
        if dist > step_size:
            q_new = wrap_angles(q_near + (diff / dist) * step_size)
        else:
            q_new = wrap_angles(target)

        if self.is_collision_free(q_new) and self.check_segment(q_near, q_new):
            if tuple(q_new) not in tree:
                nodes.append(q_new)
                tree[tuple(q_new)] = q_near
            return q_new, True
        return q_near, False

    def extract_bidirectional_path(self, tree_a, tree_b, q_a, q_b, start_q, goal_q):
        path_a = []
        curr = q_a
        while curr is not None:
            path_a.append(curr)
            curr = tree_a.get(tuple(curr))

        path_b = []
        curr = q_b
        while curr is not None:
            path_b.append(curr)
            curr = tree_b.get(tuple(curr))

        if np.allclose(path_a[-1], start_q):
            return path_a[::-1] + path_b
        else:
            return path_b[::-1] + path_a

    def plan(self, start_q: np.ndarray, goal_q: np.ndarray) -> Optional[List[np.ndarray]]:
        start_q = wrap_angles(start_q)
        goal_q = wrap_angles(goal_q)
        if not self.is_collision_free(start_q):
            print("Start configuration is in collision")
            return None
        if not self.is_collision_free(goal_q):
            print("Goal configuration is in collision")
            return None

        tree_a = {tuple(start_q): None}
        tree_b = {tuple(goal_q): None}
        nodes_a = [start_q]
        nodes_b = [goal_q]

        for _ in range(self.max_iter):
            q_rand = np.random.uniform(0.0, 2 * np.pi, size=2)
            q_new_a, success_a = self.step_towards(tree_a, nodes_a, q_rand, self.step_size)
            if success_a:
                q_new_b, success_b = self.step_towards(tree_b, nodes_b, q_new_a, self.step_size)
                while success_b:
                    if np.allclose(q_new_a, q_new_b, atol=self.step_size * 1.1):
                        return self.extract_bidirectional_path(tree_a, tree_b, q_new_a, q_new_b, start_q, goal_q)
                    q_new_b_next, success_b = self.step_towards(tree_b, nodes_b, q_new_a, self.step_size)
                    if not success_b:
                        break
                    q_new_b = q_new_b_next
            tree_a, tree_b = tree_b, tree_a
            nodes_a, nodes_b = nodes_b, nodes_a

        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D RRT in joint space (θ1, θ2)")
    parser.add_argument("--world", type=str, default="config/block_world5.yaml", help="World YAML with obstacles and robot")
    parser.add_argument("--iter", type=int, default=5000, help="Maximum number of iterations")
    parser.add_argument("--size", type=float, default=0.15, help="Step size in joint space")
    parser.add_argument("--start", type=float, nargs=2, default=[0.2, 0.2], help="Start angles")
    parser.add_argument("--goal", type=float, nargs=2, default=[2.5, 2.0], help="Goal angles")
    parser.add_argument("--out", type=str, default="output/traj/rrt.npy", help="Output path for saved trajectory")
    args = parser.parse_args()

    problem = robot_tools.PlanningProblem(args.world)
    planner = RRT2D(problem, max_iter=args.iter, step_size=args.size)

    start_q = np.array(args.start, dtype=float)
    goal_q = np.array(args.goal, dtype=float)

    path = planner.plan(start_q, goal_q)

    if path:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.save(args.out, np.array(path))
        print("RRT planning complete")
        print(f"  Saved to: {args.out}")
    else:
        print("No path found.")
