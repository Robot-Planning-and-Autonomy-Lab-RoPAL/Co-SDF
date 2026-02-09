#!/usr/bin/env python3
import os
import argparse
import numpy as np
import sys
import time
from typing import List, Optional, Tuple
from ompl import base as ob
from ompl import geometric as og
import robot_tools

def wrap_angles(q: np.ndarray) -> np.ndarray:
    """Wrap each angle to [0, 2π)."""
    return np.mod(q, 2 * np.pi)


class RRT:
    def __init__(self, problem: robot_tools.PlanningProblem):
        self.problem = problem
        self.dim = 2
        
        self.space = ob.RealVectorStateSpace(self.dim)
        bounds = ob.RealVectorBounds(self.dim)
        for i in range(self.dim):
            bounds.setLow(i, float(self.problem.robot.joint_limits[i, 0]))
            bounds.setHigh(i, float(self.problem.robot.joint_limits[i, 1]))
        self.space.setBounds(bounds)
        
        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_collision_free_ompl))

    def is_collision_free_ompl(self, state) -> bool:
        q_list = [state[i] for i in range(self.dim)]
        return self.is_collision_free(q_list)

    def is_collision_free(self, q_list: List[float]) -> bool:
        q = wrap_angles(np.array(q_list))
        self.problem.robot.set_link_angles(q)
        shapes = self.problem.robot.get_collision_shapes()
        for shape in shapes:
            if any(shape.intersect(o) for o in self.problem.workspace.obstacles):
                return False # Collision found -> Not free
        return True # No collision -> Free

    def plan(self, start_q, goal_q, timeout=1.0, planner_range=0.5) -> Optional[List[np.ndarray]]:
        """Plan a path using OMPL RRT."""
        start = ob.State(self.space)
        for i in range(self.dim): start[i] = float(start_q[i])
        
        goal = ob.State(self.space)
        for i in range(self.dim): goal[i] = float(goal_q[i])
        
        self.ss.setStartAndGoalStates(start, goal)
        
        planner = og.RRT(self.ss.getSpaceInformation())
        if planner_range > 0:
            planner.setRange(planner_range)
        self.ss.setPlanner(planner)
        
        t0 = time.time()
        solved = self.ss.solve(timeout)
        plan_time = time.time() - t0
        
        if solved:
            path = self.ss.getSolutionPath()
            path.interpolate() 
            
            n_states = path.getStateCount()
            res = []
            for i in range(n_states):
                s = path.getState(i)
                res.append(np.array([s[j] for j in range(self.dim)]))
            
            print(f"RRT: Found path with {len(res)} states in {plan_time:.4f}s.")
            return res
        else:
            print(f"RRT: Failed to find a path (Time: {plan_time:.4f}s).")
            return None

    def plan_with_time(self, start_q: np.ndarray, goal_q: np.ndarray, timeout=1.0) -> Tuple[Optional[List[np.ndarray]], float]:
        path = self.plan(start_q, goal_q, timeout=timeout)
        return path, 0.0
