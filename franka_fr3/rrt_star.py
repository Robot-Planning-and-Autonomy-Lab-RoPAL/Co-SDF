#!/usr/bin/env python3
import argparse
import numpy as np
import mujoco
from ompl import base as ob
from ompl import geometric as og
import os
import sys

def check_franka_collision(mj_model, mj_data, q, is_robot_body):
    mj_data.qpos[:7] = q
    if mj_model.nq > 7:
        mj_data.qpos[7:mj_model.nq] = 0.04
    mujoco.mj_forward(mj_model, mj_data)
    
    for i in range(mj_data.ncon):
        c = mj_data.contact[i]
        b1, b2 = mj_model.geom_bodyid[c.geom1], mj_model.geom_bodyid[c.geom2]
        if is_robot_body[b1] ^ is_robot_body[b2]:
            return True
    return False

class RRTStar:
    def __init__(self, scene_xml):
        self.scene_xml = os.path.abspath(scene_xml)
        self.mj_model = mujoco.MjModel.from_xml_path(self.scene_xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.is_robot_body = self._get_robot_bodies()
        
        self.dim = 7
        self.space = ob.RealVectorStateSpace(self.dim)
        
        bounds = ob.RealVectorBounds(self.dim)
        lower = self.mj_model.jnt_range[:7, 0]
        upper = self.mj_model.jnt_range[:7, 1]
        for i in range(self.dim):
            bounds.setLow(i, float(lower[i]))
            bounds.setHigh(i, float(upper[i]))
        self.space.setBounds(bounds)
        
        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.isStateValid))

    def _get_robot_bodies(self):
        robot_body_ids = set()
        base_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")
        for i in range(self.mj_model.nbody):
            curr = i
            while curr > 0:
                if curr == base_id:
                    robot_body_ids.add(i)
                    break
                curr = self.mj_model.body_parentid[curr]
        is_robot = np.zeros(self.mj_model.nbody, dtype=bool)
        if robot_body_ids:
            is_robot[list(robot_body_ids)] = True
        return is_robot

    def isStateValid(self, state):
        q = np.array([state[i] for i in range(self.dim)])
        return not check_franka_collision(self.mj_model, self.mj_data, q, self.is_robot_body)

    def is_collision_free(self, q):
        """Convenience method for numpy arrays."""
        return not check_franka_collision(self.mj_model, self.mj_data, q, self.is_robot_body)

    def plan(self, start_q=None, goal_q=None, output_file="franka_rrt_star_path.npy", timeout=30.0):
        if start_q is None:
            print("Sampling valid start state...")
            start_state = self.get_valid_random_state()
        else:
            start_state = start_q
            if isinstance(start_q, (np.ndarray, list)):
                q_arr = start_q
                start_state = ob.State(self.space)
                for i in range(self.dim): start_state[i] = float(q_arr[i])
        
        if goal_q is None:
            print("Sampling valid goal state...")
            goal_state = self.get_valid_random_state()
        else:
            goal_state = goal_q
            if isinstance(goal_q, (np.ndarray, list)):
                q_arr = goal_q
                goal_state = ob.State(self.space)
                for i in range(self.dim): goal_state[i] = float(q_arr[i])
            
        self.ss.setStartAndGoalStates(start_state, goal_state)
        
        # Use RRT* planner
        planner = og.RRTstar(self.ss.getSpaceInformation())
        planner.setRange(0.2)
        self.ss.setPlanner(planner)
        
        print(f"Planning with RRT* (timeout={timeout}s)...")
        solved = self.ss.solve(timeout)
        
        if solved:
            print("Found solution!")
            path = self.ss.getSolutionPath()
            path.interpolate(100)
            
            n_states = path.getStateCount()
            path_np = np.zeros((n_states, self.dim))
            for i in range(n_states):
                s = path.getState(i)
                for j in range(self.dim):
                    path_np[i, j] = s[j]
            
            # Get path cost
            solution_path = self.ss.getSolutionPath()
            cost = solution_path.length()
            print(f"Path cost: {cost:.4f}")
            
            if output_file:
                np.save(output_file, path_np)
                print(f"Path saved to {output_file} with {n_states} waypoints.")
            return path_np
        else:
            print("No solution found.")
            return None

    def get_valid_random_state(self):
        state = ob.State(self.space)
        lower = self.mj_model.jnt_range[:7, 0]
        upper = self.mj_model.jnt_range[:7, 1]
        
        while True:
            q_rand = lower + np.random.rand(self.dim) * (upper - lower)
            
            for i in range(self.dim):
                state[i] = float(q_rand[i])
                
            if self.isStateValid(state):
                return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="scenes/obs10.xml")
    parser.add_argument("--output", type=str, default="output/traj/rrt_star.npy")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    
    scene_file = args.scene
    if not os.path.exists(scene_file):
        print(f"Error: scene file {scene_file} not found.")
        sys.exit(1)
        
    planner = RRTStar(scene_file)
    planner.plan(output_file=args.output, timeout=args.timeout)
    
    sys.stdout.flush()
    os._exit(0)
