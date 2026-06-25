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

class RRTConnect:
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

    def plan(self, start_q, goal_q, output_file=None, timeout=10.0, n_waypoints=100):
        # Convert to OMPL states
        start_state = ob.State(self.space)
        for i in range(self.dim): start_state[i] = float(start_q[i])
        
        goal_state = ob.State(self.space)
        for i in range(self.dim): goal_state[i] = float(goal_q[i])
            
        self.ss.setStartAndGoalStates(start_state, goal_state)
        
        planner = og.RRTConnect(self.ss.getSpaceInformation())
        planner.setRange(0.2)
        self.ss.setPlanner(planner)
        
        print(f"Planning with RRT-Connect (timeout={timeout}s)...")
        solved = self.ss.solve(timeout)
        
        if solved:
            print("Found exact solution!")
            path = self.ss.getSolutionPath()
            # Smooth/Interpolate to requested waypoints
            path.interpolate(n_waypoints)
            
            n_states = path.getStateCount()
            path_np = np.zeros((n_states, self.dim))
            for i in range(n_states):
                s = path.getState(i)
                for j in range(self.dim):
                    path_np[i, j] = s[j]
            
            if output_file:
                np.save(output_file, path_np)
                print(f"Path saved to {output_file} with {n_states} waypoints.")
            return path_np
        else:
            print("No RRT-Connect solution found.")
            return None
