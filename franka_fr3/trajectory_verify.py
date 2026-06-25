#!/usr/bin/env python3
import argparse
import numpy as np
import mujoco
import os
from utils import ControlPoint
import torch

def densify_path(path: np.ndarray, n_interp: int) -> np.ndarray:
    if n_interp <= 0 or len(path) < 2:
        return path

    out = [path[0]]
    for i in range(len(path) - 1):
        q1, q2 = path[i], path[i + 1]
        for s in range(1, n_interp + 1):
            alpha = s / (n_interp + 1)
            out.append(q1 + (q2 - q1) * alpha)
        out.append(q2)
    return np.asarray(out)

class TrajectoryEvaluator:
    def __init__(self, scene_xml, model=None, device="cuda"):
        self.scene_xml = os.path.abspath(scene_xml)
        self.mj_model = mujoco.MjModel.from_xml_path(self.scene_xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.model = model
        self.device = device
        
        self.lo = self.mj_model.jnt_range[:7, 0]
        self.hi = self.mj_model.jnt_range[:7, 1]
        
        self.robot_body_ids = set()
        base_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")
        for i in range(self.mj_model.nbody):
            curr = i
            while curr > 0:
                if curr == base_id:
                    self.robot_body_ids.add(i)
                    break
                curr = self.mj_model.body_parentid[curr]
        self.is_robot_body = np.zeros(self.mj_model.nbody, dtype=bool)
        if self.robot_body_ids:
            self.is_robot_body[list(self.robot_body_ids)] = True
            
        self.cp_manager = ControlPoint(device=device)
        
        self.env_geoms = []
        for i in range(self.mj_model.ngeom):
            bid = self.mj_model.geom_bodyid[i]
            if not self.is_robot_body[bid]:
                self.env_geoms.append(i)

    def is_collision_free(self, q: np.ndarray) -> bool:
        self.mj_data.qpos[:7] = q
        if self.mj_model.nq > 7:
            self.mj_data.qpos[7:self.mj_model.nq] = 0.04
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            b1 = self.mj_model.geom_bodyid[c.geom1]
            b2 = self.mj_model.geom_bodyid[c.geom2]
            
            if self.is_robot_body[b1] ^ self.is_robot_body[b2]:
                return False
        return True

    def evaluate(self, path, solve_time=0.0, safety_margin=0.02):
        target_points = 512
        if len(path) < target_points:
            n_interp = max(1, (target_points - len(path)) // (len(path) - 1) + 1)
            dense = densify_path(path, n_interp)
        else:
            dense = path
            
        diffs = dense[1:] - dense[:-1]
        path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
        
        success = True
        for q in dense:
            if not self.is_collision_free(q):
                success = False
                break
        
        if len(dense) > 2:
            h = 1.0 / (target_points - 1)
            accel_norms = np.linalg.norm((dense[2:] - 2*dense[1:-1] + dense[:-2]) / (h**2), axis=1)
            sm_mean = float(np.mean(accel_norms))
            sm_p95 = float(np.percentile(accel_norms, 95))
        else:
            sm_mean = 0.0
            sm_p95 = 0.0

        return {
            "success": success,
            "path_length": path_length,
            "smoothness": sm_mean,
            "smoothness_p95": sm_p95,
            "solve_time": solve_time
        }

