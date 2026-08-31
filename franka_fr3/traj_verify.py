#!/usr/bin/env python3
import numpy as np
import mujoco
import os

def resample_path(path, count):
    path = np.asarray(path)
    if len(path) < 2:
        return np.repeat(path, count, axis=0)
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))))
    if cumulative[-1] <= 1e-12:
        return np.repeat(path[:1], count, axis=0)
    keep = np.concatenate(([True], np.diff(cumulative) > 1e-12))
    cumulative = cumulative[keep]
    path = path[keep]
    targets = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack([
        np.interp(targets, cumulative, path[:, joint])
        for joint in range(path.shape[1])
    ])

class TrajectoryEvaluator:
    def __init__(self, scene_xml):
        self.scene_xml = os.path.abspath(scene_xml)
        self.mj_model = mujoco.MjModel.from_xml_path(self.scene_xml)
        self.mj_data = mujoco.MjData(self.mj_model)
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

    def evaluate(self, path):
        metric_path = resample_path(path, 30)
        dense = resample_path(path, 512)

        diffs = metric_path[1:] - metric_path[:-1]
        path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
        
        success = True
        for q in dense:
            if not self.is_collision_free(q):
                success = False
                break
        
        h = 1.0 / (len(metric_path) - 1)
        acceleration = (metric_path[2:] - 2 * metric_path[1:-1] + metric_path[:-2]) / (h ** 2)
        accel_norms = np.linalg.norm(acceleration, axis=1)
        sm_mean = float(np.mean(accel_norms))
        sm_p95 = float(np.percentile(accel_norms, 95))

        return {
            "success": success,
            "path_length": path_length,
            "smoothness": sm_mean,
            "smoothness_p95": sm_p95,
        }
