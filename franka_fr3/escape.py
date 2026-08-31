#!/usr/bin/env python3
import argparse
import json
import os
import random
import time

import mujoco
import numpy as np
import torch
from tqdm import tqdm

from common.models import load_model
from common.utils import get_device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class SceneOracle:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(os.path.abspath(xml_path))
        self.data = mujoco.MjData(self.model)
        self.lo = self.model.jnt_range[:7, 0]
        self.hi = self.model.jnt_range[:7, 1]
        self.robot_body_mask = np.zeros(self.model.nbody, dtype=bool)
        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        for body_id in range(self.model.nbody):
            current = body_id
            while current > 0:
                if current == base_id:
                    self.robot_body_mask[body_id] = True
                    break
                current = self.model.body_parentid[current]

    def check_collision(self, q):
        self.data.qpos[:7] = q
        if self.model.nq > 7:
            self.data.qpos[7:self.model.nq] = 0.04
        mujoco.mj_forward(self.model, self.data)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            if self.robot_body_mask[body1] ^ self.robot_body_mask[body2]:
                return True
        return False

    def sample_collision(self):
        while True:
            q = self.lo + np.random.rand(7) * (self.hi - self.lo)
            if self.check_collision(q):
                return q

class GradientEscaper:
    def __init__(self, model, oracle, device):
        self.model = model
        self.oracle = oracle
        self.device = device
        self.lower = torch.as_tensor(oracle.lo, dtype=torch.float32, device=device)
        self.upper = torch.as_tensor(oracle.hi, dtype=torch.float32, device=device)

    def run(self, q0, learning_rate, max_steps):
        synchronize(self.device)
        started = time.perf_counter()
        q = torch.tensor(q0, dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([q], lr=learning_rate)
        values = []
        for step in range(max_steps + 1):
            if not self.oracle.check_collision(q.detach().cpu().numpy()):
                synchronize(self.device)
                return make_result(True, step, q0, q, started)
            if step == max_steps:
                break
            optimizer.zero_grad()
            value = self.model(q.unsqueeze(0)).squeeze()
            values.append(value.item())
            (-value).backward()
            optimizer.step()
            with torch.no_grad():
                if len(values) >= 10 and abs(np.mean(values[-5:]) - np.mean(values[-10:-5])) < 1e-4:
                    q.add_(torch.randn_like(q) * 0.01)
                q.clamp_(self.lower, self.upper)
        synchronize(self.device)
        return make_result(False, max_steps, q0, q, started)


class SamplingEscaper:
    def __init__(self, model, oracle, device):
        self.model = model
        self.oracle = oracle
        self.device = device
        self.lower = torch.as_tensor(oracle.lo, dtype=torch.float32, device=device)
        self.upper = torch.as_tensor(oracle.hi, dtype=torch.float32, device=device)

    def run(self, q0, step_size, max_steps, sample_count):
        synchronize(self.device)
        started = time.perf_counter()
        q = torch.tensor(q0, dtype=torch.float32, device=self.device)
        for step in range(max_steps + 1):
            if not self.oracle.check_collision(q.cpu().numpy()):
                synchronize(self.device)
                return make_result(True, step, q0, q, started)
            if step == max_steps:
                break
            with torch.no_grad():
                candidates = q + torch.randn(sample_count, 7, device=self.device) * step_size
                candidates = torch.maximum(torch.minimum(candidates, self.upper), self.lower)
                q = candidates[torch.argmax(self.model(candidates).squeeze())]
        synchronize(self.device)
        return make_result(False, max_steps, q0, q, started)


def make_result(success, steps, q0, q, started):
    final_q = q.detach().cpu().numpy()
    return {
        "success": success,
        "steps": steps,
        "distance": float(np.linalg.norm(final_q - q0)),
        "time": time.perf_counter() - started,
    }


def summarize(results):
    successful = [result for result in results if result["success"]]
    return {
        "success_rate": float(np.mean([result["success"] for result in results]) * 100),
        "avg_steps": float(np.mean([result["steps"] for result in results])),
        "avg_escape_dist": float(np.mean([result["distance"] for result in successful])) if successful else None,
        "avg_time_ms": float(np.mean([result["time"] for result in results]) * 1000),
    }


def serialize_results(results):
    return [
        {
            "success": result["success"],
            "steps": result["steps"],
            "escape_dist": result["distance"],
            "time_ms": result["time"] * 1000,
        }
        for result in results
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", default="obs10")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--lr-optimization", type=float, default=0.01)
    parser.add_argument("--step-sampling", type=float, default=0.01)
    parser.add_argument("--k-sampling", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="output/escape")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    oracle = SceneOracle(f"scenes/{args.world}.xml")
    model = load_model(f"output/{args.world}/model_final.pt", device)
    model.eval()
    starts = np.array([oracle.sample_collision() for _ in tqdm(range(args.n_samples), desc="Random Collision Sampling")])
    methods = {
        "Optimization": (
            GradientEscaper(model, oracle, device),
            {"learning_rate": args.lr_optimization, "max_steps": args.max_steps},
        ),
        "Sampling": (
            SamplingEscaper(model, oracle, device),
            {
                "step_size": args.step_sampling,
                "max_steps": args.max_steps,
                "sample_count": args.k_sampling,
            },
        ),
    }
    results = {
        name: [escaper.run(q, **parameters) for q in tqdm(starts, desc=name)]
        for name, (escaper, parameters) in methods.items()
    }
    summary = {name: summarize(values) for name, values in results.items()}

    print(f"{'=' * 78}")
    print(f"7D ESCAPE BENCHMARK (N={len(starts)}, World={args.world})")
    print(f"{'=' * 78}")
    print(f"{'Method':<14} | {'Success%':>9} | {'Steps':>9} | {'EscDist':>10} | {'Time/ms':>10}")
    print(f"{'-' * 78}")
    for name, metrics in summary.items():
        distance = metrics["avg_escape_dist"]
        distance_text = f"{distance:.4f}" if distance is not None else "N/A"
        print(
            f"{name:<14} | {metrics['success_rate']:>8.1f}% | "
            f"{metrics['avg_steps']:>9.1f} | {distance_text:>10} | "
            f"{metrics['avg_time_ms']:>10.2f}"
        )
    print(f"{'=' * 78}")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"results_7d_{args.world}.json")
    with open(output_path, "w") as file:
        json.dump(
            {
                "world": args.world,
                "config": vars(args),
                "summary": summary,
                "details": {name: serialize_results(values) for name, values in results.items()},
            },
            file,
            indent=2,
        )