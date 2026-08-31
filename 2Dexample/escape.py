#!/usr/bin/env python3
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from tqdm import tqdm

import robot_tools
from common.escape_utils import check_2d_collision, sample_random_collision
from common.models import load_model
from common.utils import get_device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torus_distance(q_start, q_end):
    delta = np.arctan2(np.sin(q_end - q_start), np.cos(q_end - q_start))
    return float(np.linalg.norm(delta))


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class OptimizationEscaper:
    def __init__(self, model, problem, device):
        self.model = model
        self.problem = problem
        self.device = device

    def run(self, q0, lr=0.1, max_steps=100):
        synchronize(self.device)
        start_time = time.perf_counter()

        q = torch.tensor(q0, dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([q], lr=lr)
        trajectory = [q0.copy()]
        ndf_history = []

        for _ in range(max_steps):
            if not check_2d_collision(self.problem, q.detach().cpu().numpy()):
                break

            optimizer.zero_grad()
            ndf_value = self.model(q.unsqueeze(0)).squeeze()
            ndf_history.append(float(ndf_value.detach()))
            (-ndf_value).backward()
            optimizer.step()

            with torch.no_grad():
                if len(ndf_history) >= 10:
                    recent_mean = np.mean(ndf_history[-5:])
                    previous_mean = np.mean(ndf_history[-10:-5])
                    if abs(recent_mean - previous_mean) < 1e-4:
                        q.add_(torch.randn_like(q) * 0.01)
                q.remainder_(2 * np.pi)

            trajectory.append(q.detach().cpu().numpy().copy())

        final_q = q.detach().cpu().numpy()
        success = not check_2d_collision(self.problem, final_q)
        synchronize(self.device)

        return {
            "success": success,
            "steps": len(trajectory) - 1,
            "trajectory": np.asarray(trajectory),
            "time": time.perf_counter() - start_time,
        }


class SamplingEscaper:
    def __init__(self, model, problem, device):
        self.model = model
        self.problem = problem
        self.device = device

    def run(self, q0, step_size=0.1, max_steps=100, k=50):
        synchronize(self.device)
        start_time = time.perf_counter()

        q = torch.tensor(q0, dtype=torch.float32, device=self.device)
        trajectory = [q0.copy()]

        for _ in range(max_steps):
            if not check_2d_collision(self.problem, q.cpu().numpy()):
                break

            with torch.no_grad():
                candidates = (q + torch.randn(k, 2, device=self.device) * step_size) % (2 * np.pi)
                values = self.model(candidates).flatten()
                q = candidates[torch.argmax(values)]

            trajectory.append(q.cpu().numpy().copy())

        final_q = q.cpu().numpy()
        success = not check_2d_collision(self.problem, final_q)
        synchronize(self.device)

        return {
            "success": success,
            "steps": len(trajectory) - 1,
            "trajectory": np.asarray(trajectory),
            "time": time.perf_counter() - start_time,
        }


def summarize(results):
    successes = np.asarray([result["success"] for result in results], dtype=bool)
    successful_results = [result for result in results if result["success"]]
    escape_distances = [
        torus_distance(result["trajectory"][0], result["trajectory"][-1])
        for result in successful_results
    ]

    return {
        "success_rate": float(successes.mean() * 100),
        "avg_steps": float(np.mean([result["steps"] for result in results])),
        "avg_escape_dist": float(np.mean(escape_distances)) if escape_distances else None,
        "avg_time_ms": float(np.mean([result["time"] for result in results]) * 1000),
    }


def serialize_results(results):
    return [
        {
            "success": bool(result["success"]),
            "steps": int(result["steps"]),
            "escape_dist": torus_distance(result["trajectory"][0], result["trajectory"][-1]),
            "time_ms": float(result["time"] * 1000),
        }
        for result in results
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D NDF escape benchmark")
    parser.add_argument("--world", default="block_world1", help="world name")
    parser.add_argument("--n-samples", type=int, default=1000, help="number of collision starts")
    parser.add_argument("--max-steps", type=int, default=200, help="maximum escape steps")
    parser.add_argument("--lr-optimization", type=float, default=0.1, help="optimization learning rate")
    parser.add_argument("--step-sampling", type=float, default=0.1, help="sampling step size")
    parser.add_argument("--k-sampling", type=int, default=50, help="candidates per sampling step")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--output-dir", default="output/escape", help="result directory")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    problem = robot_tools.PlanningProblem(f"config/{args.world}.yaml")
    model = load_model(f"output/{args.world}/model_final.pt", device)
    model.eval()

    print(f"Using {device}")
    q_starts = sample_random_collision(problem, args.n_samples)

    methods = {
        "Optimization": (
            OptimizationEscaper(model, problem, device),
            {"lr": args.lr_optimization, "max_steps": args.max_steps},
        ),
        "Sampling": (
            SamplingEscaper(model, problem, device),
            {
                "step_size": args.step_sampling,
                "max_steps": args.max_steps,
                "k": args.k_sampling,
            },
        ),
    }
    results = {name: [] for name in methods}

    for q0 in tqdm(q_starts, desc="Escape benchmark", ncols=100):
        for name, (escaper, run_args) in methods.items():
            results[name].append(escaper.run(q0, **run_args))

    summary = {name: summarize(method_results) for name, method_results in results.items()}

    print(f"\n{'=' * 78}")
    print(f"2D ESCAPE BENCHMARK (N={len(q_starts)}, World={args.world})")
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
    print(f"{'=' * 78}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"results_2d_{args.world}.json")
    output = {
        "world": args.world,
        "config": vars(args),
        "summary": summary,
        "details": {name: serialize_results(method_results) for name, method_results in results.items()},
    }
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
    print(f"Saved results to {output_path}")
