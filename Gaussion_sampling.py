#!/usr/bin/env python3
import os
import time
import argparse
import robot_tools
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

_GLOBAL_PROBLEM = None

def _init_oracle(world_path):
    global _GLOBAL_PROBLEM
    _GLOBAL_PROBLEM = robot_tools.PlanningProblem(world_path)

def _oracle_worker(q):
    p = _GLOBAL_PROBLEM
    p.robot.set_link_angles(q)
    fp = p.robot.get_current_polygon()
    for o in p.workspace.obstacles:
        if fp.intersect(o):
            return -1
    return 1

def extract_train_point_cloud_gaussian_pairs(n_pts, bounds, pool, sigma_pair=0.05, mix_uniform_ratio=0.0):
    
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    span = hi - lo

    def sample_uniform(m):
        if m <= 0:
            return np.zeros((0, dim))
        u = np.random.rand(m, dim)
        return lo + u * span

    def parallel_label(points, pool, desc):
        
        it = pool.imap(_oracle_worker, points, chunksize=256)
        return np.fromiter(tqdm(it, total=len(points), desc=desc, ncols=80),
                           dtype=int, count=len(points))

    # 生成成对候选，提高跨过边界的命中率
    n_pairs = max(int(10.0 * n_pts), 1)
    q1 = sample_uniform(n_pairs)
    q2 = np.clip(q1 + np.random.randn(n_pairs, dim) * (sigma_pair * span), lo, hi)

    # 并行标注碰撞检测
    l1 = parallel_label(q1, pool, "Label q1")
    l2 = parallel_label(q2, pool, "Label q2")

    mask = ((l1 == -1) & (l2 == 1)) | ((l1 == 1) & (l2 == -1))
    idx = np.where(mask)[0]

    X_in_list, X_out_list = [], []
    for i in idx:
        if l1[i] == -1 and l2[i] == 1:
            X_in_list.append(q1[i])
            X_out_list.append(q2[i])
        elif l1[i] == 1 and l2[i] == -1:
            X_in_list.append(q2[i])
            X_out_list.append(q1[i])

    X_in = np.array(X_in_list) if len(X_in_list) else np.zeros((0, dim))
    X_out = np.array(X_out_list) if len(X_out_list) else np.zeros((0, dim))

    total = X_in.shape[0] + X_out.shape[0]
    need = max(n_pts - total, 0)
    if need > 0 and mix_uniform_ratio > 0:
        m = int(need * mix_uniform_ratio)
        U = sample_uniform(m)
        Ulab = parallel_label(U, pool, "Label uniform mix")
        if np.any(Ulab == -1):
            X_in = np.vstack([X_in, U[Ulab == -1]])
        if np.any(Ulab == 1):
            X_out = np.vstack([X_out, U[Ulab == 1]])

    # 截断到不超过 n_pts
    total = X_in.shape[0] + X_out.shape[0]
    if total > n_pts and total > 0:
        ratio_in = X_in.shape[0] / total
        keep_in = min(X_in.shape[0], int(round(n_pts * ratio_in)))
        keep_out = n_pts - keep_in
        if X_in.shape[0] > keep_in:
            X_in = X_in[np.random.permutation(X_in.shape[0])[:keep_in]]
        if X_out.shape[0] > keep_out:
            X_out = X_out[np.random.permutation(X_out.shape[0])[:keep_out]]

    return X_in, X_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="C-space Gaussian Sampling Dataset Generator", description="Generate a dataset to train a neural distance field from Gaussian sampling")
    parser.add_argument('-w', '--world', default='config/block_world.yaml', help='World definition (obstacles)')
    parser.add_argument("-n", "--n-train", type=int, default=1000, help="number of samples in the training set")
    parser.add_argument("--outdir", type=str, default="inputs_gaussian", help="输出目录")
    parser.add_argument("-prefix", type=str, default=None, help="文件名前缀")
    parser.add_argument("-sigma", "--sigma-pair", type=float, default=0.05, help="noise scale for near-boundary sampling")
    parser.add_argument("-visu", action="store_true", help="generate visualization of sampled configurations")
    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    prefix = args.prefix if args.prefix is not None else Path(args.world).stem

    arm_problem = robot_tools.PlanningProblem(args.world)
    theta, c_space = arm_problem.construct_config_space(nx=101)
    bounds = [(theta[0][0], theta[0][-1]), (theta[1][0], theta[1][-1])]

    n_workers = cpu_count() or 4
    with Pool(processes=n_workers, initializer=_init_oracle, initargs=(args.world,)) as pool:
        t0 = time.time()
        X_in, X_out = extract_train_point_cloud_gaussian_pairs(
            n_pts=args.n_train,
            bounds=bounds,
            pool=pool,
            sigma_pair=args.sigma_pair,
            mix_uniform_ratio=0.3
        )
        t1 = time.time()
    print(f"Gaussian pairs sampling takes {t1 - t0:.3f}s with {n_workers} workers")


    np.save(os.path.join(outdir, f"{prefix}_Xtrain_in.npy"),  X_in)
    np.save(os.path.join(outdir, f"{prefix}_Xtrain_out.npy"), X_out)
    np.save(os.path.join(outdir, f"{prefix}_cspace.npy"),     c_space)
    np.save(os.path.join(outdir, f"{prefix}_theta0.npy"),     theta[0])
    np.save(os.path.join(outdir, f"{prefix}_theta1.npy"),     theta[1])

    if args.visu:
        plt.figure(figsize=(6, 5))
        plt.imshow(
            c_space.T, origin="lower",
            extent=[theta[0][0], theta[0][-1], theta[1][0], theta[1][-1]],
            cmap="gray_r"
        )
        if X_in.shape[0] > 0:
            plt.scatter(X_in[:, 0],  X_in[:, 1],  c='red',  s=5, label="Collision")
        if X_out.shape[0] > 0:
            plt.scatter(X_out[:, 0], X_out[:, 1], c='blue', s=5, label="Free")
        plt.xlabel(r"$\theta_1$")
        plt.ylabel(r"$\theta_2$")
        plt.title(f"C-space Gaussian Pair {args.n_train} Samples")
        plt.legend()
        out_path = os.path.join(outdir, f"{prefix}_gaussian_{args.n_train}_samples.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to {out_path}")