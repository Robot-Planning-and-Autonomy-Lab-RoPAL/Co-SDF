#!/usr/bin/env python3
import os
import time
import argparse
import robot_tools
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ────────────── 全局 oracle（用于多进程） ──────────────
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
            return -1  # collision
    return 1  # free

# ────────────── 工具函数 ──────────────
def _parallel_label(points, pool, desc, chunksize=1024):
    it = pool.imap(_oracle_worker, points, chunksize=chunksize)
    return np.fromiter(tqdm(it, total=len(points), desc=desc, ncols=80),
                       dtype=int, count=len(points))

def _sample_uniform(m, lo, hi):
    if m <= 0: return np.zeros((0, lo.shape[0]))
    u = np.random.rand(m, lo.shape[0])
    return lo + u * (hi - lo)

# ────────────── 近边界带状采样（1-Lip风格） ──────────────
def extract_train_point_cloud_band(
    n_pts,
    bounds,
    pool,
    band_ratio=0.7,        # 最终样本中有多少来自“边界带”
    pair_batch=5000,       # 每轮用多少对(q1,q2)寻找交界
    max_rounds=5,          # 最多做几轮寻找交界
    sigma_pair=0.06,       # q2 = q1 + N(0, (sigma_pair*span)^2)
    sigma_band=0.01,       # 在“边界种子”周围再抖动的噪声比例
    uniform_ratio=0.3,     # 非边界样本里，多少用全局均匀补点
    balance=False          # 是否强制 in/out 数量相近
):
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    span = hi - lo

    # 需要的数量
    n_band_target   = int(round(n_pts * band_ratio))
    n_nonband_target= n_pts - n_band_target

    # 1) 多轮：用高斯成对扰动找到“穿越边界”的配对，并提取边界种子（中点）
    seeds = []
    for r in range(max_rounds):
        # 生成成对候选
        q1 = _sample_uniform(pair_batch, lo, hi)
        q2 = np.clip(q1 + np.random.randn(pair_batch, dim) * (sigma_pair * span), lo, hi)
        # 并行标注
        l1 = _parallel_label(q1, pool, f"[Round {r+1}] Label q1 (gauss)", chunksize=512)
        l2 = _parallel_label(q2, pool, f"[Round {r+1}] Label q2 (gauss)", chunksize=512)
        # 仅保留一正一负
        cross = (l1 != l2)
        if np.any(cross):
            mid = 0.5 * (q1[cross] + q2[cross])  # 边界近似中点
            seeds.append(mid)
        # 提前停止条件：种子足够多
        if len(seeds) > 0 and sum(s.shape[0] for s in seeds) >= n_band_target:
            break

    if len(seeds) > 0:
        seeds = np.vstack(seeds)
    else:
        # 极端情况：没有任何穿越对，退化为均匀采样
        seeds = _sample_uniform(n_band_target, lo, hi)

    # 2) 围绕“边界种子”进行带状抖动采样（近似零水平集）
    if seeds.shape[0] < n_band_target:
        # 不足则重复采样
        reps = int(np.ceil(n_band_target / max(seeds.shape[0], 1)))
        seeds = np.repeat(seeds, reps, axis=0)

    band = seeds[:n_band_target]
    band = np.clip(
        band + np.random.randn(band.shape[0], dim) * (sigma_band * span),
        lo, hi
    )

    # 3) 非边界样本：少量全局均匀，其余可设为“边界远处”(这里简化为全局均匀)
    nonband = _sample_uniform(n_nonband_target, lo, hi)

    # 4) 合并并最终标注
    X = np.vstack([band, nonband]) if nonband.size > 0 else band
    labels = _parallel_label(X, pool, "Label final X", chunksize=1024)

    X_in  = X[labels == -1]
    X_out = X[labels ==  1]

    # 5) 可选：平衡 in/out
    if balance and (X_in.shape[0] > 0 and X_out.shape[0] > 0):
        k = min(X_in.shape[0], X_out.shape[0], n_pts // 2)
        perm_in  = np.random.permutation(X_in.shape[0])[:k]
        perm_out = np.random.permutation(X_out.shape[0])[:k]
        X_in, X_out = X_in[perm_in], X_out[perm_out]
    else:
        # 否则裁到总数 n_pts，按当前比例
        total = X_in.shape[0] + X_out.shape[0]
        if total > n_pts and total > 0:
            r_in = X_in.shape[0] / total
            keep_in  = min(X_in.shape[0], int(round(n_pts * r_in)))
            keep_out = n_pts - keep_in
            if X_in.shape[0] > keep_in:
                X_in = X_in[np.random.permutation(X_in.shape[0])[:keep_in]]
            if X_out.shape[0] > keep_out:
                X_out = X_out[np.random.permutation(X_out.shape[0])[:keep_out]]

    print("X_in shape :", X_in.shape)
    print("X_out shape:", X_out.shape)
    return X_in, X_out

# ────────────── 主程序 ──────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="C-space Dataset Generator",
        description="Near-boundary band sampling (1-Lip style) for C-space"
    )
    parser.add_argument('-w', '--world', default='config/block_world.yaml', help='World definition (obstacles)')
    parser.add_argument("-n", "--n-train", type=int, default=2000, help="total number of samples (in + out)")
    parser.add_argument("-visu", action="store_true", help="save visualization")
    parser.add_argument("--band-ratio", type=float, default=0.7, help="ratio of near-boundary samples")
    parser.add_argument("--pair-batch", type=int, default=5000, help="pair batch per round to find crossings")
    parser.add_argument("--max-rounds", type=int, default=5, help="max rounds to search crossings")
    parser.add_argument("--sigma-pair", type=float, default=0.06, help="std ratio for q2 = q1 + N(0,(sigma_pair*span)^2)")
    parser.add_argument("--sigma-band", type=float, default=0.01, help="jitter std ratio around boundary seeds")
    parser.add_argument("--uniform-ratio", type=float, default=0.3, help="portion of non-band samples that are uniform")
    parser.add_argument("--balance", action="store_true", help="force balanced in/out")
    args = parser.parse_args()

    outdir = "inputs_cspace"
    os.makedirs(outdir, exist_ok=True)
    prefix = Path(args.world).stem

    # 构造 C-space 栅格（仅用于可视化与保存）
    arm_problem = robot_tools.PlanningProblem(args.world)
    theta, c_space = arm_problem.construct_config_space(nx=101)
    bounds = [(theta[0][0], theta[0][-1]), (theta[1][0], theta[1][-1])]

    # 并行采样
    n_workers = cpu_count() or 4
    with Pool(processes=n_workers, initializer=_init_oracle, initargs=(args.world,)) as pool:
        t0 = time.time()
        X_in, X_out = extract_train_point_cloud_band(
            n_pts=args.n_train,
            bounds=bounds,
            pool=pool,
            band_ratio=args.band_ratio,
            pair_batch=args.pair_batch,
            max_rounds=args.max_rounds,
            sigma_pair=args.sigma_pair,
            sigma_band=args.sigma_band,
            uniform_ratio=args.uniform_ratio,
            balance=args.balance
        )
        t1 = time.time()
    print(f"Band sampling takes {t1 - t0:.3f}s with {n_workers} workers")

    # 保存
    np.save(os.path.join(outdir, f"{prefix}_Xtrain_in.npy"),  X_in)
    np.save(os.path.join(outdir, f"{prefix}_Xtrain_out.npy"), X_out)
    np.save(os.path.join(outdir, f"{prefix}_cspace.npy"),     c_space)
    np.save(os.path.join(outdir, f"{prefix}_theta0.npy"),     theta[0])
    np.save(os.path.join(outdir, f"{prefix}_theta1.npy"),     theta[1])

    # 可视化
    if args.visu:
        plt.figure(figsize=(6,5))
        plt.imshow(
            c_space.T, origin="lower",
            extent=[theta[0][0], theta[0][-1], theta[1][0], theta[1][-1]],
            cmap="gray_r"
        )
        if X_in.shape[0]  > 0: plt.scatter(X_in[:,0],  X_in[:,1],  c='red',  s=5, label="Collision")
        if X_out.shape[0] > 0: plt.scatter(X_out[:,0], X_out[:,1], c='blue', s=5, label="Free")
        plt.xlabel(r"$\theta_1$")
        plt.ylabel(r"$\theta_2$")
        plt.title(f"Near-boundary band sampling ({args.n_train})")
        plt.legend(loc='lower right', frameon=True, framealpha=0.9)
        out_path = os.path.join(outdir, f"{prefix}_band_{args.n_train}_samples.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to {out_path}")