#!/usr/bin/env python3
import os
import time
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import robot_tools  # 你的模块

# ───────────── 全局/多进程 oracle 初始化 ─────────────
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
            return -1  # 碰撞(inside obstacle)
    return 1          # 自由(free)

# 并行标注
def _parallel_label(points, pool, desc, chunksize=1024, ncols=80):
    if len(points) == 0:
        return np.zeros((0,), dtype=int)
    it = pool.imap(_oracle_worker, points, chunksize=chunksize)
    return np.fromiter(tqdm(it, total=len(points), desc=desc, ncols=80),
                       dtype=int, count=len(points))

# 论文式 Gaussian PRM 采样（循环接受-拒绝，直到凑满 n_pts）
def sample_gaussian_prm(
    n_pts, bounds, pool,
    sigma=0.05,                 # 高斯扰动比例(相对每维 span)
    p_gauss=1.0,                # 采用 Gaussian pair 的概率；其余用纯均匀
    batch_size=4096,            # 每轮生成多少对候选
    max_rounds=10               # 最多循环多少轮
):
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    span = hi - lo

    def sample_uniform(m):
        if m <= 0:
            return np.zeros((0, dim))
        u = np.random.rand(m, dim)
        return lo + u * span

    X_in_list, X_out_list = [], []
    total_target = n_pts

    for round_id in range(max_rounds):
        need = total_target - (len(X_in_list) + len(X_out_list))
        if need <= 0:
            break

        # 这轮尝试生成的成对数量：放大一些提高命中率
        n_pairs = max(batch_size, need * 2)

        # 决定这轮用多少比例做 Gaussian pair
        n_pairs_gauss = int(np.round(p_gauss * n_pairs))
        n_pairs_uniform = n_pairs - n_pairs_gauss

        # 1) Gaussian pair：q1 uniform, q2 = q1 + N(0, σ²I)
        if n_pairs_gauss > 0:
            q1 = sample_uniform(n_pairs_gauss)
            q2 = np.clip(q1 + np.random.randn(n_pairs_gauss, dim) * (sigma * span), lo, hi)
            l1 = _parallel_label(q1, pool, f"[Round {round_id+1}] Label q1 (gauss)")
            l2 = _parallel_label(q2, pool, f"[Round {round_id+1}] Label q2 (gauss)")
            # 接受：一内一外
            mask = ((l1 == -1) & (l2 == 1)) | ((l1 == 1) & (l2 == -1))
            idx = np.where(mask)[0]
            if len(idx) > 0:
                # 记录两侧（自由侧 -> X_out，碰撞侧 -> X_in）
                cond = (l1[idx] == -1)[:, None]   # (N,1)
                in_side  = np.where(cond, q1[idx], q2[idx])  # (N,2)
                out_side = np.where(cond, q2[idx], q1[idx])  # (N,2)
                # np.where 对二维数组要这样转一下
                in_side = np.vstack(in_side)
                out_side = np.vstack(out_side)
                X_in_list.append(in_side)
                X_out_list.append(out_side)

        # 2) 纯均匀补样（如果需要）
        if n_pairs_uniform > 0:
            U = sample_uniform(n_pairs_uniform)
            Ul = _parallel_label(U, pool, f"[Round {round_id+1}] Label uniform")
            if np.any(Ul == -1):
                X_in_list.append(U[Ul == -1])
            if np.any(Ul == 1):
                X_out_list.append(U[Ul == 1])

        # 汇总当前进度
        cur_in = 0 if len(X_in_list) == 0 else np.sum([a.shape[0] for a in X_in_list])
        cur_out = 0 if len(X_out_list) == 0 else np.sum([a.shape[0] for a in X_out_list])
        cur_total = cur_in + cur_out
        tqdm.write(f"  -> Accumulated: in={cur_in}, out={cur_out}, total={cur_total}/{total_target}")

    # 拼接 & 截断到恰好 n_pts（按当前比例裁剪）
    X_in = np.vstack(X_in_list) if len(X_in_list) else np.zeros((0, dim))
    X_out = np.vstack(X_out_list) if len(X_out_list) else np.zeros((0, dim))

    total = X_in.shape[0] + X_out.shape[0]
    if total == 0:
        return X_in, X_out

    if total > n_pts:
        ratio_in = X_in.shape[0] / total
        keep_in = min(X_in.shape[0], int(round(n_pts * ratio_in)))
        keep_out = n_pts - keep_in
        if X_in.shape[0] > keep_in:
            X_in = X_in[np.random.permutation(X_in.shape[0])[:keep_in]]
        if X_out.shape[0] > keep_out:
            X_out = X_out[np.random.permutation(X_out.shape[0])[:keep_out]]

    return X_in, X_out

# ───────────── 主程序 ─────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Gaussian-PRM Sampling for C-space",
        description="按 Boor et al. (1999) 思路实现的 Gaussian 近边界采样（循环接受-拒绝，直到凑满样本）"
    )
    parser.add_argument("-w", "--world", default="config/block_world.yaml", help="World 配置文件")
    parser.add_argument("-n", "--n-train", type=int, default=10000, help="希望输出的总样本数（in+out）")
    parser.add_argument("--outdir", type=str, default="inputs_gaussian", help="输出目录")
    parser.add_argument("-prefix", type=str, default=None, help="文件名前缀（默认用 world 文件名）")
    parser.add_argument("-sigma", "--sigma-pair", type=float, default=0.03, help="高斯扰动比例 σ（相对每维 span）")
    parser.add_argument("-pg", "--p-gauss", type=float, default=0.95, help="每轮用于 Gaussian pair 的比例(余下为纯 uniform)")
    parser.add_argument("-bs", "--batch-size", type=int, default=4096, help="每轮生成的候选对数量（越大越快，但占内存）")
    parser.add_argument("-mr", "--max-rounds", type=int, default=10, help="最多循环轮数（避免无限循环）")
    parser.add_argument("-visu", action="store_true", default=True, help="保存可视化图")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    prefix = args.prefix if args.prefix is not None else Path(args.world).stem

    # 构造 C-space 网格（仅用于可视化/保存）
    arm_problem = robot_tools.PlanningProblem(args.world)
    theta, c_space = arm_problem.construct_config_space(nx=101)
    bounds = [(theta[0][0], theta[0][-1]), (theta[1][0], theta[1][-1])]

    # 采样
    n_workers = cpu_count() or 4
    with Pool(processes=n_workers, initializer=_init_oracle, initargs=(args.world,)) as pool:
        t0 = time.time()
        X_in, X_out = sample_gaussian_prm(
            n_pts=args.n_train,
            bounds=bounds,
            pool=pool,
            sigma=args.sigma_pair,
            p_gauss=args.p_gauss,
            batch_size=args.batch_size,
            max_rounds=args.max_rounds
        )
        t1 = time.time()

    print(f"[Done] Gaussian-PRM sampling took {t1 - t0:.3f}s with {n_workers} workers.")
    print("X_in shape :", X_in.shape)
    print("X_out shape:", X_out.shape)

    # 保存
    np.save(os.path.join(args.outdir, f"{prefix}_Xtrain_in.npy"),  X_in)
    np.save(os.path.join(args.outdir, f"{prefix}_Xtrain_out.npy"), X_out)
    np.save(os.path.join(args.outdir, f"{prefix}_cspace.npy"),     c_space)
    np.save(os.path.join(args.outdir, f"{prefix}_theta0.npy"),     theta[0])
    np.save(os.path.join(args.outdir, f"{prefix}_theta1.npy"),     theta[1])

    # 可视化
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
        plt.title(f"Gaussian-PRM near-boundary samples ({args.n_train})")
        plt.legend(loc="lower right", frameon=True, framealpha=0.9)
        out_path = os.path.join(args.outdir, f"{prefix}_gaussianPRM_{args.n_train}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to {out_path}")