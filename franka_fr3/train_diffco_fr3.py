#!/usr/bin/env python3
"""
Train DiffCo for FR3 robot using MuJoCo collision oracle.

Usage:
    python train_diffco_fr3.py --scene obs20 --n_train 8000 --n_test 2000
"""
import os
import sys
import dill as pickle
import argparse
import numpy as np
import torch
from tqdm import tqdm
from multiprocessing import Pool

import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'diffco'))
from diffco.kernel_perceptrons import DiffCo
from diffco import kernel
from diffco.model import FR3FK

# ── MuJoCo oracle ────────────────────────────────────────────────────────────

_GLOBAL_MODEL = None
_GLOBAL_DATA  = None
_ROBOT_BODY_IDS = None

def _init_oracle(model_path):
    global _GLOBAL_MODEL, _GLOBAL_DATA, _ROBOT_BODY_IDS
    _GLOBAL_MODEL = mujoco.MjModel.from_xml_path(model_path)
    _GLOBAL_DATA  = mujoco.MjData(_GLOBAL_MODEL)

    base_id = mujoco.mj_name2id(_GLOBAL_MODEL, mujoco.mjtObj.mjOBJ_BODY, "base")
    robot_ids = set()
    if base_id != -1:
        for i in range(_GLOBAL_MODEL.nbody):
            curr = i
            while curr > 0:
                if curr == base_id:
                    robot_ids.add(i)
                    break
                curr = _GLOBAL_MODEL.body_parentid[curr]
    _ROBOT_BODY_IDS = np.zeros(_GLOBAL_MODEL.nbody, dtype=bool)
    if robot_ids:
        _ROBOT_BODY_IDS[list(robot_ids)] = True

def _oracle_worker(q):
    """Return -1 if in collision with obstacle, +1 if free."""
    _GLOBAL_DATA.qpos[:7] = q
    if _GLOBAL_MODEL.nq > 7:
        _GLOBAL_DATA.qpos[7:] = 0.04
    mujoco.mj_fwdPosition(_GLOBAL_MODEL, _GLOBAL_DATA)
    for i in range(_GLOBAL_DATA.ncon):
        c = _GLOBAL_DATA.contact[i]
        b1 = _GLOBAL_MODEL.geom_bodyid[c.geom1]
        b2 = _GLOBAL_MODEL.geom_bodyid[c.geom2]
        if _ROBOT_BODY_IDS[b1] ^ _ROBOT_BODY_IDS[b2]:
            return -1
    return 1

def label_configs(cfgs_np, model_path, n_workers, desc="Labelling"):
    with Pool(processes=n_workers, initializer=_init_oracle, initargs=(model_path,)) as pool:
        labels = list(tqdm(pool.imap(_oracle_worker, cfgs_np, chunksize=512),
                           total=len(cfgs_np), desc=desc, ncols=80))
    return np.array(labels, dtype=np.float32)

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene',    default='obs20',  help='Scene name under scenes/')
    parser.add_argument('--n_train',  type=int, default=8000)
    parser.add_argument('--n_test',   type=int, default=2000)
    parser.add_argument('--lmbda',    type=float, default=10.0, help='RQKernel bandwidth')
    parser.add_argument('--seed',     type=int, default=1917)
    parser.add_argument('--workers',  type=int, default=None)
    parser.add_argument('--out_dir',  default='results/diffco')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    scene_path = os.path.abspath(f'scenes/{args.scene}.xml')
    os.makedirs(args.out_dir, exist_ok=True)
    n_workers  = args.workers or (os.cpu_count() or 4)

    robot = FR3FK()
    lo = robot.limits[:, 0].numpy()
    hi = robot.limits[:, 1].numpy()
    span = hi - lo

    n_total = args.n_train + args.n_test
    cfgs_np = lo + np.random.rand(n_total, 7).astype(np.float32) * span
    print(f"Labelling {n_total} configs in {args.scene} with {n_workers} workers …")
    labels_np = label_configs(cfgs_np, scene_path, n_workers)

    n_col = int((labels_np == -1).sum())
    n_free = int((labels_np == 1).sum())
    print(f"Collision: {n_col}  Free: {n_free}  Ratio: {n_col/n_total:.2%}")

    cfgs   = torch.from_numpy(cfgs_np)
    labels = torch.from_numpy(labels_np)

    train_cfgs   = cfgs[:args.n_train]
    train_labels = labels[:args.n_train]
    test_cfgs    = cfgs[args.n_train:]
    test_labels  = labels[args.n_train:]

    # ── Train DiffCo ─────────────────────────────────────────────────────────
    print("Training DiffCo …")
    kernel_func = kernel.RQKernel(args.lmbda)
    checker = DiffCo(kernel_func=kernel_func, beta=1.0, transform=robot.fkine)
    checker.train(train_cfgs, train_labels,
                  max_iteration=len(train_cfgs),
                  distance=None,
                  verbose=True)

    # ── Fit continuous scorer ────────────────────────────────────────────────
    print("Fitting RBF scorer …")
    checker.fit_poly(kernel_func=kernel.Polyharmonic(1, 1.0), target='label')
    dist_est = checker.poly_score

    # ── Evaluate ─────────────────────────────────────────────────────────────
    test_preds = (dist_est(test_cfgs) > 0) * 2 - 1
    test_labels_r = test_labels.reshape(test_preds.shape)
    acc  = (test_preds == test_labels_r).float().mean()
    tpr  = (test_preds[test_labels_r ==  1] ==  1).float().mean()
    tnr  = (test_preds[test_labels_r == -1] == -1).float().mean()
    print(f'Test acc: {acc:.4f}, TPR: {tpr:.4f}, TNR: {tnr:.4f}')
    print(f'Support points: {len(checker.gains)}')

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(args.out_dir, f'fr3_{args.scene}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(checker, f)
    print(f"Checker saved → {out_path}")


if __name__ == '__main__':
    main()
