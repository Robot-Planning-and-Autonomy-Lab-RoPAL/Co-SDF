import os
import argparse
import robot_tools
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


def extract_train_point_cloud(n_pts, bounds, oracle, near_ratio=0.25, sigma=0.02):
	"""
	在 C-space 中采样训练点并打标签 (free/obs)，并包含两类样本：
	1) 全局均匀采样点 (X1)，覆盖整个关节范围
	2) 边界附近采样点 (X2)，在 free/obs 边界周围额外采样，增加训练难点样本

	Args:
		n_pts (int): 总采样点数
		bounds (list of (low, high)): 每个关节的上下界
		oracle (callable): 碰撞检测函数，输入 q(ndarray)，返回 True=碰撞, False=free
		near_ratio (float): 贴近边界的采样比例 (0~1)，决定 X2 的占比
		sigma (float): 在边界点附近加噪声的标准差，控制扰动幅度
		r_min (float, optional): 最小距离阈值，可选，过滤掉离边界过近的点
		balance (bool): 是否强制 free/obs 各取一半

	Returns:
		X_in  (ndarray): 碰撞点（inside/obs, oracle=True）
		X_out (ndarray): 可行点（outside/free, oracle=False）
	"""
	dim = len(bounds)
	lo = np.array([b[0] for b in bounds], dtype=float)
	hi = np.array([b[1] for b in bounds], dtype=float)
	
	def sample_uniform(m):
		u = np.random.rand(m, dim)
		return lo + u * (hi - lo)
	
#	X1: 全局采样
	n1 = int(n_pts * (1.0 - near_ratio))
	X1 = sample_uniform(max(n1, 0))
	
#	X2: 边界附近采样
	candidates = sample_uniform(max(10 * n_pts, 1))
	cand_labels = []
	for q in tqdm(candidates, desc="Labeling random samples", ncols=80):
		cand_labels.append(-1 if oracle(q) else 1)
	cand_labels = np.array(cand_labels)


	
#	粗略找交界：相邻标签不同
	idx_near = np.where(cand_labels[:-1] != cand_labels[1:])[0]
	k = int(n_pts * near_ratio)
	
	if len(idx_near) > 0:
		base = candidates[idx_near[:k]]
	else:
		base = sample_uniform(k)
		
	noise = sigma * (hi - lo) * np.random.randn(len(base), dim)
	X2 = np.clip(base + noise, lo, hi)
	
	X = np.vstack([X1, X2]) if len(X2) else X1
	labels = np.array([-1 if oracle(q) else 1 for q in X])
	X_in, X_out = X[labels == -1], X[labels == 1]
	return X_in, X_out

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog="C-space Dataset Generator",
		description="Generate a dataset to train a neural distance field from C-space sampling"
	)
	parser.add_argument('-w', '--world', default='config/block_world.yaml', help='World definition (obstacles)')
	
	parser.add_argument("-n", "--n-train", type=int, default=10000,
						help="number of samples in the training set")
	parser.add_argument("-visu", action="store_true",
						help="generate visualization of sampled configurations")
	parser.add_argument("-r", "--near-ratio", type=float, default=0.25,
						help="ratio of samples drawn near the free/obs boundary")
	parser.add_argument("-s", "--sigma", type=float, default=0.02,
						help="noise scale for near-boundary sampling")
	parser.add_argument("-rmin", "--min-dist", type=float, default=0.0,
						help="minimum distance from boundary to keep (optional)")
	parser.add_argument("-bal", "--balance", action="store_true",
						help="balance dataset: equal free/obs samples")
					
	args = parser.parse_args()
	
	outdir = getattr(args, "outdir", "inputs_cspace")
	os.makedirs(outdir, exist_ok=True)
	
	if hasattr(args, "prefix") and args.prefix is not None:
		prefix = args.prefix
	else:
		prefix = Path(args.world).stem
		
	arm_problem = robot_tools.PlanningProblem(args.world)
	theta, c_space = arm_problem.construct_config_space(nx=101)
	
	def oracle(q):
		arm_problem.robot.set_link_angles(q)
		fp = arm_problem.robot.get_current_polygon()
		for o in arm_problem.workspace.obstacles:
			if fp.intersect(o):
				return True
		return False
	
#	角度范围（与 construct_config_space 一致）
	bounds = [(theta[0][0], theta[0][-1]), (theta[1][0], theta[1][-1])]
	
#	采样生成训练点
	X_in, X_out = extract_train_point_cloud(
		n_pts=args.n_train,
		bounds=bounds,
		oracle=oracle,
		near_ratio=args.near_ratio,
		sigma=args.sigma
	)
	

	np.save(os.path.join(outdir, f"{prefix}_Xtrain_in.npy"),  X_in)      # 碰撞点（obs）
	np.save(os.path.join(outdir, f"{prefix}_Xtrain_out.npy"), X_out)     # 可行点（free）
	np.save(os.path.join(outdir, f"{prefix}_cspace.npy"),     c_space)   # 占据图（0/1 或 bool）
	np.save(os.path.join(outdir, f"{prefix}_theta0.npy"),     theta[0])  # θ1 轴坐标
	np.save(os.path.join(outdir, f"{prefix}_theta1.npy"),     theta[1])  # θ2 轴坐标
	
#	画图，标出采样点
	plt.figure(figsize=(6,5))
	plt.imshow(c_space.T, origin="lower",
				extent=[theta[0][0], theta[0][-1], theta[1][0], theta[1][-1]],
				cmap="gray_r")
	if len(X_in):  plt.scatter(X_in[:,0],  X_in[:,1],  c='red',  s=5, label="Collision")
	if len(X_out): plt.scatter(X_out[:,0], X_out[:,1], c='blue', s=5, label="Free")
	plt.xlabel(r"$\theta_1$"); plt.ylabel(r"$\theta_2$")
	plt.title("Configuration Space with Sampled Points"); plt.legend()
	plt.show()