import os
import numpy as np
import mujoco
from tqdm import tqdm
from multiprocessing import Pool
from skimage.measure import marching_cubes
import mouette as M

# Reuse JacoEnv logic
from jaco_arm.jaco import JacoEnv

_GLOBAL_ENV = None

def _init_oracle(model_path):
    global _GLOBAL_ENV
    _GLOBAL_ENV = JacoEnv(width=101, height=101, model_path=model_path)

def _oracle_worker(q):
    env = _GLOBAL_ENV
    for i in range(len(q)):
        env.data.qpos[i] = q[i]
    mujoco.mj_forward(env.model, env.data)
    
    is_collision = False
    for i in range(env.data.ncon):
        c = env.data.contact[i]
        g1, g2 = c.geom1, c.geom2
        fixed_geoms = {0, 1, 2, 12}
        if g1 in fixed_geoms and g2 in fixed_geoms:
            continue
        b1, b2 = env.model.geom_bodyid[g1], env.model.geom_bodyid[g2]
        if env.model.body_parentid[b1] == b2 or env.model.body_parentid[b2] == b1:
            continue
        is_collision = True
        break
    return 1 if is_collision else -1 # 1 for collision, -1 for free (for marching cubes iso=0)

def generate_true_cspace():
    model_path = "jaco_3dof.xml"
    res = 64 # High resolution for "precise" C-space
    n_workers = os.cpu_count() or 4
    
    ticks = np.linspace(-np.pi, np.pi, res)
    L = [ticks] * 3
    grid_points = np.stack(np.meshgrid(*L, indexing='ij'), axis=-1).reshape(-1, 3)
    
    print(f"Brute-force collision checking on {res}^3 grid ({len(grid_points)} points)...")
    with Pool(processes=n_workers, initializer=_init_oracle, initargs=(model_path,)) as pool:
        results = list(tqdm(pool.imap(_oracle_worker, grid_points, chunksize=512), total=len(grid_points)))
    
    dist_values = np.array(results).reshape(res, res, res)
    
    print("Extracting isosurface via Marching Cubes...")
    try:
        # iso=0 will be the boundary between collision (1) and free (-1)
        verts, faces, normals, values = marching_cubes(dist_values, level=0)
        
        # Map back to joint space
        for i in range(3):
            verts[:, i] = np.interp(verts[:, i], np.arange(res), L[i])
            
        # Create Mouette mesh
        m = M.mesh.RawMeshData()
        m.vertices += [M.Vec(*v) for v in verts]
        m.faces += list(faces)
        mesh = M.mesh.SurfaceMesh(m)
        
        out_dir = "output/ndf"
        os.makedirs(out_dir, exist_ok=True)
        
        # 1. Save Mesh (Existing format)
        out_file = os.path.join(out_dir, "true_cspace_mesh.obj")
        M.mesh.save(mesh, out_file)
        print(f"Precise C-Space mesh saved to {out_file}")

        # 2. Save Grid (Unified format like 2D)
        # Convert 1 (collision) / -1 (free) to 1 (collision) / 0 (free) for consistency
        cspace_grid = (dist_values == 1).astype(int)
        np.save(os.path.join(out_dir, "jaco_cspace.npy"), cspace_grid)
        for i in range(3):
            np.save(os.path.join(out_dir, f"jaco_theta{i}.npy"), L[i])
        print(f"Precise C-Space grid data saved to {out_dir}")
        
    except Exception as e:
        print(f"Failed to generate mesh: {e}")

if __name__ == "__main__":
    generate_true_cspace()
