#!/usr/bin/env python3
"""
Render Franka FR3 trajectory as GIF/MP4 using MuJoCo
"""
import numpy as np
import mujoco
import os
import argparse
from PIL import Image
import imageio
from scipy.interpolate import interp1d

def render_trajectory_gif(path_file, scene_xml, output_file, num_frames, fps, width, height, camera_distance, azimuth, elevation):
    print(f"Loading trajectory from {path_file}...")
    path = np.load(path_file)
    if path.ndim != 2 or path.shape[1] != 7:
        raise ValueError(f"Expected path of shape (T, 7), got {path.shape}")
    
    print(f"Trajectory has {len(path)} waypoints")
    
    if num_frames > 1:
        print(f"Interpolating to {num_frames} frames...")
        t_orig = np.linspace(0, 1, len(path))
        t_new = np.linspace(0, 1, num_frames)
        f = interp1d(t_orig, path, axis=0, kind='linear')
        path = f(t_new)
        print(f"  New path has {len(path)} waypoints")
    
    
    print(f"Loading scene from {scene_xml}...")
    abs_xml = os.path.abspath(scene_xml)
    mj_model = mujoco.MjModel.from_xml_path(abs_xml)
    mj_data = mujoco.MjData(mj_model)
    
    mj_model.vis.quality.offsamples = 8  
    
    # Ensure offscreen framebuffer is large enough for the requested resolution
    mj_model.vis.global_.offwidth = max(mj_model.vis.global_.offwidth, width)
    mj_model.vis.global_.offheight = max(mj_model.vis.global_.offheight, height)
    
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.5, 0.0, 0.6]  
    camera.distance = camera_distance
    
    camera.azimuth = azimuth
    camera.elevation = elevation
    
    frames = []
    print(f"Rendering {len(path)} frames (camera distance={camera_distance}m)...")
    
    for i, q in enumerate(path):
        mj_data.qpos[:7] = q
        if mj_model.nq > 7:
            mj_data.qpos[7:mj_model.nq] = 0.04
        mujoco.mj_forward(mj_model, mj_data)
        
        renderer.update_scene(mj_data, camera=camera)
        pixels = renderer.render()
        
        frames.append(pixels.copy())
        
        if (i + 1) % 10 == 0:
            print(f"  Rendered {i+1}/{len(path)} frames")
    
    print(f"Saving to {output_file}...")
    if output_file.endswith('.gif'):
        imageio.mimsave(output_file, frames, fps=fps, loop=0)
    elif output_file.endswith('.mp4'):
        imageio.mimsave(output_file, frames, fps=fps, codec='libx264')
    else:
        imageio.mimsave(output_file, frames, fps=fps, loop=0)
    
    print(f"✓ Saved {len(frames)} frames to {output_file}")
    print(f"  Camera: azimuth={camera.azimuth}°, elevation={camera.elevation}°, distance={camera.distance}m")
    renderer.close()


def render_trajectory_image(path_file, scene_xml, output_file, num_keyframes=8,
                            width=1920, height=1080, camera_distance=2.0,
                            azimuth=60, elevation=-20):
    """Render a static image showing ghost robot key-poses along the trajectory.

    Uses median-based background extraction: renders multiple poses, takes the
    pixel-wise median to recover a clean background, then overlays each keyframe's
    robot as a ghost with ascending opacity (faint→solid).
    """
    print(f"Loading trajectory from {path_file}...")
    path = np.load(path_file)
    if path.ndim != 2 or path.shape[1] != 7:
        raise ValueError(f"Expected path of shape (T, 7), got {path.shape}")
    print(f"Trajectory has {len(path)} waypoints")

    abs_xml = os.path.abspath(scene_xml)
    mj_model = mujoco.MjModel.from_xml_path(abs_xml)
    mj_data = mujoco.MjData(mj_model)
    mj_model.vis.quality.offsamples = 8
    mj_model.vis.global_.offwidth = max(mj_model.vis.global_.offwidth, width)
    mj_model.vis.global_.offheight = max(mj_model.vis.global_.offheight, height)

    renderer = mujoco.Renderer(mj_model, height=height, width=width)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.5, 0.0, 0.6]
    camera.distance = camera_distance
    camera.azimuth = azimuth
    camera.elevation = elevation

    # Pick evenly spaced key-frame indices across the ENTIRE trajectory
    indices = np.linspace(0, len(path) - 1, num_keyframes, dtype=int)
    key_configs = path[indices]

    # --- Build set of robot geom IDs (for segmentation mask) ---
    robot_geom_ids = set()
    for i in range(mj_model.ngeom):
        body_id = mj_model.geom_bodyid[i]
        body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name and body_name not in ('world',):
            robot_geom_ids.add(i)

    # --- Render all key frames (color + segmentation) ---
    print(f"Rendering {num_keyframes} ghost key-poses...")
    frames = []
    seg_masks = []
    for q in key_configs:
        mj_data.qpos[:7] = q
        if mj_model.nq > 7:
            mj_data.qpos[7:mj_model.nq] = 0.04
        mujoco.mj_forward(mj_model, mj_data)

        # Color render
        renderer.update_scene(mj_data, camera=camera)
        frames.append(renderer.render().copy())

        # Segmentation render for pixel-perfect robot mask
        renderer.enable_segmentation_rendering()
        renderer.update_scene(mj_data, camera=camera)
        seg = renderer.render().copy()
        renderer.disable_segmentation_rendering()

        # seg[:,:,0] = geom_id, seg[:,:,1] = geom_type
        geom_ids = seg[:, :, 0]
        robot_mask = np.isin(geom_ids, list(robot_geom_ids)).astype(np.float64)
        seg_masks.append(robot_mask)
    renderer.close()

    # --- Composite: last frame as solid base, earlier frames as ghosts ---
    base = frames[-1].astype(np.float64)
    composite = base.copy()
    ghost_alpha = 0.5

    # Overlay ghosts (all frames except the last)
    for frame_uint8, robot_mask in zip(frames[:-1], seg_masks[:-1]):
        frame = frame_uint8.astype(np.float64)
        mask_3d = robot_mask[:, :, np.newaxis]
        composite = composite * (1 - mask_3d * ghost_alpha) + frame * (mask_3d * ghost_alpha)

    # Re-stamp final pose robot pixels at full opacity
    final_mask_3d = seg_masks[-1][:, :, np.newaxis]
    composite = composite * (1 - final_mask_3d) + base * final_mask_3d

    composite = np.clip(composite, 0, 255).astype(np.uint8)

    img = Image.fromarray(composite)
    img.save(output_file, quality=95)
    print(f"✓ Saved ghost trajectory image to {output_file} ({width}×{height})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Franka trajectory as GIF/MP4/PNG")
    parser.add_argument("--scene", default="scenes/obs10.xml", help="Scene XML file")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second (default: 15)")
    parser.add_argument("--width", type=int, default=1280, help="Render width")
    parser.add_argument("--height", type=int, default=720, help="Render height")
    parser.add_argument("--distance", type=float, default=2.0, help="Camera distance from target (m)")
    parser.add_argument("--azimuth", type=float, default=60, help="Camera azimuth for far view")
    parser.add_argument("--elevation", type=float, default=-20, help="Camera elevation for far view")
    parser.add_argument("--num_frames", type=int, default=60, help="Total number of frames to render")
    parser.add_argument("--num_keyframes", type=int, default=8, help="Number of key poses for static image")
    parser.add_argument("--mode", choices=["gif", "image", "all"], default="all",
                        help="'gif' for animations, 'image' for static PNGs, 'all' for both")
    args = parser.parse_args()
    
    rrt_dir = "output/traj/rrt.npy"
    ndf_dir = "output/traj/ndf.npy"
    nnsdf_dir = "output/traj/sdf.npy"

    # ---------- GIF outputs ----------
    if args.mode in ("gif", "all"):
        rrt_output_far = "output/traj/rrt_far.gif"
        rrt_output_front = "output/traj/rrt_front.gif"
        ndf_output_far = "output/traj/ndf_far.gif"
        ndf_output_front = "output/traj/ndf_front.gif"
        nnsdf_output_far = "output/traj/sdf_far.gif"
        nnsdf_output_front = "output/traj/sdf_front.gif"

        print("="*60)
        print(f"Generating FAR view ({args.azimuth}°, {args.elevation}°)...")
        print("="*60)
        render_trajectory_gif(rrt_dir, args.scene, rrt_output_far, 
                             num_frames=args.num_frames, fps=args.fps, width=args.width, height=args.height,
                             camera_distance=args.distance, azimuth=args.azimuth, elevation=args.elevation)
        render_trajectory_gif(ndf_dir, args.scene, ndf_output_far, 
                             num_frames=args.num_frames, fps=args.fps, width=args.width, height=args.height,
                             camera_distance=args.distance, azimuth=args.azimuth, elevation=args.elevation)
        render_trajectory_gif(nnsdf_dir, args.scene, nnsdf_output_far, 
                             num_frames=args.num_frames, fps=args.fps, width=args.width, height=args.height,
                             camera_distance=args.distance, azimuth=args.azimuth, elevation=args.elevation)
        
        print("\n" + "="*60)
        print("Generating FRONT view (0°, -20°)...")
        print("="*60)
        render_trajectory_gif(rrt_dir, args.scene, rrt_output_front, 
                             num_frames=args.num_frames, fps=args.fps, width=args.width, height=args.height,
                             camera_distance=args.distance, azimuth=0, elevation=-20)
        render_trajectory_gif(ndf_dir, args.scene, ndf_output_front, 
                             num_frames=args.num_frames, fps=args.fps, width=args.width, height=args.height,
                             camera_distance=args.distance, azimuth=0, elevation=-20)
        render_trajectory_gif(nnsdf_dir, args.scene, nnsdf_output_front, 
                             num_frames=args.num_frames, fps=args.fps, width=args.width, height=args.height,
                             camera_distance=args.distance, azimuth=0, elevation=-20)

    # ---------- Static image outputs (far view only) ----------
    if args.mode in ("image", "all"):
        print("\n" + "="*60)
        print(f"Generating STATIC trajectory images ({args.num_keyframes} key poses, far view)...")
        print("="*60)
        for name, traj_file in [("rrt", rrt_dir), ("ndf", ndf_dir), ("sdf", nnsdf_dir)]:
            out_png = f"output/traj/{name}.png"
            print(f"\n--- {name.upper()} ---")
            render_trajectory_image(
                traj_file, args.scene, out_png,
                num_keyframes=args.num_keyframes,
                width=args.width, height=args.height,
                camera_distance=args.distance, azimuth=args.azimuth, elevation=args.elevation,
            )

