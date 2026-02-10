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
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Franka trajectory as GIF/MP4")
    parser.add_argument("--scene", default="scenes/obs20.xml", help="Scene XML file")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second (default: 15)")
    parser.add_argument("--width", type=int, default=1280, help="Render width")
    parser.add_argument("--height", type=int, default=720, help="Render height")
    parser.add_argument("--distance", type=float, default=2.0, help="Camera distance from target (m)")
    parser.add_argument("--azimuth", type=float, default=60, help="Camera azimuth for far view")
    parser.add_argument("--elevation", type=float, default=-20, help="Camera elevation for far view")
    parser.add_argument("--num_frames", type=int, default=30, help="Total number of frames to render")
    args = parser.parse_args()
    
    rrt_dir = "output/traj/rrt.npy"
    ndf_dir = "output/traj/ndf.npy"
    sdf_dir = "output/traj/sdf.npy"

    rrt_output_far = "output/traj/rrt_far.gif"
    rrt_output_front = "output/traj/rrt_front.gif"
    ndf_output_far = "output/traj/ndf_far.gif"
    ndf_output_front = "output/traj/ndf_front.gif"
    sdf_output_far = "output/traj/sdf_far.gif"
    sdf_output_front = "output/traj/sdf_front.gif"

    print("="*60)
    print(f"Generating FAR view ({args.azimuth}°, {args.elevation}°)...")
    print("="*60)
    render_trajectory_gif(rrt_dir, args.scene, rrt_output_far, 
                         num_frames=args.num_frames, fps=args.fps, width=args.width, height=args.height,
                         camera_distance=args.distance, azimuth=args.azimuth, elevation=args.elevation)
    render_trajectory_gif(ndf_dir, args.scene, ndf_output_far, 
                         num_frames=args.num_frames, fps=args.fps, width=args.width, height=args.height,
                         camera_distance=args.distance, azimuth=args.azimuth, elevation=args.elevation)
    render_trajectory_gif(sdf_dir, args.scene, sdf_output_far, 
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
    render_trajectory_gif(sdf_dir, args.scene, sdf_output_front, 
                         num_frames=args.num_frames, fps=args.fps, width=args.width, height=args.height,
                         camera_distance=args.distance, azimuth=0, elevation=-20)
    
    print("\n" + "="*60)
    print("✓ Both views generated successfully!")
    print(f"  Far view:   {rrt_output_far}")
    print(f"  Front view: {rrt_output_front}")
    print("="*60)
