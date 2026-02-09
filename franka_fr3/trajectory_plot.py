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

def render_trajectory_gif(path_file, scene_xml, output_file, fps=30, width=640, height=480, 
                          camera_distance=2.5, azimuth=135, elevation=-20):
    """
    Render a trajectory as GIF using MuJoCo.
    
    Args:
        path_file: Path to .npy file containing trajectory (T, 7)
        scene_xml: Path to MuJoCo scene XML
        output_file: Output GIF/MP4 file path
        fps: Frames per second
        width: Render width
        height: Render height
        camera_distance: Distance from target (meters)
        azimuth: Camera azimuth angle (0=back, 90=side, 180=front, 270=left)
        elevation: Camera elevation angle (negative=from above)
    """
    print(f"Loading trajectory from {path_file}...")
    path = np.load(path_file)
    if path.ndim != 2 or path.shape[1] != 7:
        raise ValueError(f"Expected path of shape (T, 7), got {path.shape}")
    
    print(f"Trajectory has {len(path)} waypoints")
    
    print(f"Loading scene from {scene_xml}...")
    abs_xml = os.path.abspath(scene_xml)
    mj_model = mujoco.MjModel.from_xml_path(abs_xml)
    mj_data = mujoco.MjData(mj_model)
    
    # Create renderer
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    
    # Set up camera (looking at workspace center from side-front angle)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.5, 0.0, 0.6]  # Look at workplace center
    camera.distance = camera_distance
    
    # Set view angle based on parameters
    camera.azimuth = azimuth
    camera.elevation = elevation
    
    frames = []
    print(f"Rendering {len(path)} frames (camera distance={camera_distance}m)...")
    
    for i, q in enumerate(path):
        mj_data.qpos[:7] = q
        # Set fingers to open (0.04)
        if mj_model.nq > 7:
            mj_data.qpos[7:mj_model.nq] = 0.04
        mujoco.mj_forward(mj_model, mj_data)
        
        # Render frame
        renderer.update_scene(mj_data, camera=camera)
        pixels = renderer.render()
        
        frames.append(pixels.copy())
        
        if (i + 1) % 10 == 0:
            print(f"  Rendered {i+1}/{len(path)} frames")
    
    # Save as GIF or MP4
    print(f"Saving to {output_file}...")
    if output_file.endswith('.gif'):
        imageio.mimsave(output_file, frames, fps=fps, loop=0)
    elif output_file.endswith('.mp4'):
        imageio.mimsave(output_file, frames, fps=fps, codec='libx264')
    else:
        # Default to GIF
        imageio.mimsave(output_file, frames, fps=fps, loop=0)
    
    print(f"✓ Saved {len(frames)} frames to {output_file}")
    print(f"  Camera: azimuth={camera.azimuth}°, elevation={camera.elevation}°, distance={camera.distance}m")
    renderer.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Franka trajectory as GIF/MP4")
    parser.add_argument("--scene", default="scenes/obs20.xml", help="Scene XML file")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 20)")
    parser.add_argument("--width", type=int, default=640, help="Render width (max 640)")
    parser.add_argument("--height", type=int, default=480, help="Render height")
    parser.add_argument("--distance", type=float, default=2.0, help="Camera distance from target (m)")
    parser.add_argument("--azimuth", type=float, default=60, help="Camera azimuth for far view")
    parser.add_argument("--elevation", type=float, default=-20, help="Camera elevation for far view")
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
                         fps=args.fps, width=args.width, height=args.height,
                         camera_distance=args.distance, azimuth=args.azimuth, elevation=args.elevation)
    render_trajectory_gif(ndf_dir, args.scene, ndf_output_far, 
                         fps=args.fps, width=args.width, height=args.height,
                         camera_distance=args.distance, azimuth=args.azimuth, elevation=args.elevation)
    render_trajectory_gif(sdf_dir, args.scene, sdf_output_far, 
                         fps=args.fps, width=args.width, height=args.height,
                         camera_distance=args.distance, azimuth=args.azimuth, elevation=args.elevation)
    
    # Generate front view (fixed for reference)
    print("\n" + "="*60)
    print("Generating FRONT view (0°, -20°)...")
    print("="*60)
    render_trajectory_gif(rrt_dir, args.scene, rrt_output_front, 
                         fps=args.fps, width=args.width, height=args.height,
                         camera_distance=args.distance, azimuth=0, elevation=-20)
    render_trajectory_gif(ndf_dir, args.scene, ndf_output_front, 
                         fps=args.fps, width=args.width, height=args.height,
                         camera_distance=args.distance, azimuth=0, elevation=-20)
    render_trajectory_gif(sdf_dir, args.scene, sdf_output_front, 
                         fps=args.fps, width=args.width, height=args.height,
                         camera_distance=args.distance, azimuth=0, elevation=-20)
    
    print("\n" + "="*60)
    print("✓ Both views generated successfully!")
    print(f"  Far view:   {rrt_output_far}")
    print(f"  Front view: {rrt_output_front}")
    print("="*60)
