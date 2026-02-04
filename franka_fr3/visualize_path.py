"""
Visualize a path as animated GIF
"""
import numpy as np
import mujoco
from PIL import Image, ImageDraw
import os

def render_path_animation(path, scene_xml, output_gif="path_demo.gif", fps=30):
    """Render path as GIF."""
    
    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)
    
    # Renderer
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    frames = []
    
    # Interpolate path for smooth animation
    n_frames_total = len(path) * 3  # 3 frames per waypoint
    t = np.linspace(0, 1, n_frames_total)
    path_indices = np.linspace(0, len(path)-1, n_frames_total)
    
    for frame_idx, path_idx in enumerate(path_indices):
        idx_low = int(np.floor(path_idx))
        idx_high = min(int(np.ceil(path_idx)), len(path)-1)
        alpha = path_idx - idx_low
        
        # Interpolate configuration
        q = path[idx_low] * (1 - alpha) + path[idx_high] * alpha
        
        # Set state
        data.qpos[:7] = q
        mujoco.mj_forward(model, data)
        
        # Render
        renderer.update_scene(data) # Use default camera
        pixels = renderer.render()
        
        # Add progress bar
        img = Image.fromarray(pixels)
        draw = ImageDraw.Draw(img)
        
        # Progress bar
        bar_width = 600
        bar_height = 20
        bar_x = 20
        bar_y = 440
        
        # Background
        draw.rectangle([bar_x, bar_y, bar_x + bar_width, bar_y + bar_height], 
                      fill=(50, 50, 50), outline=(255, 255, 255))
        
        # Progress
        progress = frame_idx / n_frames_total
        draw.rectangle([bar_x, bar_y, bar_x + int(bar_width * progress), bar_y + bar_height],
                      fill=(0, 200, 0))
        
        # Text
        text = f"Waypoint {int(path_idx)+1}/{len(path)}"
        draw.text((bar_x + 10, bar_y + 2), text, fill=(255, 255, 255))
        
        frames.append(np.array(img))
    
    # Save as GIF using PIL
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        output_gif,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000/fps),  # milliseconds per frame
        loop=0
    )
    print(f"Saved animation to {output_gif}")
    
    return output_gif


if __name__ == "__main__":
    # Load saved path
    path = np.load("demo_path.npy")
    print(f"Visualizing path with {len(path)} waypoints")
    
    render_path_animation(
        path, 
        "scenes/table_obs20.xml",
        output_gif="output/path_demo.gif",
        fps=30
    )
