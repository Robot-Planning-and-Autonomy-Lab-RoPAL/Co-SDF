#!/usr/bin/env python3
import os
import mujoco
import numpy as np
from PIL import Image

def render_top_view(scene_xml, output_image, width=640, height=480):
    """
    Render a top-down view of a MuJoCo scene.
    """
    print(f"Rendering {scene_xml} to {output_image}...")
    abs_xml = os.path.abspath(scene_xml)
    if not os.path.exists(abs_xml):
        print(f"Error: {abs_xml} not found.")
        return

    mj_model = mujoco.MjModel.from_xml_path(abs_xml)
    mj_data = mujoco.MjData(mj_model)

    # Set robot to a neutral position if it exists
    # For Franka, let's just use the default qpos for now.
    mujoco.mj_forward(mj_model, mj_data)

    # Create renderer
    renderer = mujoco.Renderer(mj_model, height=height, width=width)

    # Set up camera for top-down view
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.4, 0.0, 0.0]  # Look at the table area
    camera.distance = 2.0
    camera.azimuth = 90
    camera.elevation = -90

    # Render frame
    renderer.update_scene(mj_data, camera=camera)
    pixels = renderer.render()

    # Save as image
    img = Image.fromarray(pixels)
    img.save(output_image)
    print(f"✓ Saved to {output_image}")
    renderer.close()

if __name__ == "__main__":
    scenes = {
        "1": "scenes/table_obs1.xml",
        "2": "scenes/table_obstacles_2.xml",
        "5": "scenes/table_obstacles_5.xml",
        "10": "scenes/table_obstacles_10.xml",
        "20": "scenes/table_obstacles_20.xml"
    }

    output_dir = "output/scenes_topview"
    os.makedirs(output_dir, exist_ok=True)

    for name, xml_path in scenes.items():
        output_path = os.path.join(output_dir, f"table_obs_{name}_topview.png")
        render_top_view(xml_path, output_path)
