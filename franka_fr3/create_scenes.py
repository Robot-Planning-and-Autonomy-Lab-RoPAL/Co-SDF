import mujoco
import numpy as np
import os
import random

def get_base_xml():
    return """<mujoco model="fr3_clutter_scene">
  <compiler meshdir="../assets"/>
  <include file="../models/fr3.xml"/>

  <statistic center="0.2 0 0.4" extent=".8"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="obstacle_mat" rgba="0.9 0.1 0.1 1" reflectance="0.5"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" pos="0 0 0" material="groundplane"/>

    <!-- RANDOM CLUTTER -->
    {OBSTACLES}

  </worldbody>
</mujoco>
"""

def generate_random_obstacle(idx):
    # Primitives: box, sphere, cylinder
    g_type = random.choice(["box", "sphere", "cylinder"])
    
    # Random Size (0.05 to 0.1)
    size_val = random.uniform(0.04, 0.08)
    if g_type == "box":
        size = f"{size_val} {size_val} {size_val}"
    elif g_type == "cylinder":
        size = f"{size_val} {size_val}" # radius half-height
    else: # sphere
        size = f"{size_val}"
        
    # Random Position in Shell: r in [0.4, 0.9], z in [0.1, 0.8]
    r = random.uniform(0.4, 0.9)
    theta = random.uniform(0, 2 * np.pi)
    z = random.uniform(0.1, 0.8)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Random Yaw/Quat
    # Keep it simple: random axis-angle or just pos is fine for spheres
    # but for box/cylinder orientation matters.
    # Let's generate a random quat
    u, v, w = random.random(), random.random(), random.random()
    quat = f"{1.0} 0 0 0" # Default identity for now to simplify, or random
    # Let's do random Euler for simplicity of string format (MuJoCo supports euler)
    # But standard geometry definitions use pos/quat usually.
    # Random quat:
    q = np.random.randn(4)
    q /= np.linalg.norm(q)
    quat = f"{q[0]:.3f} {q[1]:.3f} {q[2]:.3f} {q[3]:.3f}"
    
    # Random Color
    rgb = np.random.rand(3)
    rgba = f"{rgb[0]:.2f} {rgb[1]:.2f} {rgb[2]:.2f} 1.0"
    
    return f'<geom name="obs_{idx}" type="{g_type}" size="{size}" pos="{x:.3f} {y:.3f} {z:.3f}" quat="{quat}" rgba="{rgba}"/>'

def is_valid_scene(xml_content, temp_filename="temp_gen_check.xml"):
    try:
        # Adjustment for validation at root dir:
        # Convert "../models" -> "models", "../assets" -> "assets"
        # This allows us to validate using a temp file in the root dir (CWD)
        # preventing complex relative path issues with ".."
        xml_val = xml_content.replace('="../', '="')
        
        with open(temp_filename, "w") as f:
            f.write(xml_val)
            
        model = mujoco.MjModel.from_xml_path(temp_filename)
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data) # Step once to resolve contacts
        
        # Check contacts
        for i in range(data.ncon):
            contact = data.contact[i]
            g1 = contact.geom1
            g2 = contact.geom2
            name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1)
            name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)
            
            is_obs1 = name1 and name1.startswith("obs_")
            is_obs2 = name2 and name2.startswith("obs_")
            is_floor1 = name1 == "floor"
            is_floor2 = name2 == "floor"
            
            # Disallow Robot-Obstacle collision
            if (is_obs1 and (not is_obs2 and not is_floor2)) or \
               (is_obs2 and (not is_obs1 and not is_floor1)):
               return False
               
        return True
    except Exception as e:
        print(f"Validation error: {e}")
        return False

def generate_and_save(n_obs, filename):
    print(f"Generating {filename} with {n_obs} obstacles...")
    max_retries = 1000
    temp_file = "temp_gen_check.xml" # Root dir
    
    for i in range(max_retries):
        obstacles_str = "\n    ".join([generate_random_obstacle(k) for k in range(n_obs)])
        full_xml = get_base_xml().replace("{OBSTACLES}", obstacles_str)
        
        if is_valid_scene(full_xml, temp_file):
            # If valid, just overwrite the temp file to the final filename
            with open(filename, "w") as f:
                f.write(full_xml)
            if os.path.exists(temp_file): os.remove(temp_file)
            print(f"Success! Saved to {filename}")
            return
    
    if os.path.exists(temp_file): os.remove(temp_file) 
    print(f"Failed to generate valid scene after {max_retries} retries.")

if __name__ == "__main__":
    os.makedirs("scenes", exist_ok=True)
    generate_and_save(30, "scenes/obs30.xml")
