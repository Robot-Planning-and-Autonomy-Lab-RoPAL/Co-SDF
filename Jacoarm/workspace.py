import os
import mujoco
import mujoco.viewer
import numpy as np
import time
from jaco_arm.jaco import JacoEnv

if __name__ == "__main__":

    env = JacoEnv(width=800, height=800, model_path="jaco_3dof.xml")
    
    # Set the configuration to the one in question
    q_test = [-2, 1, -2]
    env.data.qpos[:3] = q_test
    mujoco.mj_forward(env.model, env.data)

    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            while viewer.is_running():
                step_start = time.time()
                
                mujoco.mj_step(env.model, env.data)
                
                viewer.sync()
        
                time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    except Exception as e:
        print(f"Error during viewer execution: {e}")

