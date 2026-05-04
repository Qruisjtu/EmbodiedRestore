import robosuite as suite
import numpy as np
import time
from robosuite.controllers import load_composite_controller_config

controller_config=load_composite_controller_config("./robosuite/controllers/config/robots/default_ur5e.json")
Control_freq=20
HORIZON=500
DUMMY_ACTION=[1.0, 0.0, 0.0, 0.5, 0.0, 0.0,0.0]
env = suite.make(
        "LiftTest",
        robots="UR5e",   
        gripper_types="default",  
        controller_configs=controller_config,    
        has_renderer=False,  
        render_camera="frontview",
        camera_names=("agentview","robot0_eye_in_hand","frontview"),
        control_freq=Control_freq,
        has_offscreen_renderer=True, 
        use_camera_obs=True,  
        camera_widths=256, 
        camera_heights=256,
        horizon=HORIZON, 
        ignore_done=False,  
    )

obs = env.reset()
t=0
while t<50:
    obs, reward, done, info = env.step(DUMMY_ACTION)
    t+=1
    time.sleep(0.1)

env.close()