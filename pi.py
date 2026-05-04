import time
import collections
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.input_utils import *
from robosuite.wrappers import VisualizationWrapper
import pathlib
import numpy as np
import cv2 as cv
import tqdm,logging
import json
import random
import torch
import os
import multiprocessing as mp
from PIL import Image
from matplotlib import pyplot as plt
import imageio
from openpi_client import websocket_client_policy as _websocket_client_policy
from logs.logger import setup_logger
from pathlib import Path
# from compress import compressimg
# from compress.compressimg import MODELLIST,MODELQUADOWN

#====BENCHMARK PARAMETERS====
#Check before each run
SEED  = 42
AGENT = "Pi0_300MixLift" #Agent's name used for metadata
BENCHMARKINFO = "pi0_RealESRGAN"   #output benchmark's folder name
PROMPT="Pick up the object" #Task prompt for agent
HORIZON=250              # Maximum number of steps per episode
SAVEVIDEO=False           #save video after each episode
ISDISPLAY = False           #show Mujoco render while benchmark


do_baseline = True       #run benchmark without compress
num_steps_wait=50        #step to wait before each episode
Control_freq=10          #control freqency for robosuite
episodes_times=100       #episode times for one task
DUMMY_ACTION=[0,0,0,0,0,0,-1]
#============================

# result_save_path = pathlib.Path(f"data/benchmark/{BENCHMARKINFO}")
# muted for mp
controller_config = load_composite_controller_config("robosuite/controllers/config/robots/default_ur5e.json")
logging.basicConfig(level=logging.INFO)
seed_chain = [SEED+i for i in range(episodes_times)]


def set_seed_everywhere(seed: int) -> None:
    """
    Set random seed for all random number generators for reproducibility.

    Args:
        seed: The random seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def build_obs_ur5(obs,cpr,step:int,iscpr=False,prompt:str=PROMPT) :
    """
    convert observation from mujoco to pi's config format
    You can modify or compress the image from camera here

    :param obs: obervation from mujoco
    :param cpr: compressimg instance for image compression
    :param step: current step of this ep,onlty save first frame image
    :type step: int
    :param iscpr: compress or not
    :param prompt: task's text prompt
    :type prompt: str
    """
    state  = np.concatenate((obs["robot0_joint_pos"],[obs["robot0_gripper_qpos"][0]]),axis=0)
    agent_raw = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).astype(np.uint8)
    wrist_raw = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]).astype(np.uint8)
    # agent_raw = obs["agentview_image"][::-1, ::-1].astype(np.uint8)
    # wrist_raw = obs["robot0_eye_in_hand_image"][::-1, ::-1].astype(np.uint8)

    if iscpr: 
        agent_cpr = cpr.compress(agent_raw, saveimg=False)[0]
        wrist_cpr = cpr.compress(wrist_raw, saveimg=False)[0]
        obs_new = {
            "base_rgb": agent_cpr,   
            "wrist_rgb":wrist_cpr,    
            "state":     state,
            "prompt":   prompt,     
        }
        first_frame = [agent_cpr, wrist_cpr, agent_raw, wrist_raw] if step == 0 else None
    else:
        obs_new = {
            "base_rgb": agent_raw,  
            "wrist_rgb":wrist_raw,  
            "state":     state,
            "prompt":   prompt, 
        }
        first_frame = None
    return obs_new , first_frame

def run_task(model:str,cpr,policy,savepath,iscpr=False):
    """
    run single task
    """
    task_successes = 0
    env_metadata=[]
    env = suite.make(
        "LiftTest",
        robots="UR5e",  
        gripper_types="default", 
        controller_configs=controller_config,  
        has_renderer=ISDISPLAY, 
        render_camera="frontview",
        camera_names=("agentview","robot0_eye_in_hand"),#frontview,agentview,robot0_eye_in_hand
        control_freq=Control_freq,
        has_offscreen_renderer=True,  
        use_camera_obs=True, 
        camera_widths=256,
        camera_heights=256,
        horizon=HORIZON+num_steps_wait,  
        ignore_done=False,  
    )
    
    for episode in tqdm.tqdm(range(episodes_times)):
        for attempt in range(5):
            try:
                set_seed_everywhere(seed = seed_chain[episode])
                step = 0
                replay_images = []
                obs = env.reset()
                done = False
                action_plan = collections.deque()
                logging.info(f"Episode {episode+1}/{episodes_times},PROMPT:{PROMPT},Env:{env.env_metadata()}")
                t=0
                if t < num_steps_wait:
                    obs, reward, done, info = env.step(DUMMY_ACTION)
                    # env.render()
                    t += 1
                
                replan_steps=10
                rotate_scale=1.0
                pos_scale=1.0
                
                while step < HORIZON:
                    if SAVEVIDEO:
                        replay_images.append(obs["frontview_image"][::-1, ::-1])
                    step_strat_time = time.time()
                    if not action_plan :
                        new_obs, image_for_first_frame = build_obs_ur5(obs,cpr,step,iscpr=iscpr)
                        step_mid_time = time.time()
                        action_chunk = policy.infer(new_obs)["actions"]
                        if all(x==y for x,y in zip(action_chunk[0],action_chunk[1])):
                            replan_steps=1
                            pos_scale=2.0
                            rotate_scale=0.01
                            # logging.warning("same chunk,set step=1")
                        else:
                            replan_steps=10
                            pos_scale=1.0
                            rotate_scale=1.0   
                        action_plan.extend(action_chunk[:replan_steps])
                    action = action_plan.popleft()[:].copy()
                    action[0:3]=action[0:3]*pos_scale
                    action[3:6]=action[3:6]*rotate_scale
                    # logging.info(f"Step {step+1}/{HORIZON}, Action: {action}")
                    obs, _, _, _ = env.step(action) # play action
                    done=env._check_success() #use env._check_success() to check success directly
                    if step == 0 and iscpr:
                        for i,image in enumerate(image_for_first_frame):
                            Image.fromarray(image).save(f"{savepath}/firstframe_{model}_eps{episode+1}_{i}.png") 
                    step_end_time = time.time()
                    if step_mid_time is not None:
                        logging.info(f"Step {step+1}/{HORIZON}, Image time: {(step_mid_time - step_strat_time):.2f}s, Policy inference time: {(step_end_time - step_mid_time):.2f}s,model {model}")
                        step_mid_time = None
                    step+=1
                    if done:
                        task_successes += 1
                        break
                suffix = "success" if done else "failure"
                task_segment = PROMPT.replace(" ", "_")
                if SAVEVIDEO:
                    imageio.mimwrite(
                        pathlib.Path(savepath) / f"rollout_{env.env_metadata()['object']}_{model}_eps{episode}_{suffix}.mp4",
                        [np.asarray(x) for x in replay_images],
                        fps=Control_freq,
                        )
                env_metadata.append(env.env_metadata())
                env_metadata[-1].update({
                    "success": suffix,
                    "steps": step,
                })
                break 
            except Exception as err:
                logging.error(f"[Episode {episode+1}] Attempt {attempt+1}/{5} FAILED")
                logging.error(repr(err))
                if attempt == 5:
                    logging.error(f"[Episode {episode+1}] Gave up after max retries")
                    env_metadata.append(env.env_metadata())
                    env_metadata[-1].update({
                    "success": "failure",
                    "steps": 0,
                    })
                    break
        env.close()
        # env.render()
        # time.sleep(1)
        logging.info(f"Success: {done}")
        logging.info(f"{model} completed so far: {episode+1}")
        logging.info(f"# successes: {task_successes} ({task_successes / (episode+1) * 100:.1f}%)")

        # Log final results
    logging.info(f"Current task success rate: {float(task_successes) / float(episodes_times)}")
    return float(task_successes) / float(episodes_times),env_metadata
"""
Main code
"""

def main():
    seed_chain = [SEED+i for i in range(episodes_times)]
    start_time = time.time()
    pathlib.Path(result_save_path).mkdir(parents=True, exist_ok=True)
    policy = _websocket_client_policy.WebsocketClientPolicy(
        host="0.0.0.0",
        port=8000,
        api_key=None,
    )
    # controller_config=load_composite_controller_config("BASIC")
    controller_config=load_composite_controller_config("robosuite/controllers/config/robots/default_ur5e.json")
    # env.make
    plt.figure(figsize=(9, 6))
    plt_color_controler = 0
    ax = plt.gca()
    ax.set_title("BPP & Success Rate")
    ax.set_xlabel("Average BPP (bits per pixel)")
    ax.set_ylabel("Success Rate")
    ax.grid(True, linestyle=":", linewidth=0.8)
    cpr=compressimg.COMPRESSIMG(model="distort", dis_type=0)
    if do_baseline:
        baseline_success_rate,baseline_metadata=run_task(model = "baseline",cpr=cpr,policy=policy,iscpr=False)

        ax.axhline(
            y=baseline_success_rate, linestyle="--", linewidth=1.5,
            label=f"Baseline (no compression) = {baseline_success_rate:.2f}"
        ) 

        with open(pathlib.Path(result_save_path) / "result_default.json", "w") as f:
            json.dump({
                "model": "default",
                "quality": "default",
                "epsisodes": episodes_times,
                "success_rate": baseline_success_rate,
                "bpp": 0,  # Placeholder for bpp, as we are not using compression here
                "env_metadata" : baseline_metadata
            }, f, indent=4)
    else:
        baseline_success_rate = 0.0
    

    summary = {
        "meta": {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "episodes_per_setting": episodes_times,
            "Agent": AGENT,
            "prompt": PROMPT,
            "horizon": HORIZON,
            "control_freq": Control_freq,
        },
        "baseline": {
            "success_rate": baseline_success_rate,
            "bpp": 0.0,
        },
        "results": []
    }
    for n,model in enumerate(MODELLIST):
        logging.info(f"Running model {model} ({n+1}/{len(MODELLIST)})")
        bpplist = []
        ratelist = []
        model_save_path = pathlib.Path(result_save_path) / model
        model_save_path.mkdir(parents=True, exist_ok=True)
        for m,[q,d] in enumerate(MODELQUADOWN[model]):
            # for d in MODELQUADOWN[model][1]:
            logging.info(f"Running model {model} with quality {q} and downsample {d} ({m+1}/{len(MODELQUADOWN[model])})")
            d_suffix = d.replace("/","-")
            run_save_path = pathlib.Path(model_save_path) / f"quality{q}_down{d_suffix}"
            run_save_path.mkdir(parents=True, exist_ok=True)
            cpr.changenet(model,q,d)
            success_rate,env_metadata=run_task(model=model+"_"+str(q)+"_"+str(d_suffix),savepath=run_save_path,iscpr=True)
            avg_bpp=float(sum(cpr.bpp)/len(cpr.bpp))

            bpplist.append(avg_bpp)
            ratelist.append(success_rate)
        
            with open(run_save_path / f"result_{model}_{q}_{d_suffix}.json", "w") as f:
                json.dump({
                    "model": model,
                    "quality": q,
                    "downsample":d,
                    "epsisodes": episodes_times,
                    "success_rate": success_rate,
                    "bpp": avg_bpp,
                    "env_metadata": env_metadata,
                }, f, indent=4)
        if bpplist:
            idx = np.argsort(np.array(bpplist))
            x_sorted = [bpplist[i] for i in idx]
            y_sorted = [ratelist[i] for i in idx]
            summary["results"].append({
                "model": model,
                "quality&downsample": MODELQUADOWN[model],
                "success_rate": y_sorted,
                "bpp": x_sorted
            })
            if plt_color_controler < 10:
                ax.plot(x_sorted, y_sorted, marker="o", linewidth=1.8, label=model)
            else:
                ax.plot(x_sorted, y_sorted, marker="x", linewidth=1.8, label=model)
            plt_color_controler+=1
            ax.legend(loc="best", frameon=True)
    fig_path = pathlib.Path(result_save_path) / "bpp_successrate.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    summary_path = pathlib.Path(result_save_path) / f"summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    end_time = time.time()
    logging.info(f"Figure saved to: {fig_path}")
    logging.info(f"Summary JSON saved to: {summary_path}")
    logging.info(f"Done. Time consumed in total:{((end_time-start_time)/60):.1f} minutes")
    
#below code is for multiprocessing in GPU pod with noise and enhancement
def noise_thread(ntype:int,etype:int=0):
    # this function is used for multiprocessing, each process runs one noise type, including baseline(no noise)
    start_time = time.time()
    # cpr = compressimg.COMPRESSIMG(model="distort")
    from compress import compressimg
    # from compress.compressimg import (
    #     Distorter,
    #     InstantIREnhance,
    #     SwinIREnhance,
    #     RealESRGANEnhance,
    #     FoundIREnhance,
    #     DFPIREnhance,
    # )
    # Enhance = [InstantIREnhance, SwinIREnhance, RealESRGANEnhance, FoundIREnhance, DFPIREnhance][etype]

    from compress.compressimg import (
        Distorter,
        VarFormerEnhance,
        HypirEnhance,
    )
    Enhance = [VarFormerEnhance,HypirEnhance][etype]



    #import here for multiprocessing, avoid potential issues with CUDA initialization in child processes
    result_save_path = Path(f"data/benchmark/Pi05_{Enhance.__name__}")
    policy = _websocket_client_policy.WebsocketClientPolicy(
        host="0.0.0.0",
        port=8000,
        api_key=None,
    )
    if ntype == -1:
        logging.info(f"Running baseline")

        cpr = compressimg.Pipeline([
            # Distorter(quality=1, type=1),
            Enhance(quality=4),
        ])
        model_name = "baseline"
        base_path = result_save_path / model_name
        base_path.mkdir(parents=True, exist_ok=True)

        sr,meta = run_task(
            model=model_name,
            cpr=cpr,
            policy=policy,
            iscpr=True,
            savepath=base_path,
        )
        # share_queue.put({"type": "baseline", "sr": sr})
    else:
        logging.info(f"Running noise type {ntype}")
        cpr = compressimg.Pipeline([
        Distorter(quality=1, type=ntype),
        Enhance(quality=4), 
        ])
        dis_name = f"distort_{ntype}"
        model_name = f"distort_{ntype}&ehance_{Enhance.__name__}"
        base_path = result_save_path / dis_name
        base_path.mkdir(parents=True, exist_ok=True)
        # cpr.changenet(model="distort", dis_type=ntype)
        sr,meta = run_task(
            model=model_name,
            cpr=cpr,
            policy=policy,
            iscpr=True,
            savepath=base_path,
        )
        

    with open(base_path / f"result.json", "w") as f:
        json.dump({
            "model": model_name,
            # "quality": "demo",
            # "episodes": episodes_times,
            "success_rate": sr,
            # "bpp": avg_bpp,
            "env_metadata": meta,
        }, f, indent=4)
    end_time = time.time()
    logging.info(f"---------Noise type {ntype}&Ehance {etype} done done. Total time: {((end_time - start_time) / 60):.1f} minutes---------")
    return {"id":etype*100+ntype, "model": model_name, "sr": sr}

def init_worker():
    # initial for Pool
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    torch.cuda.set_device(0)  
def thread_manager(fulllist: list, proc_per: int = 8):
    "main process for manage multiprocessing, assign gpu for each process"
    logger = setup_logger()
    num_gpus = torch.cuda.device_count()
    logger.info(f"Starting: {num_gpus} GPUs, {proc_per} processes per GPU")
    
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if parent_cvd is not None:
        available_gpus = parent_cvd.split(",")
    else:
        available_gpus = [str(i) for i in range(num_gpus)]
        
    results = []
    pools = [] 
    async_results = []
    chunks = np.array_split(fulllist, num_gpus)

    try:
        for gpu_id in range(num_gpus):
            tasks_for_gpu = list(chunks[gpu_id])
            if len(tasks_for_gpu) == 0:
                continue
            
            os.environ["CUDA_VISIBLE_DEVICES"] = available_gpus[gpu_id]
            
            pool = mp.Pool(
                processes=proc_per,
                initializer=init_worker,  
            )
            logger.info(f"Assigned physical GPU {available_gpus[gpu_id]} to {len(tasks_for_gpu)} tasks")
    
            for task in tasks_for_gpu:
                async_results.append(pool.apply_async(noise_thread, args=(*task,)))
            
            pools.append(pool)

        if parent_cvd is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = parent_cvd
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        logger.info("All tasks submitted, waiting for results...")

        for async_res in async_results:
            result = async_res.get() 
            results.append(result)
            logger.info(f"Got result: id={result.get('id')}, sr={result.get('sr')}")

        for pool in pools:
            pool.close()
        for pool in pools:
            pool.join()
            
    except Exception as e:
        logger.error(f"Error during multiprocessing: {e}")
        for pool in pools:
            pool.terminate()
        raise e
    
    logger.info(f"All tasks completed. Collected {len(results)} results.")
    return results


if __name__ == "__main__":

    mp.set_start_method('spawn')
    task_list = [(ntype, etype) for etype in [1] for ntype in range(-1, 25) ]
    logging.info(task_list)
    thread_manager(task_list, proc_per=1)