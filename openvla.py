import time
import collections
import json
import tqdm
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple, List
import torch
import numpy as np
import math
import imageio
from PIL import Image
import random
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
import sys
sys.path.append("vla/openvla_oft")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import multiprocessing as mp
from logs.logger import setup_logger
# from compress.compressimg import MODELLIST,MODELQUADOWN
# OpenVLA-OFT 依赖
from vla.openvla_oft.experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
)
from vla.openvla_oft.experiments.robot.robot_utils import (
    get_action,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
)
from vla.openvla_oft.prismatic.vla.constants import NUM_ACTIONS_CHUNK
#COMPRESSIMG


#====BENCHMARK PARAMETERS====
#Check before each run
SEED: int = 42                                 
AGENT: str = "OpenVLA-OFT_UR5_Lift"            #Agent's name used for metadata
BENCHMARKINFO: str = "openvlainstantir"                    #output benchmark's folder name
PROMPT="Pick up the object"                    #Task prompt for agent
HORIZON: int = 250                             # Maximum number of steps per episode
SAVEVIDEO: bool = False                         #save video after each episode
ISDISPLAY = False                               #show Mujoco render while benchmark

do_baseline = True       #run benchmark without compress
num_steps_wait: int = 10        #step to wait before each episode     
Control_freq=10          #control freqency for robosuite                    
episodes_times: int = 100       #episode times for one task                       
DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
#============================

# result_save_path = Path(f"data/benchmark/{BENCHMARKINFO}")
# result_save_path.mkdir(parents=True, exist_ok=True)


controller_config = load_composite_controller_config("robosuite/controllers/config/robots/default_ur5e.json")
logging.basicConfig(level=logging.INFO)



# OpenVLA-oft setup, should align with setup of training process
@dataclass
class GenerateConfig:
    model_family: str = "openvla"            
    # Your checkpoint path
    pretrained_checkpoint: Union[str, Path] = "../qr_test/tos/zhangjianbo/Ecomp/openvla" 
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 2                
    use_proprio: bool = True                    

    center_crop: bool = False
    num_open_loop_steps: int = 8                
    lora_rank: int = 32
    unnorm_key: Union[str, Path] = "robosuite_ur5"  # same as unnorm key of training process
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    env_img_res: int = 256

    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    seed: int = SEED


def initialize_model(cfg: GenerateConfig):
    """load openvla's projector / head / processor"""
    model = get_model(cfg)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)

    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    return model, action_head, proprio_projector, noisy_action_projector, processor

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


def process_action(action: np.ndarray, model_family: str) -> np.ndarray:
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


def build_obs_ur5(obs: dict, cpr ,step: int, iscpr: bool, prompt: str) -> Tuple[dict, Optional[List[np.ndarray]]]:
    """
    convert observation from mujoco to openvla's config format
    You can modify or compress the image from camera here

    :param obs: obervation from mujoco
    :param step: current step of this ep,onlty save first frame image
    :type step: int
    :param iscpr: compress or not
    :param prompt: task's text prompt
    :type prompt: str
    """
    state = np.concatenate((obs["robot0_joint_pos"], np.array([0.0]), [obs["robot0_gripper_qpos"][0]]), axis=0)

    agent_raw = obs["agentview_image"][::-1, ::-1].astype(np.uint8)
    wrist_raw = obs["robot0_eye_in_hand_image"][::-1, ::-1].astype(np.uint8)

    if iscpr:
        # time1=time.time()
        agent_cpr = cpr.compress(agent_raw, saveimg=False)[0]
        wrist_cpr = cpr.compress(wrist_raw, saveimg=False)[0]
        # timecompress=time.time()-time1
        # logging.info(f"compress request in {timecompress},step{step}")
        obs_new = {
            "full_image": agent_cpr,
            "wrist_image": wrist_cpr,
            "state": state,
            "prompt": prompt,
        }
        first_frame = [agent_cpr, wrist_cpr, agent_raw, wrist_raw] if step == 0 else None
    else:
        obs_new = {
            "full_image": agent_raw,
            "wrist_image": wrist_raw,
            "state": state,
            "prompt": prompt,
        }
        first_frame = None

    return obs_new, first_frame


def run_episode(
    cfg: GenerateConfig,
    env,
    episode_idx: int,
    task_description: str,
    model,
    cpr,
    components: Tuple,  # (processor, action_head, proprio_projector, noisy_action_projector)
    iscpr: bool,
    cprmodel_tag: str,
    savepath: Path,
    save_video: bool,
) -> Tuple[bool, List[np.ndarray], int]:
    processor, action_head, proprio_projector, noisy_action_projector = components

    obs = env.reset()
    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    replay_images: List[np.ndarray] = []
    # logging.info(f"PROMPT:{PROMPT},Env:{env.env_metadata()}")
    t = 0
    success = False
    done = False

    while t < HORIZON + num_steps_wait:
        if t < num_steps_wait:
            obs, _, _, _ = env.step(DUMMY_ACTION)
            t += 1
            continue

        if save_video:
            replay_images.append(np.ascontiguousarray(obs["frontview_image"][::-1, ::-1]).astype(np.uint8))
        step_start_time = time.time()
        if len(action_queue) == 0:
            observation, first_frame = build_obs_ur5(
            obs,
            cpr,
            step=t - num_steps_wait,
            iscpr=iscpr,
            prompt=PROMPT
        )
            step_mid_time = time.time()
            # action_start_time = time.time()
            actions = get_action(
                cfg,
                model,
                observation,
                task_description,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                use_film=cfg.use_film,
            )
            action_queue.extend(actions)
            # action_end_time = time.time()
            # logging.info(f"inference request in {action_end_time - action_start_time:.2f}s,step{t}")
        action = action_queue.popleft()
        action = process_action(action, cfg.model_family)
        
        obs, _, _, _ = env.step(action.tolist())
        
        
        done = env._check_success()

        if (t - num_steps_wait) == 0 and iscpr and first_frame is not None:
            for i, im in enumerate(first_frame):
                Image.fromarray(im).save(savepath / f"firstframe_{cprmodel_tag}_eps{episode_idx+1}_{i}.png")
        step_end_time = time.time()
        if step_mid_time is not None:
            logging.info(f"step in {step_end_time - step_start_time:.2f}s {step_mid_time-step_start_time:.2f} for image&{step_end_time-step_mid_time:.2f} for request action,step{t},model {cprmodel_tag},")
            step_mid_time = None
        if done:
            success = True
            break

        t += 1

    steps_effective = max(0, t - num_steps_wait)
    return success, replay_images, steps_effective


def run_task(
    cfg: GenerateConfig,
    model,
    cpr,
    components: Tuple,  # (processor, action_head, proprio_projector, noisy_action_projector)
    cprmodel_tag: str,
    iscpr: bool,
    savepath: Path,
    save_video: bool,
) -> Tuple[float, list]:
    task_successes = 0
    env_metadata = []
    seed_chain = [SEED+i for i in range(episodes_times)]
    env = suite.make(
        "LiftTest",
        robots="UR5e",
        gripper_types="default",
        controller_configs=controller_config,
        has_renderer=ISDISPLAY,
        render_camera="agentview",
        camera_names=("agentview", "robot0_eye_in_hand"),
        control_freq=Control_freq,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_widths=256,
        camera_heights=256,
        horizon=HORIZON + num_steps_wait,
        ignore_done=False,
    )

    for ep in tqdm.tqdm(range(episodes_times)):
        # for attempt in range(5):
        #     try:
        set_seed_everywhere(seed = seed_chain[ep])
        
        success, replay_images, steps = run_episode(
            cfg=cfg,
            env=env,
            episode_idx=ep,
            task_description=PROMPT,
            model=model,
            cpr=cpr,
            components=components,
            iscpr=iscpr,
            cprmodel_tag=cprmodel_tag,
            savepath=savepath,
            save_video=save_video,
        )
        
        suffix = "success" if success else "failure"
        env_metadata.append(env.env_metadata())
        env_metadata[-1].update({
            "success": suffix,
            "steps": steps,
        })

        if save_video:
            imageio.mimwrite(
                savepath / f"rollout_{env.env_metadata()['object']}_eps{ep}_{cprmodel_tag}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=Control_freq,
            )

        if success:
            task_successes += 1
        logging.info(f"Success: {success}")
        logging.info(f"Model {cprmodel_tag} Completed: {ep+1}/{episodes_times}")
        logging.info(f"Successes: {task_successes} ({task_successes / (ep+1) * 100:.1f}%)")
            #     break
            # except Exception as e:
            #     logging.error(f"[Episode {ep+1}] Attempt {attempt+1}/{5} FAILED")
            #     logging.error(repr(e))
            #     if attempt == 5:
            #         logging.error(f"[Episode {ep+1}] Gave up after max retries")
            #         env_metadata.append(env.env_metadata())
            #         env_metadata[-1].update({
            #         "success": "failure",
            #         "steps": 0,
            #         })
            #         break

    env.close()

    success_rate = float(task_successes) / float(episodes_times)
    logging.info(f"Current task success rate: {success_rate:.3f}")
    return success_rate, env_metadata



def main():
    start_time = time.time()
    result_save_path.mkdir(parents=True, exist_ok=True)

    cfg = GenerateConfig()
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    components = (processor, action_head, proprio_projector, noisy_action_projector)

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(9, 6))
    # ax = plt.gca()
    # ax.set_title("BPP & Success Rate")
    # ax.set_xlabel("Average BPP (bits per pixel)")
    # ax.set_ylabel("Success Rate")
    # ax.grid(True, linestyle=":", linewidth=0.8)

    baseline_success_rate = 0.0
    baseline_metadata = []

    cpr = compressimg.COMPRESSIMG(model="distort") 
    if do_baseline:
        base_path = result_save_path / "baseline"
        base_path.mkdir(parents=True, exist_ok=True)
        baseline_success_rate, baseline_metadata = run_task(
            cfg=cfg,
            model=model,
            cpr=cpr,
            components=components,
            cprmodel_tag="baseline",
            iscpr=False,
            savepath=base_path,
            save_video=SAVEVIDEO,
        )
        # ax.axhline(y=baseline_success_rate, linestyle="--", linewidth=1.5,
        #            label=f"Baseline (no compression) = {baseline_success_rate:.2f}")
        with open(result_save_path / "result_default.json", "w") as f:
            json.dump({
                "model": "default",
                "quality": "default",
                "episodes": episodes_times,
                "success_rate": baseline_success_rate,
                "bpp": 0.0,
                "env_metadata": baseline_metadata,
            }, f, indent=4)
    
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

    # color_toggle = 0
    for dis_type in range(25):
        model_name = f"distort_{dis_type}"
        logging.info(f"Running model {model_name}")
        cpr.changenet(model="distort", dis_type=dis_type)
        bpplist: List[float] = []
        ratelist: List[float] = []

        model_dir = result_save_path / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        success_rate, env_md = run_task(
            cfg=cfg,
            model=model,
            cpr=cpr,
            components=components,
            cprmodel_tag=model_name,
            iscpr=True,
            savepath=model_dir,
            save_video=SAVEVIDEO,
        )

        avg_bpp = float(sum(cpr.bpp) / len(cpr.bpp))

        bpplist.append(avg_bpp)
        ratelist.append(success_rate)
        with open(model_dir / f"result_{model_name}.json", "w") as f:
            json.dump({
                "model": model_name,
                "quality": "demo",
                "episodes": episodes_times,
                "success_rate": success_rate,
                "bpp": avg_bpp,
                "env_metadata": env_md,
            }, f, indent=4)

    # for mi, model_name in enumerate(MODELLIST):
    #     logging.info(f"Running model {model_name} ({mi+1}/{len(MODELLIST)})")
    #     bpplist: List[float] = []
    #     ratelist: List[float] = []

    #     model_dir = result_save_path / model_name
    #     model_dir.mkdir(parents=True, exist_ok=True)

    #     for qi, (q, d) in enumerate(reversed(MODELQUADOWN[model_name])):
    #         d_suffix = d.replace("/", "-")
    #         run_dir = model_dir / f"quality{q}_down{d_suffix}"
    #         run_dir.mkdir(parents=True, exist_ok=True)

    #         logging.info(f"  - {model_name} | Q={q} | Down={d}  ({qi+1}/{len(MODELQUADOWN[model_name])})")
    #         cpr.changenet(model_name, q, d)

    #         success_rate, env_md = run_task(
    #             cfg=cfg,
    #             model=model,
    #             components=components,
    #             cprmodel_tag=f"{model_name}_{q}_{d_suffix}",
    #             iscpr=True,
    #             savepath=run_dir,
    #             save_video=SAVEVIDEO,
    #         )

    #         avg_bpp = float(sum(cpr.bpp) / len(cpr.bpp))

    #         bpplist.append(avg_bpp)
    #         ratelist.append(success_rate)
    #         with open(run_dir / f"result_{model_name}_{q}_{d_suffix}.json", "w") as f:
    #             json.dump({
    #                 "model": model_name,
    #                 "quality": q,
    #                 "downsample": d,
    #                 "episodes": episodes_times,
    #                 "success_rate": success_rate,
    #                 "bpp": avg_bpp,
    #                 "env_metadata": env_md,
    #             }, f, indent=4)

        if bpplist:
            idx = np.argsort(np.array(bpplist))
            x_sorted = [bpplist[i] for i in idx]
            y_sorted = [ratelist[i] for i in idx]

            summary["results"].append({
                "model": model_name,
                "quality&downsample": "demo",
                "success_rate": y_sorted,
                "bpp": x_sorted,
            })

    #         if color_toggle < 10:
    #             ax.plot(x_sorted, y_sorted, marker="o", linewidth=1.8, label=model_name)
    #         else:
    #             ax.plot(x_sorted, y_sorted, marker="x", linewidth=1.8, label=model_name)
    #         color_toggle += 1
    # ax.legend(loc="best", frameon=True)
    # plt.tight_layout()
    # fig_path = result_save_path / "bpp_successrate.png"
    # plt.savefig(fig_path, dpi=150)

    summary_path = result_save_path / f"summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    end_time = time.time()
    # logging.info(f"Figure saved to: {result_save_path / 'bpp_successrate.png'}")
    logging.info(f"Summary JSON saved to: {summary_path}")
    logging.info(f"Done. Total time: {((end_time - start_time) / 60):.1f} minutes")

def noise_thread(ntype:int,etype:int=0):
    # this function is used for multiprocessing, each process runs one noise type, including baseline(no noise)
    start_time = time.time()
    global _GLOBAL_MODEL_COMPONENTS
    cfg,model, components = _GLOBAL_MODEL_COMPONENTS
    from compress import compressimg
    ##Import Restoration here you want to test, and add it in Ehance list
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


    result_save_path = Path(f"data/benchmark/openvla_{Enhance.__name__}")
    if ntype == -1:
        logging.info(f"Running baseline")
        cpr = compressimg.Pipeline([
        Enhance(quality=4), 
        ])
        model_name = "baseline"
        base_path = result_save_path / model_name
        base_path.mkdir(parents=True, exist_ok=True)
        sr,meta = run_task(
            cfg=cfg,
            model=model,
            cpr=cpr,
            components=components,
            cprmodel_tag="baseline",
            iscpr=True,
            savepath=base_path,
            save_video=SAVEVIDEO,
        )
        
        # share_queue.put({"type": "baseline", "sr": sr})
    else:
        logging.info(f"Running noise type {ntype} & enhance type {etype}")
        cpr = compressimg.Pipeline([
        Distorter(quality=1, type=ntype),
        Enhance(quality=4), 
        ])
        model_name = f"distort_{ntype}"
        base_path = result_save_path / model_name
        base_path.mkdir(parents=True, exist_ok=True)
        sr,meta = run_task(
            cfg=cfg,
            model=model,
            cpr=cpr,
            components=components,
            cprmodel_tag=f"distort_{ntype}&ehance_{Enhance.__name__}",
            iscpr=True,
            savepath=base_path,
            save_video=SAVEVIDEO,
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
    logging.info(f"---------Noise type {ntype}&Ehance {etype} done. Total time: {((end_time - start_time) / 60):.1f} minutes---------")
    return {"id":etype*100+ntype, "model": str(base_path), "sr": sr}

def init_worker():
    torch.cuda.set_device(0)  
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    global _GLOBAL_MODEL_COMPONENTS
    cfg = GenerateConfig()
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    components = (processor, action_head, proprio_projector, noisy_action_projector)
    _GLOBAL_MODEL_COMPONENTS = (cfg,model, components)

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
            logger.info(f"Got result: model={result.get('model')}, sr={result.get('sr')}")

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
    thread_manager(task_list, proc_per=1)
