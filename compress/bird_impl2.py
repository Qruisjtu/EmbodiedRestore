import logging
import os
import sys
import tempfile
import traceback
import yaml
import numpy as np
import torch
from torch import nn
from PIL import Image
import cv2

USE_BIRD = True
BIRD_READY = False

BIRD_REPO_DIR = "/mnt/shared-storage-user/zhangjianbo/models/BIRD"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Inject BIRD repo to sys.path
if BIRD_REPO_DIR not in sys.path and os.path.exists(BIRD_REPO_DIR):
    sys.path.insert(0, BIRD_REPO_DIR)
    sys.path.insert(0, os.path.join(BIRD_REPO_DIR, 'guided_diffusion'))

try:
    from ddim_inversion_utils import *
    from utils import *
    BIRD_IMPORTED = True
except ImportError as e:
    BIRD_IMPORTED = False
    logging.warning(f"[BIRD] Imports failed. Error: {e}")

GLOBAL_BIRD_MODEL = None
GLOBAL_CONFIG = None
GLOBAL_TASK_CONFIG = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def init_bird():
    global BIRD_READY, GLOBAL_BIRD_MODEL, GLOBAL_CONFIG, GLOBAL_TASK_CONFIG

    if not USE_BIRD or not BIRD_IMPORTED:
        logging.info("[BIRD] USE_BIRD=False or code not found, skip init.")
        BIRD_READY = False
        return

    # 记住进入前的原始工作目录
    original_cwd = os.getcwd()
    
    try:
        logging.info("[BIRD] Initializing BIRD Optimization Pipeline...")
        
        # 瞬间切入 BIRD 目录，让原作者的相对路径生效
        os.chdir(BIRD_REPO_DIR)
        
        # 1. Load Configs (现在可以直接用相对路径了)
        task_cfg_path = 'configs/blind_deblurring.yml'
        model_cfg_path = 'data/celeba_hq.yml'
        
        with open(task_cfg_path, 'r') as f:
            GLOBAL_TASK_CONFIG = yaml.safe_load(f)
        with open(model_cfg_path, 'r') as f:
            GLOBAL_CONFIG = dict2namespace(yaml.safe_load(f))
            
        # 2. Set Seed
        torch.set_printoptions(sci_mode=False)
        ensure_reproducibility(GLOBAL_TASK_CONFIG['seed'])

        GLOBAL_BIRD_MODEL, _ = load_pretrained_diffusion_model(GLOBAL_CONFIG)
        GLOBAL_BIRD_MODEL = GLOBAL_BIRD_MODEL.to(DEVICE)
        

        for param in GLOBAL_BIRD_MODEL.parameters():
            param.requires_grad_(False)
        GLOBAL_BIRD_MODEL.eval()

        BIRD_READY = True
        logging.info("[BIRD] init finished. Model locked and loaded in VRAM.")
    except Exception as err:
        BIRD_READY = False
        logging.error(f"[BIRD] init failed: {repr(err)}")
        logging.error(traceback.format_exc())
    finally:
        # 无论成功失败，必须切回你原来的目录，以免影响你的 Pipeline 后续节点！
        os.chdir(original_cwd)

def bird_enhance(image: np.ndarray, step: int = 0, cam: str = "") -> np.ndarray:
    global BIRD_READY, GLOBAL_BIRD_MODEL, GLOBAL_CONFIG, GLOBAL_TASK_CONFIG

    if not USE_BIRD or not BIRD_READY or GLOBAL_BIRD_MODEL is None:
        return image

    if image is None or not isinstance(image, np.ndarray):
        return image

    original_cwd = os.getcwd()
    try:
        # 切入 BIRD 目录防相对路径报错
        os.chdir(BIRD_REPO_DIR)
        
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        image = image[:, :, :3]
        h_orig, w_orig = image.shape[:2]

        # Use /dev/shm (RAM Disk) to bridge BIRD's internal file reader with zero physical I/O
        ram_disk = "/dev/shm" if os.path.exists("/dev/shm") else None
        
        with tempfile.NamedTemporaryFile(suffix=".png", dir=ram_disk) as tmp_file:
            # 1. Resize to target size (BIRD natively works on specific resolution, typically 256x256)
            img_size = GLOBAL_CONFIG.data.image_size
            pil_img = Image.fromarray(image).resize((img_size, img_size), resample=Image.Resampling.LANCZOS)
            pil_img.save(tmp_file.name)
            
            # 2. Use BIRD's internal generator
            _, downsampled_torch = generate_blurry_image(tmp_file.name)
        
        downsampled_torch = downsampled_torch.to(DEVICE)



        # 3. Setup Scheduler
        ddim_scheduler = DDIMScheduler(
            beta_start=GLOBAL_CONFIG.diffusion.beta_start, 
            beta_end=GLOBAL_CONFIG.diffusion.beta_end, 
            beta_schedule=GLOBAL_CONFIG.diffusion.beta_schedule
        )
        ddim_scheduler.set_timesteps(GLOBAL_CONFIG.diffusion.num_diffusion_timesteps // GLOBAL_TASK_CONFIG['delta_t'])

        with torch.enable_grad():
            l2_loss = nn.MSELoss()
            k_size = GLOBAL_TASK_CONFIG['kernel_size']
            net_kernel = fcn(200, k_size * k_size).to(DEVICE)
            net_input_kernel = get_noise(200, 'noise', (1, 1)).to(DEVICE).squeeze_()
            
            radii = torch.ones([1, 1, 1], device=DEVICE) * (np.sqrt(img_size * img_size * 3))
            
            latent = torch.nn.Parameter(torch.randn(
                1, GLOBAL_CONFIG.model.in_channels, img_size, img_size, device=DEVICE
            ))
            
            optimizer = torch.optim.Adam([
                {'params': latent, 'lr': GLOBAL_TASK_CONFIG['lr_img']}, 
                {'params': net_kernel.parameters(), 'lr': GLOBAL_TASK_CONFIG['lr_blur']}
            ])

            opt_steps = GLOBAL_TASK_CONFIG['Optimization_steps']
            for iteration in range(opt_steps):
                optimizer.zero_grad()
                
                x_0_hat = DDIM_efficient_feed_forward(latent, GLOBAL_BIRD_MODEL, ddim_scheduler)   
                out_k = net_kernel(net_input_kernel)
                out_k_m = out_k.view(-1, 1, k_size, k_size)

                blurred_xt = nn.functional.conv2d(
                    x_0_hat.view(-1, 1, img_size, img_size), 
                    out_k_m, 
                    padding="same", 
                    bias=None
                ).view(1, 3, img_size, img_size) 
                
                loss = l2_loss(blurred_xt, downsampled_torch)
                loss.backward()  
                optimizer.step()  

                with torch.no_grad():
                    for param in latent:
                        param.data.div_((param.pow(2).sum(tuple(range(0, param.ndim)), keepdim=True) + 1e-9).sqrt())
                        param.data.mul_(radii)

            with torch.no_grad():
                out_np = process(x_0_hat, 0)

        if out_np.shape[:2] != (h_orig, w_orig):
            out_np = cv2.resize(out_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        del latent, net_kernel, optimizer, x_0_hat, blurred_xt
        return out_np

    except Exception as err:
        logging.error(f"[BIRD] enhance failed: {repr(err)}")
        logging.error(traceback.format_exc())
        return image
    finally:
        # 切回原始目录
        os.chdir(original_cwd)