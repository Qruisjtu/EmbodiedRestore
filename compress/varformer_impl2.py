import logging
import os
import sys
import traceback
import torch
import numpy as np
import cv2
from torchvision.transforms.functional import normalize

USE_VARFORMER = True
READY = False

VARFORMER_REPO_DIR = "/mnt/shared-storage-user/zhangjianbo/models/Varformer"
MODEL_PATH = os.path.join(VARFORMER_REPO_DIR, "experiments/pretrained_models/net_g_last.pth")
CONFIG_PATH = os.path.join(VARFORMER_REPO_DIR, "basicsr/options/test.yml")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

GLOBAL_VARFORMER_NET = None
GLOBAL_OPT = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class IsolateEnvironment:
    """
    Sandbox to prevent path/module conflicts with other models like BIRD/DFPIR.
    Redirects CWD to solve relative path issues inside Varformer architecture.
    """
    def __enter__(self):
        self.orig_path = sys.path.copy()
        self.orig_utils = sys.modules.get('utils')
        self.orig_cwd = os.getcwd()
        
        models_base_dir = "/mnt/shared-storage-user/zhangjianbo/models"
        clean_path =[p for p in sys.path if not (p.startswith(models_base_dir) and "Varformer" not in p)]
        
        basicsr_dir = os.path.join(VARFORMER_REPO_DIR, "basicsr")
        clean_path.insert(0, basicsr_dir)
        clean_path.insert(0, VARFORMER_REPO_DIR)
        sys.path = clean_path
        
        if 'utils' in sys.modules:
            del sys.modules['utils']
            
        os.chdir(VARFORMER_REPO_DIR)

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.path = self.orig_path
        os.chdir(self.orig_cwd)
        if self.orig_utils is not None:
            sys.modules['utils'] = self.orig_utils
        elif 'utils' in sys.modules:
            del sys.modules['utils']


def init_varformer():
    global READY, GLOBAL_VARFORMER_NET, GLOBAL_OPT

    if not USE_VARFORMER:
        READY = False
        return

    try:
        logging.info("[varformer] Initializing Varformer arch on H200 (Matching original test script)...")
        
        with IsolateEnvironment():
            # 1. Exact options parsing from varformer_test.py
            from options import options as option
            from archs import build_network
            
            opt = option.parse(CONFIG_PATH, is_train=False)
            opt = option.dict_to_nonedict(opt)
            GLOBAL_OPT = opt
            
            # 2. Build Network
            net = build_network(opt['network_g'])
            
            # 3. Load pretrain model exactly as in varformer_test.py
            # Note: The original script uses opt['path']['pretrain_model'], but we use MODEL_PATH for safety
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            ckpt = checkpoint['params_ema']
            
            strict_load = opt['path'].get('strict_load', False)
            logging.info(f"[varformer] Loading state_dict (strict={strict_load})...")
            
            net.load_state_dict(ckpt, strict=strict_load)
            net.to(DEVICE)
            net.eval()
            
            GLOBAL_VARFORMER_NET = net
            READY = True
            logging.info("[varformer] init finished. Model ready.")
        
    except Exception as e:
        READY = False
        logging.error(f"[varformer] init failed: {repr(e)}")
        logging.error(traceback.format_exc())


def varformer_enhance(image: np.ndarray, step: int = 0, cam: str = "") -> np.ndarray:
    global READY, GLOBAL_VARFORMER_NET, GLOBAL_OPT

    if not READY or GLOBAL_VARFORMER_NET is None:
        return image

    if image is None or not isinstance(image, np.ndarray):
        return image

    try:
        # 1. Match varformer_test.py reading logic (cv2 style)
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # Image is RGB from compressimg pipeline, save original size
        image = image[:, :, :3]
        h_orig, w_orig = image.shape[:2]

        input_size = GLOBAL_OPT.get('input_size', 256)
        mean = GLOBAL_OPT.get('mean',[0.5, 0.5, 0.5])
        std = GLOBAL_OPT.get('std',[0.5, 0.5, 0.5])

        # 2. CPU pre-processing mirroring varformer_test.py exactly
        img = image.astype(np.float32) / 255.
        img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        
        # Original code does BGR to RGB here, but our input `image` is ALREADY RGB.
        # So we skip `img = img[:, :, [2, 1, 0]]` to maintain correct colors!
        
        img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
        normalize(img, mean, std, inplace=True)

        with torch.no_grad():
            img = img.unsqueeze(0).to(DEVICE)

            # 3. Inference inside Sandbox
            with IsolateEnvironment():
                GLOBAL_VARFORMER_NET.eval()
                output, _ = GLOBAL_VARFORMER_NET(img)

            # 4. Post-processing mimicking `tensor2img(sr_img)`
            sr_img = output.detach().cpu().squeeze(0).float().clamp_(0, 1)
            out_np = (sr_img.numpy() * 255.0).round().astype(np.uint8)
            out_np = np.transpose(out_np, (1, 2, 0)) # CHW -> HWC (RGB)

        # 5. Restore original resolution
        if out_np.shape[:2] != (h_orig, w_orig):
            out_np = cv2.resize(out_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        return out_np

    except Exception as e:
        logging.error(f"[varformer] enhance failed: {repr(e)}")
        logging.error(traceback.format_exc())
        return image