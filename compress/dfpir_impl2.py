import logging
import os
import sys
import math
import traceback
import torch
import torch.nn.functional as F
import numpy as np

# ==== DFPIR PARAMETERS ====
USE_DFPIR = True
DFPIR_READY = False

DFPIR_REPO_DIR = "/mnt/shared-storage-user/zhangjianbo/models/DFPIR"
DFPIR_WEIGHT_PATH = "/mnt/shared-storage-user/zhangjianbo/models/DFPIR/DFPIR-3D-SP_p_n34.12-0.9348_31.45-0.8911_28.18-0.8009_p_r38.63-0.9821_p_h31.93-0.9805avr32.86-0.9179.pt"
DFPIR_PROMPT = "Gaussian noise with a standard deviation of 25"

# ==========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

if DFPIR_REPO_DIR not in sys.path and os.path.exists(DFPIR_REPO_DIR):
    sys.path.insert(0, DFPIR_REPO_DIR)

try:
    import clip
    from net.model import ChannelShuffle_skip_textguaid
    DFPIR_IMPORTED = True
except ImportError as e:
    DFPIR_IMPORTED = False
    logging.warning(f"DFPIR src could not be imported. USE_DFPIR will be disabled. Error: {e}")

GLOBAL_DFPIR_MODEL = None
GLOBAL_DFPIR_TEXT_CODE = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def pad_to_multiple(x: torch.Tensor, base: int = 8):
    """确保高宽是 base 的整数倍（官方实现）"""
    _, _, h, w = x.shape
    new_h = math.ceil(h / base) * base
    new_w = math.ceil(w / base) * base
    pad_h = new_h - h
    pad_w = new_w - w
    # pad order: left, right, top, bottom
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, h, w

def crop_back(x: torch.Tensor, h: int, w: int):
    """切回原本的大小"""
    return x[:, :, :h, :w]

def init_dfpir():
    global DFPIR_READY, GLOBAL_DFPIR_MODEL, GLOBAL_DFPIR_TEXT_CODE

    if not USE_DFPIR or not DFPIR_IMPORTED:
        logging.info("USE_DFPIR=False or code not found, skip DFPIR init.")
        DFPIR_READY = False
        return

    try:
        logging.info("Initializing DFPIR model into VRAM...")
        
        # 为了让 clip 能找到缓存目录
        os.environ.setdefault("HOME", "/mnt/shared-storage-user/zhangjianbo")

        # 1. 加载 CLIP 提取固定 prompt 的 text_code
        logging.info("Loading CLIP for text prompt encoding...")
        CLIP_WEIGHT_PATH = "/mnt/shared-storage-user/zhangjianbo/.cache/clip/ViT-B-32.pt"
        clip_model, _ = clip.load(CLIP_WEIGHT_PATH, device=DEVICE)
        clip_model.eval()
        
        text_token = clip.tokenize(DFPIR_PROMPT).to(DEVICE)
        with torch.no_grad():
            GLOBAL_DFPIR_TEXT_CODE = clip_model.encode_text(text_token).to(dtype=torch.float32)

        # 完美优化：既然有了 text_code，CLIP 模型就不需要了，直接释放腾出显存！
        del clip_model
        torch.cuda.empty_cache()

        # 2. 加载主模型 DFPIR
        model = ChannelShuffle_skip_textguaid()
        ckpt = torch.load(DFPIR_WEIGHT_PATH, map_location=DEVICE)

        # 解析 ckpt 结构（复刻官方的 load_model 逻辑）
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                state = ckpt["state_dict"]
            elif "params" in ckpt:
                state = ckpt["params"]
            elif "model" in ckpt:
                state = ckpt["model"]
            else:
                state = ckpt
        else:
            state = ckpt

        # 剥离 module. 前缀
        new_state = {}
        for k, v in state.items():
            if k.startswith("module."):
                new_state[k[len("module."):]] = v
            else:
                new_state[k] = v

        model.load_state_dict(new_state, strict=False)
        model.to(DEVICE)
        model.eval()

        GLOBAL_DFPIR_MODEL = model
        DFPIR_READY = True
        logging.info("DFPIR init finished. Model and Text Code loaded in VRAM.")
    except Exception as err:
        DFPIR_READY = False
        logging.error(f"DFPIR init failed: {repr(err)}")
        logging.error(traceback.format_exc())

def dfpir_enhance(image: np.ndarray, step: int = 0, cam: str = "") -> np.ndarray:
    global DFPIR_READY, GLOBAL_DFPIR_MODEL, GLOBAL_DFPIR_TEXT_CODE

    if not USE_DFPIR or not DFPIR_READY or GLOBAL_DFPIR_MODEL is None:
        return image

    try:
        if image is None or not isinstance(image, np.ndarray):
            return image
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # 确保通道是 RGB
        image = image[:, :, :3]

        # 1. 预处理 Numpy HWC -> Tensor NCHW [0, 1]
        arr = image.astype(np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # 2. 官方 Padding: 填充到 8 的倍数
            x, h0, w0 = pad_to_multiple(x, base=8)
            
            # 3. 推理 (传入图和在 init 阶段就算好的 text_code)
            restored = GLOBAL_DFPIR_MODEL(x, GLOBAL_DFPIR_TEXT_CODE)
            
            # 4. 后处理: 裁回原大小并裁切值域
            restored = crop_back(restored, h0, w0)
            restored = restored.clamp(0, 1)

        # 5. Tensor NCHW -> Numpy HWC [0, 255]
        out = restored.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        out = (out * 255.0).round().astype(np.uint8)

        return out

    except Exception as err:
        logging.error(f"DFPIR enhance failed: {repr(err)}")
        logging.error(traceback.format_exc())
        return image