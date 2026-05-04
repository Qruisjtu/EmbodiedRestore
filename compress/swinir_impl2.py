import logging
import os
import sys
import traceback
import torch
import numpy as np

# ==== SwinIR PARAMETERS ====
USE_SWINIR = True
SWINIR_READY = False

SWINIR_REPO = "/mnt/shared-storage-user/zhangjianbo/models/SwinIR"
SWINIR_MODEL_PATH = "/mnt/shared-storage-user/zhangjianbo/models/SwinIR/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x3.pth"
SWINIR_SCALE = 3
# ==========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

if SWINIR_REPO not in sys.path and os.path.exists(SWINIR_REPO):
    sys.path.insert(0, SWINIR_REPO)

try:
    from models.network_swinir import SwinIR as net
    SWINIR_IMPORTED = True
except ImportError as e:
    SWINIR_IMPORTED = False
    logging.warning(f"SwinIR src could not be imported. USE_SWINIR will be disabled. Error: {e}")

GLOBAL_SWINIR_MODEL = None
DEVICE = "cuda" 


def init_swinir():
    global SWINIR_READY, GLOBAL_SWINIR_MODEL

    if not USE_SWINIR or not SWINIR_IMPORTED:
        logging.info("USE_SWINIR=False or code not found, skip SwinIR init.")
        SWINIR_READY = False
        return

    try:
        logging.info("Initializing SwinIR model into VRAM...")
        
        # 对应 task = 'lightweight_sr' 的网络配置
        model = net(
            upscale=SWINIR_SCALE, 
            in_chans=3, 
            img_size=64, 
            window_size=8,
            img_range=1., 
            depths=[6, 6, 6, 6], 
            embed_dim=60, 
            num_heads=[6, 6, 6, 6],
            mlp_ratio=2, 
            upsampler='pixelshuffledirect', 
            resi_connection='1conv'
        )

        # 加载权重
        pretrained_model = torch.load(SWINIR_MODEL_PATH, map_location=DEVICE)
        param_key_g = 'params'
        model.load_state_dict(
            pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, 
            strict=True
        )

        model.eval()
        GLOBAL_SWINIR_MODEL = model.to(DEVICE)
        
        SWINIR_READY = True
        logging.info("SwinIR init finished. Model loaded in VRAM.")
    except Exception as err:
        SWINIR_READY = False
        logging.error(f"SwinIR init failed: {repr(err)}")
        logging.error(traceback.format_exc())


def swinir_enhance(image: np.ndarray, step: int = 0, cam: str = "") -> np.ndarray:
    global SWINIR_READY, GLOBAL_SWINIR_MODEL

    if not USE_SWINIR or not SWINIR_READY or GLOBAL_SWINIR_MODEL is None:
        return image

    try:
        if image is None or not isinstance(image, np.ndarray):
            return image
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # 只取前三通道，确保是 RGB
        image = image[:, :, :3]

        # 1. 预处理: HWC (uint8) -> CHW (float32) [0, 1]
        img_lq = image.astype(np.float32) / 255.0
        img_lq = np.transpose(img_lq, (2, 0, 1))  # HWC to CHW
        
        # 2. 转换为 Tensor 并加上 Batch 维度
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # 3. Padding (为了适应 SwinIR window_size=8 的要求，复刻官方逻辑)
            window_size = 8
            _, _, h_old, w_old = img_lq.size()
            
            # 计算需要 padding 的大小
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            
            # 镜像填充 (Reflection Pad)
            if h_pad != window_size:
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            if w_pad != window_size:
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            
            # 4. 执行推理
            output = GLOBAL_SWINIR_MODEL(img_lq)
            
            # 5. 去除 Padding 产生的部分 (注意这里尺寸已经被放大了 SWINIR_SCALE 倍)
            output = output[..., :h_old * SWINIR_SCALE, :w_old * SWINIR_SCALE]

        # 6. 后处理: CHW Tensor -> HWC numpy array
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output, (1, 2, 0))  # CHW to HWC
        output = (output * 255.0).round().astype(np.uint8)
        
        return output

    except Exception as err:
        logging.error(f"SwinIR enhance failed: {repr(err)}")
        logging.error(traceback.format_exc())
        return image