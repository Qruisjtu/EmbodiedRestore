import logging
import os
import sys
import numpy as np
from PIL import Image

# ==== Real-ESRGAN PARAMETERS ====
USE_REALESRGAN = True
REALESRGAN_READY = False
REALESRGAN_MODEL = None  # 全局模型实例，避免重复加载

REALESRGAN_PYTHON = "/mnt/shared-storage-user/zhangjianbo/conda-envs/withflash/bin/python"
REALESRGAN_SCRIPT = "/mnt/shared-storage-user/zhangjianbo/tos/Real-ESRGAN/inference_realesrgan.py"
REALESRGAN_REPO = "/mnt/shared-storage-user/zhangjianbo/tos/Real-ESRGAN"
BASICSR_REPO = "/mnt/shared-storage-user/zhangjianbo/models/BasicSR"

REALESRGAN_MODEL_NAME = "RealESRGAN_x4plus"
REALESRGAN_MODEL_PATH = "/mnt/shared-storage-user/zhangjianbo/models/weights/RealESRGAN_x4plus.pth"
# ================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def _setup_pythonpath():
    """设置 PYTHONPATH 以导入 realesrgan 模块"""
    current_path = os.environ.get("PYTHONPATH", "")
    new_path = f"{REALESRGAN_REPO}:{BASICSR_REPO}"
    if current_path:
        new_path = f"{new_path}:{current_path}"
    if new_path not in sys.path:
        sys.path = new_path.split(":") + sys.path


def init_realesrgan():
    """初始化 Real-ESRGAN 模型（只加载一次）"""
    global REALESRGAN_READY, REALESRGAN_MODEL

    if not USE_REALESRGAN:
        logging.info("USE_REALESRGAN=False, skip Real-ESRGAN init.")
        REALESRGAN_READY = False
        return

    try:
        # 检查路径
        assert os.path.exists(REALESRGAN_REPO), f"repo not found: {REALESRGAN_REPO}"
        assert os.path.exists(BASICSR_REPO), f"BasicSR repo not found: {BASICSR_REPO}"
        assert os.path.exists(REALESRGAN_MODEL_PATH), f"model path not found: {REALESRGAN_MODEL_PATH}"

        # 设置 PYTHONPATH 并导入
        _setup_pythonpath()
        
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        import torch

        # 构建模型架构 (根据 RealESRGAN_x4plus)
        model = RRDBNet(
            num_in_ch=3, 
            num_out_ch=3, 
            num_feat=64, 
            num_block=23, 
            num_grow_ch=32, 
            scale=4
        )
        
        # 创建 RealESRGANer 实例
        # half=True 使用半精度推理，速度更快，显存更少 [[2]]
        REALESRGAN_MODEL = RealESRGANer(
            scale=4,
            model_path=REALESRGAN_MODEL_PATH,
            model=model,
            tile=0,          # 0=不切片，大图可设为 400 避免 OOM [[21]]
            tile_pad=10,
            pre_pad=0,
            half=True,       # 启用 FP16 加速 [[24]]
            gpu_id=0,        # 指定 GPU
        )
        
        # 预热模型（可选，避免首次推理延迟）
        if torch.cuda.is_available():
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            REALESRGAN_MODEL.enhance(dummy, outscale=4)
        
        REALESRGAN_READY = True
        logging.info("Real-ESRGAN init finished (model loaded in memory).")
        
    except ImportError as e:
        REALESRGAN_READY = False
        logging.error(f"Failed to import realesrgan: {e}")
        logging.error("Please ensure PYTHONPATH includes Real-ESRGAN and BasicSR repos.")
    except Exception as err:
        REALESRGAN_READY = False
        logging.error(f"Real-ESRGAN init failed: {repr(err)}")
        logging.error("Fallback to raw image.")


def realesrgan_enhance(image: np.ndarray, step: int = 0, cam: str = "") -> np.ndarray:
    """
    使用内存中的 Real-ESRGAN 模型直接增强图像
    输入: RGB numpy array (H, W, 3), dtype=uint8
    输出: RGB numpy array (H*4, W*4, 3), dtype=uint8
    """
    global REALESRGAN_READY, REALESRGAN_MODEL

    if not USE_REALESRGAN or not REALESRGAN_READY or REALESRGAN_MODEL is None:
        return image

    try:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return image
            
        # 确保输入格式
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        image = image[:, :, :3]  # 只取 RGB 通道

        # RealESRGANer 内部使用 BGR，但 enhance() 会自动处理 RGB->BGR 转换
        # 直接传入 numpy 数组，完全在内存中处理，无磁盘 IO
        output, _ = REALESRGAN_MODEL.enhance(image, outscale=4)
        
        # output 已是 uint8 numpy array (BGR 格式)，转回 RGB
        return output.copy()  

    except RuntimeError as e:
        # 常见于显存不足，可尝试启用 tile 模式
        if "CUDA out of memory" in str(e):
            logging.warning("CUDA OOM, try setting tile>0 in init_realesrgan()")
        logging.error(f"Real-ESRGAN enhance failed: {repr(e)}")
        return image
    except Exception as err:
        logging.error(f"Real-ESRGAN enhance failed: {repr(err)}")
        return image