import logging
import os
import sys
import tempfile
import shutil
import traceback
import torch
import numpy as np
from PIL import Image
from argparse import Namespace

USE_FOUNDIR = True
FOUNDIR_READY = False

# 确保路径指向你的仓库
FOUNDIR_REPO = "/mnt/shared-storage-user/zhangjianbo/models/FoundIR"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

if FOUNDIR_REPO not in sys.path and os.path.exists(FOUNDIR_REPO):
    sys.path.insert(0, FOUNDIR_REPO)

try:
    from src.model import ResidualDiffusion, Trainer, UnetRes, set_seed
    from data.combined_dataset import CombinedDataset
    FOUNDIR_IMPORTED = True
except ImportError as e:
    FOUNDIR_IMPORTED = False
    logging.warning(f"FoundIR src could not be imported. USE_FOUNDIR will be disabled. Error: {e}")

GLOBAL_FOUNDIR_TRAINER = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def init_foundir():
    global FOUNDIR_READY, GLOBAL_FOUNDIR_TRAINER

    if not USE_FOUNDIR or not FOUNDIR_IMPORTED:
        logging.info("USE_FOUNDIR=False or code not found, skip FoundIR init.")
        FOUNDIR_READY = False
        return

    try:
        logging.info("Initializing FoundIR model into VRAM...")
        set_seed(10)

        # 构造跟 test.py 一模一样的配置
        opt = Namespace(
            dataroot="/tmp", # 占位
            phase="test",
            max_dataset_size=int(1e18),
            load_size=256,
            crop_size=256,
            direction="AtoB",
            preprocess="none",
            no_flip=True,
            meta=None,
            bsize=1,
        )

        model = UnetRes(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            num_unet=1,
            condition=True,
            objective="pred_res",
            test_res_or_noise="res",
        )

        diffusion = ResidualDiffusion(
            model,
            image_size=1024,
            timesteps=1000,
            delta_end=1.4e-3,
            sampling_timesteps=4,
            ddim_sampling_eta=0.0,
            objective="pred_res",
            loss_type="l1",
            condition=True,
            sum_scale=0.01,
            test_res_or_noise="res",
        ).to(DEVICE)

        # 构造一个临时的官方 Dataset 用于初始化 Trainer（因为里面必须计算长度等）
        dummy_dir = tempfile.mkdtemp(prefix="foundir_dummy_")
        lq_dir = os.path.join(dummy_dir, "LQ")
        gt_dir = os.path.join(dummy_dir, "GT")
        os.makedirs(lq_dir)
        os.makedirs(gt_dir)
        # 写入一张全黑占位图
        Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).save(os.path.join(lq_dir, "dummy.png"))
        shutil.copy(os.path.join(lq_dir, "dummy.png"), os.path.join(gt_dir, "dummy.png"))
        
        opt.dataroot = dummy_dir

        dummy_dataset = CombinedDataset(
            opt,
            image_size=1024,
            augment_flip=False,
            equalizeHist=True,
            crop_patch=False,
            generation=False,
            task=None,
        )

        trainer = Trainer(
            diffusion,
            dummy_dataset,
            opt,
            train_batch_size=1,
            num_samples=1,
            train_lr=2e-4,
            train_num_steps=100000,
            gradient_accumulate_every=2,
            ema_decay=0.995,
            amp=False,
            convert_image_to="RGB",
            results_folder=os.path.join(FOUNDIR_REPO, "premodel"),
            condition=True,
            save_and_sample_every=1000,
            num_unet=1,
        )

        # 加载权重
        trainer.load(2000)
        GLOBAL_FOUNDIR_TRAINER = trainer
        GLOBAL_FOUNDIR_TRAINER.model.eval()

        # 扫尾删除占位文件夹
        shutil.rmtree(dummy_dir, ignore_errors=True)

        FOUNDIR_READY = True
        logging.info("FoundIR init finished. Model loaded in VRAM.")
    except Exception as err:
        FOUNDIR_READY = False
        logging.error(f"FoundIR init failed: {repr(err)}")
        logging.error(traceback.format_exc())

def _find_best_output(out_root: str):
    candidates =[]
    for root, _, files in os.walk(out_root):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                candidates.append(os.path.join(root, f))

    if not candidates:
        return None

    best_img = None
    best_area = -1
    for p in candidates:
        try:
            img = Image.open(p).convert("RGB")
            area = img.size[0] * img.size[1]
            if area > best_area:
                best_area = area
                best_img = np.array(img).astype(np.uint8)
        except Exception:
            pass
    return best_img

def foundir_enhance(image: np.ndarray, step: int = 0, cam: str = "") -> np.ndarray:
    global FOUNDIR_READY, GLOBAL_FOUNDIR_TRAINER

    if not USE_FOUNDIR or not FOUNDIR_READY or GLOBAL_FOUNDIR_TRAINER is None:
        return image

    try:
        if image is None or not isinstance(image, np.ndarray):
            return image
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # 保证去除Alpha通道
        image = image[:, :, :3]
        h, w = image.shape[:2]

        with tempfile.TemporaryDirectory(prefix="foundir_run_") as tmpdir:
            # 1. 构造本次推理的短命 Dataset (利用 OS 内存缓存，I/O 耗时不到 1ms)
            dataroot = os.path.join(tmpdir, "dataset")
            lq_dir = os.path.join(dataroot, "LQ")
            gt_dir = os.path.join(dataroot, "GT")
            os.makedirs(lq_dir)
            os.makedirs(gt_dir)

            input_path = os.path.join(lq_dir, "frame.png")
            Image.fromarray(image).save(input_path)
            # 拷贝一份给 GT 目录以安抚 Dataset 强制校验逻辑
            shutil.copy(input_path, os.path.join(gt_dir, "frame.png"))

            opt = Namespace(
                dataroot=dataroot,
                phase="test",
                max_dataset_size=int(1e18),
                load_size=256,
                crop_size=256,
                direction="AtoB",
                preprocess="none",
                no_flip=True,
                meta=None,
                bsize=1,
            )

            dataset = CombinedDataset(
                opt,
                image_size=1024,
                augment_flip=False,
                equalizeHist=True,
                crop_patch=False,
                generation=False,
                task=None,
            )

            # 2. 挂载数据集并指定输出目录
            GLOBAL_FOUNDIR_TRAINER.sample_dataset = dataset
            out_root = os.path.join(tmpdir, "output")
            os.makedirs(out_root)
            GLOBAL_FOUNDIR_TRAINER.set_results_folder(out_root)

            # 3. 动态自适应切块 (修复小图片报错)
            crop_size = 512
            crop_stride = 256
            if h < crop_size or w < crop_size:
                crop_phase = "none"  # 图片太小，不开切块
            else:
                crop_phase = "im2overlap"

            # 4. 执行推理
            with torch.no_grad():
                GLOBAL_FOUNDIR_TRAINER.test(
                    last=True,
                    crop_phase=crop_phase,
                    crop_size=crop_size,
                    crop_stride=crop_stride,
                )

            # 5. 捞取结果
            out_array = _find_best_output(out_root)
            if out_array is not None:
                return out_array
            else:
                logging.error("FoundIR finished but no output generated. Fallback.")
                return image

    except Exception as err:
        logging.error(f"FoundIR enhance failed: {repr(err)}")
        # 打印完整的报错栈，方便排查后续任何其他问题
        logging.error(traceback.format_exc())
        return image