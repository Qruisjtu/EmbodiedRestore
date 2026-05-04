import logging
import os
import sys

import numpy as np
import torch
from PIL import Image

# ==== InstantIR PARAMETERS ====
USE_INSTANTIR = True
INSTANTIR_DEVICE = "cuda"
INSTANTIR_READY = False

INSTANTIR_REPO_DIR = "/mnt/shared-storage-user/zhangjianbo/models/InstantIR"
if INSTANTIR_REPO_DIR not in sys.path:
    sys.path.insert(0, INSTANTIR_REPO_DIR)

INSTANTIR_WEIGHTS_DIR = "/mnt/shared-storage-user/zhangjianbo/models/InstantIR/models"
INSTANTIR_DINO_PATH = "/mnt/shared-storage-user/zhangjianbo/models/InstantIR/dinov2-large"
INSTANTIR_SDXL_PATH = "/mnt/shared-storage-user/zhangjianbo/models/sdxl-base-1.0"


instantir_runner = {
    "pipe": None,
    "lcm_scheduler": None,
    "device": None,
    "dtype": None,
    "resize_img": None,
}
# ==============================


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def _instantir_default_prompt():
    return (
        "Photorealistic, highly detailed, hyper detailed photo - realistic maximum detail, 32k, "
        "ultra HD, extreme meticulous detailing, skin pore detailing, "
        "hyper sharpness, perfect without deformations, "
        "taken using a Canon EOS R camera, Cinematic, High Contrast, Color Grading."
    )


def _instantir_default_neg_prompt():
    return (
        "blurry, out of focus, unclear, depth of field, over-smooth, "
        "sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, "
        "dirty, messy, worst quality, low quality, frames, painting, illustration, drawing, art, "
        "watermark, signature, jpeg artifacts, deformed, lowres"
    )


def init_instantir():
    global instantir_runner, INSTANTIR_READY

    if not USE_INSTANTIR:
        logging.info("USE_INSTANTIR=False, skip InstantIR init.")
        INSTANTIR_READY = False
        return

    try:
        logging.info("Initializing InstantIR (real)...")

        if INSTANTIR_REPO_DIR not in sys.path:
            sys.path.insert(0, INSTANTIR_REPO_DIR)

        from infer import resize_img
        from schedulers.lcm_single_step_scheduler import LCMSingleStepScheduler
        from diffusers import DDPMScheduler
        from module.ip_adapter.utils import load_adapter_to_pipe
        from pipelines.sdxl_instantir import InstantIRPipeline

        device = torch.device(INSTANTIR_DEVICE if torch.cuda.is_available() else "cpu")
        dtype = torch.float32

        pipe = InstantIRPipeline.from_pretrained(
            INSTANTIR_SDXL_PATH,
            torch_dtype=dtype,
        )

        adapter_path = os.path.join(INSTANTIR_WEIGHTS_DIR, "adapter.pt")
        aggregator_path = os.path.join(INSTANTIR_WEIGHTS_DIR, "aggregator.pt")

        assert os.path.exists(adapter_path), f"adapter.pt not found: {adapter_path}"
        assert os.path.exists(aggregator_path), f"aggregator.pt not found: {aggregator_path}"
        assert os.path.exists(INSTANTIR_DINO_PATH), f"DINO path not found: {INSTANTIR_DINO_PATH}"
        assert os.path.exists(INSTANTIR_SDXL_PATH), f"SDXL path not found: {INSTANTIR_SDXL_PATH}"

        load_adapter_to_pipe(
            pipe,
            adapter_path,
            INSTANTIR_DINO_PATH,
            use_clip_encoder=False,
        )

        lora_alpha = pipe.prepare_previewers(INSTANTIR_WEIGHTS_DIR)
        logging.info(f"InstantIR previewer LoRA alpha = {lora_alpha}")
        pipe.to(device=device, dtype=dtype)

        pipe.scheduler = DDPMScheduler.from_pretrained(INSTANTIR_SDXL_PATH, subfolder="scheduler")
        lcm_scheduler = LCMSingleStepScheduler.from_config(pipe.scheduler.config)

        state_dict = torch.load(aggregator_path, map_location="cpu")
        pipe.aggregator.load_state_dict(state_dict)
        pipe.aggregator.to(device=device, dtype=dtype)

        instantir_runner = {
            "pipe": pipe,
            "lcm_scheduler": lcm_scheduler,
            "device": device,
            "dtype": dtype,
            "resize_img": resize_img,
        }

        INSTANTIR_READY = True
        logging.info("InstantIR init finished (real).")

    except Exception as err:
        INSTANTIR_READY = False
        instantir_runner["pipe"] = None
        instantir_runner["lcm_scheduler"] = None
        logging.error(f"InstantIR init failed: {repr(err)}")
        logging.error("Fallback to raw image.")


def instantir_enhance(image: np.ndarray, step: int = 0, cam: str = "") -> np.ndarray:
    global instantir_runner, INSTANTIR_READY

    if not USE_INSTANTIR:
        return image
    if not INSTANTIR_READY or instantir_runner is None:
        return image

    try:
        if image is None or not isinstance(image, np.ndarray):
            return image
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        pipe = instantir_runner["pipe"]
        lcm_scheduler = instantir_runner["lcm_scheduler"]
        device = instantir_runner["device"]
        resize_img = instantir_runner["resize_img"]

        lq_pil = Image.fromarray(image[:, :, :3], mode="RGB")
        lq_pil, out_size = resize_img(lq_pil.convert("RGB"), width=None, height=None)

        prompt = _instantir_default_prompt()
        neg_prompt = _instantir_default_neg_prompt()

        generator = torch.Generator(device=device).manual_seed(42)

        result = pipe(
            prompt=[prompt],
            image=[lq_pil],
            num_inference_steps=8,
            generator=generator,
            timesteps=None,
            negative_prompt=[neg_prompt],
            guidance_scale=0.0,
            previewer_scheduler=lcm_scheduler,
            preview_start=0.0,
            control_guidance_end=1.0,
        ).images[0]

        result = result.resize([out_size[0], out_size[1]], Image.BILINEAR)
        out = np.array(result).astype(np.uint8)
        return out

    except Exception as err:
        logging.error(f"InstantIR enhance failed: {repr(err)}")
        return image
