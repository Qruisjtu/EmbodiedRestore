import logging
import os
import sys
import traceback
import torch
import torch.nn.functional as F
import numpy as np

USE_SD21 = True
READY = False

MODEL_ID = "/mnt/shared-storage-user/zhangjianbo/models/hf_models/sd2-1-base"

# Tuned prompts for VLA physical fidelity
PROMPT = "high quality, clear, sharp, realistic photography, clean robotic scene"
NEGATIVE_PROMPT = "blurry, distorted, artifacts, extra objects, wrong structure, hallucination, cartoon, painting"
STRENGTH = 0.2  # Kept at 0.2 as requested
GUIDANCE_SCALE = 5.0
STEPS = 20
SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

try:
    from diffusers import StableDiffusionImg2ImgPipeline
    DIFFUSERS_IMPORTED = True
except ImportError as e:
    DIFFUSERS_IMPORTED = False
    logging.warning(f"[sd21] diffusers not found. Error: {e}")

GLOBAL_SD21_PIPE = None
GLOBAL_PROMPT_EMBEDS = None
GLOBAL_NEGATIVE_PROMPT_EMBEDS = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def init_sd21():
    global READY, GLOBAL_SD21_PIPE
    global GLOBAL_PROMPT_EMBEDS, GLOBAL_NEGATIVE_PROMPT_EMBEDS

    if not USE_SD21 or not DIFFUSERS_IMPORTED:
        READY = False
        return

    try:
        logging.info("[sd21] Initializing SD2.1 Img2Img Pipeline on H200 (Optimized for Throughput)...")
        
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            local_files_only=True,
            safety_checker=None,
        ).to(DEVICE)

        # Disable CPU-blocking progress bars for multiprocessing
        pipe.set_progress_bar_config(disable=True)

        # Pre-encode text prompts to save CPU tokenization and GPU compute
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=PROMPT,
            device=DEVICE,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=NEGATIVE_PROMPT,
        )

        GLOBAL_PROMPT_EMBEDS = prompt_embeds
        GLOBAL_NEGATIVE_PROMPT_EMBEDS = negative_prompt_embeds

        # Fully loaded in H200 VRAM, no need to offload or delete anything
        GLOBAL_SD21_PIPE = pipe
        READY = True
        logging.info("[sd21] init finished. Pipeline fully loaded in VRAM.")
    except Exception as e:
        READY = False
        logging.error(f"[sd21] init failed: {repr(e)}")
        logging.error(traceback.format_exc())


def sd21_enhance(image: np.ndarray, step=0, cam="compressimg") -> np.ndarray:
    global READY, GLOBAL_SD21_PIPE

    if not READY or GLOBAL_SD21_PIPE is None:
        return image

    if image is None or not isinstance(image, np.ndarray):
        return image

    try:
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        image = image[:, :, :3]
        h_orig, w_orig = image.shape[:2]

        with torch.no_grad():
            # 1. Numpy HWC -> Tensor NCHW (Everything runs on GPU)
            img_tensor = torch.from_numpy(image).to(device=DEVICE, dtype=DTYPE)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0  # Range: [0, 1]

            # 2. Resize to 512x512 on GPU (Zero CPU overhead)
            img_tensor_resized = F.interpolate(
                img_tensor, 
                size=(512, 512), 
                mode='bilinear', 
                align_corners=False
            )

            generator = torch.Generator(device=DEVICE).manual_seed(SEED)
            
            # 3. Inference (output_type="pt" bypasses internal PIL conversions)
            out_tensor = GLOBAL_SD21_PIPE(
                image=img_tensor_resized,
                prompt_embeds=GLOBAL_PROMPT_EMBEDS,
                negative_prompt_embeds=GLOBAL_NEGATIVE_PROMPT_EMBEDS,
                strength=STRENGTH,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=STEPS,
                generator=generator,
                output_type="pt"  
            ).images  # Shape: (1, 3, 512, 512)

            # 4. Restore original resolution on GPU
            if out_tensor.shape[-2:] != (h_orig, w_orig):
                out_tensor = F.interpolate(
                    out_tensor, 
                    size=(h_orig, w_orig), 
                    mode='bilinear', 
                    align_corners=False
                )

            # 5. Tensor NCHW (GPU) -> Numpy HWC (CPU) (Return to memory)
            out_tensor = out_tensor.squeeze(0).permute(1, 2, 0)
            out_tensor = (out_tensor * 255.0).clamp(0, 255).to(torch.uint8)
            out_np = out_tensor.cpu().numpy()

        return out_np

    except Exception as e:
        logging.error(f"[sd21] enhance failed: {repr(e)}")
        logging.error(traceback.format_exc())
        return image