import logging
import os
import sys
import traceback
import torch
import torch.nn.functional as F
import numpy as np

USE_SSD1B = True
READY = False

MODEL_DIR = "/mnt/shared-storage-user/zhangjianbo/models/ssd-1b-ms"

# Tuned prompts to prevent hallucination (removing "robotics scene")
PROMPT = "high quality, clear, sharp, realistic photography, clean robotic scene"
NEGATIVE_PROMPT = "blurry, distorted, artifacts, extra objects, wrong structure, hallucination, cartoon, painting"
STRENGTH = 0.2
GUIDANCE_SCALE = 5.0
STEPS = 20
SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

try:
    from diffusers import AutoPipelineForImage2Image
    DIFFUSERS_IMPORTED = True
except ImportError as e:
    DIFFUSERS_IMPORTED = False
    logging.warning(f"[ssd1b] diffusers not found. Error: {e}")

GLOBAL_SSD1B_PIPE = None
GLOBAL_PROMPT_EMBEDS = None
GLOBAL_NEGATIVE_PROMPT_EMBEDS = None
GLOBAL_POOLED_PROMPT_EMBEDS = None
GLOBAL_NEGATIVE_POOLED_PROMPT_EMBEDS = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def init_ssd1b():
    global READY, GLOBAL_SSD1B_PIPE
    global GLOBAL_PROMPT_EMBEDS, GLOBAL_NEGATIVE_PROMPT_EMBEDS
    global GLOBAL_POOLED_PROMPT_EMBEDS, GLOBAL_NEGATIVE_POOLED_PROMPT_EMBEDS

    if not USE_SSD1B or not DIFFUSERS_IMPORTED:
        READY = False
        return

    try:
        logging.info("[ssd1b] Initializing SSD-1B Img2Img Pipeline on H200 (Optimized for Throughput)...")
        
        # SSD-1B is a distilled SDXL model, fully compatible with SDXL pipelines
        pipe = AutoPipelineForImage2Image.from_pretrained(
            MODEL_DIR,
            torch_dtype=DTYPE,
            variant="fp16",
            local_files_only=True,
            use_safetensors=True,
        ).to(DEVICE)

        # Disable CPU-blocking progress bars for 64-process concurrency
        pipe.set_progress_bar_config(disable=True)

        # Pre-encode text prompts (SSD-1B shares SDXL's dual text-encoder architecture)
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=PROMPT,
            device=DEVICE,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=NEGATIVE_PROMPT,
        )

        GLOBAL_PROMPT_EMBEDS = prompt_embeds
        GLOBAL_NEGATIVE_PROMPT_EMBEDS = negative_prompt_embeds
        GLOBAL_POOLED_PROMPT_EMBEDS = pooled_prompt_embeds
        GLOBAL_NEGATIVE_POOLED_PROMPT_EMBEDS = negative_pooled_prompt_embeds

        # Leave everything in VRAM (H200 optimization)
        GLOBAL_SSD1B_PIPE = pipe
        READY = True
        logging.info("[ssd1b] init finished. Pipeline fully loaded in VRAM.")
    except Exception as e:
        READY = False
        logging.error(f"[ssd1b] init failed: {repr(e)}")
        logging.error(traceback.format_exc())


def ssd1b_enhance(image: np.ndarray, step=0, cam="compressimg") -> np.ndarray:
    global READY, GLOBAL_SSD1B_PIPE

    if not READY or GLOBAL_SSD1B_PIPE is None:
        return image

    if image is None or not isinstance(image, np.ndarray):
        return image

    try:
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        image = image[:, :, :3]
        h_orig, w_orig = image.shape[:2]

        with torch.no_grad():
            # 1. Numpy HWC -> Tensor NCHW [0, 1] mapped to GPU
            img_tensor = torch.from_numpy(image).to(device=DEVICE, dtype=DTYPE)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0

            # 2. Resize to 1024x1024 on GPU (Zero CPU overhead)
            img_tensor_resized = F.interpolate(
                img_tensor, 
                size=(1024, 1024), 
                mode='bilinear', 
                align_corners=False
            )

            generator = torch.Generator(device=DEVICE).manual_seed(SEED)
            
            # 3. Inference (output_type="pt" keeps output as GPU Tensor)
            out_tensor = GLOBAL_SSD1B_PIPE(
                image=img_tensor_resized,
                prompt_embeds=GLOBAL_PROMPT_EMBEDS,
                negative_prompt_embeds=GLOBAL_NEGATIVE_PROMPT_EMBEDS,
                pooled_prompt_embeds=GLOBAL_POOLED_PROMPT_EMBEDS,
                negative_pooled_prompt_embeds=GLOBAL_NEGATIVE_POOLED_PROMPT_EMBEDS,
                strength=STRENGTH,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=STEPS,
                generator=generator,
                output_type="pt"  
            ).images  # Shape: (1, 3, 1024, 1024)

            # 4. Restore original resolution on GPU
            if out_tensor.shape[-2:] != (h_orig, w_orig):
                out_tensor = F.interpolate(
                    out_tensor, 
                    size=(h_orig, w_orig), 
                    mode='bilinear', 
                    align_corners=False
                )

            # 5. Tensor NCHW (GPU) -> Numpy HWC (CPU) 
            out_tensor = out_tensor.squeeze(0).permute(1, 2, 0)
            out_tensor = (out_tensor * 255.0).clamp(0, 255).to(torch.uint8)
            out_np = out_tensor.cpu().numpy()

        return out_np

    except Exception as e:
        logging.error(f"[ssd1b] enhance failed: {repr(e)}")
        logging.error(traceback.format_exc())
        return image