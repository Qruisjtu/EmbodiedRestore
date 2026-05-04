import logging
import os
import sys
import traceback
import torch
import numpy as np
from PIL import Image

USE_KANDINSKY = True
READY = False

PRIOR_PATH = "/mnt/shared-storage-user/zhangjianbo/models/kandinsky-2-2-prior"
DECODER_PATH = "/mnt/shared-storage-user/zhangjianbo/models/kandinsky-2-2-decoder"

PROMPT =  "high quality, clear, sharp, realistic photography, clean robotic scene"
NEGATIVE_PROMPT = "blurry, distorted, artifacts, extra objects, wrong structure"
STRENGTH = 0.15
GUIDANCE_SCALE = 2.0
STEPS = 25
SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

try:
    from diffusers import KandinskyV22PriorPipeline, KandinskyV22Img2ImgPipeline
    DIFFUSERS_IMPORTED = True
except ImportError as e:
    DIFFUSERS_IMPORTED = False
    logging.warning(f"[kandinsky] diffusers not found. USE_KANDINSKY will be disabled. Error: {e}")

GLOBAL_IMG2IMG_PIPE = None
GLOBAL_IMAGE_EMBEDS = None
GLOBAL_NEGATIVE_IMAGE_EMBEDS = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

def init_kandinsky():
    global READY, GLOBAL_IMG2IMG_PIPE, GLOBAL_IMAGE_EMBEDS, GLOBAL_NEGATIVE_IMAGE_EMBEDS

    if not USE_KANDINSKY or not DIFFUSERS_IMPORTED:
        logging.info("[kandinsky] USE_KANDINSKY=False or code not found, skip init.")
        READY = False
        return

    try:
        if not os.path.exists(PRIOR_PATH) or not os.path.exists(DECODER_PATH):
            raise FileNotFoundError("Kandinsky model paths not found.")

        logging.info("[kandinsky] Initializing Prior Pipeline to cache text embeddings...")
        

        prior_pipe = KandinskyV22PriorPipeline.from_pretrained(
            PRIOR_PATH,
            torch_dtype=DTYPE,
            local_files_only=True,
        ).to(DEVICE)


        generator = torch.Generator(device=DEVICE).manual_seed(SEED)
        with torch.no_grad():
            prior_out = prior_pipe(
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=STEPS,
                generator=generator,
            )
        
        GLOBAL_IMAGE_EMBEDS = prior_out.image_embeds
        GLOBAL_NEGATIVE_IMAGE_EMBEDS = prior_out.negative_image_embeds


        del prior_pipe
        torch.cuda.empty_cache()
        logging.info("[kandinsky] Prior embeddings cached. Prior model removed from VRAM.")


        logging.info("[kandinsky] Initializing Img2Img Pipeline...")
        img2img_pipe = KandinskyV22Img2ImgPipeline.from_pretrained(
            DECODER_PATH,
            torch_dtype=DTYPE,
            local_files_only=True,
        ).to(DEVICE)
        

        img2img_pipe.set_progress_bar_config(disable=True)

        GLOBAL_IMG2IMG_PIPE = img2img_pipe
        READY = True
        logging.info("[kandinsky] init finished. Model loaded in VRAM.")
        
    except Exception as e:
        READY = False
        logging.error(f"[kandinsky] init failed: {repr(e)}")
        logging.error(traceback.format_exc())

def kandinsky_enhance(image: np.ndarray, step=0, cam="compressimg") -> np.ndarray:
    global READY, GLOBAL_IMG2IMG_PIPE, GLOBAL_IMAGE_EMBEDS, GLOBAL_NEGATIVE_IMAGE_EMBEDS

    if not READY or GLOBAL_IMG2IMG_PIPE is None:
        return image

    if image is None or not isinstance(image, np.ndarray):
        return image

    try:
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)


        image = image[:, :, :3]
        h_orig, w_orig = image.shape[:2]


        pil_img = Image.fromarray(image).resize((768, 768), resample=Image.Resampling.LANCZOS)


        generator = torch.Generator(device=DEVICE).manual_seed(SEED)


        with torch.no_grad():
            out_pil = GLOBAL_IMG2IMG_PIPE(
                image=pil_img,
                image_embeds=GLOBAL_IMAGE_EMBEDS,
                negative_image_embeds=GLOBAL_NEGATIVE_IMAGE_EMBEDS,
                strength=STRENGTH,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=STEPS,
                height=768,
                width=768,
                generator=generator,
            ).images[0]


        if out_pil.size != (w_orig, h_orig):
            out_pil = out_pil.resize((w_orig, h_orig), resample=Image.Resampling.LANCZOS)

        return np.array(out_pil, dtype=np.uint8)

    except Exception as e:
        logging.error(f"[kandinsky] enhance failed: {repr(e)}")
        logging.error(traceback.format_exc())
        return image