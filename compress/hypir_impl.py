import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# 让 Python 能 import 到 /mnt/shared-storage-user/zhangjianbo/models/HYPIR/HYPIR/...
HYPIR_REPO = "/mnt/shared-storage-user/zhangjianbo/models/HYPIR"
if HYPIR_REPO not in sys.path:
    sys.path.insert(0, HYPIR_REPO)

from HYPIR.enhancer.sd2 import SD2Enhancer

_MODEL = None
_DEVICE = None
_TO_TENSOR = transforms.ToTensor()
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def init_hypir(
    base_model_type='sd2',
    base_model_path='/mnt/shared-storage-user/zhangjianbo/models/hf_models/sd2-1-base',
    weight_path='/mnt/shared-storage-user/zhangjianbo/models/HYPIR/weights/HYPIR_sd2.pth',
    lora_rank=256,
    lora_modules="to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj",
    model_t=200,
    coeff_t=200,
    device='cuda',
):
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    _DEVICE = device

    model = SD2Enhancer(
        base_model_path=base_model_path,
        weight_path=weight_path,
        lora_modules=lora_modules.split(","),
        lora_rank=lora_rank,
        model_t=model_t,
        coeff_t=coeff_t,
        device=device,
    )

    print("[INFO] Start loading HYPIR models (in-process) ...")
    model.init_models()
    print("[INFO] HYPIR init done.")
    _MODEL = model
    return _MODEL

@torch.no_grad()
def hypir_enhance(
    image_array,
    fixed_prompt="make the scientific image clear and readable",
    scale_by="factor",
    upscale=4,
    target_longest_side=None,
    patch_size=512,
    stride=256,
):
    global _MODEL
    if _MODEL is None:
        init_hypir()

    if image_array is None:
        raise ValueError("image_array is None")

    if not isinstance(image_array, np.ndarray):
        image_array = np.array(image_array)

    if image_array.dtype != np.uint8:
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)

    if image_array.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape={image_array.shape}")

    lq_pil = Image.fromarray(image_array).convert("RGB")
    lq_tensor = _TO_TENSOR(lq_pil).unsqueeze(0)

    result_pil = _MODEL.enhance(
        lq=lq_tensor,
        prompt=fixed_prompt,
        scale_by=scale_by,
        upscale=upscale,
        target_longest_side=target_longest_side,
        patch_size=patch_size,
        stride=stride,
        return_type="pil",
    )[0]

    out_np = np.array(result_pil.convert("RGB"))
    return out_np

def _list_images_recursively(folder: Path):
    imgs = []
    for root, _, files in os.walk(folder):
        rootp = Path(root)
        for fn in files:
            ext = Path(fn).suffix.lower()
            if ext in IMG_EXTS:
                imgs.append(rootp / fn)
    imgs.sort(key=lambda x: str(x))
    return imgs

def hypir_enhance_all(
    input_root,
    output_root,
    fixed_prompt="make the scientific image clear and readable",
    scale_by="factor",
    upscale=4,
    target_longest_side=None,
    patch_size=512,
    stride=256,
    out_ext=".png",
):
    global _MODEL
    if _MODEL is None:
        init_hypir()

    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    subfolders = [p for p in sorted(input_root.iterdir()) if p.is_dir() and p.name.endswith("_8")]
    if not subfolders:
        raise RuntimeError(f"No *_8 folders found in {input_root}")

    for sub in subfolders:
        imgs = _list_images_recursively(sub)
        for ip in imgs:
            rel = ip.relative_to(sub)
            out_dir = output_root / sub.name / rel.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / (rel.stem + out_ext)

            img_np = np.array(Image.open(ip).convert("RGB"))
            out_np = hypir_enhance(
                img_np,
                fixed_prompt=fixed_prompt,
                scale_by=scale_by,
                upscale=upscale,
                target_longest_side=target_longest_side,
                patch_size=patch_size,
                stride=stride,
            )
            Image.fromarray(out_np).save(out_path)
