from abc import ABC, abstractmethod
import io
import math
import os
import subprocess
import time
from compress.instantir_impl import init_instantir, instantir_enhance
from compress.dfpir_impl2 import init_dfpir, dfpir_enhance
from compress.swinir_impl2 import init_swinir, swinir_enhance
from compress.realesrgan_impl2 import init_realesrgan, realesrgan_enhance
from compress.foundir_impl2 import init_foundir, foundir_enhance
from compress.hypir_impl import init_hypir, hypir_enhance
from compress.bird_impl2 import init_bird, bird_enhance
from compress.sdxl_impl2 import init_sdxl, sdxl_enhance
from compress.sd21_impl2 import init_sd21, sd21_enhance
from compress.playground_impl2 import init_playground, playground_enhance
from compress.ssd1b_impl2 import init_ssd1b, ssd1b_enhance
from compress.kandinsky_impl2 import init_kandinsky, kandinsky_enhance
from compress.varformer_impl2 import init_varformer, varformer_enhance
from pathlib import Path
import importlib
import cv2 as cv
import imageio
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from compressai.zoo import (
    bmshj2018_factorized,
    bmshj2018_hyperprior,
    cheng2020_anchor,
    cheng2020_attn,
    mbt2018,
    mbt2018_mean,
)
import uuid, shutil
from img_distort import distort_lib
from img_distort.distort_lib import STRONG_PARAMS
CAMERARESOLUTION = 256 * 256 
#Below are parameters for EmbodiedComp
# MODELLIST = [
#     'mbt2018_mean',
#     'mbt2018',
#     'cheng2020_attn',
#     'bmshj2018_factorized',
#     'bmshj2018_hyperprior',
#     'cheng2020_anchor',
#     'instantir',
#     'jpeg',
#     'webp',
#     'lichpcm',
#     'dcae',
#     'rwkv',
# ]
# MODELQUALITY = {
#     'mbt2018_mean': [1, 2, 3, 4, 5, 6, 7, 8],
#     'mbt2018': [1, 2, 3, 4, 5, 6, 7, 8],
#     'cheng2020_attn': [1, 2, 3, 4, 5, 6],
#     'bmshj2018_factorized': [1, 2, 3, 4, 5, 6, 7, 8],
#     'bmshj2018_hyperprior': [1, 2, 3, 4, 5, 6, 7, 8],
#     'cheng2020_anchor': [1, 2, 3, 4, 5, 6],
#     'jpeg': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90],
#     'webp': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90],
#     'lichpcm': [1, 2, 3, 4, 5, 6],
#     'dcae': [1, 2, 3, 4, 5, 6],
#     'rwkv': [1, 2, 3, 4, 5, 6],
# }
# DOWNSAMPLEGRADE = {
#     '1':8,
#     '7/8':7,
#     '3/4':6,
#     '5/8':5,
#     '1/2':4,
#     '3/8':3,
#     '1/4':2,
#     '1/8':1
#     }
# MODELQUADOWN={
#     "mbt2018":[[1,'1/4'],[1,'1/2'],[1,'3/4'],[1,'1'],[2,'1']],
#     "cheng2020_anchor":[[1,'1/4'],[1,'1/2'],[1,'3/4'],[1,'1'],[2,'1']],
#     "bmshj2018_hyperprior":[[1,'1/4'],[1,'1/2'],[1,'3/4'],[2,'1']],
#     "jpeg":[[5,'1/8'],[10,'1/8'],[20,'1/8'],[40,'1/8'],[10,'1/4']],
#     "webp":[[5,'1/8'],[5,'1/4'],[10,'3/8'],[10,'1/2'],[20,'1/2'],[40,'1/2']],
#     'lichpcm':[[1,'1/8'],[1,'1/4'],[1,'1/2'],[1,'3/4'],[1,'1'],[3,'1']],
#     'dcae':[[1,'1/8'],[1,'1/4'],[1,'1/2'],[1,'3/4'],[1,'1'],[3,'1']],
#     'rwkv':[[1,'1/8'],[1,'1/4'],[1,'1/2'],[1,'3/4'],[1,'1'],[3,'1']],
# }
assert torch.cuda.is_available(),print("cuda is not available")
device = "cuda" 
CODEC_DIR = Path(__file__).parent / 'codec'


def cal_psnr(a: torch.Tensor, b: torch.Tensor):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)
def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)
def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )

def scale_n_over_base(img, is_down: bool, n: int, base: int = 8):
    """
    将图像按 n/base 比例进行上下采样
    img
    """
    if n <= 0 or n > base:
        raise ValueError(f"n must be in 1..{base}, but got n={n}")

    scale = n / base

    if not is_down:  
        scale = 1 / scale


    h, w = img.shape[:2]

    # 目标大小
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))


    if is_down:
        interp = cv.INTER_AREA    
    else:
        interp = cv.INTER_LINEAR    

    resized = cv.resize(img, (new_w, new_h), interpolation=interp)
    return resized

class Codec(ABC):
    def __init__(self, quality):
        self.quality = quality

    @abstractmethod
    def compress(self, image_array)->tuple:
        pass

class Distorter(Codec):
    def __init__(self,quality=1,type=0):
        super().__init__(quality)
        noise_names = ["additive_gaussian_noise", "color_component_noise", "spatially_correlated_noise", "masked_noise", "high_frequency_noise", "impulse_noise", "quantization_noise", "gaussian_blur", "image_denoising", "jpeg_compression", "jpeg2000_compression", "jpeg_transmission_errors", "jpeg2000_transmission_errors", "non_eccentricity_pattern_noise", "local_block_distortions", "mean_shift", "contrast_change", "color_saturation_change", "multiplicative_gaussian_noise", "comfort_noise", "lossy_compression_noisy_images", "color_quantization_dither", "chromatic_aberrations", "sparse_sampling_reconstruction", "ringing_artifacts"]
        assert type < len(noise_names) , print(f"type should be in range 0-{len(noise_names)-1}")
        self.type = noise_names[type]
    def compress(self, image_array):
        distorter = distort_lib.ImageDistortion(image_array)
        add_noise = getattr(distorter, self.type)
        img_dis = add_noise(**STRONG_PARAMS[self.type])
        return img_dis,24

class Enhance(Codec):
    def __init__(self, quality, type=None):
        super().__init__(quality)
        self.type = type
        self.init()

    def init(self):
        pass

    def compress(self, image_array):
        raise NotImplementedError

class InstantIREnhance(Enhance):
    def __init__(self, quality):
        super().__init__(quality, "instantir")

    def init(self):
        init_instantir()

    def compress(self, image_array):
        if image_array is None:
            return image_array, 24

        if not isinstance(image_array, np.ndarray):
            image_array = np.array(image_array)

        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        # x = transforms.ToTensor()(image_array).unsqueeze(0).to(device)
        rec_np = instantir_enhance(image_array, step=0, cam="compressimg")
        # rec_img =(rec_np*255).clamp(0, 255).byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
        return rec_np, 24

class SwinIREnhance(Enhance):
    def __init__(self, quality):
        super().__init__(quality, "swinir")

    def init(self):
        init_swinir()

    def compress(self, image_array):

        if image_array is None:
            return image_array, 24

        if not isinstance(image_array, np.ndarray):
            image_array = np.array(image_array)

        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        # x = transforms.ToTensor()(image_array).unsqueeze(0).to(device)
        rec_np = swinir_enhance(image_array, step=0, cam="compressimg")
        # rec_img =(rec_np*255).clamp(0, 255).byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
        return rec_np, 24


class DFPIREnhance(Enhance):
    def __init__(self, quality):
        super().__init__(quality, "dfpir")

    def init(self):
        init_dfpir()

    def compress(self, image_array):
        if image_array is None:
            return image_array, 24

        if not isinstance(image_array, np.ndarray):
            image_array = np.array(image_array)

        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)

        rec_np = dfpir_enhance(image_array, step=0, cam="compressimg")

        return rec_np, 24


class RealESRGANEnhance(Enhance):
    def __init__(self, quality):
        super().__init__(quality, "realesrgan")

    def init(self):
        init_realesrgan()

    def compress(self, image_array):
        if image_array is None:
            return image_array, 24

        if not isinstance(image_array, np.ndarray):
            image_array = np.array(image_array)

        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        # x = transforms.ToTensor()(image_array).unsqueeze(0).to(device)
        rec_np = realesrgan_enhance(image_array, step=0, cam="compressimg")
        # rec_img =(rec_np*255).clamp(0, 255).byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
        return rec_np, 24


class FoundIREnhance(Enhance):
    def __init__(self, quality):
        super().__init__(quality, "foundir")

    def init(self):
        init_foundir()

    def compress(self, image_array):
        if image_array is None:
            return image_array, 24

        if not isinstance(image_array, np.ndarray):
            image_array = np.array(image_array)

        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)

        rec_np = foundir_enhance(image_array, step=0, cam="compressimg")

        return rec_np, 24

class VarFormerEnhance(Enhance):
    def __init__(self, quality):
        super().__init__(quality, "varformer")

    def init(self):
        init_varformer()

    def compress(self, image_array):
        if image_array is None:
            return image_array, 24

        if not isinstance(image_array, np.ndarray):
            image_array = np.array(image_array)

        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)

        rec_np = varformer_enhance(image_array, step=0, cam="compressimg")
        return rec_np, 24

class HypirEnhance(Enhance):
    def __init__(self, quality):
        super().__init__(quality, "hypir")

    def init(self):
        init_hypir()

    def compress(self, image_array):
        import numpy as np

        if image_array is None:
            return image_array, 24

        if not isinstance(image_array, np.ndarray):
            image_array = np.array(image_array)

        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)

        rec_np = hypir_enhance(
            image_array,
            fixed_prompt="make the scientific image clear and readable",
            patch_size=512,
            stride=256,
        )
        return rec_np, 24

class BirdEnhance(Enhance):
    def __init__(self, quality):
        super().__init__(quality, "bird")

    def init(self):
        init_bird()

    def compress(self, image_array):
        if image_array is None:
            return image_array, 24

        if not isinstance(image_array, np.ndarray):
            image_array = np.array(image_array)

        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)

        rec_np = bird_enhance(image_array, step=0, cam="compressimg")
        return rec_np, 24

class SDXLEnhance(Enhance):
    def __init__(self, quality):
        super().__init__(quality, "sdxl")

    def init(self):
        init_sdxl()

    def compress(self, image_array):
        rec_np = sdxl_enhance(image_array, step=0, cam="compressimg")
        return rec_np, 24

class SD21Enhance(Enhance):
    def __init__(self, quality):
        super().__init__(quality, "sd21")

    def init(self):
        init_sd21()

    def compress(self, image_array):
        rec_np = sd21_enhance(image_array, step=0, cam="compressimg")
        return rec_np, 24

class KandinskyEnhance(Enhance):
    def __init__(self, quality):
        super().__init__(quality, "kandinsky")

    def init(self):
        init_kandinsky()

    def compress(self, image_array):
        rec_np = kandinsky_enhance(image_array, step=0, cam="compressimg")
        return rec_np, 24

class PlaygroundEnhance(Enhance):
    def __init__(self, quality):
        super().__init__(quality, "playground")

    def init(self):
        init_playground()

    def compress(self, image_array):
        rec_np = playground_enhance(image_array, step=0, cam="compressimg")
        return rec_np, 24

class SSD1BEnhance(Enhance):
    def __init__(self, quality):
        super().__init__(quality, "ssd1b")

    def init(self):
        init_ssd1b()

    def compress(self, image_array):
        rec_np = ssd1b_enhance(image_array, step=0, cam="compressimg")
        return rec_np, 24

class PillowCodec(Codec):
    def __init__(self, quality, fmt):
        super().__init__(quality)
        self.fmt = fmt

    def compress(self, image_array):
        img = Image.fromarray(image_array)
        tmp = io.BytesIO()
        img.save(tmp, format=self.fmt, quality=self.quality)
        tmp.seek(0)
        filesize = tmp.getbuffer().nbytes
        # bpp = filesize * 8 / (img.size[0] * img.size[1])
        bpp = filesize * 8 / CAMERARESOLUTION
        rec = Image.open(tmp)
        return np.array(rec, dtype=np.uint8), bpp


class JpegCodec(PillowCodec):
    def __init__(self, quality):
        super().__init__(quality, 'jpeg')


class WebpCodec(PillowCodec):
    def __init__(self, quality):
        super().__init__(quality, 'webp')


class CompressAICodec(Codec):
    def __init__(self, quality, model_name):
        super().__init__(quality)
        model_map = {
  'hypir': hypir_enhance,
            'mbt2018': mbt2018,
            'mbt2018_mean': mbt2018_mean,
            'cheng2020_attn': cheng2020_attn,
            'bmshj2018_factorized': bmshj2018_factorized,
            'bmshj2018_hyperprior': bmshj2018_hyperprior,
            'cheng2020_anchor': cheng2020_anchor,
        }
        self.net = model_map[model_name](
            quality=quality, metric='mse', pretrained=True, progress=True
        ).eval().to(device)

    def compress(self, image_array):
        tensor = transforms.ToTensor()(image_array).unsqueeze(0).to(device)
        with torch.no_grad():
            out_net = self.net(tensor)
        rec = out_net['x_hat'].clamp_(0, 1).squeeze().cpu()
        rec_np = (rec * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        bpp = self._compute_bpp(out_net)
        return rec_np, bpp

    def _compute_bpp(self, out_net):
        size = out_net['x_hat'].size()
        # num_pixels = size[0] * size[2] * size[3]
        num_pixels = CAMERARESOLUTION
        return sum(
            torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
            for likelihoods in out_net['likelihoods'].values()
        ).item()

class AICODECEXT(Codec):
    def __init__(self,quality,model_name):
        super().__init__(quality)
        dir_map = {
            'lichpcm':'LIC-HPCM',
            'dcae' :'DCAE',
            'rwkv': 'RwkvCompress'
        }
        checkpoint_map = {
            'lichpcm': {
                1: '0.0018.pth.tar',
                2: '0.0035.pth.tar',
                3: '0.0067.pth.tar',
                4: '0.013.pth.tar',
                5: '0.025.pth.tar',
                6: '0.0483.pth.tar',
            },
            'dcae': {
                1: '0.0018checkpoint_best.pth.tar',
                2: '0.0035checkpoint_best.pth.tar',
                3: '0.0067checkpoint_best.pth.tar',
                4: '0.013checkpoint_best.pth.tar',
                5: '0.025checkpoint_best.pth.tar',
                6: '0.05checkpoint_best.pth.tar',
            },
            'rwkv':{
                1: 'lalic-q1.pth',
                2: 'lalic-q2.pth',
                3: 'lalic-q3.pth',
                4: 'lalic-q4.pth',
                5: 'lalic-q5.pth',
                6: 'lalic-q6.pth',
            }
        }
        self.codec_dir = CODEC_DIR / dir_map[model_name]
        assert quality in list(checkpoint_map[model_name].keys()) , f"quality not in range{list(checkpoint_map[model_name].keys())}"
        self.checkpoint = self.codec_dir / f'checkpoints/{checkpoint_map[model_name][quality]}'

    
            
class LICHPCMCodec(AICODECEXT):
    def __init__(self, quality):
        """
        quality: LICHPCM provide pretrained checkpoint list on https://github.com/lyq133/LIC-HPCM .
        """
        super().__init__(quality,'lichpcm')
        import sys
        sys.path.append(str(self.codec_dir))
        self.model_name = 'HPCM_Base'
        self.num = 60
        # from codec.LIC_HPCM.src.models import HPCM_Base
        net = importlib.import_module(f'.{self.model_name}', 'codec.LIC-HPCM.src.models').HPCM
        # net = HPCM_Base.HPCM
        checkpoint = torch.load(self.checkpoint, map_location=device)
        model = net()
        model.eval()
        model.load_state_dict(checkpoint, strict=True)
        model.update(self._get_scale_table(0.12, 64, self.num))
        self.model = model.to(device)
    def _get_scale_table(self,min, max, levels):
        """Returns table of logarithmically scales."""
        return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

    def compress(self, image_array):
        

        x = transforms.ToTensor()(image_array).unsqueeze(0).to(device)
        h,w = x.shape[2],x.shape[3]
        x,padding = pad(x,256)
        assert x.size(2) == x.size(3) == 256 , f"assert receive 256*256 image,but receive {x.size(2)}*{x.size(3)}"
        with torch.no_grad():
                out_enc = self.model.compress(x)
        with torch.no_grad():
                out_dec = self.model.decompress(out_enc["strings"], out_enc["shape"])
        out_dec["x_hat"]= crop(out_dec["x_hat"], padding)
        bpp_img = sum(len(s) for s in out_enc["strings"]) * 8.0 / (x.size(2) * x.size(3))
        rec_img = (out_dec["x_hat"] * 255).clamp(0, 255).byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
        return rec_img, bpp_img
        
class DCAECodec(AICODECEXT):
    """https://github.com/CVL-UESTC/DCAE"""
    def __init__(self, quality):
        super().__init__(quality,'dcae')
        net = importlib.import_module('.dcae', 'codec.DCAE.models').DCAE
        checkpoint = torch.load(self.checkpoint, map_location=device)
        dictory = {}
        model = net()
        model.eval()
        for k, v in checkpoint["state_dict"].items():
            dictory[k.replace("module.", "")] = v
        model.load_state_dict(state_dict=dictory)
        model.update()
        self.model = model.to(device)
    def compress(self, image_array):
        x = transforms.ToTensor()(image_array).unsqueeze(0).to(device)
        x,padding = pad(x,256)
        with torch.no_grad():
            out_enc = self.model.compress(x)
        with torch.no_grad():
            out_dec = self.model.decompress(out_enc["strings"], out_enc["shape"])
        out_dec["x_hat"]= crop(out_dec["x_hat"], padding)
        bpp_img = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / (x.size(2) * x.size(3))
        
        rec_img = (out_dec["x_hat"] * 255).clamp(0, 255).byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
        return rec_img, bpp_img

class RwkvCodec(AICODECEXT):
    """https://github.com/sjtu-medialab/RwkvCompress"""
    def __init__(self, quality):
        super().__init__(quality,'rwkv')
        LALIC = importlib.import_module('.lalic', 'codec.RwkvCompress.models').LALIC

        checkpoint = torch.load(self.checkpoint, map_location=device)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        state_dict = {self._rename_key(k): v for k, v in state_dict.items()}
        
        model = LALIC.from_state_dict(state_dict)
        model.eval()
        model.update()
        self.model = model.to(device)
    def _rename_key(self, key: str) -> str:
        """Rename state_dict key from RwkvCompress/eval.py."""
        # Deal with modules trained with DataParallel
        if key.startswith("module."):
            key = key[7:]

        # ResidualBlockWithStride: 'downsample' -> 'skip'
        if ".downsample." in key:
            return key.replace("downsample", "skip")

        # EntropyBottleneck: nn.ParameterList to nn.Parameters
        if "entropy_bottleneck" in key:
            key = key.replace("entropy_bottleneck.matrices.", "entropy_bottleneck._matrix")
            key = key.replace("entropy_bottleneck.biases.", "entropy_bottleneck._bias")
            key = key.replace("entropy_bottleneck.factors.", "entropy_bottleneck._factor")

        return key

    def compress(self, image_array):

        x = transforms.ToTensor()(image_array).unsqueeze(0).to(device)
        h, w = x.size(2), x.size(3)
        
        p = 256 
        x, padding = pad(x, p)

        with torch.no_grad():
            out_enc = self.model.compress(x)
            out_dec = self.model.decompress(out_enc["strings"], out_enc["shape"])

        out_dec["x_hat"] = crop(out_dec["x_hat"], padding)

        num_pixels = CAMERARESOLUTION
        bpp_img = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

        rec_img = (out_dec["x_hat"] * 255).clamp(0, 255).byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
        return rec_img, bpp_img

class COMPRESSIMG:
    def __init__(self, model='mbt2018_mean', quality=1 , downsample:str = '1',dis_type=0):
        self.changenet(model, quality , downsample, dis_type)
        self.initcache()

    def changenet(self, model='mbt2018_mean', quality=1, downsample:str = '1',dis_type=0):
        codec_map = {
                'jpeg': JpegCodec,
                'webp': WebpCodec,
                'instantir': InstantIREnhance,
                'swinir': SwinIREnhance,
                'dfpir': DFPIREnhance,
                'realesrgan': RealESRGANEnhance,
                'varformer': VarFormerEnhance,
                'hypir': HypirEnhance,
                'bird': BirdEnhance,
                'foundir': FoundIREnhance,
                # 'hevc': HEVCCodec,
                # 'vvc': VVCCodec,
                'sdxl': SDXLEnhance,
                'sd21': SD21Enhance,
                'playground': PlaygroundEnhance,
                'kandinsky': KandinskyEnhance,
                'ssd1b': SSD1BEnhance,
                'lichpcm': LICHPCMCodec,
                'dcae': DCAECodec,
                'rwkv': RwkvCodec,
                'distort': Distorter
            }
        if model in codec_map.keys():
            if model == 'distort':
                self.net = codec_map[model](quality,dis_type)
            else:
                self.net = codec_map[model](quality)
        else:
            self.net = CompressAICodec(quality, model)
        assert downsample in list(DOWNSAMPLEGRADE.keys()),print(f"downsample range should in list{(DOWNSAMPLEGRADE.keys())}")
        self.downsample = downsample
        self.initcache()

    def compress(self, imagearray, saveimg=False):
        downsample = self.downsample
        if downsample and downsample != '1':
            imagearray = scale_n_over_base(imagearray,True,DOWNSAMPLEGRADE[downsample],8)

        rec_np, bpp = self.net.compress(imagearray)
        
        
        if downsample and downsample != '1':
            # rec_np = transforms.ToTensor()(rec_np).unsqueeze(0)
            # rec_np = crop(rec_np, padding)
            # rec_np = rec_np.squeeze(0).permute(1,2,0).numpy()
            rec_np = scale_n_over_base(rec_np,False,DOWNSAMPLEGRADE[downsample],8)

        if saveimg:
            self.rawimage.append(Image.fromarray(imagearray))
            self.recimage.append(Image.fromarray(rec_np))

        self.bpp.append(bpp)
        return rec_np, bpp

    def storevideo(self, path='data/videos/compress', fps=10):
        os.makedirs(path, exist_ok=True)
        nowtime = time.time()
        imageio.mimwrite(os.path.join(f"{path}/raw_{nowtime:.0f}.mp4"), self.rawimage, fps=fps)
        imageio.mimwrite(os.path.join(f"{path}/rec_{nowtime:.0f}.mp4"), self.recimage, fps=fps)
        self.recimage, self.rawimage, self.diff = [], [], []

    def initcache(self):
        self.recimage, self.rawimage, self.diff = [], [], []
        self.bpp = []

    def calculate_psnr(self, original_image_array, reconstructed_image_array):
        original_tensor = transforms.ToTensor()(original_image_array).unsqueeze(0).to(device)
        reconstructed_tensor = transforms.ToTensor()(reconstructed_image_array).unsqueeze(0).to(device)
        return cal_psnr(original_tensor, reconstructed_tensor)
class Pipeline(Codec):
    def __init__(self, codecs: list=None):
        self.codecs = []
        if codecs:
            for c in codecs:
                self.add(c)
    
    def add(self,codec: Codec):
        if not isinstance(codec, Codec):
            raise TypeRrror(f"Expected a Codec instance, but got {type(codec)}")
        self.codecs.append(codec)
        return self

    def compress(self, image_array,saveimg=False):
        result = image_array
        h,w = result.shape[:2]
        # print(h,w)
        for i, codec in enumerate(self.codecs):
            # try:
            preresult = result
            result,_= codec.compress(result)
            # except Exception as e:
            #     raise RuntimeError(
            #         f"Error at step {i} ({codec}): {e}"
            #     ) from e
        
        result = cv.resize(result, (w, h), interpolation=cv.INTER_LINEAR)
        return result,_
    
def model_test_all():
    import matplotlib.pyplot as plt
    # Load the example image 'a.png'
    try:
        image = Image.open(CODEC_DIR /'test'/ 'a.png')
    except FileNotFoundError:
        print(f"Error: Could not find 'a.png' in the '{CODEC_DIR}' directory.")
        exit()
        
    imagearray = np.array(image, dtype=np.uint8)


    all_models_bpp = {}
    all_models_psnr = {}

    for model_name in MODELLIST:
        print(f"--- Testing {model_name} Codec ---")
        bpps = []
        psnrs = []
        for downsample in ['1','1/2']:
            for quality in MODELQUALITY[model_name]:
                
                compressor = COMPRESSIMG(model=model_name, quality=quality,downsample=downsample)
                rec_np, bpp = compressor.compress(imagearray)
                psnr = compressor.calculate_psnr(imagearray, rec_np)
                bpps.append(bpp)
                psnrs.append(psnr)
                print(f"  Quality: {quality}, Downsample:{downsample}, BPP: {bpp:.4f}, PSNR: {psnr:.2f} dB")
            
        all_models_bpp[model_name] = bpps
        all_models_psnr[model_name] = psnrs

    plt.figure(figsize=(12, 8))
     
    for model_name in MODELLIST:
        if all_models_bpp[model_name] and all_models_psnr[model_name]:
            half = len(all_models_bpp[model_name]) // 2
            line1, = plt.plot(all_models_bpp[model_name][:half], all_models_psnr[model_name][:half], marker='o', label=model_name)
            base_color = line1.get_color()
            plt.plot(all_models_bpp[model_name][half:], all_models_psnr[model_name][half:],color=base_color, linestyle='--', marker='x', label=model_name + ' (downsampled)')

    plt.xlabel("Bits Per Pixel (BPP)")
    plt.ylabel("Peak Signal-to-Noise Ratio (PSNR) [dB]")
    plt.title("PSNR-BPP Curve for Different Compression Models")
    plt.legend()
    plt.grid(True)
    plt.show()
def single_psnr_bpp():
    import matplotlib.pyplot as plt
    import mplcursors
    # Load the example image 'a.png'
    try:
        image = Image.open(CODEC_DIR /'test'/ 'a.png')
    except FileNotFoundError:
        print(f"Error: Could not find 'a.png' in the '{CODEC_DIR}' directory.")
        exit()
        
    imagearray = np.array(image, dtype=np.uint8)

    plt.figure(figsize=(12, 8))
    all_downsample_bpp = {}
    all_downsample_psnr = {}

    model_name = 'vvc'
    print(f"--- Testing {model_name} Codec ---")
    full_list = ['1','7/8','3/4','1/2','5/8','3/8','1/4','1/8']
    half_list =['1','3/4','1/2','1/4']
    for downsample in full_list:
        bpps = []
        psnrs = []
        for quality in MODELQUALITY[model_name]: 
            compressor = COMPRESSIMG(model=model_name, quality=quality,downsample=downsample)
            rec_np, bpp = compressor.compress(imagearray)
            psnr = compressor.calculate_psnr(imagearray, rec_np)
            bpps.append(bpp)
            psnrs.append(psnr)
            print(f"  Quality: {quality}, Downsample:{downsample}, BPP: {bpp:.4f}, PSNR: {psnr:.2f} dB")
        all_downsample_bpp[downsample] = bpps
        all_downsample_psnr[downsample] = psnrs
    
    
        scatter = plt.scatter(all_downsample_bpp[downsample], all_downsample_psnr[downsample], marker='x', label=f"{model_name}_{downsample}")
        cursor = mplcursors.cursor(scatter, hover=True)

    plt.xlabel("Bits Per Pixel (BPP)")
    plt.ylabel("Peak Signal-to-Noise Ratio (PSNR) [dB]")
    plt.title("PSNR-BPP Curve for Compression Models")
    plt.legend()
    plt.grid(True)
    plt.show()
if __name__ == '__main__':
    model_test_all()


class HypirEnhance(Enhance):
    def __init__(self, quality):
        super().__init__(quality, "hypir")

    def init(self):
        init_hypir()

    def compress(self, image_array):
        import numpy as np

        if image_array is None:
            return image_array, 24

        if not isinstance(image_array, np.ndarray):
            image_array = np.array(image_array)

        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)

        rec_np = hypir_enhance(
            image_array,
            fixed_prompt="make the scientific image clear and readable",
            patch_size=512,
            stride=256,
        )
        return rec_np, 24
