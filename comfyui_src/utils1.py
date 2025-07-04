import json
import os
import time

import torch
import numpy as np
import torch.nn.functional as F

import torch

# Import the DeepCacheSDHelper
from DeepCache import DeepCacheSDHelper
import cv2
import numpy as np
from PIL import Image
import torch
import gc

import os
import sys
import json
import hashlib
import traceback
import math
import time
import random
import logging
import time

from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo

import numpy as np
import safetensors.torch

from PIL import Image
import os
import io

from transformers.dynamic_module_utils import get_imports
from unittest.mock import patch
from typing import Union

#def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
from typing import List, Union
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import importlib
# from compel import Compel, ReturnedEmbeddingsType

from diffusers import DiffusionPipeline
import torch
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import itertools
from spandrel import ModelLoader, ImageModelDescriptor
# import checkpoint_pickle
import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from io import BytesIO
import base64

def resize_image_to_long_side(image, target_long_side):
    # 获取图像的宽和高
    width, height = image.size
    
    # 确定短边的长度
    if width > height:
        long_side = width
        scale_factor = target_long_side / width  # 缩放比例
    else:
        long_side = height
        scale_factor = target_long_side / height  # 缩放比例

    # 计算新的宽和高，按比例缩放
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # 调整图像大小并保持长宽比
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    return resized_image

def resize_image_to_short_side(image, target_short_side):
    # 获取图像的宽和高
    width, height = image.size
    
    # 确定短边的长度
    if width < height:
        short_side = width
        scale_factor = target_short_side / width  # 缩放比例
    else:
        short_side = height
        scale_factor = target_short_side / height  # 缩放比例

    # 计算新的宽和高，按比例缩放
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # 调整图像大小并保持长宽比
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    return resized_image

def load_torch_file(ckpt, safe_load=False, device=None):
    if device is None:
        device = torch.device("cpu")
    if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    else:
        if safe_load:
            if not 'weights_only' in torch.load.__code__.co_varnames:
                logging.warning("Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely.")
                safe_load = False
        if safe_load:
            pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
        else:
            pl_sd = torch.load(ckpt, map_location=device, pickle_module=checkpoint_pickle)
        if "global_step" in pl_sd:
            logging.debug(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    return sd

def tiled_scale_multidim(samples, function, tile=(64, 64), overlap = 8, upscale_amount = 4, out_channels = 3, output_device="cpu", pbar = None):
    dims = len(tile)
    output = torch.empty([samples.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), samples.shape[2:])), device=output_device)

    for b in range(samples.shape[0]):
        s = samples[b:b+1]
        out = torch.zeros([s.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), s.shape[2:])), device=output_device)
        out_div = torch.zeros([s.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), s.shape[2:])), device=output_device)

        for it in itertools.product(*map(lambda a: range(0, a[0], a[1] - overlap), zip(s.shape[2:], tile))):
            s_in = s
            upscaled = []

            for d in range(dims):
                pos = max(0, min(s.shape[d + 2] - overlap, it[d]))
                l = min(tile[d], s.shape[d + 2] - pos)
                s_in = s_in.narrow(d + 2, pos, l)
                upscaled.append(round(pos * upscale_amount))
            ps = function(s_in).to(output_device)
            mask = torch.ones_like(ps)
            feather = round(overlap * upscale_amount)
            for t in range(feather):
                for d in range(2, dims + 2):
                    m = mask.narrow(d, t, 1)
                    m *= ((1.0/feather) * (t + 1))
                    m = mask.narrow(d, mask.shape[d] -1 -t, 1)
                    m *= ((1.0/feather) * (t + 1))

            o = out
            o_d = out_div
            for d in range(dims):
                o = o.narrow(d + 2, upscaled[d], mask.shape[d + 2])
                o_d = o_d.narrow(d + 2, upscaled[d], mask.shape[d + 2])

            o += ps * mask
            o_d += mask

            if pbar is not None:
                pbar.update(1)

        output[b:b+1] = out/out_div
    return output

def tiled_scale(samples, function, tile_x=64, tile_y=64, overlap = 8, upscale_amount = 4, out_channels = 3, output_device="cpu", pbar = None):
    return tiled_scale_multidim(samples, function, (tile_y, tile_x), overlap, upscale_amount, out_channels, output_device, pbar)

def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    if filter_keys:
        out = {}
    else:
        out = state_dict
    for rp in replace_prefix:
        replace = list(map(lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp):])), filter(lambda a: a.startswith(rp), state_dict.keys())))
        for x in replace:
            w = state_dict.pop(x[0])
            out[x[1]] = w
    return out

def load_upscale_model(model_path):
    sd = load_torch_file(model_path, safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = state_dict_prefix_replace(sd, {"module.":""})
    out = ModelLoader().load_from_state_dict(sd).eval()
    if not isinstance(out, ImageModelDescriptor):
        raise Exception("Upscale model must be a single-image model.")
    return out

class ProgressBar:
    def __init__(self, total):
        global PROGRESS_BAR_HOOK
        self.total = total
        self.current = 0
        self.hook = PROGRESS_BAR_HOOK

    def update_absolute(self, value, total=None, preview=None):
        if total is not None:
            self.total = total
        if value > self.total:
            value = self.total
        self.current = value
        if self.hook is not None:
            self.hook(self.current, self.total, preview)

    def update(self, value):
        self.update_absolute(self.current + value)
        
def module_size(module):
    module_mem = 0
    sd = module.state_dict()
    for k in sd:
        t = sd[k]
        module_mem += t.nelement() * t.element_size()
    return module_mem

def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    return math.ceil((height / (tile_y - overlap))) * math.ceil((width / (tile_x - overlap)))

def fixed_get_imports(filename: Union[str, os.PathLike]) -> List[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    try:
        imports.remove("flash_attn")
    except:
        print(f"No flash_attn import to remove")
        pass
    return imports

def bislerp(samples, width, height):
    def slerp(b1, b2, r):
        '''slerps batches b1, b2 according to ratio r, batches should be flat e.g. NxC'''
        
        c = b1.shape[-1]

        #norms
        b1_norms = torch.norm(b1, dim=-1, keepdim=True)
        b2_norms = torch.norm(b2, dim=-1, keepdim=True)

        #normalize
        b1_normalized = b1 / b1_norms
        b2_normalized = b2 / b2_norms

        #zero when norms are zero
        b1_normalized[b1_norms.expand(-1,c) == 0.0] = 0.0
        b2_normalized[b2_norms.expand(-1,c) == 0.0] = 0.0

        #slerp
        dot = (b1_normalized*b2_normalized).sum(1)
        omega = torch.acos(dot)
        so = torch.sin(omega)

        #technically not mathematically correct, but more pleasing?
        res = (torch.sin((1.0-r.squeeze(1))*omega)/so).unsqueeze(1)*b1_normalized + (torch.sin(r.squeeze(1)*omega)/so).unsqueeze(1) * b2_normalized
        res *= (b1_norms * (1.0-r) + b2_norms * r).expand(-1,c)

        #edge cases for same or polar opposites
        res[dot > 1 - 1e-5] = b1[dot > 1 - 1e-5] 
        res[dot < 1e-5 - 1] = (b1 * (1.0-r) + b2 * r)[dot < 1e-5 - 1]
        return res
    
    def generate_bilinear_data(length_old, length_new, device):
        coords_1 = torch.arange(length_old, dtype=torch.float32, device=device).reshape((1,1,1,-1))
        coords_1 = torch.nn.functional.interpolate(coords_1, size=(1, length_new), mode="bilinear")
        ratios = coords_1 - coords_1.floor()
        coords_1 = coords_1.to(torch.int64)
        
        coords_2 = torch.arange(length_old, dtype=torch.float32, device=device).reshape((1,1,1,-1)) + 1
        coords_2[:,:,:,-1] -= 1
        coords_2 = torch.nn.functional.interpolate(coords_2, size=(1, length_new), mode="bilinear")
        coords_2 = coords_2.to(torch.int64)
        return ratios, coords_1, coords_2

    orig_dtype = samples.dtype
    samples = samples.float()
    n,c,h,w = samples.shape
    h_new, w_new = (height, width)
    
    #linear w
    ratios, coords_1, coords_2 = generate_bilinear_data(w, w_new, samples.device)
    coords_1 = coords_1.expand((n, c, h, -1))
    coords_2 = coords_2.expand((n, c, h, -1))
    ratios = ratios.expand((n, 1, h, -1))

    pass_1 = samples.gather(-1,coords_1).movedim(1, -1).reshape((-1,c))
    pass_2 = samples.gather(-1,coords_2).movedim(1, -1).reshape((-1,c))
    ratios = ratios.movedim(1, -1).reshape((-1,1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h, w_new, c).movedim(-1, 1)

    #linear h
    ratios, coords_1, coords_2 = generate_bilinear_data(h, h_new, samples.device)
    coords_1 = coords_1.reshape((1,1,-1,1)).expand((n, c, -1, w_new))
    coords_2 = coords_2.reshape((1,1,-1,1)).expand((n, c, -1, w_new))
    ratios = ratios.reshape((1,1,-1,1)).expand((n, 1, -1, w_new))

    pass_1 = result.gather(-2,coords_1).movedim(1, -1).reshape((-1,c))
    pass_2 = result.gather(-2,coords_2).movedim(1, -1).reshape((-1,c))
    ratios = ratios.movedim(1, -1).reshape((-1,1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h_new, w_new, c).movedim(-1, 1)
    return result.to(orig_dtype)

def lanczos(samples, width, height):
    images = [Image.fromarray(np.clip(255. * image.movedim(0, -1).cpu().numpy(), 0, 255).astype(np.uint8)) for image in samples]
    images = [image.resize((width, height), resample=Image.Resampling.LANCZOS) for image in images]
    images = [torch.from_numpy(np.array(image).astype(np.float32) / 255.0).movedim(-1, 0) for image in images]
    result = torch.stack(images)
    return result.to(samples.device, samples.dtype)

def common_upscale(samples, width, height, upscale_method, crop):
    if crop == "center":
        old_width = samples.shape[3]
        old_height = samples.shape[2]
        old_aspect = old_width / old_height
        new_aspect = width / height
        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        s = samples[:,:,y:old_height-y,x:old_width-x]
    else:
        s = samples

    if upscale_method == "bislerp":
        return bislerp(s, width, height)
    elif upscale_method == "lanczos":
        return lanczos(s, width, height)
    else:
        return torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)
        
def latent_upscale(latent, upscale_method, width, height, crop):
    if width == 0 and height == 0:
        s = latent
    else:
        s = latent.clone()
        if width == 0:
            height = max(64, height)
            width = max(64, round(latent.shape[2] * height / latent.shape[1]))
        elif height == 0:
            width = max(64, width)
            height = max(64, round(latent.shape[1] * width / latent.shape[2]))
        else:
            width = max(64, width)
            height = max(64, height)
        s = common_upscale(latent.unsqueeze(0), width // 8, height // 8, upscale_method, crop)
    return s.squeeze()
