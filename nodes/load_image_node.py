import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import hashlib
from .wtz_node_helpers import pillow

class LoadImagePathNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image_path": ("STRING", {"default": "/root/autodl-tmp/diffuser-pipe/input/test1.jpg"})}}

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image_from_path"

    def load_image_from_path(self, image_path):
        img = pillow(Image.open, image_path)
        output_images = []
        output_masks = []
        w, h = None, None
        excluded_formats = ['MPO']
        for i in ImageSequence.Iterator(img):
            i = pillow(ImageOps.exif_transpose, i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            if image.size[0] != w or image.size[1] != h:
                continue
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))
        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]
        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(cls, image_path):
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image_path):
        if not image_path:
            return f"Invalid image file: {image_path}"
        return True 