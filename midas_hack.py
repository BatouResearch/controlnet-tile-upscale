# Controlnet_aux 0.0.3 doesn't support normals, this is a copy
# of the latest version of the file from the controlnet_aux repo
# FIXME(ja): remove this when controlnet_aux is released with normals

import cv2
import numpy as np
from PIL import Image
import torch
import os 

from huggingface_hub import hf_hub_download
from einops import rearrange
from controlnet_aux.midas.api import MiDaSInference
from controlnet_aux.util import HWC3

class MidasDetector:
    def __init__(self, model_type="dpt_hybrid", model_path=None):
        self.model = MiDaSInference(model_type=model_type, model_path=model_path)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        
    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, model_type="dpt_hybrid", filename=None, cache_dir=None):
        if pretrained_model_or_path == "lllyasviel/ControlNet":
            filename = filename or "annotator/ckpts/dpt_hybrid-midas-501f0c75.pt"
        else:
            filename = filename or "dpt_hybrid-midas-501f0c75.pt"

        if os.path.isdir(pretrained_model_or_path):
            model_path = os.path.join(pretrained_model_or_path, filename)
        else:
            model_path = hf_hub_download(pretrained_model_or_path, filename, cache_dir=cache_dir)

        return cls(model_type=model_type, model_path=model_path)
        
    def __call__(self, input_image, a=np.pi * 2.0, bg_th=0.1, depth_and_normal=False):
        
        input_type = "np"
        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)
            input_type = "pil"
            
        input_image = HWC3(input_image)
        image_depth = input_image
        with torch.no_grad():
            image_depth = torch.from_numpy(image_depth).float()
            if torch.cuda.is_available():
                image_depth = image_depth.cuda()
            image_depth = image_depth / 127.5 - 1.0
            image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
            depth = self.model(image_depth)[0]

            depth_pt = depth.clone()
            depth_pt -= torch.min(depth_pt)
            depth_pt /= torch.max(depth_pt)
            depth_pt = depth_pt.cpu().numpy()
            depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)

            depth_np = depth.cpu().numpy()
            if depth_and_normal:
                x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
                y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)
                z = np.ones_like(x) * a
                x[depth_pt < bg_th] = 0
                y[depth_pt < bg_th] = 0
                normal = np.stack([x, y, z], axis=2)
                normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
                normal_image = (normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        
        if input_type == "pil":
            depth_image = Image.fromarray(depth_image)
            depth_image = depth_image.convert("RGB")
            if depth_and_normal:
                normal_image = Image.fromarray(normal_image)
        
        if depth_and_normal:
            return depth_image, normal_image
        else:
            return depth_image
