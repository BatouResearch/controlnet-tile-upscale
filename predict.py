import torch
import os
from typing import List
from RealESRGAN import RealESRGAN
import shutil
import time
from cog import BasePredictor, Input, Path
from diffusers.utils import load_image
from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LCMScheduler
)
from PIL import Image, ImageEnhance, ImageOps
import cv2
import numpy as np
import math

SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
}

SD15_WEIGHTS = "weights"
CONTROLNET_CACHE = "controlnet-cache"


class Predictor(BasePredictor):
    def setup(self):

        """Load the model into memory to make running multiple predictions efficient"""

        print("Loading pipeline...")
        st = time.time()

        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_CACHE,
            torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            SD15_WEIGHTS,
            torch_dtype=torch.float16,
            controlnet=controlnet
        ).to("cuda")

        self.ESRGAN_models = {}

        for scale in [2, 4]:
            self.ESRGAN_models[scale] = RealESRGAN("cuda", scale=scale)
            self.ESRGAN_models[scale].load_weights(
                f"weights/RealESRGAN_x{scale}.pth", download=False
            )

        self.pipe.unload_lora_weights()

        print("Setup complete in %f" % (time.time() - st))

    def resize_for_condition_image(self, input_image, resolution):
        scale = 2
        if (resolution == 2048) :
            init_w = 1024
        elif (resolution == 2560) :
            init_w = 1280
        elif (resolution == 3072):
            init_w = 1536
        elif resolution == 4096:
            init_w = 2048
        else:
            init_w = 1024
            scale = 4
        input_image = input_image.convert("RGB")
        W, H = input_image.size
        k = float(init_w) / min(H, W)
        H *= k
        W *= k
        H = int(round(H / 64.0)) * 64
        W = int(round(W / 64.0)) * 64
        img = input_image.resize((W, H), resample=Image.LANCZOS)
        model = self.ESRGAN_models[scale]
        img = model.predict(img)
        return img
    
    def calculate_brightness_factors(self, hdr_intensity):
        factors = [1.0] * 9
        if hdr_intensity > 0:
            factors = [1.0 - 0.9 * hdr_intensity, 1.0 - 0.7 * hdr_intensity, 1.0 - 0.45 * hdr_intensity,
                       1.0 - 0.25 * hdr_intensity, 1.0, 1.0 + 0.2 * hdr_intensity,
                       1.0 + 0.4 * hdr_intensity, 1.0 + 0.6 * hdr_intensity, 1.0 + 0.8 * hdr_intensity]
        return factors
    
    def pil_to_cv(self, pil_image):
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def adjust_brightness(self, cv_image, factor):
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        v = np.clip(v * factor, 0, 255).astype('uint8')
        adjusted_hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)

    def create_hdr_effect(self, original_image, hdr):
        # Convert PIL image to OpenCV format
        cv_original = self.pil_to_cv(original_image)
        
        brightness_factors = self.calculate_brightness_factors(hdr)
        images = [self.adjust_brightness(cv_original, factor) for factor in brightness_factors]

        merge_mertens = cv2.createMergeMertens()
        hdr_image = merge_mertens.process(images)
        hdr_image_8bit = np.clip(hdr_image*255, 0, 255).astype('uint8')
        hdr_image_color = cv2.cvtColor(hdr_image_8bit, cv2.COLOR_BGR2RGB)
        hdr_image_pil = Image.fromarray(hdr_image_color)

        return hdr_image_pil.convert("RGB")

    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        image = load_image("/tmp/image.png").convert("RGB")
        return image

    def set_tile(self, image, x, y, tile):
        image = image.convert("RGB")
        image.paste(tile.convert("RGB"), (x, y))
        return image

    def create_masks(self, image, tile_width, tile_height, row, col, pad):
        tx, ty = col, row
        tx2, ty2 = tx + tile_width, ty + tile_height

        mx, my = tx - pad, ty - pad
        mx2, my2 = tx + tile_width + pad, ty + tile_height + pad

        if tx == 0 or mx < 0: mx = 0
        if tx2 > image.width : tx2, mx2 = image.width, image.width
        elif mx2 > image.width: mx2 = image.width
        if ty == 0 or my < 0: my = 0
        if ty2 > image.height: ty2, my2 = image.height, image.height
        elif my2 > image.height: my2 = image.height

        mask = Image.new("L", (mx2 - mx, my2 - my), 0)
        mask_tile = Image.new("L", (tx2 - tx, ty2 - ty), 255)
        mask.paste(mask_tile, (abs(mx-tx), abs(ty-my)))

        return mask, image.crop((mx, my, mx2, my2)), mx, my, mx2, my2

    def create_seam_masks(self, image_width, image_height, tile_width, tile_height, rows, cols, is_horizontal):
        if is_horizontal:
            mask = Image.new("L", (image_width, image_height), 0)
            gradient = Image.linear_gradient("L").resize((tile_width, tile_height // 2))
            for yi in range(rows - 1):
                for xi in range(cols):
                    mask.paste(gradient, (xi * tile_width, yi * tile_height + tile_height // 2))
                    mask.paste(gradient.rotate(180), (xi * tile_width, (yi + 1) * tile_height))
        else:
            mask = Image.new("L", (image_width, image_height), 0)
            gradient = Image.linear_gradient("L").rotate(90).resize((tile_width // 2, tile_height))
            for xi in range(cols - 1):
                for yi in range(rows):
                    mask.paste(gradient, (xi * tile_width + tile_width // 2, yi * tile_height))
                    mask.paste(gradient.rotate(180), ((xi + 1) * tile_width, yi * tile_height))
        
        return mask

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Prompt for the model",
            default=None
        ),
        image: Path = Input(
            description="Control image for scribble controlnet", 
            default=None
        ),
        resolution: int = Input(
            description="Image resolution",
            default=2560,
            choices=[2048,2560,4096]
        ),
        resemblance: float = Input(
            description="Conditioning scale for controlnet",
            default=0.85,
            ge=0,
            le=1,
        ),
        creativity: float = Input(
            description="Denoising strength. 1 means total destruction of the original image",
            default=0.35,
            ge=0.1,
            le=1,
        ),
        hdr: float = Input(
            description="HDR improvement over the original image",
            default=0,
            ge=0,
            le=1,
        ),
        scheduler: str = Input(
            default="DDIM",
            choices=SCHEDULERS.keys(),
            description="Choose a scheduler.",
        ),
        steps: int = Input(
            description="Steps", default=8
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance",
            default=0,
            ge=0,
            le=30.0,
        ),
        seed: int = Input(
            description="Seed", default=None
        ),
        negative_prompt: str = Input(  # FIXME
            description="Negative prompt",
            default="teeth, tooth, open mouth, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, mutant",
        ),
        guess_mode: bool = Input(
            description="In this mode, the ControlNet encoder will try best to recognize the content of the input image even if you remove all prompts. The `guidance_scale` between 3.0 and 5.0 is recommended.",
            default=False,
        ),
        tile_size: int = Input(
            description="Size of partitions of the image. A 1/4 of the final resolution is recommended for optimal.",
            default=512,
            choices=[128, 256, 374, 512, 768, 1024, 1280]
        ),
        lora_sharpness_strength: float = Input(
            description="Strength of the image's sharpness. For it to be noticeable, it is recommended to use values between 2 and 5.",
            default=8,
            ge=-3.0,
            le=10.0,
        ),
        lora_details_strength: float = Input(
            description="Strength of the image's details",
            default=2.75,
            ge=-3.0,
            le=3.0,
        ),
    ) -> Path:
        
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        self.pipe.unload_lora_weights()

        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.load_lora_weights("lora/add_sharpness.safetensors", adapter_name="sharp")
        self.pipe.load_lora_weights("lora/add_detail.safetensors", adapter_name="detail")
        self.pipe.load_lora_weights("lora/more_details.safetensors", adapter_name="more")
        self.pipe.set_adapters(["sharp","detail","more"], adapter_weights=[lora_sharpness_strength, lora_details_strength, lora_details_strength])

        
        self.pipe.load_lora_weights("lora/pytorch_lora_weights.safetensors")
        self.pipe.fuse_lora()
        
        self.pipe.enable_xformers_memory_efficient_attention()
        
        generator = torch.Generator("cuda").manual_seed(seed)
        loaded_image = self.load_image(image)
        loaded_image = loaded_image.convert("RGB")
        control_image = self.resize_for_condition_image(loaded_image, resolution)
        final_image = self.create_hdr_effect(control_image, hdr)

        tile_width = tile_size
        tile_height=math.ceil(tile_size * (final_image.height/final_image.width))

        pad = int(tile_size/2)

        rows = math.ceil(final_image.height / tile_height)
        cols = math.ceil(final_image.width / tile_width)

        tiles = []
        for yi in range(rows):
            for xi in range(cols):
                if xi == 0:
                    tiles.append([])
                color = xi % 2 == 0
                if yi > 0 and yi % 2 != 0:
                    color = not color
                tiles[yi].append(color)

        for row in range(len(tiles)):
            for col in range(len(tiles[row])):
                if not tiles[row][col]:
                    continue
                
                mask, tile, mx, my, mx2, my2 = self.create_masks(final_image, tile_width, tile_height, row * tile_height, col * tile_width, pad)
    
                args = {
                    "prompt": prompt,
                    "image": tile,
                    "control_image": tile,
                    "mask_image": mask,
                    "strength": creativity,
                    "controlnet_conditioning_scale": resemblance,
                    "negative_prompt": negative_prompt,
                    "guidance_scale": guidance_scale,
                    "generator": generator,
                    "num_inference_steps": steps,
                    "guess_mode": guess_mode,
                }

                outputs = self.pipe(**args)
                processed_tile = outputs.images[0]
                final_image = self.set_tile(final_image, mx, my, processed_tile)

        for row in range(len(tiles)):
            for col in range(len(tiles[row])):
                if tiles[row][col]:
                    continue
                
                mask, tile, mx, my, mx2, my2 = self.create_masks(final_image, tile_width, tile_height, row * tile_height, col * tile_width, pad)
    
                args = {
                    "prompt": prompt,
                    "image": tile,
                    "control_image": tile,
                    "mask_image": mask,
                    "strength": creativity,
                    "controlnet_conditioning_scale": resemblance,
                    "negative_prompt": negative_prompt,
                    "guidance_scale": guidance_scale,
                    "generator": generator,
                    "num_inference_steps": steps,
                    "guess_mode": guess_mode,
                }

                outputs = self.pipe(**args)
                processed_tile = outputs.images[0]
                final_image = self.set_tile(final_image, mx, my, processed_tile)

        if resolution==4096:
            args["num_inference_steps"] = 10
            h_mask = self.create_seam_masks(final_image.width, final_image.height, tile_width, tile_height, rows, cols, True)
            v_mask = self.create_seam_masks(final_image.width, final_image.height, tile_width, tile_height, rows, cols, False)
            full_mask = Image.composite(h_mask, v_mask, Image.new("L", final_image.size, 127))
            for row in range(0,2):
                for col in range(0,2):
                    _, tile, mx, my, mx2, my2 = self.create_masks(final_image, 2048, math.ceil(2048 * (final_image.height/final_image.width)), row * 2048, col * 2048, 700)
                    mask = full_mask.crop((mx, my, mx2, my2))
                    
                    args["image"] = tile
                    args["control_image"] = tile
                    args["mask_image"] = mask

                    outputs = self.pipe(**args)
                    processed_tile = outputs.images[0]
                    new_image = self.set_tile(final_image, mx, my, processed_tile)
                    final_image = Image.composite(final_image, new_image, full_mask) 

        else:
            args["num_inference_steps"] = 15
            args["image"] = final_image
            args["control_image"] = final_image
            args["mask_image"] = Image.new("L", final_image.size, 255)

            outputs = self.pipe(**args)
            final_image = outputs.images[0]

        output_path = f"output.jpg"
        final_image.save(output_path, "jpeg")


        
        return Path(output_path)