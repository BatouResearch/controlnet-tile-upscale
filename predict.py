import torch
import os
from typing import List
from RealESRGAN import RealESRGAN
import shutil
import time
from cog import BasePredictor, Input, Path
from diffusers.utils import load_image
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    StableDiffusionPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
)
from PIL import Image, ImageEnhance
import cv2
import numpy as np

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

        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
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

        print("Setup complete in %f" % (time.time() - st))

    def resize_for_condition_image(self, input_image, resolution):
        scale = 2
        if (resolution == 2048) :
            init_w = 1024
        elif (resolution == 2560) :
            init_w = 1280
        elif (resolution == 3072):
            init_w = 1536
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
        hdr_image_pil = Image.fromarray(cv2.cvtColor(hdr_image_8bit, cv2.COLOR_BGR2RGB))

        return hdr_image_pil

    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

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
            default=2048,
            choices=[2048,2560]
        ),
        resemblance: float = Input(
            description="Conditioning scale for controlnet",
            default=0.5,
            ge=0,
            le=1,
        ),
        creativity: float = Input(
            description="Denoising strength. 1 means total destruction of the original image",
            default=0.5,
            ge=0,
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
            description="Steps", default=20
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance",
            default=7.0,
            ge=0.1,
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
    ) -> List[Path]:
        
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        self.pipe.scheduler = SCHEDULERS[scheduler].from_config(self.pipe.scheduler.config)
        generator = torch.Generator("cuda").manual_seed(seed)
        loaded_image = self.load_image(image)
        control_image = self.resize_for_condition_image(loaded_image, resolution)
        final_image = self.create_hdr_effect(control_image, hdr)
        
        args = {
            "prompt": prompt,
            "image": final_image,
            "control_image": final_image,
            "strength": creativity,
            "controlnet_conditioning_scale": resemblance,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": steps,
            "guess_mode": guess_mode,
        }
        
        w,h = control_image.size
        
        if (w*h > 2560*2560):
            self.pipe.enable_vae_tiling()
        else:
            self.pipe.disable_vae_tiling()
        
        self.pipe.enable_xformers_memory_efficient_attention()
        outputs = self.pipe(**args)
        output_paths = []
        for i, sample in enumerate(outputs.images):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))
        return output_paths