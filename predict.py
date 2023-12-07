import torch
import os
from typing import List
import numpy as np
from PIL import Image
import time

from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    StableDiffusionPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)
from controlnet_aux import (
    HEDdetector,
    OpenposeDetector,
    MLSDdetector,
    CannyDetector,
    LineartDetector,
)
from midas_hack import MidasDetector


AUX_IDS = {
    "canny"    : "lllyasviel/control_v11p_sd15_canny",
    "depth"    : "lllyasviel/control_v11f1p_sd15_depth",
    "normal"   : "lllyasviel/control_v11p_sd15_normalbae",
    "lineart"  : "lllyasviel/control_v11p_sd15_lineart",
    "scribble" : "lllyasviel/control_v11p_sd15_scribble",
}

SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "KLMS": LMSDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "UniPCMultistep": UniPCMultistepScheduler,
}

SD15_WEIGHTS = "weights"
CONTROLNET_CACHE = "controlnet-cache"
PROCESSORS_CACHE = "processors-cache"
MISSING_WEIGHTS = []

if not os.path.exists(CONTROLNET_CACHE) or not os.path.exists(PROCESSORS_CACHE):
    print(
        "controlnet weights missing, use `cog run python script/download_weights` to download"
    )
    MISSING_WEIGHTS.append("controlnet")

if not os.path.exists(SD15_WEIGHTS):
    print(
        "sd15 weights missing, use `cog run python` and then load and save_pretrained('weights')"
    )
    MISSING_WEIGHTS.append("sd15")


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        if len(MISSING_WEIGHTS) > 0:
            print("skipping setup... missing weights: ", MISSING_WEIGHTS)
            return

        print("Loading pipeline...")
        st = time.time()

        self.pipe = StableDiffusionPipeline.from_pretrained(
            SD15_WEIGHTS, torch_dtype=torch.float16, local_files_only=True
        ).to("cuda")

        self.controlnets = {}
        for name in AUX_IDS.keys():
            self.controlnets[name] = ControlNetModel.from_pretrained(
                os.path.join(CONTROLNET_CACHE, name),
                torch_dtype=torch.float16,
                local_files_only=True,
            ).to("cuda")

        self.canny = CannyDetector()

        # Depth + Normal
        self.midas = MidasDetector.from_pretrained(
            "lllyasviel/ControlNet", cache_dir=PROCESSORS_CACHE
        )

        self.hed = HEDdetector.from_pretrained(
            "lllyasviel/ControlNet", cache_dir=PROCESSORS_CACHE
        )

        self.lineart = LineartDetector.from_pretrained(
            "lllyasviel/Annotators", cache_dir=PROCESSORS_CACHE
        )

        print("Setup complete in %f" % (time.time() - st))

    def canny_preprocess(self, img):
        return self.canny(img)

    def depth_preprocess(self, img):
        return self.midas(img)

    def lineart_preprocess(self, img):
        return self.lineart(img)

    def normal_preprocess(self, img):
        return self.midas(img, depth_and_normal=True)[1]

    def scribble_preprocess(self, img):
        return self.hed(img, scribble=True)

    def build_pipe(
        self, inputs, low_threshold=100, high_threshold=200, guess_mode=False
    ):
        control_nets = []
        processed_control_images = []
        conditioning_scales = []

        for name, [image, conditioning_scale] in inputs.items():
            if image is None:
                continue
            control_nets.append(self.controlnets[name])
            img = Image.open(image)
            if name == "canny":
                img = self.canny_preprocess(img)
            else:
                img = getattr(self, "{}_preprocess".format(name))(img)

            processed_control_images.append(img)
            conditioning_scales.append(conditioning_scale)

        if len(control_nets) == 0:
            pipe = self.pipe
            kwargs = {}
        else:
            pipe = StableDiffusionControlNetPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                unet=self.pipe.unet,
                scheduler=self.pipe.scheduler,
                safety_checker=None,  # self.pipe.safety_checker,
                feature_extractor=self.pipe.feature_extractor,
                controlnet=control_nets,
            )
            kwargs = {
                "image": processed_control_images,
                "controlnet_conditioning_scale": conditioning_scales,
                "guess_mode": guess_mode,
            }

        return pipe, kwargs

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Prompt for the model"),
        canny_image: Path = Input(
            description="Control image for canny controlnet", default=None
        ),
        canny_conditioning_scale: float = Input(
            description="Conditioning scale for canny controlnet",
            default=1,
        ),
        depth_image: Path = Input(
            description="Control image for depth controlnet", default=None
        ),
        depth_conditioning_scale: float = Input(
            description="Conditioning scale for depth controlnet",
            default=1,
        ),
        lineart_image: Path = Input(
            description="Control image for hed controlnet", default=None
        ),
        lineart_conditioning_scale: float = Input(
            description="Conditioning scale for hed controlnet", default=1
        ),
        normal_image: Path = Input(
            description="Control image for normal controlnet", default=None
        ),
        normal_conditioning_scale: float = Input(
            description="Conditioning scale for normal controlnet",
            default=1,
        ),
        scribble_image: Path = Input(
            description="Control image for scribble controlnet", default=None
        ),
        scribble_conditioning_scale: float = Input(
            description="Conditioning scale for scribble controlnet",
            default=1,
        ),

        num_samples: int = Input(
            description="Number of samples (higher values may OOM)",
            ge=1,
            le=4,
            default=1,
        ),
        image_resolution: int = Input(
            description="Resolution of image (smallest dimension)",
            choices=[256, 512, 768],
            default=512,
        ),
        scheduler: str = Input(
            default="DDIM",
            choices=SCHEDULERS.keys(),
            description="Choose a scheduler.",
        ),
        steps: int = Input(description="Steps", default=20),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance",
            default=9.0,
            ge=0.1,
            le=30.0,
        ),
        seed: int = Input(description="Seed", default=None),
        eta: float = Input(
            description="Controls the amount of noise that is added to the input data during the denoising diffusion process. Higher value -> more noise",
            default=0.0,
        ),
        negative_prompt: str = Input(  # FIXME
            description="Negative prompt",
            default="Longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        ),
        # Only applicable when model type is 'canny'
        low_threshold: int = Input(
            description="[canny only] Line detection low threshold",
            default=100,
            ge=1,
            le=255,
        ),
        # Only applicable when model type is 'canny'
        high_threshold: int = Input(
            description="[canny only] Line detection high threshold",
            default=200,
            ge=1,
            le=255,
        ),
        guess_mode: bool = Input(
            description="In this mode, the ControlNet encoder will try best to recognize the content of the input image even if you remove all prompts. The `guidance_scale` between 3.0 and 5.0 is recommended.",
            default=False,
        ),
    ) -> List[Path]:
        if len(MISSING_WEIGHTS) > 0:
            raise Exception("missing weights")

        pipe, kwargs = self.build_pipe(
            {
                "canny": [canny_image, canny_conditioning_scale],
                "depth": [depth_image, depth_conditioning_scale],
                "lineart": [lineart_image, lineart_conditioning_scale],
                "normal": [normal_image, normal_conditioning_scale],
                "scribble": [scribble_image, scribble_conditioning_scale],
            },
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            guess_mode=guess_mode,
        )
        pipe.enable_xformers_memory_efficient_attention()
        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        output_paths = []

        if "image" in kwargs:
            img = kwargs["image"][0]
            scale = float(image_resolution) / (min(img.size))
            for i, sample in enumerate(kwargs["image"]):
                output_path = f"/tmp/processed-{i}.png"
                sample.save(output_path)
                output_paths.append(Path(output_path))

            def quick_rescale(dim):
                """quick rescale to a multiple of 64, as per original controlnet"""
                dim *= scale
                return int(np.round(dim / 64.0)) * 64

            width = quick_rescale(img.size[0])
            height = quick_rescale(img.size[1])
        else:
            width = height = image_resolution

        generator = torch.Generator("cuda").manual_seed(seed)

        outputs = pipe(
            prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            eta=eta,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_samples,
            generator=generator,
            **kwargs,
        )
        for i, sample in enumerate(outputs.images):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))
        return output_paths
