from diffusers import ControlNetModel, DiffusionPipeline
import torch

SD15_WEIGHTS = "weights"
CONTROLNET_CACHE = "controlnet-cache"

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1e_sd15_tile", torch_dtype=torch.float16, cache_dir=CONTROLNET_CACHE
)
controlnet.save_pretrained(CONTROLNET_CACHE)

pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, cache_dir=SD15_WEIGHTS
)
pipe.save_pretrained(SD15_WEIGHTS)

