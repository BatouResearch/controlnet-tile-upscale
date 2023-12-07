# Cog Implementation of ControlNet 

This is an implementation of the [Diffusers ControlNet](https://huggingface.co/blog/controlnet) as a Cog model. [Cog](https://github.com/replicate/cog) packages machine learning models as standard containers.

First, download the controlnet / processor weights:

`cog run python script/download_weights`

Next, download your desired SD1.5 based weights to weights folder:

`cog run python`

followed by:

    >>> from diffusers import StableDiffusionPipeline
    >>> import torch

    >>> p = StableDiffusionPipeline.from_pretrained('SG161222/Realistic_Vision_V1.3', torch_dtype=torch.float16)

    Downloading (…)ain/model_index.json: 100%|███████████████████████████████████████████████████████████████████████████████| 577/577 [00:00<00:00, 64.4kB/s]
    Downloading (…)cheduler_config.json: 100%|███████████████████████████████████████████████████████████████████████████████| 341/341 [00:00<00:00, 33.2kB/s]
    Downloading (…)_checker/config.json: 100%|████████████████████████████████████████████████████████████████████████████| 4.89k/4.89k [00:00<00:00, 504kB/s]
    Downloading (…)cial_tokens_map.json: 100%|███████████████████████████████████████████████████████████████████████████████| 472/472 [00:00<00:00, 14.3kB/s]
    Downloading (…)_encoder/config.json: 100%|███████████████████████████████████████████████████████████████████████████████| 612/612 [00:00<00:00, 49.9kB/s]
    Downloading (…)rocessor_config.json: 100%|███████████████████████████████████████████████████████████████████████████████| 518/518 [00:00<00:00, 20.3kB/s]
    Downloading (…)tokenizer/merges.txt: 100%|█████████████████████████████████████████████████████████████████████████████| 525k/525k [00:00<00:00, 6.33MB/s]
    Downloading (…)okenizer_config.json: 100%|████████████████████████████████████████████████████████████████████████████████| 806/806 [00:00<00:00, 410kB/s]
    Downloading (…)998/unet/config.json: 100%|████████████████████████████████████████████████████████████████████████████████| 901/901 [00:00<00:00, 427kB/s]
    Downloading (…)b998/vae/config.json: 100%|████████████████████████████████████████████████████████████████████████████████| 548/548 [00:00<00:00, 295kB/s]
    Downloading (…)tokenizer/vocab.json: 100%|███████████████████████████████████████████████████████████████████████████| 1.06M/1.06M [00:00<00:00, 7.83MB/s]
    Downloading (…)on_pytorch_model.bin: 100%|██████████████████████████████████████████████████████████████████████████████| 335M/335M [00:03<00:00, 105MB/s]
    Downloading pytorch_model.bin: 100%|████████████████████████████████████████████████████████████████████████████████████| 492M/492M [00:03<00:00, 127MB/s]
    Downloading pytorch_model.bin: 100%|██████████████████████████████████████████████████████████████████████████████████| 1.22G/1.22G [00:09<00:00, 128MB/s]
    Downloading (…)on_pytorch_model.bin: 100%|███████████████████████████████████████████████████████████████████████████| 3.44G/3.44G [01:02<00:00, 54.8MB/s]
    Fetching 15 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [01:03<00:00,  4.23s/it]
    `text_config_dict` is provided which will be used to initialize `CLIPTextConfig`. The value `text_config["id2label"]` will be overriden.9<00:47, 60.3MB/s]

    >>> p.save_pretrained('weights')


Then, you can run predictions:

`cog predict -i image=@monkey.png -i prompt="monkey scuba diving" -i structure='canny'`

## Issues

- [ ] support aspect ratio from image (currently it is resized to a square?)
- [ ] safety results aren't checked (resulting in a black image?)
- [ ] ability to return processed control image(s)
- [ ] ability to send pre-processed control image(s)
- [ ] support for multiple control nets / images
- [ ] support for controlnet guidance scale