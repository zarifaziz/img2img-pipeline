import gc
import time
from typing import Optional

import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    SchedulerMixin,
    StableDiffusionImg2ImgPipeline,
)
from loguru import logger
from PIL import Image

from .constants import CACHE_DIR, memory_efficient_compute
from .interfaces import ModelInterface

torch.backends.cuda.matmul.allow_tf32 = True


class Img2ImgModel(ModelInterface):
    def __init__(
        self,
        model_path: str,
        scheduler_config: Optional[dict] = None,
        scheduler: SchedulerMixin = DPMSolverMultistepScheduler,
        strength: float = 0.75,
        guidance_scale: float = 7.5,
        cache_dir: str = str(CACHE_DIR),
    ) -> None:
        model_load_time = time.time()
        if torch.cuda.is_available():
            logger.info("Using CUDA")
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                torch_dtype=torch.float16,
            ).to("cuda")
        else:
            logger.info("Using CPU")
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_path, cache_dir=cache_dir
            )
        model_load_time = round((time.time() - model_load_time) * 1000)

        logger.info(f"Loaded model in {model_load_time} ms")

        if scheduler_config is None:
            self.pipe.scheduler = scheduler.from_config(self.pipe.scheduler.config)
        else:
            self.pipe.scheduler = scheduler.from_config(scheduler_config)

        # applying memory efficient settings
        if memory_efficient_compute:
            self.pipe.enable_attention_slicing()
            self.pipe.enable_sequential_cpu_offload()
            self.pipe.enable_xformers_memory_efficient_attention()

        self.generator = torch.Generator("cuda").manual_seed(0)
        self.strength = strength
        self.guidance_scale = guidance_scale

    def predict(
        self, image: Image.Image, prompt: str = "in the style of picasso"
    ) -> Image.Image:
        with torch.no_grad():
            images = self.pipe(
                prompt=prompt,
                image=image,
                generator=self.generator,
                strength=self.strength,
                guidance_scale=self.guidance_scale,
            ).images
        logger.info("clearing cache")
        torch.cuda.empty_cache()
        gc.collect()

        return images[0]
