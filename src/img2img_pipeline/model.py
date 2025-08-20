import gc
import time
from typing import Optional

import numpy as np
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
        strength: float = 0.3,  # Reduced from 0.75 - less noise means more similar to original
        guidance_scale: float = 12.0,  # Increased from 7.5 - better prompt adherence
        num_inference_steps: int = 20,  # Number of denoising steps
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
        elif torch.backends.mps.is_available():
            logger.info("Using MPS (Apple Silicon)")
            try:
                # Use float32 for MPS to avoid black image issues
                self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    model_path,
                    cache_dir=cache_dir,
                    torch_dtype=torch.float32,
                ).to("mps")
            except Exception as e:
                logger.warning(f"MPS failed, falling back to CPU: {e}")
                self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    model_path, 
                    cache_dir=cache_dir,
                    torch_dtype=torch.float32,
                )
        else:
            logger.info("Using CPU")
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_path, 
                cache_dir=cache_dir,
                torch_dtype=torch.float32,
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
            try:
                # Only enable CPU offload if we have CUDA available and accelerate is properly configured
                if torch.cuda.is_available():
                    self.pipe.enable_sequential_cpu_offload()
                    logger.info("Enabled sequential CPU offload")
            except Exception as e:
                logger.warning(f"Could not enable sequential CPU offload: {e}")
            
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory efficient attention")
            except Exception as e:
                logger.warning(f"Could not enable xformers memory efficient attention: {e}")

        # Set up generator based on available device
        if torch.cuda.is_available():
            self.generator = torch.Generator("cuda").manual_seed(0)
            self.device = "cuda"
        elif torch.backends.mps.is_available() and str(self.pipe.device).startswith("mps"):
            # MPS doesn't support Generator, use CPU generator
            self.generator = torch.Generator().manual_seed(0)
            self.device = "mps"
        else:
            self.generator = torch.Generator().manual_seed(0)
            self.device = "cpu"
            
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps

    def predict(
        self, image: Image.Image, prompt: str = "in the style of picasso"
    ) -> Image.Image:
        logger.info(f"Running prediction with prompt: '{prompt}' on device: {self.device}")
        logger.info(f"Input image size: {image.size}, mode: {image.mode}")
        
        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.info("Converted input image to RGB")
        
        # Validate image size (diffusion models work best with certain sizes)
        width, height = image.size
        if width < 64 or height < 64:
            logger.warning(f"Image very small ({width}x{height}), results may be poor")
        
        with torch.no_grad():
            images = self.pipe(
                prompt=prompt,
                image=image,
                generator=self.generator,
                strength=self.strength,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
            ).images
        
        output_image = images[0]
        logger.info(f"Output image size: {output_image.size}, mode: {output_image.mode}")
        
        # Check if the image is all black (debugging)
        img_array = np.array(output_image)
        if img_array.max() == 0:
            logger.warning("Generated image appears to be all black!")
        else:
            logger.info(f"Image pixel range: {img_array.min()} to {img_array.max()}")
        
        logger.info("clearing cache")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

        return output_image
