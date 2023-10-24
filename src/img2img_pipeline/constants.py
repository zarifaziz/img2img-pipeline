from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent.parent / "model_cache"

DATA_DIR = Path(__file__).parent.parent.parent / "data"

INPUT_IMAGE_DIR = DATA_DIR / "input_images"

OUTPUT_IMAGE_DIR = DATA_DIR / "output_images"

memory_efficient_compute = True

model_list = [
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2",
    "stabilityai/stable-diffusion-2-1",
    "CompVis/stable-diffusion-v1-4",
]

prompt_list = [
    "with a cat in the corner",
    "in the style of picasso",
    "45mm focus with a studio camera",
    "ghibli style, a fantasy landscape with castles",
]
