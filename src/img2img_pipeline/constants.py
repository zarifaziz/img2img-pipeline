from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent.parent / "model_cache"

DATA_DIR = Path(__file__).parent.parent.parent / "data"

INPUT_IMAGE_DIR = DATA_DIR / "input_images"

OUTPUT_IMAGE_DIR = DATA_DIR / "output_images"

memory_efficient_compute = True

model_list = [
    "runwayml/stable-diffusion-v1-5",        # Best for artistic styles
    "stabilityai/stable-diffusion-2",         # Good balance
    "stabilityai/stable-diffusion-2-1",       # Latest, best quality
    "CompVis/stable-diffusion-v1-4",          # Creative/dreamy
    # Add more models here if desired:
    # "nitrosocke/Arcane-Diffusion",          # Arcane style
    # "prompthero/openjourney",               # Midjourney-like
]

prompt_list = [
    "with a cat in the corner",
    "in the style of picasso",
    "45mm focus with a studio camera",
    "ghibli style, a fantasy landscape with castles",
]
