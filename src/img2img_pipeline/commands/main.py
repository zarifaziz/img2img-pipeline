from dotenv import load_dotenv
import time
import warnings

import typer
from loguru import logger

load_dotenv()

from ..model import Img2ImgModel
from ..pipeline import DiffusionAllImagesPipeline, DiffusionSingleImagePipeline

app = typer.Typer()


@app.command("run_all_images_pipeline")
def run_all_images_pipeline():
    """
    Run the pipeline for all images in the input_images directory
    
    Uses a randomly chosen prompt and model for each image
    """
    logger.info("Running pipeline")
    run_time = time.time()
    pipeline = DiffusionAllImagesPipeline(model_class=Img2ImgModel)
    pipeline.run()
    run_time = round((time.time() - run_time) * 1000)
    logger.info(f"Finished running pipeline in {run_time} ms")


@app.command("run_single_image_pipeline")
def run_single_image_pipeline(
    filename: str,
    prompt: str = "in the style of picasso",
    model: str = "stabilityai/stable-diffusion-2",
    strength: float = 0.3,
    guidance_scale: float = 12.0,
    num_inference_steps: int = 20,
    output_filename: str = None,
):
    """
    Run the pipeline for a single image.

    Args:
        filename (str): The name of the file to process.
        prompt (str, optional): The style prompt for the image. 
            Defaults to "in the style of picasso".
        model (str, optional): The model to use for the diffusion. 
            Defaults to "stabilityai/stable-diffusion-2".
        strength (float, optional): How much to transform the image (0.0-1.0).
            Lower values preserve more of the original. Defaults to 0.3.
        guidance_scale (float, optional): How closely to follow the prompt.
            Higher values follow the prompt more strictly. Defaults to 12.0.
        num_inference_steps (int, optional): Number of denoising steps.
            More steps = higher quality but slower. Defaults to 20.
        output_filename (str, optional): Custom output filename. If None,
            auto-generates based on input name and parameters.
    """
    logger.info("Running pipeline")
    run_time = time.time()
    pipeline = DiffusionSingleImagePipeline(model_class=Img2ImgModel)
    pipeline.run(
        filename=filename, 
        prompt=prompt, 
        model_repo=model, 
        strength=strength, 
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        output_filename=output_filename
    )
    run_time = round((time.time() - run_time) * 1000)
    logger.info(f"Finished running pipeline in {run_time} ms")


@app.callback()
def callback():
    """This is a callback function for the Typer app."""


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        app()
