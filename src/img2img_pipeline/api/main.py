import warnings
import uuid
from fastapi import FastAPI, File, UploadFile
from starlette.responses import FileResponse
from PIL import Image
from loguru import logger
import os

from ..model import Img2ImgModel
from ..pipeline import DiffusionSingleImagePipeline

app = FastAPI()

# Set the path for input and output directories
input_dir = "data/input_images"
output_dir = "data/output_images"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)


# Instantiate the DiffusionSingleImagePipeline during API app startup
@app.on_event("startup")
async def startup_event():
    global pipeline
    pipeline = DiffusionSingleImagePipeline(model_class=Img2ImgModel)


@app.post("/images")
async def generate_image(model_repo: str, prompt: str, image: UploadFile = File(...)):
    logger.info(
        f"Received request for image generation using {model_repo} repo and prompt: {prompt}"
    )
    # Save the uploaded image to the input directory
    image_path = os.path.join(input_dir, image.filename)
    with open(image_path, "wb") as f:
        f.write(image.file.read())

    # Generate a unique filename for the processed image
    output_filename = f"{uuid.uuid4().hex}.jpg"
    processed_image_path = os.path.join(output_dir, output_filename)

    # Run the diffusion pipeline for the single image with the provided prompt and model
    pipeline.run(image_path, processed_image_path, prompt, model_repo)

    # Return the processed image for download
    return FileResponse(processed_image_path, media_type="image/jpeg")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        app()
