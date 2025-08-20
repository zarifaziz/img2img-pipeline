import os
import re
from typing import Generator, Tuple

from PIL import Image

from .constants import INPUT_IMAGE_DIR


def generate_output_filename(
    input_filename: str,
    prompt: str = None,
    strength: float = None,
    guidance_scale: float = None,
    num_inference_steps: int = None,
    model_repo: str = None,
) -> str:
    """
    Generates a descriptive output filename based on input parameters.
    
    Args:
        input_filename: Original input filename
        prompt: The style prompt used
        strength: Strength parameter used
        guidance_scale: Guidance scale used
        num_inference_steps: Number of inference steps used
        model_repo: Model repository used
        
    Returns:
        str: Generated output filename
    """
    # Get base name without extension
    base_name = os.path.splitext(input_filename)[0]
    extension = os.path.splitext(input_filename)[1]
    
    # Clean prompt for filename (remove special characters, limit length)
    if prompt:
        clean_prompt = re.sub(r'[^\w\s-]', '', prompt)
        clean_prompt = re.sub(r'[-\s]+', '_', clean_prompt)
        clean_prompt = clean_prompt[:30]  # Limit length
    else:
        clean_prompt = "styled"
    
    # Extract model organization and name from repo
    if model_repo:
        if '/' in model_repo:
            model_org, model_name = model_repo.split('/', 1)
            # Clean up organization and model names for filename
            model_org = model_org.replace('-', '_')
            model_name = model_name.replace('-', '_')
            model_identifier = f"{model_org}_{model_name}"
        else:
            # Handle case where there's no organization (just model name)
            model_identifier = model_repo.replace('-', '_')
    else:
        model_identifier = "unknown"
    
    # Build filename parts
    parts = [base_name]
    
    if clean_prompt:
        parts.append(clean_prompt)
    
    if strength is not None:
        parts.append(f"str{strength:.1f}")
    
    if guidance_scale is not None:
        parts.append(f"guide{guidance_scale:.1f}")
    
    if num_inference_steps is not None:
        parts.append(f"steps{num_inference_steps}")
    
    parts.append(model_identifier)
    
    # Join parts and add extension
    output_filename = "_".join(parts) + extension
    
    return output_filename


def get_all_images() -> Generator[Tuple[Image.Image, Tuple[int, int], str], None, None]:
    """
    Opens and yields all images from the image directory using a generator.

    Yields:
        Tuple[Image.Image, Tuple[int, int], str]: A tuple containing the PIL Image
        object representing an image from the directory, a tuple of the original image
        dimensions (width, height), and the filename of the image.
    """
    image_dir = str(INPUT_IMAGE_DIR)
    image_list = [
        image
        for image in os.listdir(image_dir)
        if image.endswith((".jpg", ".jpeg", ".png"))
    ]

    for filename in image_list:
        image_path = os.path.join(image_dir, filename)
        image = load_img(image_path)
        original_dimensions = image.size

        yield image, original_dimensions, filename


def load_img(path: str) -> Image.Image:
    """Loads image from path.

    Args:
        path : str

    Returns:
        Image.Image
    """
    image = Image.open(path).convert("RGB")

    return image


def save_image(image: Image.Image, destination_path: str) -> None:
    """
    Saves a Pillow Image object to the specified destination path.

    Args:
        image (Image.Image): The Pillow Image object to be saved.
        destination_path (str): The destination path where the image will be saved.
    """
    image.save(destination_path)


def resize_image(image: Image.Image, max_dimension: int) -> Image.Image:
    """
    Resizes the input image so that the maximum dimension is max_dimension
    while maintaining the aspect ratio.

    Args:
        image (Image.Image): The PIL Image object to be resized.
        max_dimension (int): The maximum dimension (width or height) of the
        resized image.

    Returns:
        Image.Image: The resized PIL Image object.
    """
    width, height = image.size
    aspect_ratio = width / height

    if width >= height:
        new_width = max_dimension
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = max_dimension
        new_width = int(new_height * aspect_ratio)

    resized_image = image.resize((new_width, new_height), resample=Image.LANCZOS)
    return resized_image
