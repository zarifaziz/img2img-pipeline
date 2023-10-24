import os
from typing import Generator, Tuple

from PIL import Image

from .constants import INPUT_IMAGE_DIR


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
