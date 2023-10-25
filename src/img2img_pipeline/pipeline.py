import random

from loguru import logger
from PIL import Image
from tqdm import tqdm

from .constants import INPUT_IMAGE_DIR, OUTPUT_IMAGE_DIR, model_list, prompt_list
from .interfaces import PipelineInterface, ModelInterface
from .model import Img2ImgModel
from .utils import get_all_images, resize_image, save_image


class DiffusionAllImagesPipeline(PipelineInterface):
    def __init__(self, model_class: ModelInterface, max_dimension=512) -> None:
        """
        Initializes the DiffusionAllImagesPipeline class.

        Args:
            model_class: class that implements ModelInterface to be used in the pipeline
            max_dimension: Maximum dimension (width or height) of input to model
        """
        self.model_class = model_class
        self.max_dimension = max_dimension

    def run(self) -> None:
        """
        Runs the diffusion pipeline for every image
        """
        for input_image, original_dimensions, filename in tqdm(get_all_images()):
            # Save the image dimensions
            width, height = original_dimensions

            # Resize the input_image so that the max dimension is self.max_dimension
            # while keeping the aspect ratio
            resized_image = resize_image(input_image, max_dimension=self.max_dimension)

            model_repo = random.choice(model_list)
            logger.info(f"randomly picked model: {model_repo}")

            model = self.model_class(model_repo)

            output_image = model.predict(
                resized_image, prompt=random.choice(prompt_list)
            )
            logger.info("inference for image finished")

            # Resize the output_image back to its original dimensions
            resized_output = output_image.resize(
                (width, height), resample=Image.LANCZOS
            )

            save_image(resized_output, str(OUTPUT_IMAGE_DIR) + f"/{filename}")


class DiffusionSingleImagePipeline(PipelineInterface):
    def __init__(self, model_class: ModelInterface, max_dimension=512) -> None:
        """
        Initializes the DiffusionSingleImagePipeline class.

        Args:
            model_class: class that implements ModelInterface to be used in the pipeline
            max_dimension: Maximum dimension (width or height) of input to model
        """
        self.model_class = model_class
        self.max_dimension = max_dimension

    def run(
        self,
        filename: str,
        prompt: str,
        model_repo: str,
    ) -> None:
        """
        Runs the diffusion pipeline for a single image.

        Args:
            file_name: filename of the input image
            prompt: The prompt for model prediction
            model_repo: The model repository to use
        """
        # Load the input image
        input_image = Image.open(str(INPUT_IMAGE_DIR / filename))

        # Save the original dimensions
        width, height = input_image.size

        # Resize the input image so that the max dimension is self.max_dimension
        # while keeping the aspect ratio
        resized_image = resize_image(input_image, max_dimension=self.max_dimension)

        model = self.model_class(model_repo)

        output_image = model.predict(resized_image, prompt=prompt)
        logger.info("Inference for image finished")

        # Resize the output image back to its original dimensions
        resized_output = output_image.resize((width, height), resample=Image.LANCZOS)

        save_image(resized_output, str(OUTPUT_IMAGE_DIR) + f"/{filename}")
