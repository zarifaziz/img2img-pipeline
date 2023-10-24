import time
import warnings

import typer
from loguru import logger

from ..model import Img2ImgModel
from ..pipeline import DiffusionAllImagesPipeline

app = typer.Typer()


@app.command("run_pipeline")
def run_pipeline():
    logger.info("Running pipeline")
    run_time = time.time()
    pipeline = DiffusionAllImagesPipeline(model_class=Img2ImgModel)
    pipeline.run()
    run_time = round((time.time() - run_time) * 1000)
    logger.info(f"Finished running pipeline in {run_time} ms")


@app.callback()
def callback():
    pass


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        app()
