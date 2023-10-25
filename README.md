 # img2img-pipeline

Stable Diffusion img2img pipeline, supporting various models and images and 
tested on NVIDIA / CUDA devices.

This pipeline:

1. Loads images from `data/input_images` directory
2. For each image, selects a random model from `model_list` in `constants.py`.
3. Performs img2img generation for each image
4. Saves output to `data/output_images` directory

## Run it locally

**Set up python environment**

It is required to have Python3.8, CUDA, and Pytorch installed on the system

Install requirements
```
pip install -r requirements.txt
```
You must have CUDA enabled pytorch. You can check by running the following
```
import torch; print(torch.cuda.is_available())
```

**Add images to the `data/input_images` directory
```
cp example_image.png data/input_images/
```

**Run the pipeline**
Either over all the images in `data/input_images`

```
python -m src.img2img_pipeline.commands.main run_all_images_pipeline
```

Or on a specific image by providing the `[filename]` and extra arguments
```
python -m src.img2img_pipeline.commands.main run_single_image_pipeline example_image.png --prompt "in the style of picasso" --model "stabilityai/stable-diffusion-2"
```
There are a list of prompts and models in `src/constants.py`. If `--filename` or `--prompt` are not provided,
a default is chosen from the lists. In which case, the command can be simplified into
```
python -m src.img2img_pipeline.commands.main run_single_image_pipeline example_image.png
```

## Project Structure
```
.
├── README.md
├── data
│   ├── input_images
│   └── output_images
├── metrics.md
│    Metrics of the pipeline runs such as time, memory
├── requirements.txt
└── src
    └── img2img_pipeline
         Application source code
```

## Design considerations

### Code structure

All the source code sits under `src/img2img_pipeline` and uses relative paths so
it can be packaged in a pip package and integrated with any another orchestration repo
or an API service for example.

The files `model.py` and `pipeline.py` are there to provide clean levels of
abstraction and separation of responsibilities. Interfaces are defined in
`interfaces.py` to formalise these abstractions. The pipeline class can accept
any image model as long as it inherits from `ModelInterface`, as it knows that
the instance will have implemented the `predict()` function.

This carries on in making it very simple to run the pipeline as it only has to be
initialised with a model and then the `.run()` method has to be called in order
to run it. This can be seen in [commands/main.py](./src/img2img_pipeline/commands/main.py).

The current `main.py` is put inside a `commands` module because we might want to add an `api/` folder using the same modules to create a Stable Diffusion API service. Finally, The `typer` library has been used to implement the CLI command for running the pipeline as it is very simple and easily extensible.


### Work towards increasing GPU memory efficiency

1. Images are downsampled to 512px before inference and then resampled back to it's original dimensions after
inference. This is done because stable diffusion models have only been trained on 512px images or less and memory
gets exceeded very rapidly if images are higher resolution than this.

2. The pipeline has been configured to be memory efficient. The current settings are recommended from my [research](https://huggingface.co/docs/diffusers/optimization/fp16) and they are the following:
```
self.pipe.enable_attention_slicing()
self.pipe.enable_sequential_cpu_offload()
self.pipe.enable_xformers_memory_efficient_attention()
```
It was seen that installing `xformers` resulted in a significant memory boost.

This can be turned toggled using the
`memory_efficient_compute` flag in `constants.py` to see the difference in GPU utilisation. 

3. Specified `torch_dtype=torch.float16` which improved memory efficiency and speed

4. A generator function has been used to load the images one by one. If this pipeline scales to thousands of images, loading them all into memory one by one will result in exceeding available memory. Using a generator allows the pipeline to release memmory after each image is processed.

5. In `model.py` line 71 we clear the cache after every prediction
```
logger.info("clearing cache")
torch.cuda.empty_cache()
gc.collect()
```
This works to clear intermediate results, such as activations, gradients, and other temporary variables that are stored in the GPU memory or system memory. 

**Future Work**
- Research on `torch.compile`


### Work towards increasing computation speed
1. The pipeline and image generator are set to run on GPU

2. Added a fast scheduler `DPMSolverMultistepScheduler` which requires
less inference steps (~25 compared to ~50 by default).

2. Specified `torch_dtype=torch.float16` which improved memory efficiency and speed by loading the models in half precision

3. Used the TensorFloat32 (TF32) mode for faster but slightly less accurate computations
```
torch.backends.cuda.matmul.allow_tf32 = True
```

3. There is a `CACHE_DIR` specified in `constants.py`. The models are downloaded 
into `CACHE_DIR` directory and loaded from there.

**Future Work**
- figure out batch prediction with many images at once
- relying on the [sequential nature of diffusion models](https://lightning.ai/pages/community/optimize-inference-scheduler/) to run batch inference

## Discussion of pipeline results

Please see [metrics.md](./metrics.md) for metrics of the pipeline. 
The machine used was the following:
```
1 x A100 (40 GB SXM4)
30 vCPUs, 200GiB RAM, 512 GiB SSD
```

The results show different performance metrics for various configurations, including different data types, memory-efficient settings, and garbage collection settings. These configurations present trade-offs in terms of speed, quality, and GPU memory utilization.

When using the float32 data type, the pipeline runtime was 65.6 seconds with an average GPU usage of 70%. The peak GPU usage reached 97%, indicating high GPU utilization. The average GPU memory usage was 25.4%, with a peak of 30.8%. This configuration provides a balance between speed and memory usage.

By switching to the float16 data type, the pipeline runtime improved to 52.9 seconds. However, the average GPU usage decreased to 52%, and the peak GPU usage was 88%. The memory efficiency increased significantly, with average GPU memory usage at 15.3% and a peak of 18.7%. This configuration sacrifices some GPU usage for improved memory efficiency and faster computations.

Incorporating memory-efficient settings alongside float16 data type further enhanced memory efficiency. The pipeline runtime increased to 186.6 seconds, indicating a longer processing time. The average GPU usage reduced to 47%, and the peak GPU usage decreased to 62%. However, the average GPU memory usage dropped to 6.4%, with a peak of 11.5%. This configuration focuses on minimizing GPU memory usage at the expense of longer processing time.

Disabling garbage collection had a surprisingly negligible impact on memory. The pipeline runtime was 181.9 seconds, similar to the previous configuration. The average and peak GPU usage remained relatively unchanged, and the GPU memory usage remained at 6.38% on average and 11.5% at peak. Disabling garbage collection can be beneficial when dealing with large-scale pipelines, but for our case of loading one image at a time, it is not that effective.

Overall there are trade-offs in every configuration and the choice of the appropriate configuration will depend on whether we are constrained by speed, quality or cost/GPU memory.

## Future architecture for scaling

Although a handful of images are run in the pipeline, this pipeline is designed to scale to
thousands of images and/or models.

To scale the img2img pipeline for processing thousands of images, a scalable architecture can be implemented using tools such as Docker, AWS SQS, load balancing, and cloud storage like Amazon S3. 

Docker containers can be used to deploy and manage multiple pipeline instances. Images can be sent as messages to an SQS queue, where pipeline instances in Docker containers can process them. Load balancing distributes the workload evenly among instances, ensuring efficient resource utilization. Processed images can be stored in scalable cloud storage like Amazon S3. 

This architecture enables parallel image processing, resource efficiency, fault tolerance, and easy integration with other AWS services.
