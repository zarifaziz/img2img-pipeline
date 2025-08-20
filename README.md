 # img2img-pipeline

Stable Diffusion img2img pipeline, supporting various models and images and 
tested on Apple MPS device.

<img src="https://github.com/mwheeler235/img2img-pipeline-bananas/blob/main/data/pipeline_architecture_example.png" width=100% height=100%>


This pipeline:

1. Loads images from `data/input_images` directory
2. For each image, selects a random model from `model_list` in `constants.py`.
3. Performs img2img generation for each image
4. Saves output to `data/output_images` directory

## Fork Updates

This fork includes several significant improvements to enhance functionality, reliability, and user experience:

#### **Enhanced CLI Interface**
- Added configurable parameters for fine-tuning generation:
  - `--strength` (0.0-1.0): Controls how much to transform the original image
  - `--guidance-scale`: Controls prompt adherence strength  
  - `--num-inference-steps`: Controls generation quality vs speed
  - `--output-filename`: Custom output filename control

#### **Cross-Platform Device Support**
- **Apple Silicon (M1/M2) Support**: Added native MPS (Metal Performance Shaders) acceleration
- **Improved CPU Fallback**: Graceful degradation when GPU acceleration unavailable
- **Device-Aware Configuration**: Automatic detection and optimization for CUDA/MPS/CPU
- **Data Type Optimization**: Uses float32 for MPS/CPU to prevent black image artifacts

#### **Intelligent Output Management**
- **Smart Filename Generation**: Auto-generates descriptive filenames including all parameters
  - Example: `image_van_gogh_style_str0.7_guide18.0_steps40_stable_diffusion_v1_5.jpg`
- **Parameter Preservation**: Filenames encode generation settings for reproducibility
- **Collision Prevention**: Unique names prevent accidental overwrites

#### **Robustness & Error Handling**
- **Graceful Memory Management**: Improved error handling for memory optimization features
- **Dependency Compatibility**: Fixed typer/click version conflicts
- **Input Validation**: Better image preprocessing and format handling
- **Comprehensive Logging**: Detailed debug information for troubleshooting

## Run it locally

**Set up python environment**

This pipeline supports multiple platforms and acceleration methods:
- **NVIDIA/CUDA**: Original GPU acceleration (requires CUDA-enabled PyTorch)
- **Apple Silicon (M1/M2)**: Native MPS acceleration
- **CPU**: Fallback support for any system

Install requirements
```
pip install -r requirements.txt
```

**For NVIDIA GPU users:**
You must have CUDA enabled pytorch. You can check by running:
```python
import torch; print(torch.cuda.is_available())
```

**For Apple Silicon users:**
MPS acceleration is automatically detected. You can verify by running:
```python
import torch; print(torch.backends.mps.is_available())
```

**For CPU-only systems:**
The pipeline will automatically fall back to CPU processing.

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

**Advanced usage with all parameters:**
```
python -m src.img2img_pipeline.commands.main run_single_image_pipeline example_image.png \
  --prompt "Van Gogh painting with thick brushstrokes" \
  --model "runwayml/stable-diffusion-v1-5" \
  --strength 0.7 \
  --guidance-scale 18.0 \
  --num-inference-steps 40 \
  --output-filename "my_van_gogh_masterpiece.jpg"
```

There are a list of prompts and models in `src/constants.py`. If `--filename` or `--prompt` are not provided,
a default is chosen from the lists. In which case, the command can be simplified into
```
python -m src.img2img_pipeline.commands.main run_single_image_pipeline example_image.png
```

**Parameter Guidelines:**
- `--strength 0.1-0.4`: Subtle style transfer, preserves original content
- `--strength 0.5-0.7`: Balanced artistic transformation  
- `--strength 0.8-1.0`: Dramatic style changes, may lose original details
- `--guidance-scale 10-15`: Standard prompt following
- `--guidance-scale 16-25`: Strong prompt adherence for specific styles
- `--num-inference-steps 15-25`: Fast generation
- `--num-inference-steps 30-50`: High quality results

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