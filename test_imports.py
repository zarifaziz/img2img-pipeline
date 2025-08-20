#!/usr/bin/env python3
"""Test script to verify all imports work correctly."""

print("Testing imports...")

try:
    import typer
    print("✅ typer imported successfully")
except ImportError as e:
    print(f"❌ Failed to import typer: {e}")

try:
    import torch
    print(f"✅ torch imported successfully (version: {torch.__version__})")
    print(f"   CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"❌ Failed to import torch: {e}")

try:
    import diffusers
    print(f"✅ diffusers imported successfully (version: {diffusers.__version__})")
except ImportError as e:
    print(f"❌ Failed to import diffusers: {e}")

try:
    import accelerate
    print(f"✅ accelerate imported successfully (version: {accelerate.__version__})")
except ImportError as e:
    print(f"❌ Failed to import accelerate: {e}")

try:
    from loguru import logger
    print("✅ loguru imported successfully")
except ImportError as e:
    print(f"❌ Failed to import loguru: {e}")

print("\nTesting model import...")
try:
    from src.img2img_pipeline.model import Img2ImgModel
    print("✅ Img2ImgModel imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Img2ImgModel: {e}")

print("\nAll tests completed!")
