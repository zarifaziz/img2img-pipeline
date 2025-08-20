#!/usr/bin/env python3
"""Test script to check device detection and model loading."""

import torch
from loguru import logger

logger.info("Testing device detection...")

if torch.cuda.is_available():
    logger.info("✅ CUDA is available")
    logger.info(f"   CUDA device: {torch.cuda.get_device_name()}")
elif torch.backends.mps.is_available():
    logger.info("✅ MPS (Apple Silicon) is available")
    logger.info("   This should work well for Stable Diffusion")
else:
    logger.info("ℹ️  Using CPU (this will be slow)")

logger.info(f"PyTorch version: {torch.__version__}")

# Test if MPS works
if torch.backends.mps.is_available():
    try:
        # Simple MPS test
        x = torch.randn(2, 3).to("mps")
        y = torch.randn(2, 3).to("mps")
        z = x + y
        logger.info("✅ MPS basic operations work")
    except Exception as e:
        logger.error(f"❌ MPS test failed: {e}")

logger.info("Device detection test completed!")
