"""
https://github.com/huggingface/transformers/issues/30547
"""

import torch
from transformers.utils import is_flash_attn_2_available
import sys

# Check Python version
print(f"Python version: {sys.version}")

# Check what version of PyTorch is installed
print(f"PyTorch version: {torch.__version__}")

# Check the current CUDA version being used
print(f"CUDA Version: {torch.version.cuda}")

# Check if CUDA is available and if so, print the device name
print(f"Device name: {torch.cuda.get_device_properties('cuda').name}")

# Check if CUDA is available and if so, print the device memory
print(
    f"Device memory: {torch.cuda.get_device_properties('cuda').total_memory/1024/1024/1024:0.3f} GB"
)

# Check if SDPA is enabled (flash scaled dot product attention)
print(f"SDPA (FlashAttention) enabled: {torch.backends.cuda.flash_sdp_enabled()}")

# Check if FlashAttention-2 available
print(f"FlashAttention-2 available: {is_flash_attn_2_available()}")
