"""Smoke test to verify CUDA availability and core library imports."""

# Verify PyTorch and CUDA availability
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")

# Verify OpenMMLab packages
import mmcv
import mmdet
import mmengine

print("OpenMMLab (mmengine, mmcv, mmdet): OK")

# Verify Segment Anything (SAM)
import segment_anything

print("Segment Anything: OK")

# Verify Hugging Face Transformers
import transformers

print("Transformers: OK")