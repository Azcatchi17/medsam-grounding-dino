import torch
print(f"CUDA disponible : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Périphérique GPU : {torch.cuda.get_device_name(0)}")

# Test OpenMMLab
import mmengine
import mmcv
import mmdet
print("OpenMMLab (mmengine, mmcv, mmdet) : OK")

# Test MedSAM / SAM
import segment_anything
print("Segment Anything : OK")

# Test Transformers (utilisé par DINO)
import transformers
print("Transformers : OK")