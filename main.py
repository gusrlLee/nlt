import mitsuba as mi
import drjit as dr
import torch

mi.set_variant("cuda_ad_rgb")

print("Mitsuba:", mi.__version__)
print("Dr.Jit:", dr.__version__)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.is_available())