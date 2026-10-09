"""Select independent PyTorch and Mitsuba backends across Windows/macOS."""
import torch


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_renderer():
    import drjit as dr
    import mitsuba as mi

    variants = {"cuda": "cuda_ad_rgb", "metal": "metal_ad_rgb", "llvm": "llvm_ad_rgb"}
    backends = {"cuda": dr.JitBackend.CUDA,
                "metal": getattr(dr.JitBackend, "Metal", None), "llvm": dr.JitBackend.LLVM}
    for name in ("cuda", "metal", "llvm"):
        variant = variants[name]
        backend = backends[name]
        if variant in mi.variants() and backend is not None and dr.has_backend(backend):
            mi.set_variant(variant)
            return variant
    raise RuntimeError(
        "No supported Mitsuba renderer is available. CUDA requires an NVIDIA GPU "
        "and driver; Metal requires a supported Apple GPU and a Metal-enabled "
        "Mitsuba/Dr.Jit build; LLVM CPU rendering requires a shared LLVM library. "
        "Install LLVM and, if necessary, set DRJIT_LIBLLVM_PATH to the library file "
        "(libLLVM.dylib on macOS, LLVM-C.dll on Windows).")
