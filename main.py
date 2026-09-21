import sys
from pathlib import Path
import drjit as dr
import mitsuba as mi

# 1. Variant configuration (CUDA AD preferred, fallback to LLVM AD)
try:
    mi.set_variant("cuda_ad_rgb")
except Exception as e:
    print(f"[Warning] Failed to set 'cuda_ad_rgb': {e}")
    mi.set_variant("llvm_ad_rgb")

from PathTracer import CustomPathTracer, render_scene

def find_cornell_box_scene() -> Path:
    """Find the Cornell Box scene XML in assets directory."""
    candidates = [
        Path("assets/cornell-box/cornell-box/scene.xml"),
        Path("assets/cornell-box/scene.xml"),
        Path("../assets/cornell-box/cornell-box/scene.xml"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not locate Cornell Box scene.xml. Checked: {[str(c) for c in candidates]}"
    )

def main():
    print("=" * 60)
    print("[Mitsuba 3 & Dr.Jit Custom Path Tracer Pipeline]")
    print(f" Mitsuba Version: {mi.__version__}")
    print(f" Dr.Jit Version : {dr.__version__}")
    print(f" Active Variant : {mi.variant()}")
    print("=" * 60)

    # 2. Locate scene
    scene_path = find_cornell_box_scene()
    print(f"[Main] Scene file found at: {scene_path}")

    # 3. Render parameters
    spp = 64
    max_depth = 8
    output_image = "cornell_box_custom_pt.png"

    # 4. Render with custom path tracer
    render_scene(
        scene_or_path=scene_path,
        spp=spp,
        max_depth=max_depth,
        rr_depth=5,
        use_nee=True,
        output_path=output_image,
    )

    print("=" * 60)
    print(f"[Success] Rendering finished! Output saved to: {output_image}")
    print("=" * 60)

if __name__ == "__main__":
    main()