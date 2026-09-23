import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

mi.set_variant('cuda_ad_rgb')

class PathTracer(mi.SamplingIntegrator):
    def __init__(self, props):
        super().__init__(props)
        self.max_depth = 10
        self.rr_depth = 5

    def mis_weight(self, pdf_a, pdf_b):
        a2 = pdf_a * pdf_a
        b2 = pdf_b * pdf_b

        w = a2 / (a2 + b2)
        return dr.detach(dr.select(dr.isfinite(w), w, 0.0))

    @dr.syntax
    def sample(self, scene, sampler, ray, medium=None, active=True):
        # ----------------------------------------------
        # Initialization 
        # ----------------------------------------------
        ctx = mi.BSDFContext()
        ray = mi.Ray3f(ray)
        L = mi.Spectrum(0.0)
        beta = mi.Spectrum(1.0)

        eta = mi.Float(1.0)
        depth = mi.UInt32(0)

        valid_ray = mi.Bool(False)

        # Information from the previous vertex.
        prev_si = dr.zeros(mi.Interaction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)

        # ----------------------------------------------
        # Path Tracing Main Loop
        # ----------------------------------------------
        while active:
            si = scene.ray_intersect(ray, active=active)
            emitter = si.emitter(scene, active=active)
            is_emitter = emitter != None

            # Probability that NEE could have generated 
            direct_sample = mi.DirectionSample3f(scene, si, prev_si)
            emitter_pdf = scene.pdf_emitter_direction(
                prev_si, 
                direct_sample, 
                active=active & is_emitter & ~prev_bsdf_delta
            )

            mis_bsdf = self.mis_weight(prev_bsdf_pdf, emitter_pdf)

            Le = emitter.eval(si, active=active & is_emitter)

            L += beta * Le * mis_bsdf
            valid_ray |= active & is_emitter

            is_hit = si.is_valid()
            active_next = (active & is_hit & (depth + 1 < self.max_depth))

            bsdf = si.bsdf()
            active_emitter = (active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth))
            ds, emitter_weight = scene.sample_emitter_direction(
                si, 
                sampler.next_2d(active_emitter),
                test_visibility=True,
                active=active_emitter
            )

            active_emitter &= ds.pdf > 0
            wo_emitter = si.to_local(ds.d)

            bsdf_value, bsdf_pdf = bsdf.eval_pdf(
                ctx, 
                si,
                wo_emitter,
                active_emitter
            )

            mis_emitter = dr.select(ds.delta, 1.0, self.mis_weight(ds.pdf, bsdf_pdf))

            direct_contribution = (beta * bsdf_value * emitter_weight * mis_emitter)
            L += dr.select(active_emitter, direct_contribution, 0.0)

            bs, bsdf_weight = bsdf.sample(ctx, si, sampler.next_1d(), sampler.next_2d(), active_next)
            active_bsdf = active_next & (bs.pdf > 0)

            new_direction = si.to_world(bs.wo)
            ray = si.spawn_ray(new_direction)

            beta *= bsdf_weight
            eta *= bs.eta

            non_null = (active_bsdf & ~mi.has_flag(bs.sampled_type, mi.BSDFFlags.Null))
            valid_ray |= non_null

            prev_si = mi.Interaction3f(si)
            prev_bsdf_pdf = bs.pdf
            prev_bsdf_delta = mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta)

            depth += 1

            # R.R.
            throughput_max = dr.max(mi.unpolarized_spectrum(beta))
            rr_probability = dr.minimum(throughput_max * dr.square(eta), 0.95)
            rr_active = depth >= self.rr_depth

            rr_continue = (sampler.next_1d(rr_active) < rr_probability)
            beta[rr_active] *= dr.rcp(dr.detach(rr_probability))

            active = (active_bsdf & (throughput_max != 0.0) & (~rr_active | rr_continue))

        return dr.select(valid_ray, L, 0.0), valid_ray, []

### define setting 
ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets"
OUTPUT_DIR = Path(__file__).resolve().parent

REFERENCE_SEEDS = (100, 101, 102, 103)
SPP1_SEEDS = (0, 1, 2)

EXPOSURE_STOPS = 0.0
ERROR_PERCENTILE = 99.0
LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

def render_linear(scene, integrator, spp, seed):
    image = mi.render(
        scene,
        integrator=integrator,
        spp=spp,
        seed=seed,
    )

    dr.eval(image)
    return np.array(image, dtype=np.float32, copy=True)

def srgb_to_linear(image):
    image = np.asarray(image, dtype=np.float32)[..., :3]

    return np.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4,
    )

def to_display(linear_image):
    image = np.maximum(
        linear_image * (2.0 ** EXPOSURE_STOPS),
        0.0,
    )

    image = np.where(
        image <= 0.0031308,
        12.92 * image,
        1.055 * image ** (1.0 / 2.4) - 0.055,
    )

    return np.clip(image, 0.0, 1.0)

def render_reference(scene, integrator):
    """256 spp 렌더 4장을 linear RGB에서 평균한다."""
    reference_sum = None

    for seed in REFERENCE_SEEDS:
        print(f"    reference: 256 spp, seed={seed}")

        image = render_linear(
            scene,
            integrator,
            spp=256,
            seed=seed,
        )

        if reference_sum is None:
            reference_sum = image.astype(np.float64)
        else:
            reference_sum += image

    return (
        reference_sum / len(REFERENCE_SEEDS)
    ).astype(np.float32)


def luminance_error(reference, image):
    return np.abs(reference - image) @ LUMINANCE_WEIGHTS


def show_rgb(axis, image, title):
    axis.imshow(to_display(image))
    axis.set_title(title)
    axis.axis("off")


def show_error(axis, error, title, vmax):
    plot = axis.imshow(error, cmap="magma", vmin=0.0, vmax=vmax)
    axis.set_title(title)
    axis.axis("off")
    return plot


def save_comparison(
    scene_name,
    tungsten,
    reference,
    spp1_images,
    error_maps,
):
    mean_spp1 = np.mean(spp1_images, axis=0)
    mean_error = np.mean(error_maps, axis=0)

    finite_errors = np.concatenate([
        error[np.isfinite(error)] for error in [*error_maps, mean_error]
    ])
    vmax = np.percentile(finite_errors, ERROR_PERCENTILE) if finite_errors.size else 1.0
    vmax = max(float(vmax), np.finfo(np.float32).eps)

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    grid = fig.add_gridspec(3, 4)

    full_rgb_axes = [
        (fig.add_subplot(grid[0, 0:2]), tungsten, "Tungsten"),
        (fig.add_subplot(grid[0, 2:4]), reference, "Reference 1024 spp"),
        *[
            (fig.add_subplot(grid[1, column]), image, f"1 spp s{seed}")
            for column, (seed, image) in enumerate(zip(SPP1_SEEDS, spp1_images))
        ],
        (fig.add_subplot(grid[1, 3]), mean_spp1, "Mean 1 spp"),
    ]
    for axis, image, title in full_rgb_axes:
        show_rgb(axis, image, title)

    error_axes = []
    for column, (seed, error) in enumerate(zip(SPP1_SEEDS, error_maps)):
        axis = fig.add_subplot(grid[2, column])
        show_error(axis, error, f"Error s{seed}", vmax)
        error_axes.append(axis)

    mean_error_axis = fig.add_subplot(grid[2, 3])
    error_plot = show_error(mean_error_axis, mean_error, "Mean Error", vmax)
    error_axes.append(mean_error_axis)

    fig.colorbar(error_plot, ax=error_axes, label="Absolute luminance error", shrink=0.8)
    fig.suptitle(scene_name)

    output_path = OUTPUT_DIR / f"{scene_name}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"    saved: {output_path}")


def process_scene(scene_dir):
    scene_name = scene_dir.name
    print(f"[{scene_name}]")

    scene = mi.load_file(
        str(scene_dir / "scene.xml")
    )

    # PNG는 visual reference로만 사용한다.
    tungsten = srgb_to_linear(
        plt.imread(scene_dir / "TungstenRender.png")
    )

    integrator = PathTracer(mi.Properties())

    reference = render_reference(
        scene,
        integrator,
    )

    spp1_images = []

    for seed in SPP1_SEEDS:
        print(f"    1 spp, seed={seed}")

        spp1_images.append(
            render_linear(
                scene,
                integrator,
                spp=1,
                seed=seed,
            )
        )

    if tungsten.shape != reference.shape:
        raise ValueError(
            f"Tungsten/reference shape mismatch: {tungsten.shape} != {reference.shape}"
        )

    # RGB absolute error를 linear RGB에서 계산한 뒤 luminance로 축약한다.
    error_maps = [luminance_error(reference, image) for image in spp1_images]

    save_comparison(
        scene_name,
        tungsten,
        reference,
        spp1_images,
        error_maps,
    )


def main():
    scene_dirs = sorted(
        xml_path.parent
        for xml_path in ASSETS_DIR.glob("*/scene.xml")
    )

    for scene_dir in scene_dirs:
        try:
            process_scene(scene_dir)
        except Exception as error:
            print(
                f"[FAILED] {scene_dir.name}: "
                f"{type(error).__name__}: {error}"
            )
        finally:
            dr.flush_malloc_cache()


if __name__ == "__main__":
    main()
