import mitsuba as mi
import drjit as dr

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

mi.set_variant(
    'cuda_ad_rgb',
    'metal_ad_rgb',
    'llvm_ad_rgb',
    'scalar_rgb'
)

class PathTracer(mi.SamplingIntegrator):
    def __init__(self, props, nee_only=False, hit_depth=-1, bsdf_depth=-1):
        super().__init__(props)
        self.max_depth = 10
        self.rr_depth = 100

        self.nee_only = nee_only
        self.hit_depth = hit_depth
        self.bsdf_depth = bsdf_depth

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

        # For experiment
        L_nee = mi.Spectrum(0.0)
        L_hit = mi.Spectrum(0.0)
        L_type = mi.Spectrum(0.0)

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
            emitter_contribution = beta * Le * mis_bsdf
            L += emitter_contribution

            if self.hit_depth >= 0:
                L_hit += dr.select(active & is_emitter & (depth == self.hit_depth), emitter_contribution, 0.0)

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

            direct_contribution = beta * bsdf_value * emitter_weight * mis_emitter
            direct_contribution = dr.select(active_emitter, direct_contribution, 0.0)

            L += direct_contribution
            L_nee += direct_contribution

            bs, bsdf_weight = bsdf.sample(ctx, si, sampler.next_1d(), sampler.next_2d(), active_next)
            active_bsdf = active_next & (bs.pdf > 0)

            if self.bsdf_depth >= 0:
                code = mi.Float(0.0)
                code = dr.select(mi.has_flag(bs.sampled_type, mi.BSDFFlags.DiffuseReflection), 1.0, code)
                code = dr.select(mi.has_flag(bs.sampled_type, mi.BSDFFlags.DiffuseTransmission), 2.0, code)
                code = dr.select(mi.has_flag(bs.sampled_type, mi.BSDFFlags.GlossyReflection), 3.0, code)
                code = dr.select(mi.has_flag(bs.sampled_type, mi.BSDFFlags.GlossyTransmission), 4.0, code)
                code = dr.select(mi.has_flag(bs.sampled_type, mi.BSDFFlags.DeltaReflection), 5.0, code)
                code = dr.select(mi.has_flag(bs.sampled_type, mi.BSDFFlags.DeltaTransmission), 6.0, code)
                code = dr.select(mi.has_flag(bs.sampled_type, mi.BSDFFlags.Null), 7.0, code)
                L_type += dr.select(active_bsdf & (depth == self.bsdf_depth), mi.Spectrum(code), 0.0)

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

        if self.bsdf_depth >= 0:
            result = L_type
        elif self.hit_depth >= 0:
            result = L_hit
        elif self.nee_only:
            result = L_nee
        else:
            result = L
        return dr.select(valid_ray, result, 0.0), valid_ray, []


if __name__ == "__main__":
    scene = mi.load_file("assets/kitchen/scene.xml")
    integrator = PathTracer(mi.Properties())

    images = []

    for seed in tqdm(range(64), desc="Rendering"):
        image = mi.render(scene, integrator=integrator, spp=1, seed=seed)
        images.append(image)

    stack = np.stack([np.array(img) for img in images])

    # ---------------------------------------------------------------------------------- 
    # Check Variance 
    # ---------------------------------------------------------------------------------- 
    # RGB -> luminance
    Y = (0.2126 * stack[..., 0] + 0.7152 * stack[..., 1] + 0.0722 * stack[..., 2])

    variance = np.var(Y, axis=0)
    plt.imshow(np.log1p(variance))
    plt.colorbar()
    plt.title("1-spp Variance")
    plt.axis("off")

    plt.savefig("1-spp variance.png", dpi=300, bbox_inches="tight")
    plt.close()

    y, x = np.unravel_index(np.argmax(variance), variance.shape)
    values = Y[:, y, x]

    print("pixel:", x, y)
    print("mean:", values.mean())
    print("std:", values.std())
    print("min:", values.min())
    print("max:", values.max())

    plt.hist(values, bins=32)
    plt.xlabel("Luminance")
    plt.ylabel("Count")
    plt.title(f"Pixel ({x}, {y})")
    plt.savefig("variance_pixel_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()

    mean = np.mean(Y, axis=0)
    gy, gx = np.gradient(mean)
    gradient = np.sqrt(gx * gx + gy * gy)
    threshold = np.percentile(gradient, 90)
    mask = gradient < threshold

    masked_variance = np.where(mask, variance, 0.0)
    y, x = np.unravel_index(np.argmax(masked_variance), masked_variance.shape)
    values = Y[:, y, x]

    print("\n[Non-edge max variance pixel]")
    print("pixel:", x, y)
    print("mean:", values.mean())
    print("std:", values.std())
    print("min:", values.min())
    print("max:", values.max())

    plt.hist(values, bins=32)
    plt.xlabel("Luminance")
    plt.ylabel("Count")
    plt.title(f"Non-edge Pixel ({x}, {y})")
    plt.savefig("non_edge_variance_pixel_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.imshow(np.log1p(masked_variance))
    plt.colorbar()
    plt.title("1-spp Variance (Edges Removed)")
    plt.axis("off")
    plt.savefig("1-spp variance non-edge.png", dpi=300, bbox_inches="tight")
    plt.close()

    order = np.argsort(values)[::-1]

    """
    대부분의 path는 거의 아무 것도 못찾고, 극 소수 path가 큰 contribution 을 발견하는 특징, 
    """

    for i in order[:10]:
        print(f"seed {i:2d}: {values[i]:.6f}")
    

    high_seed = np.argmax(values)
    low_seed = np.argmin(values)

    print("high seed:", high_seed, values[high_seed])
    print("low seed :", low_seed, values[low_seed]) 

    # Check Primary ray of 
    aov_integrator = mi.load_dict({
        'type' : 'aov',
        'aovs' : 'position:position,normal:sh_normal'
    })


    for seed in [48, 11]:
        mi.render(scene, integrator=aov_integrator, spp=1, seed=seed)
        bitmap = scene.sensors()[0].film().bitmap()
        layers = dict(bitmap.split())

        position = np.array(layers['position'])
        normal = np.array(layers['normal'])

        print(f"\nseed {seed}")
        print("position:", position[y, x])
        print("normal  :", normal[y, x])


    nee_integrator = PathTracer(mi.Properties(), nee_only=True)
    for seed in [int(high_seed), int(low_seed)]:
        image = mi.render(scene, integrator=nee_integrator, spp=1, seed=seed)

        img = np.array(image)
        value = (0.2126 * img[y, x, 0] + 0.7152 * img[y, x, 1] + 0.0722 * img[y, x, 2])
        print(f"seed {seed}: NEE = {value}")
    
    for depth in range(10):
        hit_depth_check_integrator = PathTracer(mi.Properties(), hit_depth=depth)
        image = mi.render(scene, integrator=hit_depth_check_integrator, spp=1, seed=int(high_seed))
        img = np.array(image)
        value = (0.2126 * img[y, x, 0] + 0.7152 * img[y, x, 1] + 0.0722 * img[y, x, 2])

        print(f"depth {depth}: {value}")

        

    type_names = {
        0: "None",
        1: "Diffuse Reflection",
        2: "Diffuse Transmission",
        3: "Glossy Reflection",
        4: "Glossy Transmission",
        5: "Delta Reflection",
        6: "Delta Transmission",
        7: "Null"
    }

    print("\n[High-energy path]")

    for depth in range(7):
        type_integrator = PathTracer(mi.Properties(),bsdf_depth=depth)
        image = mi.render(scene, integrator=type_integrator, spp=1, seed=int(high_seed))
        img = np.array(image)
        code = int(round(img[y, x, 0]))
        print(f"depth {depth}: " f"{type_names.get(code, 'Unknown')}")

    print("depth 7: Emitter")