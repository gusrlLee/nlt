import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt

mi.set_variant(
    'cuda_ad_rgb',
    'metal_ad_rgb',
    'llvm_ad_rgb',
    'scalar_rgb'
)

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


if __name__ == "__main__":
    scene = mi.load_file("assets/glass-of-water/scene.xml")
    integrator = PathTracer(mi.Properties())
    image = mi.render(scene, integrator=integrator, spp=2048)
    mi.util.write_bitmap("path_tracing.png", image)