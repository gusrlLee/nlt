import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
import numpy as np
import torch

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

    def collect_paths(self, scene, sampler, ray, active=True):
        ctx = mi.BSDFContext()

        ray = mi.Ray3f(ray)
        beta = mi.Spectrum(1.0)
        eta = mi.Float(1.0)

        prev_si = dr.zeros(mi.Interaction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)

        records = []
        L_check = mi.Spectrum(0.0)

        for depth in range(self.max_depth):
            # Trace current ray
            active_before = mi.Bool(active)
            si = scene.ray_intersect(ray, active=active_before)
            valid_hit = active_before & si.is_valid()

            # Evaluate surface or environment emitter
            emitter = si.emitter(scene, active=active_before)
            is_emitter = emitter != None

            # Current path state
            dir_in = -ray.d
            beta_before = mi.Spectrum(beta)

            # Emitter contribution from the current ray
            direct_sample = mi.DirectionSample3f(scene, si, prev_si)
            emitter_pdf = scene.pdf_emitter_direction(
                prev_si,
                direct_sample,
                active=active_before & is_emitter & ~prev_bsdf_delta
            )

            mis_bsdf = self.mis_weight(prev_bsdf_pdf, emitter_pdf)
            Le = emitter.eval(si, active=active_before & is_emitter)
            emitter_hit = Le * mis_bsdf

            # Continue only from a valid surface
            active_next = valid_hit & (depth + 1 < self.max_depth)
            bsdf = si.bsdf()

            # Next-event estimation
            active_emitter = (
                active_next &
                mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
            )

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

            mis_emitter = dr.select(
                ds.delta,
                1.0,
                self.mis_weight(ds.pdf, bsdf_pdf)
            )

            nee = dr.select(
                active_emitter,
                bsdf_value * emitter_weight * mis_emitter,
                0.0
            )

            # Contribution generated at this bounce
            bounce_estimator = emitter_hit + nee
            path_contribution = beta_before * bounce_estimator
            L_check += path_contribution

            # Sample the next BSDF direction
            bs, bsdf_weight = bsdf.sample(
                ctx,
                si,
                sampler.next_1d(),
                sampler.next_2d(),
                active_next
            )

            valid_bsdf = active_next & (bs.pdf > 0)
            dir_out = si.to_world(bs.wo)
            next_ray = si.spawn_ray(dir_out)

            # Update transport state
            beta *= bsdf_weight
            eta *= bs.eta

            # Russian roulette
            next_depth = depth + 1
            throughput_max = dr.max(mi.unpolarized_spectrum(beta))

            rr_probability = dr.minimum(
                throughput_max * dr.square(eta),
                0.95
            )

            rr_active = (
                valid_bsdf &
                (throughput_max != 0.0) &
                (next_depth >= self.rr_depth)
            )

            rr_continue = (
                sampler.next_1d(rr_active) < rr_probability
            )

            beta[rr_active] *= dr.rcp(
                dr.detach(rr_probability)
            )

            # Rays that actually continue after Russian roulette
            valid_next = (
                valid_bsdf &
                (throughput_max != 0.0) &
                (~rr_active | rr_continue)
            )

            # Save current bounce
            records.append({
                "depth": depth,

                # Path validity
                "active": mi.Bool(active_before),
                "valid_hit": mi.Bool(valid_hit),
                "is_emitter": mi.Bool(is_emitter),
                "valid_bsdf": mi.Bool(valid_bsdf),
                "valid_next": mi.Bool(valid_next),

                # Geometry
                "position": mi.Point3f(si.p),
                "normal": mi.Normal3f(si.sh_frame.n),

                # Directions
                "dir_in": mi.Vector3f(dir_in),
                "dir_out": mi.Vector3f(dir_out),

                # Transport state
                "beta": mi.Spectrum(beta_before),
                "bsdf_weight": mi.Spectrum(bsdf_weight),
                "bsdf_pdf": mi.Float(bs.pdf),

                # Light estimator
                "emitter_hit": mi.Spectrum(emitter_hit),
                "nee": mi.Spectrum(nee),
                "bounce_estimator": mi.Spectrum(bounce_estimator),
                "path_contribution": mi.Spectrum(path_contribution),

                # Russian roulette
                "rr_active": mi.Bool(rr_active),
                "rr_probability": mi.Float(rr_probability),
            })

            # Save information needed for MIS at the next vertex
            prev_si = mi.Interaction3f(si)
            prev_bsdf_pdf = bs.pdf
            prev_bsdf_delta = mi.has_flag(
                bs.sampled_type,
                mi.BSDFFlags.Delta
            )

            # Move to the next bounce
            ray = next_ray
            active = valid_next

        return records, L_check

    @dr.syntax
    def sample(self, scene, sampler, ray, medium=None, active=True):
        # init 
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

        # main loop
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

            throughput_max = dr.max(mi.unpolarized_spectrum(beta))

            rr_probability = dr.minimum(
                throughput_max * dr.square(eta),
                0.95
            )

            rr_active = (
                active_bsdf &
                (throughput_max != 0.0) &
                (depth >= self.rr_depth)
            )

            rr_continue = (
                sampler.next_1d(rr_active) < rr_probability
            )

            beta[rr_active] *= dr.rcp(
                dr.detach(rr_probability)
            )

            active = (
                active_bsdf &
                (throughput_max != 0.0) &
                (~rr_active | rr_continue)
            )

        return dr.select(valid_ray, L, 0.0), valid_ray, []