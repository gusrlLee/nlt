import csv
import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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
        self.rr_depth = 100

        self.analysis_depth = 5
        self.analysis_width = None
        self.analysis_height = None

        self.hit_density = None
        self.throughput_density = None
        self.valid_hits = None
        self.camera_to_screen = None
        self.world_to_camera = None

    def initialize_analysis(self, scene):
        camera = scene.sensors()[0]
        if not isinstance(camera, mi.ProjectiveCamera):
            raise TypeError("EXP03 requires a projective camera")

        # Mitsuba's projection includes the film crop and camera intrinsics.
        self.world_to_camera = camera.world_transform().inverse()
        self.camera_to_screen = camera.projection_transform()
        self.analysis_width, self.analysis_height = camera.film().crop_size()

        # Store integer hit counts and float throughput weights in flat buffers.
        size = self.analysis_depth * self.analysis_width * self.analysis_height
        self.hit_density = dr.zeros(mi.UInt32, size)
        self.throughput_density = dr.zeros(mi.Float, size)
        self.valid_hits = dr.zeros(mi.UInt32, self.analysis_depth)

    def analysis_index(self, p, slot):
        width, height = self.analysis_width, self.analysis_height
        camera_p = self.world_to_camera @ p
        screen_p = self.camera_to_screen @ camera_p
        u, v = screen_p.x, screen_p.y

        # Exclude points behind the camera, outside the frame, or non-finite.
        inside = (camera_p.z > 0) & dr.isfinite(camera_p.z)
        inside &= dr.isfinite(u) & dr.isfinite(v)
        inside &= (u >= 0) & (u < 1) & (v >= 0) & (v < 1)

        # Use safe coordinates for masked lanes before converting to integers.
        x = mi.UInt32(dr.floor(dr.select(inside, u, 0) * width))
        y = mi.UInt32(dr.floor(dr.select(inside, v, 0) * height))
        return slot * width * height + y * width + x, inside

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
            # Slots 0..4 represent path depths 1..5; depth 0 is omitted.
            hit_active = active & si.is_valid() & (depth >= 1) & (depth <= self.analysis_depth)
            slot = dr.select(depth > 0, depth - 1, 0)
            dr.scatter_add(self.valid_hits, mi.UInt32(1), slot, active=hit_active)
            hit_index, in_frame = self.analysis_index(si.p, slot)
            projected = hit_active & in_frame
            # Count each surface arrival before sampling the next bounce.
            dr.scatter_add(
                self.hit_density, mi.UInt32(1), hit_index,
                active=projected
            )
            # Record the throughput that reached this surface.
            dr.scatter_add(
                self.throughput_density, mi.luminance(beta), hit_index,
                active=projected
            )
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


def normalize_hits(hits):
    """Turn raw hit counts into spatial probabilities; empty maps are undefined."""
    total = hits.sum(dtype=np.uint64)
    if total == 0:
        return None
    return hits.astype(np.float64) / total


def block_sum(hits, factor):
    """Aggregate all counts, including partial blocks at image edges."""
    y = np.arange(0, hits.shape[0], factor)
    x = np.arange(0, hits.shape[1], factor)
    return np.add.reduceat(np.add.reduceat(hits.astype(np.uint64), y, axis=0), x, axis=1)


def similarity(p, q):
    """Return base-2 Jensen-Shannon divergence and cosine similarity."""
    if p is None or q is None:
        return None, None
    m = (p + q) / 2
    p_mask = p > 0
    q_mask = q > 0
    js = (np.sum(p[p_mask] * np.log2(p[p_mask] / m[p_mask]))
          + np.sum(q[q_mask] * np.log2(q[q_mask] / m[q_mask]))) / 2
    cosine = np.sum(p * q) / (np.linalg.norm(p) * np.linalg.norm(q))
    return float(np.clip(js, 0, 1)), float(np.clip(cosine, 0, 1))


def analyze_hits(scene_name, hits):
    """Compare only depth pairs present in the raw hit buffer."""
    rows = []
    for scale, factor in (("native", 1), ("medium", 4), ("coarse", 16)):
        distributions = [normalize_hits(block_sum(depth_hits, factor)) for depth_hits in hits]
        print(f"Scale: {scale}")
        for slot in range(len(distributions) - 1):
            js, cosine = similarity(distributions[slot], distributions[slot + 1])
            rows.append({"scene": scene_name, "scale": scale, "depth_a": slot + 1,
                         "depth_b": slot + 2, "js_divergence": js,
                         "cosine_similarity": cosine})
            js_text = "undefined" if js is None else f"{js:.6f}"
            cosine_text = "undefined" if cosine is None else f"{cosine:.6f}"
            print(f"d{slot + 1}->d{slot + 2}: JS={js_text}, cosine={cosine_text}")
        available = [row for row in rows if row["scale"] == scale
                     and row["js_divergence"] is not None]
        if available:
            print(f"Mean: JS={np.mean([row['js_divergence'] for row in available]):.6f}, "
                  f"cosine={np.mean([row['cosine_similarity'] for row in available]):.6f}")
        else:
            print("Mean: undefined (no valid adjacent pairs)")
    return rows


def save_summary_figure(rows, output_path):
    """Use the native scale for every scene, chosen before viewing results."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    scenes = sorted({row["scene"] for row in rows})
    pairs = sorted({row["depth_a"] for row in rows if row["scale"] == "native"})
    for scene in scenes:
        scene_rows = {row["depth_a"]: row for row in rows
                      if row["scene"] == scene and row["scale"] == "native"}
        for ax, field in zip(axes, ("js_divergence", "cosine_similarity")):
            ax.plot(pairs, [np.nan if scene_rows.get(d) is None or
                             scene_rows[d][field] is None else scene_rows[d][field]
                             for d in pairs], marker="o", markersize=3, label=scene)
    for ax, title in zip(axes, ("JS divergence", "Cosine similarity")):
        ax.set_title(title)
        ax.set_xticks(pairs, [f"{d}→{d + 1}" for d in pairs])
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    fig.suptitle("Adjacent-depth spatial similarity (native resolution)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(scenes)), fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    # Run every scene that has a scene.xml file.
    scenes = sorted(path for path in Path("assets").iterdir()
                    if (path / "scene.xml").is_file())

    results = Path("results")
    results.mkdir(exist_ok=True)
    csv_path = results / "exp03-1_adjacent_depth_similarity.csv"
    all_rows = []
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=("scene", "scale", "depth_a", "depth_b",
                                                    "js_divergence", "cosine_similarity"))
        writer.writeheader()
        for scene_path in scenes:
            name = scene_path.name
            print(f"Scene: {name}")
            scene = mi.load_file(str(scene_path / "scene.xml"))
            integrator = PathTracer(mi.Properties())
            integrator.initialize_analysis(scene)
            mi.render(scene, integrator=integrator, spp=2048)

            shape = (integrator.analysis_depth, integrator.analysis_height,
                     integrator.analysis_width)
            hits = integrator.hit_density.numpy().reshape(shape)
            rows = analyze_hits(name, hits)
            writer.writerows(rows)
            csv_file.flush()
            all_rows.extend(rows)

    if all_rows:
        save_summary_figure(all_rows, results / "exp03-1_adjacent_depth_similarity.png")
