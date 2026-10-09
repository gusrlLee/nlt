"""Cornell Box path sequences. Run: python dataset.py"""
import argparse
from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np
import torch
from tqdm import tqdm
from runtime import select_renderer


ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "assets/cornell-box/scene.xml"
FEATURES = "position_xyz,normal_xyz,dir_in_xyz,diffuse_rgb,depth/9"
TARGET = "inclusive local continuation: C[d] = emitter_hit + NEE + survived_weight * C[d+1]"


def setup_scene(resolution=256):
    select_renderer()
    from path_tracer import DataPathTracer

    scene = mi.load_file(str(SCENE), resx=resolution, resy=resolution,
                         spp=64, max_depth=10)
    tracer = DataPathTracer(mi.Properties())
    assert tracer.max_depth == 10 and tracer.rr_depth == 5
    return scene, tracer


def lanes(value, count, rgb=False):
    a = np.asarray(value)
    if rgb:
        return np.broadcast_to(a.reshape(3, -1).T, (count, 3)).copy()
    return np.broadcast_to(a.reshape(-1), (count,)).copy()


def camera_rays(scene, pixels, resolution, seed):
    sampler = mi.load_dict({"type": "independent"})
    sampler.seed(seed, len(pixels))
    jitter = sampler.next_2d()
    uv = mi.Point2f((mi.Float(pixels % resolution) + jitter.x) / resolution,
                   (mi.Float(pixels // resolution) + jitter.y) / resolution)
    ray, weight = scene.sensors()[0].sample_ray(
        0.0, sampler.next_1d(), uv, sampler.next_2d())
    return ray, weight, sampler


def path_features(scene, ray, records, count):
    """Replay intersections only to query materials; never resample the path."""
    features, masks = [], []
    for d, record in enumerate(records):
        active, valid = record["active"], record["valid_hit"]
        si = scene.ray_intersect(ray, active=active)
        mask = lanes(valid, count).astype(bool)
        assert np.array_equal(lanes(active & si.is_valid(), count), mask)
        material = si.bsdf().eval_diffuse_reflectance(si, active=valid)
        # ponytail: diffuse RGB suffices for Cornell; extend for other BSDFs.
        values = [lanes(record[key], count, True)
                  for key in ("position", "normal", "dir_in")]
        values += [lanes(material, count, True), np.full((count, 1), d / 9)]
        x = np.concatenate(values, axis=1).astype(np.float32)
        x[~mask] = 0
        assert np.isfinite(x).all()
        features.append(x)
        masks.append(mask)
        ray = si.spawn_ray(record["dir_out"])
    x = torch.from_numpy(np.stack(features, axis=1))
    mask = torch.from_numpy(np.stack(masks, axis=1))
    assert not (mask[:, 1:] & ~mask[:, :-1]).any()
    return x, mask


def continuation_targets(records, radiance, count):
    """Inclusive tail in local units, without dividing by accumulated beta.

    Survived RR carries BSDF weight / survival probability; killed paths
    have no tail. Miss emission belongs to the preceding surface's tail.
    The terminal depth keeps emission but no NEE, exactly as collect_paths.
    Each local estimator already includes the original NEE/MIS weights.
    """
    depth = len(records)
    target = np.zeros((count, depth, 3), dtype=np.float64)
    tail = np.zeros((count, 3), dtype=np.float64)
    contribution_sum = np.zeros_like(tail)
    for d in reversed(range(depth)):
        r = records[d]
        active = lanes(r["active"], count).astype(bool)
        continues = lanes(r["valid_next"], count).astype(bool)
        rr = lanes(r["rr_active"], count).astype(bool)
        q = lanes(r["rr_probability"], count).astype(np.float64)
        assert np.all((q[rr] > 0) & (q[rr] <= 0.950001))
        if d == depth - 1:
            assert not continues.any()
        else:
            assert np.array_equal(continues, lanes(records[d + 1]["active"], count))
        w = lanes(r["bsdf_weight"], count, True).astype(np.float64)
        w[rr] /= q[rr, None]
        w[~continues] = 0
        local = lanes(r["bounce_estimator"], count, True).astype(np.float64)
        assert np.allclose(local, lanes(r["emitter_hit"], count, True)
                           + lanes(r["nee"], count, True), rtol=2e-5, atol=2e-6)
        tail = local + w * tail
        tail[~active] = 0
        target[:, d] = tail
        contribution_sum += lanes(r["path_contribution"], count, True)
        beta = lanes(r["beta"], count, True)
        assert np.allclose(beta * tail, contribution_sum, rtol=5e-5, atol=1e-5), (
            f"Continuation reconstruction failed at depth {d}")
    reference = lanes(radiance, count, True)
    assert np.allclose(target[:, 0], reference, rtol=5e-5, atol=1e-5)
    assert np.isfinite(target).all() and (target >= -1e-6).all()
    return torch.from_numpy(target.astype(np.float32))


def validate_dataset(data):
    x, y, mask = data["features"], data["targets"], data["mask"]
    n = len(x)
    assert data["feature_layout"] == FEATURES and data["target_definition"] == TARGET
    assert x.shape == (n, 10, 13) and y.shape == (n, 10, 3)
    assert x.dtype == y.dtype == torch.float32
    assert mask.shape == (n, 10) and mask.dtype == torch.bool
    assert data["pixel_id"].shape == data["train_ray"].shape == (n,)
    assert data["train_ray"].dtype == torch.bool
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    assert (y >= -1e-6).all() and (x[~mask] == 0).all()
    assert not (mask[:, 1:] & ~mask[:, :-1]).any()
    split = data["train_ray"]
    assert split.any() and (~split).any()
    assert mask[split].any() and mask[~split].any()
    assert not set(data["pixel_id"][split].tolist()) & set(data["pixel_id"][~split].tolist())
    bounds = data["bounds"]
    assert bounds.shape == (2, 3) and torch.isfinite(bounds).all()
    assert (bounds[1] > bounds[0]).all()


def save_tensor_file(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/cornell.pt")
    parser.add_argument("--batch-size", type=int, default=4096, help="Rays per tracing batch")
    args = parser.parse_args()
    assert args.batch_size > 0
    scene, tracer = setup_scene()
    print(f"Mitsuba renderer: {mi.variant()}")
    rng = np.random.default_rng(42)
    pixels = rng.choice(256 * 256, 1024, replace=False)
    validation_pixels = pixels[rng.permutation(1024)[:205]]
    ray_pixels = np.repeat(pixels, 64)
    all_x, all_y, all_mask = [], [], []
    for start in tqdm(range(0, len(ray_pixels), args.batch_size), desc="Dataset batches"):
        ids = ray_pixels[start:start + args.batch_size]
        ray, _, sampler = camera_rays(scene, ids, 256, 42 + start)
        records, radiance = tracer.collect_paths(
            scene, sampler, ray, active=dr.full(mi.Bool, True, len(ids)))
        x, mask = path_features(scene, ray, records, len(ids))
        y = continuation_targets(records, radiance, len(ids))
        all_x.append(x)
        all_y.append(y)
        all_mask.append(mask)
    bbox = scene.bbox()
    data = {
        "features": torch.cat(all_x), "targets": torch.cat(all_y),
        "mask": torch.cat(all_mask), "pixel_id": torch.from_numpy(ray_pixels.copy()),
        "train_ray": torch.from_numpy(~np.isin(ray_pixels, validation_pixels)),
        "bounds": torch.tensor([list(bbox.min), list(bbox.max)], dtype=torch.float32),
        "feature_layout": FEATURES, "target_definition": TARGET,
        "settings": {"resolution": 256, "spp": 64, "selected_pixels": 1024,
                     "max_depth": 10, "rr_depth": 5, "seed": 42,
                     "scene": "cornell-box", "variant": mi.variant(),
                     "generation_batch_size": args.batch_size},
    }
    validate_dataset(data)
    save_tensor_file(data, args.output)
    print(f"Saved {len(ray_pixels)} ordered ray sequences to {args.output}")


if __name__ == "__main__":
    main()
