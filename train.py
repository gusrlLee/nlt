"""Train both baselines, then write only loss.png and inference.png as figures."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dataset import ROOT, validate_dataset, save_tensor_file
from model import RadianceModel, masked_mse
from runtime import select_device


def train_model(kind, data, args):
    torch.manual_seed(42)
    device = args.device
    use_cuda = device.type == "cuda"
    model = RadianceModel(kind, data["bounds"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loaders = []
    for split in (data["train_ray"], ~data["train_ray"]):
        # Only entirely empty rays are dropped; bounce order stays intact.
        selected = split & data["mask"].any(dim=1)
        samples = TensorDataset(*(data[key][selected] for key in ("features", "targets", "mask")))
        loaders.append(DataLoader(samples, batch_size=args.batch_size,
                                  shuffle=len(loaders) == 0, pin_memory=use_cuda))
    history = {"train": [], "validation": []}
    best = float("inf")
    start = 0
    last_path = args.output / f"{kind}_last.pt"
    best_path = args.output / f"{kind}_best.pt"
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        assert checkpoint["kind"] == kind and checkpoint["settings"] == data["settings"]
        assert checkpoint["dataset_path"] == str(args.dataset.resolve())
        assert torch.equal(checkpoint["bounds"], data["bounds"])
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history, best, start = checkpoint["history"], checkpoint["best_loss"], checkpoint["epoch"]
        assert best_path.exists(), "Resume requires the matching best checkpoint"
    for epoch in tqdm(range(start, args.epochs), desc=f"{kind.upper()} epochs"):
        for training, loader, name in zip((True, False), loaders, ("train", "validation")):
            model.train(training)
            total, elements = 0., 0
            batches = tqdm(loader, desc=f"{kind} {epoch + 1} {name}", leave=False)
            with torch.set_grad_enabled(training):
                for x, y, mask in batches:
                    x, y, mask = (v.to(device, non_blocking=use_cuda) for v in (x, y, mask))
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                    loss = masked_mse(model(x, mask), y, mask)
                    assert torch.isfinite(loss), "Non-finite loss"
                    if training:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step()
                    count = int(mask.sum().item()) * 3
                    total += loss.item() * count
                    elements += count
                    batches.set_postfix(loss=total / elements)
            assert elements > 0
            history[name].append(total / elements)
        checkpoint = {"kind": kind, "model": model.state_dict(),
                      "bounds": data["bounds"], "settings": data["settings"],
                      "dataset_path": str(args.dataset.resolve()),
                      "epoch": epoch + 1, "history": history}
        if history["validation"][-1] < best:
            best = history["validation"][-1]
            save_tensor_file(checkpoint, best_path)
        checkpoint.update(optimizer=optimizer.state_dict(), best_loss=best)
        save_tensor_file(checkpoint, last_path)
    assert best_path.exists()
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best_checkpoint["model"])
    model.eval()
    return model, history


@torch.no_grad()
def render_inference(model, args):
    """Validate GRU on full traced sequences; do not form a neural estimator."""
    import drjit as dr
    import mitsuba as mi
    from dataset import setup_scene, camera_rays, path_features, continuation_targets, lanes

    resolution = 256
    scene, tracer = setup_scene(resolution)
    print(f"Mitsuba renderer: {mi.variant()}")
    assert model.kind == "gru"
    model.eval()
    device = next(model.parameters()).device
    image = np.zeros((resolution * resolution, 3), dtype=np.float64)
    squared_error = np.zeros(len(image), dtype=np.float64)
    vertex_count = np.zeros(len(image), dtype=np.int64)
    depth_error = np.zeros(tracer.max_depth, dtype=np.float64)
    depth_count = np.zeros(tracer.max_depth, dtype=np.int64)
    # ponytail: replay collected paths for validation; a live traversal hook and
    # incident-radiance Two-Level MC remain for the full hybrid renderer.
    for sample in tqdm(range(args.inference_spp), desc="Full-path GRU validation SPP"):
        for start in range(0, len(image), 4096):
            pixels = np.arange(start, min(start + 4096, len(image)))
            ray, weight, sampler = camera_rays(
                scene, pixels, resolution, 42000 + sample * len(image) + start)
            # Preserve every original geometry, BSDF, NEE/MIS and RR decision.
            records, radiance = tracer.collect_paths(
                scene, sampler, ray, active=dr.full(mi.Bool, True, len(pixels)))
            x, mask = path_features(scene, ray, records, len(pixels))
            target = continuation_targets(records, radiance, len(pixels))
            x_device, mask_device = x.to(device), mask.to(device)
            hidden, predictions = None, []
            for d in range(len(records)):
                prediction, hidden = model.step(x_device[:, d], mask_device[:, d], hidden)
                assert torch.isfinite(prediction).all() and torch.isfinite(hidden).all()
                predictions.append(prediction)
            predicted = torch.stack(predictions, dim=1).cpu()
            assert predicted.shape == target.shape and (predicted[~mask] == 0).all()
            if sample == 0 and start == 0:
                assert torch.allclose(predicted.to(device), model(x_device, mask_device),
                                      rtol=1e-5, atol=1e-6), "Streaming GRU differs from training"
            # Inclusive continuations overlap. Compare them locally, never sum
            # them as pixel contributions or substitute them into the tracer.
            error = (predicted.double() - target.double()).square().mean(dim=-1)
            error = (error * mask).numpy()
            squared_error[pixels] += error.sum(axis=1)
            vertex_count[pixels] += mask.sum(dim=1).numpy()
            depth_error += error.sum(axis=0)
            depth_count += mask.sum(dim=0).numpy()
            result = lanes(radiance, len(pixels), True) * lanes(weight, len(pixels), True)
            assert np.isfinite(result).all() and (result >= -1e-6).all()
            image[pixels] += result / args.inference_spp
    linear = image.reshape(resolution, resolution, 3).clip(0, None)
    srgb = np.where(linear <= .0031308, 12.92 * linear,
                    1.055 * linear ** (1 / 2.4) - .055).clip(0, 1)
    assert depth_count.sum() > 0
    rmse = np.sqrt(squared_error / np.maximum(vertex_count, 1)).reshape(resolution, resolution)
    rmse = np.ma.masked_where(vertex_count.reshape(resolution, resolution) == 0, rmse)
    assert np.isfinite(rmse.compressed()).all()
    description = ("Full-path GRU continuation validation; left: original path tracer; "
                   "right: per-pixel surface-vertex RGB RMSE in local linear radiance units. "
                   "No neural rendering estimator or Incident Radiance Two-Level MC.")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(srgb, origin="upper")
    axes[0].set_title(f"Original path tracer ({args.inference_spp} SPP, max depth 10)")
    heatmap = axes[1].imshow(rmse, origin="upper", cmap="magma", vmin=0)
    axes[1].set_title("GRU continuation RMSE (all surface bounces)")
    fig.colorbar(heatmap, ax=axes[1], label="Local linear RGB radiance RMSE")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Full traced paths + recurrent prediction validation\n"
                 "Continuation targets; no Incident Radiance / Two-Level MC estimator")
    fig.tight_layout()
    fig.savefig(args.output / "inference.png", dpi=150,
                metadata={"Description": description})
    plt.close(fig)
    print(description)
    for d, count in enumerate(depth_count):
        if count:
            print(f"Bounce {d}: {count} valid vertices, continuation MSE={depth_error[d] / count:.6g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/cornell.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256, help="Complete rays per batch")
    parser.add_argument("--inference-spp", type=int, default=16)
    parser.add_argument("--resume", action="store_true", help="Continue from last checkpoints")
    args = parser.parse_args()
    assert args.epochs > 0 and args.batch_size > 0 and args.inference_spp > 0
    args.device = select_device()
    print(f"PyTorch device: {args.device}")
    torch.manual_seed(42)
    data = torch.load(args.dataset, map_location="cpu", weights_only=True)
    validate_dataset(data)
    args.output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    gru = None
    for kind in ("mlp", "gru"):
        model, history = train_model(kind, data, args)
        for split, label in (("train", "Train"), ("validation", "Validation")):
            ax.plot(range(1, len(history[split]) + 1), history[split],
                    label=f"{kind.upper()} {label} Loss")
        if kind == "gru":
            gru = model
        else:
            del model
    ax.set(xlabel="Epoch", ylabel="Loss", title="Masked RGB continuation MSE")
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(args.output / "loss.png", dpi=150)
    plt.close(fig)
    render_inference(gru, args)
    print(f"Saved best/last checkpoints, loss.png and inference.png to {args.output}")


if __name__ == "__main__":
    main()
