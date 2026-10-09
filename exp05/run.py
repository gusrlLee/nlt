"""EXP05: paired recurrent-memory ablation; writes only report.md after completion.

Run from the project root: conda activate mi; python exp05/run.py
No tracing, package installation, or modification of the original dataset/code.
"""
import argparse
import hashlib
import os
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import FEATURES, TARGET, validate_dataset
from model import RadianceModel, masked_mse
from runtime import select_device

KINDS = ("mlp", "gru_reset", "gru")
SEEDS = (42, 123, 2026)
SPLIT_SEED = 202605


class MemoryAblation(RadianceModel):
    """Reuse the existing grid, encoding, output head and complete-path forward.

    Both GRU modes contain exactly the same GRUCell(26, 64). Resetting happens
    inside the recurrent step in BOTH training and evaluation, never after it.
    """
    def __init__(self, mode, bounds):
        super().__init__("mlp" if mode == "mlp" else "gru", bounds)
        self.mode = mode
        if mode == "mlp":
            # GRU core: 17,664 parameters; closest integer width here: 17,627.
            # Output width/head remain identical; only MLP's intermediate width changes.
            self.core = nn.Sequential(nn.Linear(26, 193), nn.ReLU(),
                                      nn.Linear(193, 64), nn.ReLU())

    def _recurrent_step(self, encoded, mask, hidden):
        if self.mode == "gru_reset":
            hidden = torch.zeros_like(hidden)
        return super()._recurrent_step(encoded, mask, hidden)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(mode, bounds, seed):
    # Every model starts with identical grid/head weights within a seed, and
    # the two GRUs start with identical ALL weights, not just similar budgets.
    seed_everything(seed)
    reference = RadianceModel("gru", bounds)
    model = MemoryAblation(mode, bounds)
    if mode == "mlp":
        model.grid.load_state_dict(reference.grid.state_dict())
        model.head.load_state_dict(reference.head.state_dict())
    else:
        model.load_state_dict(reference.state_dict())
    return model


def audit_models(bounds):
    """User-run preflight: history isolation, shared initialization and gradients."""
    models = {k: make_model(k, bounds, SEEDS[0]) for k in KINDS}
    reset, gru = models["gru_reset"], models["gru"]
    assert reset.state_dict().keys() == gru.state_dict().keys()
    assert all(torch.equal(v, gru.state_dict()[k]) for k, v in reset.state_dict().items())
    for mode in KINDS:
        for part in ("grid", "head"):
            assert all(torch.equal(v, getattr(gru, part).state_dict()[k])
                       for k, v in getattr(models[mode], part).state_dict().items())
    seed_everything(901)
    x = torch.rand(3, 4, 13)
    x[..., :3] = bounds[0] + x[..., :3] * (bounds[1] - bounds[0])
    mask = torch.tensor([[True, True, True, True], [True, True, False, False],
                         [False, False, False, False]])
    changed = x.clone()
    changed[0, 0, 3:12] += 2
    for mode, model in models.items():
        model.train()  # Explicitly test the train-time reset, not only eval.
        y = model(x, mask)
        altered = model(changed, mask)
        assert (y[~mask] == 0).all()
        assert torch.allclose(y[0], model(x[:1], mask[:1])[0], atol=1e-6)
        if mode == "gru":
            assert not torch.allclose(y[0, 1:], altered[0, 1:], atol=1e-7, rtol=1e-6)
        else:
            assert torch.equal(y[0, 1:], altered[0, 1:])
        if mode == "gru_reset":
            independent = torch.stack([model(x[:, d:d+1], mask[:, d:d+1])[:, 0]
                                       for d in range(x.shape[1])], dim=1)
            assert torch.allclose(y, independent, atol=1e-6)
            y[mask].square().mean().backward()
            assert torch.count_nonzero(model.core.weight_hh.grad) == 0
            assert torch.count_nonzero(model.core.weight_ih.grad) > 0
            assert model.grid.tables[0].weight.grad is not None
    assert torch.equal(reset(x, mask)[:, 0], gru(x, mask)[:, 0])
    counts = {}
    for mode, model in models.items():
        counts[mode] = {part: sum(p.numel() for p in getattr(model, part).parameters())
                        for part in ("grid", "core", "head")}
        counts[mode]["total"] = sum(counts[mode].values())
    assert counts["gru_reset"] == counts["gru"]
    assert abs(counts["mlp"]["core"] / counts["gru"]["core"] - 1) < .01
    return counts


def make_splits(data):
    """Keep original train; divide original held-out PIXELS into val/test.

    Grouping entire pixels is stronger than path-only separation: all 64 paths
    from the same pixel stay together. Split is fixed across all training seeds.
    """
    ids = data["pixel_id"]
    heldout = torch.unique(ids[~data["train_ray"]], sorted=True)
    assert len(heldout) >= 2
    order = torch.randperm(len(heldout), generator=torch.Generator().manual_seed(SPLIT_SEED))
    val_ids = heldout[order[:len(heldout) // 2]]
    splits = {"train": data["train_ray"].clone(),
              "validation": (~data["train_ray"]) & torch.isin(ids, val_ids),
              "test": (~data["train_ray"]) & ~torch.isin(ids, val_ids)}
    assert torch.stack(list(splits.values())).sum(0).eq(1).all()
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not set(ids[splits[a]].tolist()) & set(ids[splits[b]].tolist())
    assert all((s & data["mask"].any(1)).any() for s in splits.values())
    return splits


def make_loader(data, selected, batch_size, seed, training=False):
    selected = selected & data["mask"].any(1)
    samples = TensorDataset(*(data[k][selected] for k in ("features", "targets", "mask")))
    return DataLoader(samples, batch_size=batch_size, shuffle=training,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    squared = torch.zeros(10, dtype=torch.float64)
    absolute = torch.zeros_like(squared)
    counts = torch.zeros(10, dtype=torch.int64)
    for x, y, mask in loader:
        prediction = model(x.to(device), mask.to(device)).cpu()
        assert torch.isfinite(prediction).all()
        # Accumulate on CPU in float64; measure raw local linear RGB radiance.
        error = prediction.double() - y.double()
        squared += (error.square() * mask[..., None]).sum((0, 2))
        absolute += (error.abs() * mask[..., None]).sum((0, 2))
        counts += mask.sum(0)
    elements = counts.sum().item() * 3
    assert elements > 0
    return {"mse": squared.sum().item() / elements,
            "mae": absolute.sum().item() / elements,
            "depth_mse": [s.item() / (3 * n.item()) if n else None
                          for s, n in zip(squared, counts)],
            "depth_count": counts.tolist()}


def train_one(mode, seed, data, splits, args, device):
    model = make_model(mode, data["bounds"], seed).to(device)
    # Reset RNG after construction; batch shuffling has its own paired generator.
    seed_everything(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    train = make_loader(data, splits["train"], args.batch_size, seed, True)
    validation = make_loader(data, splits["validation"], args.batch_size, seed)
    best, best_state, best_epoch = float("inf"), None, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, elements = 0., 0
        for x, y, mask in train:
            x, y, mask = (v.to(device) for v in (x, y, mask))
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mse(model(x, mask), y, mask)
            assert torch.isfinite(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            count = mask.sum().item() * 3
            total += loss.item() * count
            elements += count
        metrics = evaluate(model, validation, device)
        if metrics["mse"] < best:
            best, best_epoch = metrics["mse"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"seed={seed} {mode} epoch={epoch}/{args.epochs} "
              f"train={total / elements:.7g} val={metrics['mse']:.7g}", flush=True)
    model.load_state_dict(best_state)
    # Test is queried once per fitted model, after validation-only selection.
    result = {"seed": seed, "mode": mode, "best_epoch": best_epoch,
              "validation": evaluate(model, validation, device),
              "test": evaluate(model, make_loader(data, splits["test"], args.batch_size, seed), device)}
    del model, optimizer, best_state
    return result


def mean_sd(values):
    return f"{statistics.mean(values):.7g} ± {statistics.stdev(values):.3g}"


def slope(depths, values):
    if len(depths) < 2:
        return None
    center = statistics.mean(depths)
    return sum((d - center) * v for d, v in zip(depths, values)) / sum((d - center)**2 for d in depths)


def prior_experiment():
    observations = []
    for mode in ("mlp", "gru"):
        path = ROOT / "outputs" / f"{mode}_best.pt"
        if path.exists():
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            observations.append(f"- 기존 {mode.upper()}: best epoch {checkpoint['epoch']}, "
                                f"validation MSE {checkpoint['history']['validation'][-1]:.7g}.")
    return observations


def write_report(results, data, splits, counts, args, device, digest, versions, prior):
    lookup = {(r["seed"], r["mode"]): r for r in results}
    lines = ["# EXP05 — Recurrent Memory Ablation", "", "## 목표와 통제 조건", "",
             "Hash Grid + GRU의 개선이 이전 bounce의 hidden state 유지에서 오는지 검증한다. "
             "GRU Reset은 학습·평가 모두 매 bounce 직전에 h=0을 사용한다.", "",
             f"- Dataset: `{args.dataset.resolve()}`; SHA256 `{digest}`.",
             f"- Dataset settings: `{data['settings']}`.",
             f"- 환경: conda `{os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}`; "
             f"{versions}; device `{device}`.",
             f"- Feature: `{FEATURES}`. 세 모델 모두 같은 현재 bounce 입력, 같은 Hash Grid와 head.",
             f"- Target (변경 없음): `{TARGET}`.",
             f"- Seeds: {SEEDS}; split seed {SPLIT_SEED}는 모든 run에서 고정.",
             f"- 학습: {args.epochs} epochs, batch {args.batch_size} complete paths, "
             "Adam lr=0.001, 기본 betas/eps, weight decay=0, clip norm=1. "
             "scheduler/early stopping/target 변환 없음.",
             "- 동일 seed의 grid/head 초기값 및 epoch별 batch 순서 동일. GRU/Reset은 전체 초기값 동일.",
             "- 전체 유효 vertex의 RGB 원소 평균 MSE로 학습·validation 선택; 최소 validation MSE epoch를 "
             "모델별 선택한 뒤 test를 한 번 평가. Padding/empty paths는 loss·metric에서 제외.",
             "- CPU float64 합산 MSE/MAE; 깊이별 MSE도 RGB 원소 평균. Depth 0은 첫 surface bounce.",
             "- deterministic algorithms는 warn_only=True; 장치별 비결정적 연산 경고는 콘솔에 표시된다. "
             "동일 seed 재실행의 bitwise 일치는 보장하지 않는다.", "",
             "## 분리와 공정성", "",
             "기존 train pixels를 보존하고 기존 held-out pixels를 validation/test로 절반 분할했다. "
             "동일 pixel의 모든 paths를 한 split에 묶어 path 및 pixel 중복을 차단했다.", "",
             "| Split | Pixels | Paths | Nonempty paths | Valid vertices |", "|---|---:|---:|---:|---:|"]
    for name, selected in splits.items():
        lines.append(f"| {name} | {len(torch.unique(data['pixel_id'][selected]))} | "
                     f"{selected.sum().item()} | {(selected & data['mask'].any(1)).sum().item()} | "
                     f"{data['mask'][selected].sum().item()} |")
    lines += ["", "| Model | Grid | Core | Head | Total |", "|---|---:|---:|---:|---:|"]
    for mode in KINDS:
        c = counts[mode]
        lines.append(f"| {mode} | {c['grid']} | {c['core']} | {c['head']} | {c['total']} |")
    lines += ["", "MLP는 26→193→64, GRU/Reset은 GRUCell(26,64), 공통 head 64→32→3+Softplus. "
              "MLP core는 GRU 대비 37 parameters 적다 (약 0.21%). 전체 budget뿐 아니라 core budget도 맞췄다.",
              "GRU Reset의 weight_hh는 구조에 포함되지만 h=0 때문에 gradient가 0이다. "
              "이는 reset intervention 자체의 필연적 차이이며 유효 용량/최적화 경로까지 같다는 뜻은 아니다.",
              "실행 전 train-mode reset, 이전 입력 격리, 첫 bounce 일치, "
              "GRU/Reset 구조·초기값 일치와 reset recurrent-weight gradient=0 assertions를 통과했다.", "",
              "## 기존 실험과 해석 범위", ""]
    lines += prior or ["기존 best checkpoint가 없어 수치 비교를 기록하지 않았다."]
    lines += ["", "이전 학습 체크포인트는 EXP05 초기화에 사용하지 않았다. 기존 실험의 validation에 "
              "이번 test pixels도 포함되어 있었으므로 완전히 새로운 confirmatory holdout은 아니다. "
              "EXP05 안에서는 test로 튜닝/epoch 선택을 하지 않는다. 결과는 고정 Cornell dataset의 "
              "training-seed 변동이며 dataset/scene 일반화나 엄밀한 통계적 유의성을 입증하지 않는다. "
              "Target은 단일 sampled continuation의 noisy estimator이고 깊은 bounce는 생존 path가 적다.", "",
              "## 실제 측정값", "", "| Seed | Model | Best epoch | Val MSE | Val MAE | Test MSE | Test MAE |",
              "|---:|---|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['seed']} | {r['mode']} | {r['best_epoch']} | "
                     f"{r['validation']['mse']:.7g} | {r['validation']['mae']:.7g} | "
                     f"{r['test']['mse']:.7g} | {r['test']['mae']:.7g} |")
    lines += ["", "Seed 평균 ± sample SD (n=3):", "",
              "| Model | Val MSE | Val MAE | Test MSE | Test MAE |", "|---|---:|---:|---:|---:|"]
    for mode in KINDS:
        rows = [lookup[s, mode] for s in SEEDS]
        lines.append("| " + mode + " | " + " | ".join(mean_sd([r[split][metric] for r in rows])
                     for split in ("validation", "test") for metric in ("mse", "mae")) + " |")
    lines += ["", "## Depth별 Test MSE", "",
              "| Seed | Depth | Vertices | MLP | GRU Reset | GRU | Reset−GRU | Reduction % |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    trends = []
    for seed in SEEDS:
        depths, absolute, relative, relative_depths = [], [], [], []
        for d in range(10):
            metrics = [lookup[seed, k]["test"] for k in KINDS]
            assert all(m["depth_count"] == metrics[0]["depth_count"] for m in metrics)
            n = metrics[0]["depth_count"][d]
            if not n:
                lines.append(f"| {seed} | {d} | 0 | N/A | N/A | N/A | N/A | N/A |")
                continue
            mlp, reset, gru = [m["depth_mse"][d] for m in metrics]
            gain = reset - gru
            percent = 100 * gain / reset if reset > 0 else None
            lines.append(f"| {seed} | {d} | {n} | {mlp:.7g} | {reset:.7g} | {gru:.7g} | "
                         f"{gain:.7g} | {percent:.4g} |" if percent is not None else
                         f"| {seed} | {d} | {n} | {mlp:.7g} | {reset:.7g} | {gru:.7g} | {gain:.7g} | N/A |")
            # First bounce has no history; depth trend concerns subsequent vertices.
            if d >= 1:
                depths.append(d)
                absolute.append(gain)
                if percent is not None:
                    relative_depths.append(d)
                    relative.append(percent)
        trends.append((seed, slope(depths, absolute), slope(relative_depths, relative)))
    lines += ["", "Depth별 평균 ± sample SD:", "",
              "| Depth | Test vertices | MLP | GRU Reset | GRU | Paired Reset−GRU |",
              "|---:|---:|---:|---:|---:|---:|"]
    for d in range(10):
        n = lookup[SEEDS[0], "gru"]["test"]["depth_count"][d]
        if n:
            columns = [mean_sd([lookup[s, k]["test"]["depth_mse"][d] for s in SEEDS]) for k in KINDS]
            columns.append(mean_sd([lookup[s, "gru_reset"]["test"]["depth_mse"][d] -
                                    lookup[s, "gru"]["test"]["depth_mse"][d] for s in SEEDS]))
            lines.append(f"| {d} | {n} | " + " | ".join(columns) + " |")
        else:
            lines.append(f"| {d} | 0 | N/A | N/A | N/A | N/A |")
    lines += ["", "## 질문에 대한 관측 기반 답변", ""]
    for baseline, question in (("mlp", "GRU가 MLP보다 좋은가?"),
                               ("gru_reset", "GRU가 GRU Reset보다 좋은가?")):
        lines.append(f"**{question}**")
        for metric in ("mse", "mae"):
            deltas = [lookup[s, baseline]["test"][metric] - lookup[s, "gru"]["test"][metric] for s in SEEDS]
            wins = sum(v > 0 for v in deltas)
            lines.append(f"- Test {metric.upper()}: {baseline}−GRU = {mean_sd(deltas)}; "
                         f"GRU 개선 {wins}/3 seeds. " +
                         ("세 seed에서 일관된 개선을 관측했다." if wins == 3 else
                          "세 seed에서 일관된 개선은 관측되지 않았다."))
        lines.append("")
    lines += ["**깊어질수록 memory 효과가 커지는가?**", "",
              "Depth ≥ 1의 Reset−GRU 절대 MSE 차이와 상대 감소율에 대한 동일 가중 선형 기울기. "
              "양수는 증가 경향이며, 단조 증가나 인과적 증명은 아니다.", "",
              "| Seed | Absolute gain slope / bounce | Relative gain slope (pp / bounce) |",
              "|---:|---:|---:|"]
    for seed, a, r in trends:
        lines.append(f"| {seed} | {a:.7g} | {r:.7g} |" if a is not None and r is not None else
                     f"| {seed} | {a if a is not None else 'N/A'} | {r if r is not None else 'N/A'} |")
    for index, label in ((1, "절대 효과"), (2, "상대 효과")):
        values = [t[index] for t in trends]
        lines.append(f"- {label}: " + (f"양의 기울기 {sum(v > 0 for v in values)}/3 seeds; {mean_sd(values)}."
                     if all(v is not None for v in values) else "관측 가능한 depth 부족으로 판단 불가."))
    lines += ["", "Memory에 대한 핵심 비교는 동일 GRU 구조의 GRU vs GRU Reset이다. "
              "MLP 비교만으로 memory 원인을 확정하지 않는다. 위 paired 차이와 깊이별 표가 "
              "혼재하면 memory 가설의 일관된 지지로 해석하지 않는다.", ""]
    path = Path(__file__).resolve().parent / "report.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/cornell.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--check-only", action="store_true", help="Audit data/models; do not train or write results")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch-size must be positive")
    import drjit
    import mitsuba
    if mitsuba.__version__ != "3.9.1" or drjit.__version__ != "1.5.0":
        raise RuntimeError("PLAN.md requires Mitsuba 3.9.1 and Dr.Jit 1.5.0; no environment changes were made")
    data = torch.load(args.dataset, map_location="cpu", weights_only=True)
    validate_dataset(data)
    assert data["settings"]["scene"] == "cornell-box"
    assert data["settings"]["max_depth"] == 10 and data["settings"]["rr_depth"] == 5
    counts = audit_models(data["bounds"])
    splits = make_splits(data)
    device = select_device()
    print(f"Preflight passed; auto device={device}; seeds={SEEDS}; parameters={counts}", flush=True)
    if args.check_only:
        return
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    with args.dataset.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    prior = prior_experiment()
    results = []
    for seed in SEEDS:
        for mode in KINDS:
            results.append(train_one(mode, seed, data, splits, args, device))
    versions = f"Mitsuba {mitsuba.__version__}, Dr.Jit {drjit.__version__}, PyTorch {torch.__version__}"
    write_report(results, data, splits, counts, args, device, digest, versions, prior)


if __name__ == "__main__":
    main()
