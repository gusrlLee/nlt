# EXP05 — Validation / Test 차이 진단

## 현재 결론과 근거의 범위

`report.md`, `run.py`, `dataset.py`, `model.py`, `path_tracer.py`, Cornell Box scene과 현재 파일 목록을 확인했다. 아래 기존 결과 수치는 **완료된 EXP05 보고서에서 읽은 값**이다. 요청대로 추가 분석 프로그램·추론·학습을 실행하지 않았다. Dataset의 평균·분산·분위수는 아직 측정하지 않았으며, 아래 실행 블록으로 실제 값을 이 파일에 추가하도록 준비했다. 확인하지 않은 수치를 채우지 않았다.

현재 가장 중요한 제약은 **EXP05가 학습된 가중치와 pixel별 예측을 저장하지 않았다는 점**이다. `train_one()`은 최적 가중치를 메모리에만 보관하고 평가 후 삭제한다. 현재 `exp05/`에는 실행 코드와 집계 보고서만 있다. `outputs/*_best.pt`는 이전 실험이며 EXP05 모델이 아니다. 특히 이전 MLP는 26→64→64, EXP05 MLP는 26→193→64로 구조도 다르다. 이 가중치를 사용해 EXP05 pixel별 MSE를 계산하면 다른 실험을 분석하게 된다.

따라서 Dataset만으로 target 분포와 고휘도 pixel의 위치는 측정할 수 있지만, **EXP05의 pixel별 모델 MSE·고휘도 구간별 모델 오차는 현재 자료에서 복원할 수 없다.** 아래에 그 항목을 미측정으로 명시했다. 재학습은 하지 않으며, 외부에 따로 보관된 EXP05 가중치/예측이 있는 경우에만 무학습 재평가가 가능하다.

## 1. 기존 결과에서 확인되는 현상

3개 training seed 평균이다. Validation / Test의 역할이나 단위가 다른 값이 아니라, 동일하게 유효 surface vertex RGB 원소의 평균 오차다.

| Model | Validation MSE | Test MSE | Val/Test MSE 비율 | Validation MAE | Test MAE |
|---|---:|---:|---:|---:|---:|
| MLP | 0.3082361 | 0.0171952 | 약 17.93 | 0.07495061 | 0.04877739 |
| GRU Reset | 0.2753802 | 0.0261506 | 약 10.53 | 0.07964104 | 0.05421604 |
| GRU | 0.1015773 | 0.02332481 | 약 4.36 | 0.08067413 | 0.06561407 |

- GRU는 Validation MSE에서 MLP 대비 약 67.0% 개선하지만, Test MSE에서는 약 35.6% 악화한다. 이 방향은 세 seed 모두 동일하다.
- Validation MAE에서는 GRU가 MLP보다 좋은 seed가 2026 하나뿐이다. **Validation에서 GRU의 우위는 모든 오차 지표의 우위가 아니라 MSE의 우위**다.
- MLP의 Val/Test MAE 비율은 약 1.54인 반면 MSE 비율은 약 17.93이다. 큰 residual이 MSE에 강하게 영향을 주는 패턴과 양립한다. 하지만 집계값만으로 residual의 위치·개수나 target outlier를 특정할 수는 없다.
- Training seeds는 모두 같은 Validation/Test split을 사용한다. 3개 seed의 일관성은 이 split에서의 초기화·최적화 변동에 대한 결과이며, 서로 다른 세 데이터 분할에서 재현된 결과가 아니다.

### Test 차이가 어느 depth에서 생기는가

보고서의 depth별 평균에서 GRU−MLP 차이를 전체 vertex 수로 가중하면, Depth 0 차이는 전체 Test MSE 격차 약 0.00612961 중 약 0.004369, 즉 **약 71.3%**를 차지한다.

| 항목 | 보고서에 근거한 값 |
|---|---:|
| 전체 Test 유효 vertices | 23,578 |
| Depth 0 vertices | 6,592 (약 28.0%) |
| Depth 0 MLP MSE | 0.01004746 |
| Depth 0 GRU MSE | 0.02567484 |
| Depth 0 GRU−MLP MSE | 0.01562738 |

Depth 0에는 이전 bounce가 없다. 따라서 Test 역전의 큰 부분을 **평가 시 이전 hidden state의 오염**으로 설명할 수는 없다. 다만 GRU의 첫 bounce 예측에도 전체 sequence loss로 학습된 공유 가중치와 validation으로 선택한 epoch가 영향을 주므로, 이 사실이 recurrent 학습의 간접 영향을 배제하는 것은 아니다.

GRU는 Test Depth 1/2/3에서 GRU Reset보다 세 seed 모두 좋은 MSE를 보인다. 반면 GRU와 MLP 비교에서는 Depth 1/2에서도 평균 MSE가 더 높다. 이는 일부 depth에서 recurrence의 이득이 관측된다는 것과 MLP 대비 전체 일반화 우위를 구분해야 함을 보여준다.

Depth 4는 전체 vertices의 약 13.5%지만 MLP Test SSE의 약 40.6%를 차지한다. Depth 5~9는 합계 697 vertices, 전체의 약 3.0%다. 후반 depth의 불안정성과 큰 상대 감소율을 전체 역전의 주원인으로 단정해서는 안 된다.

## 2. 분할·평가 코드 점검

| 점검 항목 | 소스에서 확인한 동작 | 판단 |
|---|---|---|
| Path 분리 | 하나의 Dataset 행이 전체 10-bounce path이며 행 전체가 같은 split에 속함 | bounce 단위 누출 없음 |
| Pixel 분리 | 기존 train 보존, held-out pixel 205개를 고정 permutation으로 Val 102 / Test 103개 분할 | 같은 pixel의 64 paths가 split을 넘지 않음 |
| 분할 중복·누락 | 세 split 합이 행마다 1인지, pixel 집합이 서로 겹치지 않는지 assert | 코드에 검사 존재; 아래 실행에서 현 Dataset 재검증 |
| 그룹의 실질 크기 | Val 6,528 paths / Test 6,592 paths지만 pixel은 각각 102 / 103개 | pixel 내 64 paths를 독립 공간 표본으로 간주하면 안 됨 |
| 밝기 균형 | held-out pixel 무작위 분할, target 밝기·분산·광원 위치 층화 없음 | 표본 분포 불균형 가능; 실제 크기는 미측정 |
| Loss / MSE 분모 | `masked_mse`는 valid RGB 평균, evaluate는 SSE / (valid vertices × 3) | 동일한 정의 |
| 평균 합산 | evaluate는 CPU float64로 SSE를 합한 뒤 전체 원소 수로 나눔 | batch별 MSE의 단순 평균 오류 없음 |
| Padding | invalid vertices는 합산과 분모에서 제외 | 경로 길이 차이를 padding 오차로 계산하지 않음 |
| GRU state | forward 시작마다 path별 h=0, mask에 따라 업데이트 | batch·path·split 사이 state 이월 없음 |
| Reset | recurrent step 직전 h=0, train/eval 동일 | 평가 시에만 reset하는 오류 없음 |
| Checkpoint | 각 모델의 최소 Validation MSE epoch를 선택하고 같은 상태로 Val/Test 평가 | test epoch selection 없음 |
| Target | 기존 inclusive local continuation 그대로, RR survival weight 포함 | 이번 평가가 pixel radiance나 log target으로 바뀐 것 아님 |

보고서의 Val/Test vertex 수 차이는 약 5.8%이고 평가가 각각 원소 수로 정규화되므로, **단순히 Test 행 수가 더 많은 것이 4~18배 MSE 차이를 만드는 설명은 아니다.** 다만 서로 다른 depth 구성·target 분포는 정규화 이후에도 영향을 준다.

소스 검토에서 잘못된 분모, Val/Test 교환, state 누출, batch 평균 오류는 발견하지 않았다. 이것은 runtime 결과 전체의 검증 완료를 의미하지 않는다. Dataset hash, 실제 split 개수, depth 집계와 전체 Test MSE의 일치 여부는 아래 실행에서 확인한다. Validation depth별 MSE는 보고서에 없으므로 같은 방식의 복원 검사는 불가능하다.

## 3. 원인 후보와 확정에 필요한 증거

| 원인 후보 | 현재 근거 | 아직 필요한 확인 |
|---|---|---|
| Val/Test target 분포·고휘도 pixel 구성 차이 | 102/103 pixel 무층화 분할, MSE/MAE의 다른 순위 | valid target의 평균·분산·P90/P99/Max, 공통 밝기 구간 점유율, pixel별 target energy |
| 소수 큰 residual이 Val MSE를 지배 | MLP Val MSE 차이에 비해 MAE 차이는 작음 | EXP05 pixel별 SSE와 상위 pixel의 SSE 점유율; target 통계만으로는 확정 불가 |
| Validation에 특화된 epoch 선택 | MLP best epoch 2/2/1, GRU 2/3/4; 선정 목적은 Val MSE | 동일 기존 checkpoint들의 밝기·depth별 오차와, 보관된 epoch별 예측이 있다면 epoch 선택 민감도 |
| Recurrent 학습이 첫 bounce fitting에 영향 | 전체 Test 차이의 약 71.3%가 h=0인 Depth 0에서 발생 | 동일 trained weights의 Depth 0 pixel별 오차와 밝기 구간별 분해 |
| Target 자체의 Monte Carlo 변동 | 64 paths/pixel의 개별 sampled target, local RR weight, NEE/MIS | 같은 pixel 내 target 분산과 상위 target의 path/depth 집중도; 이는 noise floor의 직접 추정치는 아님 |

**현 시점의 답변:** GRU가 Validation에서는 MLP보다 좋은데 Test에서는 나쁜 현상은 확인되었다. 그러나 “Validation에 광원 pixel이 몰려서 GRU가 유리했다” 또는 “GRU가 고휘도에 과적합했다”는 결론은 아직 증거가 없다. 데이터 측정과 모델 residual 측정을 분리해야 한다. 큰 target은 모델이 잘 맞출 수도 있으므로 target energy를 모델 오차로 대체하면 안 된다.

Cornell scene의 면광원 RGB radiance는 (17,12,4)다. 직접 emitter hit가 포함되는 target 정의를 고려하면 분포를 분석할 이유는 충분하지만, 이 값이 Dataset의 관측 Max라는 뜻은 아니다. NEE/MIS·RR·continuation에 의해 target 값이 달라질 수 있다. 임의 clipping이나 target 정의 변경은 하지 않는다.

## 4. 실행할 최소 분석 — 학습·추론 없음

아래 블록은 기존 Dataset과 기존 분할 함수만 사용한다. Dataset 생성, 모델 생성, optimizer, forward/backward, package 설치는 호출하지 않는다. 유효 RGB 원소 전체와 R/G/B 각각, linear RGB의 진단용 luminance `Y=0.2126R+0.7152G+0.0722B`의 분포를 측정한다. 분산은 population variance, 분위수는 linear interpolation이다.

각 split의 분위수는 분포 요약에 사용하고, 밝기 구간 비교는 **Train에서 계산한 공통 P90/P99**를 사용한다. Pixel 밝기는 Depth 0 target luminance의 64-path 평균이다. 모든 depth의 continuation을 더해 pixel radiance로 해석하지 않는다. Pixel별 `E[target²]`는 zero predictor의 참고 MSE이며 **EXP05 모델 MSE가 아니다.**

추가 파일을 만들지 않도록 분석 코드를 이 문서에 포함했다. 프로젝트 루트에서 아래 명령을 실행하면 측정값이 이 문서 마지막에 추가되며, 재실행하면 측정 섹션만 갱신된다. 원래 `report.md`, Dataset, 코드는 변경하지 않는다.

```bash
conda activate mi
python -B -c 'from pathlib import Path; s=Path("exp05/analysis.md").read_text(); exec(compile(s.split("```python\n",1)[1].split("```",1)[0], "exp05/analysis.md", "exec"))'
```

```python
import hashlib
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
root = Path.cwd()
assert (root / "exp05/run.py").is_file(), "프로젝트 루트에서 실행하세요"
sys.path.insert(0, str(root / "exp05"))
import numpy as np
import torch
from run import make_splits
from dataset import validate_dataset

dataset_path = root / "data/cornell.pt"
report = (root / "exp05/report.md").read_text(encoding="utf-8")
expected = re.search(r"SHA256 `([0-9a-f]{64})`", report).group(1)
with dataset_path.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
assert digest == expected, "Dataset이 EXP05 report의 hash와 다릅니다"
data = torch.load(dataset_path, map_location="cpu", weights_only=True)
validate_dataset(data)
splits = make_splits(data)
y = data["targets"].numpy().astype(np.float64)
mask = data["mask"].numpy()
pixel = data["pixel_id"].numpy()
lum = y @ np.array([.2126, .7152, .0722])
lines = ["## 사용자 실행 측정값", "", f"Dataset SHA256: `{digest}`", "",
         "아래 수치는 실제 Dataset 계산값이다. 모델 inference/학습은 실행하지 않았다.", ""]

def table(header, rows):
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    lines.append("")

def summary(a):
    a = np.asarray(a).reshape(-1)
    if not len(a):
        return [0] + ["N/A"] * 6
    return [len(a)] + [f"{v:.9g}" for v in
        (a.mean(), a.var(ddof=0), np.quantile(a, .9), np.quantile(a, .99), a.max(), (a*a).mean())]

counts, stats, depth_stats, pixel_rows = [], [], [], {}
depth_counts = {}
for name, selected in splits.items():
    sel = selected.numpy()
    valid = sel[:, None] & mask
    ids, group = np.unique(pixel[sel], return_inverse=True)
    paths_per_pixel = np.bincount(group)
    assert np.all(paths_per_pixel == data["settings"]["spp"])
    counts.append([name, len(ids), int(sel.sum()), int(valid.sum()),
                   int((sel & mask.any(1)).sum()), int(paths_per_pixel.min()), int(paths_per_pixel.max())])
    for label, values in [("RGB pooled", y[valid].reshape(-1)),
                          ("R", y[..., 0][valid]), ("G", y[..., 1][valid]),
                          ("B", y[..., 2][valid]), ("Y", lum[valid])]:
        stats.append([name, label] + summary(values))
    depth_counts[name] = valid.sum(0)
    for d in range(10):
        depth_stats.append([name, d] + summary(lum[:, d][valid[:, d]]))
    per_pixel = []
    for p in ids:
        rows = pixel == p
        m = mask[rows]
        a = lum[rows][m]
        first = mask[rows, 0]
        first_lum = lum[rows, 0][first]
        energy = (y[rows][m] ** 2).mean()
        per_pixel.append({"id": int(p), "n": int(m.sum()),
                          "first_y": float(first_lum.mean()) if first.any() else float("nan"),
                          "mean_y": float(a.mean()), "var_y": float(a.var()),
                          "p99_y": float(np.quantile(a, .99)), "max_y": float(a.max()),
                          "energy": float(energy)})
    pixel_rows[name] = per_pixel

table(["Split", "Pixels", "Paths", "Vertices", "Nonempty paths", "Paths/pixel min", "max"], counts)
expected_counts = {"train": (819, 52416, 180784),
                   "validation": (102, 6528, 22293), "test": (103, 6592, 23578)}
assert all(tuple(row[1:4]) == expected_counts[row[0]] for row in counts), "EXP05 report의 표본 수와 불일치"
lines += ["Dataset hash·split 배타성·path/pixel grouping·report 표본 수 일치: PASS.", "",
          "### Target 분포 (padding 제외)", ""]
table(["Split", "Quantity", "N", "Mean", "Variance", "P90", "P99", "Max", "E[value²]"], stats)
lines += ["### Depth별 target luminance와 생존 표본 수", ""]
table(["Split", "Depth", "Vertices", "Mean Y", "Variance Y", "P90 Y", "P99 Y", "Max Y", "E[Y²]"], depth_stats)

train_valid = splits["train"].numpy()[:, None] & mask
q90, q99 = np.quantile(lum[train_valid], [.9, .99])
lines += ["### 공통 밝기 구간 비교", "",
          f"Train valid-vertex Y 기준 P90={q90:.9g}, P99={q99:.9g}. "
          "경계값 동률은 낮은 구간에 포함하므로 정확히 10%/1%가 아닐 수 있다.", ""]
band_rows = []
for name, selected in splits.items():
    valid = selected.numpy()[:, None] & mask
    total_energy = (y[valid]**2).sum()
    for label, band in [("Y <= Train P90", lum <= q90),
                        ("Train P90 < Y <= P99", (lum > q90) & (lum <= q99)),
                        ("Y > Train P99", lum > q99)]:
        v = valid & band
        n = int(v.sum())
        energy = (y[v]**2).sum()
        band_rows.append([name, label, n, f"{100*n/valid.sum():.6g}",
                          f"{lum[v].mean():.9g}" if n else "N/A",
                          f"{100*energy/total_energy:.6g}" if total_energy else "N/A"])
table(["Split", "Band", "Vertices", "Vertex %", "Mean Y", "Target RGB energy % (모델 SSE 아님)"], band_rows)

lines += ["### Pixel 수준의 target 구성", "",
          "First-bounce Y는 camera pixel 밝기의 proxy이고 all-depth Y는 local continuation 분포다. "
          "각 pixel의 target 분산에는 서로 다른 깊이/상태의 변화도 포함되므로 순수 Monte Carlo noise variance가 아니다.", ""]
pixel_stats = []
for name, rows in pixel_rows.items():
    for quantity in ("first_y", "energy"):
        pixel_stats.append([name, quantity] + summary([r[quantity] for r in rows if np.isfinite(r[quantity])]))
table(["Split", "Pixel quantity", "Pixels", "Mean", "Variance", "P90", "P99", "Max", "Second moment"], pixel_stats)
train_first = np.array([r["first_y"] for r in pixel_rows["train"]])
train_first = train_first[np.isfinite(train_first)]
bright_threshold = np.quantile(train_first, .9)
lines += [f"고휘도 pixel의 공통 기준: Train pixel mean Depth-0 Y > {bright_threshold:.9g} (P90).", ""]
bright_rows = []
for name, rows in pixel_rows.items():
    high = [r for r in rows if r["first_y"] > bright_threshold]
    bright_rows.append([name, len(high), len(rows), f"{100*len(high)/len(rows):.6g}"])
table(["Split", "Bright pixels", "All pixels", "Bright pixel %"], bright_rows)
for name, rows in pixel_rows.items():
    lines += [f"#### {name}: 가장 밝은 pixel 10개", ""]
    brightest = sorted(rows, key=lambda r: r["first_y"], reverse=True)[:10]
    table(["Pixel", "(x,y)", "Vertices", "Mean Depth-0 Y", "All-depth mean Y", "Var Y", "P99 Y", "Max Y", "E[RGB²], not model MSE"],
          [[r["id"], (r["id"] % data["settings"]["resolution"], r["id"] // data["settings"]["resolution"]),
            r["n"]] + [f"{r[k]:.9g}" for k in ("first_y", "mean_y", "var_y", "p99_y", "max_y", "energy")]
           for r in brightest])
    ranked = sorted(rows, key=lambda r: r["energy"]*r["n"], reverse=True)
    denominator = sum(r["energy"]*r["n"] for r in rows)
    lines.append("Target RGB energy 집중도 (모델 residual 아님): " + ", ".join(
        f"상위 {n} pixels {100*sum(r['energy']*r['n'] for r in ranked[:n])/denominator:.6g}%"
        for n in (1, 5, 10)) + ".")
    lines.append("")

# Parse only already measured EXP05 rows. No model/checkpoint is loaded.
section = report.split("## 실제 측정값", 1)[1].split("Seed 평균", 1)[0]
overall = {}
for line in section.splitlines():
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) == 7 and cells[0] in ("42", "123", "2026"):
        overall[int(cells[0]), cells[1]] = list(map(float, cells[3:]))
section = report.split("## Depth별 Test MSE", 1)[1].split("Depth별 평균", 1)[0]
by_depth = {}
for line in section.splitlines():
    c = [v.strip() for v in line.strip().strip("|").split("|")]
    if len(c) == 8 and c[0] in ("42", "123", "2026"):
        s, d, n = map(int, c[:3])
        assert n == depth_counts["test"][d]
        by_depth[s, d] = list(map(float, c[3:6]))
assert len(overall) == 9 and len(by_depth) == 30
lines += ["### 기존 Test 평가 집계의 산술 검증", ""]
audit = []
for s in (42, 123, 2026):
    for index, mode in enumerate(("mlp", "gru_reset", "gru")):
        reconstructed = sum(by_depth[s, d][index] * depth_counts["test"][d] for d in range(10)) / depth_counts["test"].sum()
        measured = overall[s, mode][2]
        assert np.isclose(reconstructed, measured, rtol=2e-6, atol=2e-8), "depth/global MSE inconsistency"
        audit.append([s, mode, f"{measured:.9g}", f"{reconstructed:.9g}", f"{abs(measured-reconstructed):.3g}"])
table(["Seed", "Model", "Reported Test MSE", "Depth-weighted MSE", "Difference"], audit)
lines += ["### Test GRU−MLP 격차의 depth별 분해", ""]
contribution_rows = []
for d in range(10):
    delta = np.mean([by_depth[s, d][2] - by_depth[s, d][0] for s in (42, 123, 2026)])
    contribution = delta * depth_counts["test"][d] / depth_counts["test"].sum()
    contribution_rows.append([d, int(depth_counts["test"][d]), f"{delta:.9g}", f"{contribution:.9g}"])
table(["Depth", "Vertices", "GRU−MLP depth MSE", "Contribution to overall difference"], contribution_rows)

lines += ["### 측정의 해석과 남아 있는 항목", "",
          "Target 분포·밝기 구간·pixel energy는 Dataset에 근거한 관측이다. "
          "이 표만으로 GRU residual의 고휘도 집중이나 validation checkpoint의 편향을 확정하지 않는다.", "",
          "EXP05 pixel별 모델 MSE, 고휘도 구간별 모델 MSE/MAE, 상위-error pixel의 SSE 점유율: "
          "**측정 불가 — 해당 trained weights/예측이 저장되어 있지 않음.** "
          "기존 outputs checkpoint로 대체하거나 새로운 학습을 수행하지 않았다.", ""]
path = root / "exp05/analysis.md"
marker = "\n<!-- ACTUAL_DATA_MEASUREMENTS -->\n"
base = path.read_text(encoding="utf-8").split(marker, 1)[0].rstrip()
path.write_text(base + marker + "\n" + "\n".join(lines), encoding="utf-8")
print("Dataset-only analysis completed; updated exp05/analysis.md; no training/inference")
```

## 5. Pixel별 모델 오차를 확인하는 최소 후속 절차

추가 학습 없이 가능한 범위는 위 Dataset 진단과 기존 집계값의 재검산까지다. 따로 보관된 EXP05 best weights 또는 per-path predictions가 있다면 다음 순서로 평가만 수행한다. **현재는 해당 자료가 없으므로 아래 결과는 작성하지 않았다.**

1. Dataset SHA256, seed, mode, best epoch가 기존 보고서와 같은지 확인한다. 원래 split 및 target을 유지한다.
2. 모델을 eval/no_grad로 두고 기존 전체 path forward를 사용한다. Prediction과 path 행 번호를 매칭한 뒤 pixel ID로 묶는다. 단순 초기화 모델이나 이전 실험 weights를 대신 사용하지 않는다.
3. Pixel `p`마다 `SSE_p = sum_valid_RGB(pred−target)²`, `N_p = 3 × valid vertices`, `MSE_p = SSE_p/N_p`를 계산한다. 전체 MSE는 `sum(SSE_p)/sum(N_p)`로 재구성한다. Pixel MSE의 균등 평균은 다른 지표로 별도 표시한다.
4. Train target에서 정한 공통 밝기 P90/P99로 밝은 vertex 오차를 나누고, Depth 0 평균 Y의 Train P90으로 밝은 camera pixel 그룹을 나눈다. 모델마다 다른 error 기반 bright threshold를 사용하지 않는다.
5. 상위 1/5/10 error pixels의 전체 SSE 점유율, 동일 pixel에서 GRU−MLP 차이, 밝은/나머지 그룹의 MSE·MAE·표본 수·SSE 점유율을 비교한다. Outlier 제외 MSE는 민감도 분석으로만 보고하고 원래 metric을 교체하지 않는다.
6. Validation/Test를 합친 기존 held-out 영역에서 **고정된 예측**의 pixel 단위 bootstrap/repartition으로 순위 민감도를 확인할 수 있다. 이는 checkpoint가 기존 Validation으로 이미 선택된 뒤의 사후 분석이며, 새로운 독립 Test 검증으로 해석하지 않는다. Paths나 bounce를 독립 bootstrap 표본으로 쓰지 않는다.

현재 최우선 작업은 Dataset 분포를 측정하는 것이다. 분포 차이가 확인되어도 그것만으로 모델 순위 역전의 원인을 확정하지 않는다. EXP05 residual 자료가 없다는 한계는 최소 결과물 정책으로 weights/예측까지 버린 데서 발생했다. 새 실험이나 재학습을 실행하지 않고 이 한계를 명시하는 것이 현재 자료에 맞는 결론이다.
