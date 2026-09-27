"""Paired bootstrap over patients for the dual-image comparison.

Comparing 0.824 / 0.836 / 0.840 by eye is meaningless when the fold-to-fold
spread is ~0.017.  This resamples patients with replacement, recomputes every
model's macro-AUROC on the same resample, and reports the distribution of the
paired differences.
"""

import glob
import os
import sys

import numpy as np

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import mm_eval

RESULTS = r"E:\模型\mm4class\results"
TARGETS = {
    "单图 stacking": "oof_stack_Stacking_融合.npz",
    "表格+DR": "oof_dual_表格_DR__无超声_.npz",
    "双图 stacking": "oof_dual_双图_stacking__表格_DR_超声_.npz",
}


def load(name):
    for path in glob.glob(os.path.join(RESULTS, "oof_*.npz")):
        if os.path.basename(path) == name:
            z = np.load(path)
            return z["y"], z["oof"], z["patient_id"]
    raise FileNotFoundError(name)


data = {k: load(v) for k, v in TARGETS.items()}
y, _, groups = next(iter(data.values()))
patients = np.unique(groups)
pat_index = {p: np.flatnonzero(groups == p) for p in patients}

print(f"样本 {len(y)}, 患者 {len(patients)}\n")
print("各模型在完整数据上的 macro-AUROC:")
for k, (yy, oof, _) in data.items():
    print(f"  {k:<16} {mm_eval.compute_metrics(yy, oof)['macro_auroc']:.3f}")

rng = np.random.default_rng(0)
N_BOOT = 2000
diffs = {f"{a} - {b}": [] for a in data for b in data if a < b}

for _ in range(N_BOOT):
    pick = rng.choice(patients, size=len(patients), replace=True)
    idx = np.concatenate([pat_index[p] for p in pick])
    if len(np.unique(y[idx])) < 4:
        continue
    scores = {k: mm_eval.compute_metrics(y[idx], oof[idx])["macro_auroc"]
              for k, (_, oof, _) in data.items()}
    for a in data:
        for b in data:
            if a < b:
                diffs[f"{a} - {b}"].append(scores[a] - scores[b])

print()
print("配对自助检验 (2000 次患者级重采样):")
print(f"{'比较':<34}{'差值均值':>10}{'95% CI':>20}{'P(>0)':>10}")
for k, vals in diffs.items():
    arr = np.array(vals)
    ci = np.percentile(arr, [2.5, 97.5])
    print(f"{k:<34}{arr.mean():>+10.3f}"
          f"{'['+format(ci[0],'+.3f')+', '+format(ci[1],'+.3f')+']':>20}"
          f"{(arr > 0).mean():>10.3f}")
print()
print("P(>0) 接近 0.5 = 无差异; 95% CI 跨 0 = 不显著")
