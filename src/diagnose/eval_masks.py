"""Evaluate the center-only and border-only DR variants against the full image."""

import os
import sys

import numpy as np

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import mm_data
import mm_eval
from mm_models import LightGBMHead

N_REPEATS = 3
FDIR = r"E:\模型\mm4class\features"


def run(name, X, y, groups, folds):
    res = mm_eval.run_cv(
        name, y, groups,
        lambda tr, te: LightGBMHead(seed=0).fit(X[tr], y[tr])
        .predict_proba(X[te]),
        folds=folds,
    )
    s = res.summary()
    print(f"  {name:<22} macro-AUROC {s['macro_auroc_mean']:.3f} ± "
          f"{s['macro_auroc_std']:.3f}   ARDS AUPRC {s['ards_auprc_mean']:.3f}")
    return s["macro_auroc_mean"]


cohort = mm_data.load_cohort()
dr = cohort.subset(cohort.modality == "DR")
fold_y = dr.label
folds = mm_eval.make_folds(fold_y, dr.groups, n_repeats=N_REPEATS)
print(f"DR 子集 {len(dr)} 条, ARDS {int((dr.label == 1).sum())} 例, {len(folds)} 折\n")

z = np.load(f"{FDIR}/features_dr_masks.npz")
order = {int(p): i for i, p in enumerate(z["center_ids"])}
sel = np.array([order[p] for p in dr.patient_id])

print("=== 结论看这两行 ===")
full = run("完整图像", np.stack(dr.pooled), fold_y, dr.groups, folds)
center = run("只留中心 60%", z["center_features"][sel], fold_y, dr.groups, folds)
border = run("只留外圈 20%", z["border_features"][sel], fold_y, dr.groups, folds)

print()
print(f"  去掉边缘后: {center - full:+.3f}")
print(f"  只用边缘:   {border - full:+.3f}")
print()
print("解读:")
print("  去掉边缘几乎不掉分 -> 模型本来就主要看中心，Grad-CAM 的'外圈'")
print("     其实是插值后的视觉假象，没有捷径")
print("  只用边缘仍能拿高分 -> 边缘里有判别信息，需要进一步查是不是文字水印")
