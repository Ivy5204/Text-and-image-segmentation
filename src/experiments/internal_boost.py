"""Remaining cheap levers for the INTERNAL macro-AUROC.

(a) Feature concatenation: DR full + DR centre-crop, and DR + ultrasound for
    the samples that have both.  Free -- no re-extraction.
(b) Swap the stacking combiner from logistic regression to TabPFN.
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import mm_data
import mm_eval

FDIR = r"E:\模型\mm4class\features"
N_REPEATS = 3


def corrected(cohort):
    p = os.path.join(FDIR, "label_conflicts_text.csv")
    fix = {int(r["patient"]) for r in
           csv.DictReader(open(p, encoding="utf-8-sig"))
           if r["starts_with_ARDS"] == "True"}
    y = cohort.label.copy()
    for i, pid in enumerate(cohort.patient_id):
        if int(pid) in fix and y[i] != 1:
            y[i] = 1
    return y


def eval_feats(tag, X, y, groups, folds):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def fp(tr, te):
        sc = StandardScaler()
        clf = LogisticRegression(max_iter=600, C=0.05, class_weight="balanced",
                                 multi_class="multinomial", random_state=0)
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        return clf.predict_proba(sc.transform(X[te]))

    res = mm_eval.run_cv(tag, y, groups, fp, folds=folds)
    s = res.summary()
    print(f"  {tag:<26} macro-AUROC {s['macro_auroc_mean']:.3f} ± "
          f"{s['macro_auroc_std']:.3f}   ARDS AUPRC {s['ards_auprc_mean']:.3f}")
    return s["macro_auroc_mean"]


cohort = mm_data.load_cohort()
y = corrected(cohort)
dr_mask = cohort.modality == "DR"
dr_idx = np.flatnonzero(dr_mask)
y_dr = y[dr_idx]
folds_dr = mm_eval.make_folds(y_dr, cohort.patient_id[dr_idx],
                              n_repeats=N_REPEATS)
zc = np.load(os.path.join(FDIR, "features_crop.npz"))

print("=" * 78)
print("(a) DR 特征组合（威宁 DR 子集 %d 条）" % len(dr_idx))
print("=" * 78)
full = np.stack(cohort.pooled)[dr_idx]
crop = zc["wn_dr"]
a = eval_feats("完整图像", full, y_dr, cohort.patient_id[dr_idx], folds_dr)
b = eval_feats("中心裁剪", crop, y_dr, cohort.patient_id[dr_idx], folds_dr)
c = eval_feats("完整 + 裁剪 (拼接)", np.hstack([full, crop]), y_dr,
               cohort.patient_id[dr_idx], folds_dr)
print(f"\n  拼接 vs 完整: {c - a:+.3f}")

print()
print("=" * 78)
print("(b) 双图：DR + 超声特征拼接（两种图都有的 %d 条）" % len(dr_idx))
print("=" * 78)
zall = np.load(os.path.join(FDIR, "features_us_all.npz"))
both = np.hstack([full, zall["features"][dr_idx]])
d = eval_feats("DR + 超声 (拼接)", both, y_dr, cohort.patient_id[dr_idx],
               folds_dr)
print(f"\n  加超声 vs 只用 DR: {d - a:+.3f}")
