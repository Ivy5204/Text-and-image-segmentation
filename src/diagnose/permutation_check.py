"""Prove the image model really uses image content.

Two runs on the same DR-only subset, same folds:
  real     - features as extracted from the actual image files
  shuffled - same features, labels permuted within the cohort

If the model were only exploiting label priors rather than image content, the
two runs would score the same.  With shuffle the discriminative signal must
collapse towards 0.5.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import mm_data
import mm_eval
from mm_models import LightGBMHead, LogisticHead

N_REPEATS = 1


def evaluate_dr(cohort, y, note):
    folds = mm_eval.make_folds(y, cohort.groups, n_repeats=N_REPEATS)
    feats = cohort.pooled
    t0 = time.time()
    res = mm_eval.run_cv(
        note, y, cohort.groups,
        lambda tr, te: LightGBMHead(seed=0, n_estimators=150)
        .fit(np.stack([feats[i] for i in tr]), y[tr])
        .predict_proba(np.stack([feats[i] for i in te])),
        folds=folds,
    )
    s = res.summary()
    print(f"  {note:<28} macro-AUROC {s['macro_auroc_mean']:.3f}   "
          f"({time.time()-t0:.0f}s)")
    return s["macro_auroc_mean"]


cohort = mm_data.load_cohort()
dr = cohort.subset(cohort.modality == "DR")
print(f"DR 子集: {len(dr)} 条, ARDS {int((dr.label == 1).sum())} 例")
print("同一套折, 同一套超参, 唯一区别是标签有没有被打乱\n")

real = evaluate_dr(dr, dr.label, "真实特征 + 真实标签")

# Record-level permutation: keeps the class counts exactly identical, which is
# what makes this a valid null (a patient-level permutation would not).
rng = np.random.default_rng(0)
shuffled = rng.permutation(dr.label)
assert (np.bincount(shuffled, minlength=4) == np.bincount(dr.label, minlength=4)).all()

shuf = evaluate_dr(dr, shuffled, "真实特征 + 打乱标签")

print()
print(f"图像内容的真实贡献: {real - shuf:+.3f} macro-AUROC")
print("（若接近 0，说明模型没在用图像内容；大幅下降说明图像确实被读取并使用了）")
