"""Fast screening of the DR feature variants.

Multinomial logistic on 3072 dimensions is far too slow to sweep, so the
screen uses a lightweight linear probe (few lbfgs iterations, single repeat).
The point is only to rank the variants; the winner is confirmed afterwards
with the full protocol and LightGBM.
"""

import os
import sys
import time

import numpy as np

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import mm_data
import mm_eval

FDIR = r"E:\模型\mm4class\features"


def probe(X, y, groups, folds):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    aucs = []
    for _, _, tr, te in folds:
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        clf = LogisticRegression(max_iter=250, C=0.05, class_weight="balanced",
                                 multi_class="multinomial", random_state=0)
        clf.fit(Xtr, y[tr])
        p = clf.predict_proba(Xte)
        m = mm_eval.compute_metrics(y[te], p)
        if np.isfinite(m["macro_auroc"]):
            aucs.append(m["macro_auroc"])
    return float(np.mean(aucs)), float(np.std(aucs))


cohort = mm_data.load_cohort()
dr = cohort.subset(cohort.modality == "DR")
folds = mm_eval.make_folds(dr.label, dr.groups, n_repeats=1)
print(f"DR 子集 {len(dr)} 条, ARDS {int((dr.label == 1).sum())} 例, "
      f"单重复 {len(folds)} 折 (快速筛)\n")

v2 = np.load(f"{FDIR}/features_dr_v2.npz")
order = {int(p): i for i, p in enumerate(v2["patient_id"])}
sel = np.array([order[p] for p in dr.patient_id])

variants = [
    ("224 last (baseline)", np.stack(dr.pooled)),
    ("320 last", v2["f_last"][sel]),
    ("320 last + TTA", v2["f_last_tta"][sel]),
    ("320 multi", v2["f_multi"][sel]),
    ("320 multi + TTA", v2["f_multi_tta"][sel]),
]

print(f"{'variant':<24}{'dim':>6}{'macro-AUROC':>14}{'Δ vs base':>12}")
base_auc = None
for name, X in variants:
    t0 = time.time()
    mean, std = probe(X, dr.label, dr.groups, folds)
    if base_auc is None:
        base_auc = mean
    print(f"{name:<24}{X.shape[1]:>6}{mean:>9.3f} ±{std:.3f}"
          f"{mean - base_auc:>+12.3f}   ({time.time()-t0:.0f}s)")

print()
print("提示: 单重复 5 折, 噪声较大, 只用于排序")
