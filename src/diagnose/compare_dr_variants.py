"""Measure each DR upgrade on its own, on identical folds.

    224 (current)      vs  320               -> isolates resolution
    320 last          vs  320 multi          -> isolates multi-layer
    320 multi         vs  320 multi + TTA    -> isolates TTA
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
from mm_models import LightGBMHead

N_REPEATS = 3
FDIR = r"E:\模型\mm4class\features"


class LogisticPipe:
    def __init__(self, C=0.05):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1500, C=C, class_weight="balanced",
                               multi_class="multinomial", random_state=0),
        )

    def fit(self, X, y):
        self.median = np.nanmedian(X, axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        self.model.fit(np.where(np.isnan(X), self.median, X), y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(np.where(np.isnan(X), self.median, X))


HEADS = {
    "Logistic": lambda: LogisticPipe(0.05),
    "LightGBM": lambda: LightGBMHead(seed=0),
}


def evaluate(name, X, y, groups, folds, heads=("Logistic",)):
    row = {"variant": name, "dim": X.shape[1]}
    for head_name in heads:
        factory = HEADS[head_name]
        t0 = time.time()
        res = mm_eval.run_cv(
            name, y, groups,
            lambda tr, te: factory().fit(X[tr], y[tr]).predict_proba(X[te]),
            folds=folds,
        )
        s = res.summary()
        row[head_name] = s
        print(f"  {name:<22} {head_name:<9} macro-AUROC "
              f"{s['macro_auroc_mean']:.3f} ± {s['macro_auroc_std']:.3f}   "
              f"ARDS AUPRC {s['ards_auprc_mean']:.3f}   ({time.time()-t0:.0f}s)")
    return row


cohort = mm_data.load_cohort()
dr = cohort.subset(cohort.modality == "DR")
print(f"DR 子集 {len(dr)} 条, ARDS {int((dr.label == 1).sum())} 例")
folds = mm_eval.make_folds(dr.label, dr.groups, n_repeats=N_REPEATS)
print(f"折数 {len(folds)}\n")

v2 = np.load(f"{FDIR}/features_dr_v2.npz")
order = {int(p): i for i, p in enumerate(v2["patient_id"])}
sel = np.array([order[p] for p in dr.patient_id])

variants = [
    ("224 last (现状)", np.stack(dr.pooled)),
    ("320 last", v2["f_last"][sel]),
    ("320 last + TTA", v2["f_last_tta"][sel]),
    ("320 multi", v2["f_multi"][sel]),
    ("320 multi + TTA", v2["f_multi_tta"][sel]),
]

rows = []
for name, X in variants:
    print(f"=== {name}  (dim {X.shape[1]}) ===")
    rows.append(evaluate(name, X, dr.label, dr.groups, folds, heads=("Logistic",)))
    print()

print("=" * 78)
print("第一阶段: Logistic 快速筛 (macro-AUROC)")
print("=" * 78)
base = rows[0]
print(f"{'variant':<24}{'dim':>6}{'Logistic':>12}{'Δ vs 224':>12}")
for r in rows:
    lg = r["Logistic"]["macro_auroc_mean"]
    d1 = lg - base["Logistic"]["macro_auroc_mean"]
    print(f"{r['variant']:<24}{r['dim']:>6}{lg:>12.3f}{d1:>+12.3f}")

# LightGBM only on the baseline and the best Logistic variant, to confirm the
# ranking holds for a non-linear head without paying for all five.
best = max(rows, key=lambda r: r["Logistic"]["macro_auroc_mean"])
confirm = [v for v in variants if v[0] in (base["variant"], best["variant"])]
print()
print("=" * 78)
print(f"第二阶段: 用 LightGBM 确认 {base['variant']} vs {best['variant']}")
print("=" * 78)
for name, X in confirm:
    evaluate(name, X, dr.label, dr.groups, folds, heads=("LightGBM",))

print()
print("提示: ARDS 30 例, 置信区间约 ±0.09, 小于该幅度的差异不可靠")
