"""Table branch: TabPFN v2 vs v3, identical folds."""

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
CACHE = r"E:\模型\torch_cache\tabpfn"


class TabPFNHead:
    def __init__(self, ckpt, n_estimators=4):
        from tabpfn import TabPFNClassifier

        self.model = TabPFNClassifier(
            device="cpu",
            n_estimators=n_estimators,
            ignore_pretraining_limits=True,
            model_path=os.path.join(CACHE, ckpt),
        )

    def fit(self, X, y):
        self.median = np.nanmedian(X, axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        self.model.fit(np.where(np.isnan(X), self.median, X), y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(np.where(np.isnan(X), self.median, X))


cohort = mm_data.load_cohort()
y, groups = cohort.label, cohort.groups
X = cohort.tabular
folds = mm_eval.make_folds(y, groups, n_repeats=N_REPEATS)
print(f"全量 {len(y)} 条, {X.shape[1]} 维, {len(folds)} 折")
print(f"类别分布: {dict(zip(mm_data.CLASSES, np.bincount(y)))}\n")

CANDIDATES = [
    ("LightGBM", lambda: LightGBMHead(seed=0)),
    ("TabPFN v2", lambda: TabPFNHead("tabpfn-v2-classifier.ckpt")),
    ("TabPFN v3", lambda: TabPFNHead("tabpfn-v3-classifier-v3_default.ckpt")),
]

rows = []
for name, factory in CANDIDATES:
    t0 = time.time()
    try:
        res = mm_eval.run_cv(
            name, y, groups,
            lambda tr, te: factory().fit(X[tr], y[tr]).predict_proba(X[te]),
            folds=folds,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  {name:<14} 失败: {type(exc).__name__}: {str(exc)[:160]}")
        continue
    s = res.summary()
    rows.append((name, s))
    print(f"  {name:<14} macro-AUROC {s['macro_auroc_mean']:.3f} ± "
          f"{s['macro_auroc_std']:.3f}   ARDS AUPRC {s['ards_auprc_mean']:.3f}"
          f"   ({time.time()-t0:.0f}s)")
    if res.pooled_oof is not None:
        safe = name.replace(" ", "_")
        np.savez_compressed(
            rf"E:\模型\mm4class\results\oof_tablev3_{safe}.npz",
            y=y, oof=res.pooled_oof, patient_id=groups,
        )

if rows:
    base = rows[0][1]["macro_auroc_mean"]
    print()
    print("=" * 70)
    print(f"{'model':<16}{'macro-AUROC':>13}{'Δ vs LightGBM':>16}")
    for name, s in rows:
        print(f"{name:<16}{s['macro_auroc_mean']:>13.3f}"
              f"{s['macro_auroc_mean']-base:>+16.3f}")
