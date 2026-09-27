"""Table branch bake-off: LightGBM / Logistic vs TabPFN.

TabPFN is a tabular foundation model that does in-context learning over the
training set; its sweet spot is exactly n < 1000 with tens of features, which
is this dataset.  The table branch is currently stuck at 0.778 across every
GBDT and linear configuration, so a different inductive bias is the only
untested lever left.
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


class TabPFNHead:
    """TabPFN needs NaN-free input and a bounded feature count."""

    def __init__(self, n_estimators=8):
        from tabpfn import TabPFNClassifier

        self.model = TabPFNClassifier(
            device="cpu",
            n_estimators=n_estimators,
            ignore_pretraining_limits=True,
            # Default is %APPDATA%/tabpfn (unwritable here) and its auto
            # download resolves to a name that does not exist in the repo,
            # so point straight at the standard v2 classifier checkpoint.
            model_path=r"E:\模型\torch_cache\tabpfn\tabpfn-v2-classifier.ckpt",
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
print(f"全量 {len(y)} 条, 特征 {X.shape[1]} 维, 折数 {len(folds)}")
print(f"类别分布: {dict(zip(mm_data.CLASSES, np.bincount(y)))}")
print()

CANDIDATES = [
    ("Logistic C=0.05", lambda: LogisticPipe(0.05)),
    ("LightGBM (无类别权重)", lambda: LightGBMHead(seed=0, class_weight=None)),
    ("TabPFN v2 (n_est=4)", lambda: TabPFNHead(n_estimators=4)),
    ("TabPFN v2 (n_est=8)", lambda: TabPFNHead(n_estimators=8)),
]

rows = []
for name, factory in CANDIDATES:
    t0 = time.time()
    res = mm_eval.run_cv(
        name, y, groups,
        lambda tr, te: factory().fit(X[tr], y[tr]).predict_proba(X[te]),
        folds=folds,
    )
    s = res.summary()
    rows.append((name, s))
    print(f"  {name:<24} macro-AUROC {s['macro_auroc_mean']:.3f} ± "
          f"{s['macro_auroc_std']:.3f}   ARDS AUPRC {s['ards_auprc_mean']:.3f}   "
          f"({time.time()-t0:.0f}s)")
    if res.pooled_oof is not None:
        safe = "".join(c if c.isalnum() else "_" for c in name)
        np.savez_compressed(
            rf"E:\模型\mm4class\results\oof_table_{safe}.npz",
            y=y, oof=res.pooled_oof, patient_id=groups,
        )

print()
print("=" * 72)
base = rows[1][1]["macro_auroc_mean"]
print(f"{'model':<26}{'macro-AUROC':>13}{'Δ vs LightGBM':>16}")
for name, s in rows:
    d = s["macro_auroc_mean"] - base
    print(f"{name:<26}{s['macro_auroc_mean']:>13.3f}{d:>+16.3f}")
print()
print("说明: 折叠标准差约 ±0.02, 小于该幅度的差异不可靠")
