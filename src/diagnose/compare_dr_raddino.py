"""DR encoder bake-off: DenseNet121-CheXpert vs RAD-DINO, identical folds."""

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


class Probe:
    def __init__(self, C=0.05):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=600, C=C, class_weight="balanced",
                               multi_class="multinomial", random_state=0),
        )

    def fit(self, X, y):
        self.median = np.nanmedian(X, axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        self.model.fit(np.where(np.isnan(X), self.median, X), y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(np.where(np.isnan(X), self.median, X))


def run(name, X, y, groups, folds):
    r = {}
    for head, factory in (("Logistic", lambda: Probe(0.05)),
                          ("LightGBM", lambda: LightGBMHead(seed=0))):
        t0 = time.time()
        res = mm_eval.run_cv(
            name, y, groups,
            lambda tr, te: factory().fit(X[tr], y[tr]).predict_proba(X[te]),
            folds=folds,
        )
        s = res.summary()
        r[head] = s
        print(f"  {name:<22} {head:<9} macro-AUROC {s['macro_auroc_mean']:.3f} ± "
              f"{s['macro_auroc_std']:.3f}   ARDS AUPRC {s['ards_auprc_mean']:.3f}"
              f"   ({time.time()-t0:.0f}s)")
        if res.pooled_oof is not None:
            np.savez_compressed(
                f"{FDIR}/../results/oof_dr_{name.split()[0]}_{head}.npz",
                y=y, oof=res.pooled_oof, patient_id=groups,
            )
    return r


cohort = mm_data.load_cohort()
dr = cohort.subset(cohort.modality == "DR")
print(f"DR 子集 {len(dr)} 条, ARDS {int((dr.label == 1).sum())} 例")
folds = mm_eval.make_folds(dr.label, dr.groups, n_repeats=N_REPEATS)
print(f"折数 {len(folds)}\n")

ckpt = np.load(f"{FDIR}/features_dr_raddino.npz")
order = {int(p): i for i, p in enumerate(ckpt["patient_id"])}
sel = np.array([order[p] for p in dr.patient_id])

print("=== DenseNet121 + CheXpert (当前) ===")
base = run("DenseNet-CheXpert", np.stack(dr.pooled), dr.label, dr.groups, folds)
print()
print("=== RAD-DINO (自监督, 80 万张胸片) ===")
rd = run("RAD-DINO", ckpt["features"][sel], dr.label, dr.groups, folds)

print()
print("=" * 72)
for head in ("Logistic", "LightGBM"):
    d = rd[head]["macro_auroc_mean"] - base[head]["macro_auroc_mean"]
    print(f"  {head:<9} RAD-DINO - DenseNet = {d:+.3f}")
print("  折间标准差约 ±0.025, 小于该幅度的差异不可靠")
