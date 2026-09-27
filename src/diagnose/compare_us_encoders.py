"""Ultrasound encoder bake-off on identical folds.

ConvNeXt-Tiny (ImageNet supervised)  vs  USF-MAE (ViT-B, ultrasound MAE).
Only the ultrasound records are used; the two feature sets are otherwise
processed identically.
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
from mm_models import LightGBMHead, LogisticHead

N_REPEATS = 3
FEATURE_DIR = r"E:\模型\mm4class\features"


class LogisticPipe:
    """Median-impute -> standardise -> multinomial logistic."""

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


def evaluate(name, feats, y, groups, folds):
    X = np.stack(feats)
    out = {}
    for head_name, factory in (("Logistic", lambda: LogisticPipe(0.05)),
                               ("LightGBM", lambda: LightGBMHead(seed=0))):
        t0 = time.time()
        res = mm_eval.run_cv(
            head_name, y, groups,
            lambda tr, te: factory().fit(X[tr], y[tr]).predict_proba(X[te]),
            folds=folds,
        )
        s = res.summary()
        out[head_name] = s
        print(f"  {name:<12} {head_name:<9} macro-AUROC "
              f"{s['macro_auroc_mean']:.3f} ± {s['macro_auroc_std']:.3f}   "
              f"ARDS AUPRC {s['ards_auprc_mean']:.3f}   ({time.time()-t0:.0f}s)")
    return out


cohort = mm_data.load_cohort()
us = cohort.subset(cohort.modality == "US")
print(f"超声子集 {len(us)} 条 (院号 {len(np.unique(us.patient_id))} 个), "
      f"ARDS {int((us.label == 1).sum())} 例")
print("类别分布:", {mm_data.CLASSES[k]: int((us.label == k).sum()) for k in range(4)})
print()

folds = mm_eval.make_folds(us.label, us.groups, n_repeats=N_REPEATS)

print("=== ConvNeXt-Tiny (ImageNet), 当前方案 ===")
convnext = evaluate("ConvNeXt", us.pooled, us.label, us.groups, folds)

print()
print("=== USF-MAE (ViT-B, 超声 MAE 预训练), 新方案 ===")
z = np.load(f"{FEATURE_DIR}/features_us_usfmae.npz")
order = {int(p): f for p, f in zip(z["patient_id"], z["features"])}
usfmae = [order[p] for p in us.patient_id]
print(f"  特征维度 {usfmae[0].shape}")
usfmae_res = evaluate("USF-MAE", usfmae, us.label, us.groups, folds)

print()
print("=== 汇总 delta (USF-MAE - ConvNeXt) ===")
for head in ("Logistic", "LightGBM"):
    d_auc = usfmae_res[head]["macro_auroc_mean"] - convnext[head]["macro_auroc_mean"]
    d_ap = usfmae_res[head]["ards_auprc_mean"] - convnext[head]["ards_auprc_mean"]
    print(f"  {head:<9} macro-AUROC {d_auc:+.3f}   ARDS AUPRC {d_ap:+.3f}")
