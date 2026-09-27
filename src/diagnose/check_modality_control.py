"""Decisive control: remove the modality shortcut and re-measure.

Finding from the crosstab: "has a chest X-ray" is itself strongly associated
with the label (Cramer's V = 0.405, chi2 p < 1e-4), and a model that only
knows the modality already scores macro-AUROC 0.644.

No patient has DR without ultrasound, and 390 patients have ultrasound only.
Of those, 78% are 阴性 -- i.e. "no chest X-ray" is a proxy for "healthy baby".

The fix: keep only the 483 records whose patient has BOTH modalities.  Those
records already use DR, so the modality is constant and cannot be exploited.
Anything the model still achieves there is real signal.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import mm_data
import mm_eval
import mm_train
from mm_models import FusionConfig, LightGBMHead

N_REPEATS = 3
EPOCHS = 18


def per_modality_head(feats, modalities, y):
    def fit_predict(tr, te):
        preds = np.zeros((len(te), 4))
        for mod in sorted(set(modalities)):
            tr_m = [i for i in tr if modalities[i] == mod]
            te_m = [k for k, i in enumerate(te) if modalities[i] == mod]
            if not tr_m or not te_m:
                continue
            head = LightGBMHead(seed=0).fit(
                np.stack([feats[i] for i in tr_m]), y[tr_m]
            )
            preds[te_m] = head.predict_proba(np.stack([feats[te[k]] for k in te_m]))
        return preds
    return fit_predict


def run_subset(name, cohort, fusion_cfg):
    y, groups = cohort.label, cohort.groups
    tab = cohort.tabular
    pooled = cohort.pooled
    modalities = cohort.modality

    print(f"\n{'='*70}\n{name}  (n={len(cohort)}, "
          f"ARDS={int((y == 1).sum())})\n{'='*70}")
    for k, cls in enumerate(mm_data.CLASSES):
        print(f"  {cls:<4} {int((y == k).sum()):>4}")
    if len(set(modalities)) == 1:
        print(f"  模态: 全部为 {modalities[0]}（无模态捷径可用）")

    folds = mm_eval.make_folds(y, groups, n_repeats=N_REPEATS)
    out = {}

    t0 = time.time()
    res = mm_eval.run_cv("T-only", y, groups,
                         lambda tr, te: LightGBMHead(seed=0)
                         .fit(tab[tr], y[tr]).predict_proba(tab[te]),
                         folds=folds)
    out["T-only"] = res.summary()
    print(f"  [{time.time()-t0:5.1f}s] T-only          "
          f"macro-AUROC {res.summary()['macro_auroc_mean']:.3f}")

    t0 = time.time()
    res = mm_eval.run_cv("I-only", y, groups,
                         per_modality_head(pooled, modalities, y), folds=folds)
    out["I-only"] = res.summary()
    print(f"  [{time.time()-t0:5.1f}s] I-only          "
          f"macro-AUROC {res.summary()['macro_auroc_mean']:.3f}")

    arrays = mm_train.build_arrays(cohort)

    def fusion_fit(tr, te):
        return mm_train.train_fusion(
            (arrays[0][tr], arrays[1][tr], tab[tr], arrays[2][tr]),
            (arrays[0][te], arrays[1][te], tab[te], arrays[2][te]),
            y[tr], y[te], fusion_cfg, seed=0)["proba"]

    t0 = time.time()
    res = mm_eval.run_cv("T+I Gated", y, groups, fusion_fit, folds=folds)
    out["T+I Gated"] = res.summary()
    print(f"  [{time.time()-t0:5.1f}s] T+I Gated       "
          f"macro-AUROC {res.summary()['macro_auroc_mean']:.3f}  "
          f"ARDS AUPRC {res.summary()['ards_auprc_mean']:.3f}")
    return out


torch.set_flush_denormal(True)
cohort = mm_data.load_cohort()
cfg = FusionConfig(epochs=EPOCHS)

dr_only = run_subset("仅 DR 子集 483 条（模态无捷径）",
                     cohort.subset(cohort.modality == "DR"), cfg)

# Reference numbers from results/summary_quick.json (same folds, same config,
# whole cohort).  Re-running them here would cost another ~15 minutes.
full = {
    "T-only": {"macro_auroc_mean": 0.770},
    "I-only": {"macro_auroc_mean": 0.753},
    "T+I Gated": {"macro_auroc_mean": 0.802},
}

print("\n" + "=" * 70)
print("对照结果：去掉模态捷径后掉了多少")
print("=" * 70)
print(f"{'模型':<14}{'全量879':>12}{'仅DR483':>12}{'差值':>10}")
for key in ("T-only", "I-only", "T+I Gated"):
    a = full[key]["macro_auroc_mean"]
    b = dr_only[key]["macro_auroc_mean"]
    print(f"{key:<14}{a:>12.3f}{b:>12.3f}{b - a:>+10.3f}")
print()
print("解释：I-only 的跌幅主要来自模态捷径被切断；")
print("      T-only 不接触模态，跌幅只反映子集的类别分布变化。")
