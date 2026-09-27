"""Does centre-cropping fix the border shortcut and improve cross-centre transfer?"""

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
CLASSES = ["阴性", "ARDS", "湿肺", "肺炎"]


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


def internal(name, feats, y, groups, folds):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def fp(tr, te):
        sc = StandardScaler()
        clf = LogisticRegression(max_iter=600, C=0.05, class_weight="balanced",
                                 multi_class="multinomial", random_state=0)
        clf.fit(sc.fit_transform(feats[tr]), y[tr])
        return clf.predict_proba(sc.transform(feats[te]))

    res = mm_eval.run_cv(name, y, groups, fp, folds=folds)
    s = res.summary()
    print(f"  {name:<26} macro-AUROC {s['macro_auroc_mean']:.3f} ± "
          f"{s['macro_auroc_std']:.3f}"
          f"   ARDS AUPRC {s['ards_auprc_mean']:.3f}")
    return s["macro_auroc_mean"]


def external(name, Xs, ys, Xt, yt):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    clf = LogisticRegression(max_iter=800, C=0.05, class_weight="balanced",
                             multi_class="multinomial", random_state=0)
    clf.fit(sc.fit_transform(Xs), ys)
    p = clf.predict_proba(sc.transform(Xt))
    m = mm_eval.compute_metrics(yt, p)
    print(f"  {name:<26} macro-AUROC {m['macro_auroc']:.3f}   "
          f"ARDS AUROC {m['auroc_ARDS']:.3f}   湿肺 {m['auroc_湿肺']:.3f}")
    return m["macro_auroc"]


cohort = mm_data.load_cohort()
y = corrected(cohort)
dr = cohort.modality == "DR"
dr_idx = np.flatnonzero(dr)
y_dr = y[dr_idx]
folds = mm_eval.make_folds(y_dr, cohort.patient_id[dr_idx], n_repeats=N_REPEATS)

z = np.load(os.path.join(FDIR, "features_crop.npz"))
zfull = np.load(os.path.join(FDIR, "features_dr_raddino.npz"))
pos_full = {int(p): i for i, p in enumerate(zfull["patient_id"])}
sel_full = np.array([pos_full[p] for p in cohort.patient_id[dr_idx]])

# crop features are stored in manifest order (DR rows only)
wn_rows = list(csv.DictReader(
    open(os.path.join(FDIR, "manifest_dual.csv"), encoding="utf-8-sig")))
dr_rows = [r for r in wn_rows if int(r["has_dr"]) == 1]
assert len(dr_rows) == len(dr_idx), (len(dr_rows), len(dr_idx))

print("=" * 74)
print("内部验证：威宁 DR 子集（%d 条）" % len(dr_idx))
print("=" * 74)
a = internal("完整图像（基线）", np.stack(cohort.pooled)[dr_idx],
             y_dr, cohort.patient_id[dr_idx], folds)
b = internal("中心裁剪 70%", z["wn_dr"], y_dr, cohort.patient_id[dr_idx], folds)
print(f"\n  裁剪后变化: {b - a:+.3f}")

# external
an_rows = list(csv.DictReader(
    open(os.path.join(FDIR, "manifest_ani.csv"), encoding="utf-8-sig")))
an_dr = [r for r in an_rows if int(r["has_dr"]) == 1]
y_an = np.array([int(r["label"]) for r in an_dr])
af = np.load(os.path.join(FDIR, "features_ani.npz"))
apos = {int(p): i for i, p in enumerate(af["dr_id"])}
sel_an = np.array([apos[int(r["patient_id"])] for r in an_dr])

print()
print("=" * 74)
print("外部验证：威宁 → 安医（DR %d 条）" % len(an_dr))
print("=" * 74)
c = external("完整图像（基线）", np.stack(cohort.pooled)[dr_idx], y_dr,
             af["dr_features"][sel_an], y_an)
d = external("中心裁剪 70%", z["wn_dr"], y_dr, z["an_dr"], y_an)
print(f"\n  裁剪后变化: {d - c:+.3f}")
