"""Does matching 安医's preprocessing to 威宁's (CLAHE) improve transfer?

    source (威宁) : CLAHE-enhanced, fixed
    target (安医) : raw  vs  CLAHE-enhanced , each full and centre-cropped
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


def probe(Xs, ys, Xt):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    clf = LogisticRegression(max_iter=800, C=0.05, class_weight="balanced",
                             multi_class="multinomial", random_state=0)
    clf.fit(sc.fit_transform(Xs), ys)
    return clf.predict_proba(sc.transform(Xt))


def show(tag, y, p):
    m = mm_eval.compute_metrics(y, p)
    print(f"  {tag:<18} macro-AUROC {m['macro_auroc']:.3f}   "
          f"平衡正确率 {m['balanced_accuracy']:.3f}   "
          + " ".join(f"{c[:2]}{m['auroc_' + c]:.2f}" for c in CLASSES))
    return m["macro_auroc"]


cohort = mm_data.load_cohort()
y_s = corrected(cohort)
dr_mask = cohort.modality == "DR"
dr_idx = np.flatnonzero(dr_mask)
y_s_dr = y_s[dr_idx]
y_s_us = y_s[~dr_mask]

zc = np.load(os.path.join(FDIR, "features_crop.npz"))
zr = np.load(os.path.join(FDIR, "features_dr_raddino.npz"))
zall = np.load(os.path.join(FDIR, "features_us_all.npz"))
zclahe = np.load(os.path.join(FDIR, "features_ani_clahe.npz"))

an_rows = list(csv.DictReader(
    open(os.path.join(FDIR, "manifest_ani.csv"), encoding="utf-8-sig")))
y_t = np.array([int(r["label"]) for r in an_rows])
an_dr_rows = [i for i, r in enumerate(an_rows) if int(r["has_dr"]) == 1]
an_us_rows = [i for i, r in enumerate(an_rows) if int(r["has_dr"]) == 0]

af = np.load(os.path.join(FDIR, "features_ani.npz"))
dmap = {int(p): f for p, f in zip(af["dr_id"], af["dr_features"])}
umap = {int(p): f for p, f in zip(af["us_id"], af["us_features"])}
raw_dr_full = np.stack([dmap[int(an_rows[i]["patient_id"])] for i in an_dr_rows])
raw_us_full = np.stack([umap[int(an_rows[i]["patient_id"])] for i in an_us_rows])
raw_dr_crop = zc["an_dr"]
raw_us_crop = zc["an_us"][an_us_rows]

print("=" * 90)
print("外部验证 · DR 子集（%d 条）" % len(an_dr_rows))
print("=" * 90)
src_full = zr["features"]
src_crop = zc["wn_dr"]
r1 = show("原始 · 完整", y_t[an_dr_rows], probe(src_full, y_s_dr, raw_dr_full))
r2 = show("原始 · 裁剪", y_t[an_dr_rows], probe(src_crop, y_s_dr, raw_dr_crop))
c1 = show("CLAHE · 完整", y_t[an_dr_rows],
          probe(src_full, y_s_dr, zclahe["dr_full"]))
c2 = show("CLAHE · 裁剪", y_t[an_dr_rows],
          probe(src_crop, y_s_dr, zclahe["dr_crop"]))

print()
print("=" * 90)
print("外部验证 · 超声子集（%d 条）" % len(an_us_rows))
print("=" * 90)
s1 = show("原始 · 完整", y_t[an_us_rows],
          probe(zall["features"], y_s, raw_us_full))
s2 = show("原始 · 裁剪", y_t[an_us_rows],
          probe(zc["wn_us"][~dr_mask], y_s_us, raw_us_crop))
t1 = show("CLAHE · 完整", y_t[an_us_rows],
          probe(zall["features"], y_s, zclahe["us_full"][an_us_rows]))
t2 = show("CLAHE · 裁剪", y_t[an_us_rows],
          probe(zc["wn_us"][~dr_mask], y_s_us, zclahe["us_crop"][an_us_rows]))

print()
print("=" * 90)
print("CLAHE 带来的变化")
print("=" * 90)
print(f"  DR   完整 {c1-r1:+.3f}   裁剪 {c2-r2:+.3f}")
print(f"  超声 完整 {t1-s1:+.3f}   裁剪 {t2-s2:+.3f}")
print(f"\n  最佳: DR {max(r1, r2, c1, c2):.3f}   超声 {max(s1, s2, t1, t2):.3f}")
