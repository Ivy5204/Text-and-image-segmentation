"""One consolidated external comparison across every preprocessing variant."""

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


cohort = mm_data.load_cohort()
y_s = corrected(cohort)
dr_mask = cohort.modality == "DR"
dr_idx = np.flatnonzero(dr_mask)
us_idx = np.flatnonzero(~dr_mask)
y_s_dr, y_s_us = y_s[dr_idx], y_s[us_idx]
# Source features must be restricted to the same samples the label vector
# covers: the 483 DR-assigned rows, and the 396 ultrasound-assigned rows.
SRC_DR = zc["wn_dr"] if False else None      # assigned below
SRC_US = None

an = list(csv.DictReader(
    open(os.path.join(FDIR, "manifest_ani.csv"), encoding="utf-8-sig")))
y_t = np.array([int(r["label"]) for r in an])
an_dr = [i for i, r in enumerate(an) if int(r["has_dr"]) == 1]
an_us = [i for i, r in enumerate(an) if int(r["has_dr"]) == 0]

zc = np.load(os.path.join(FDIR, "features_crop.npz"))
zcl = np.load(os.path.join(FDIR, "features_ani_clahe.npz"))
zm = np.load(os.path.join(FDIR, "features_ani_match.npz"))
zr = np.load(os.path.join(FDIR, "features_dr_raddino.npz"))
zall = np.load(os.path.join(FDIR, "features_us_all.npz"))
af = np.load(os.path.join(FDIR, "features_ani.npz"))
SRC_DR = zr["features"]                 # 483 rows, DR-assigned
SRC_US = zall["features"][us_idx]       # 396 rows, US-assigned
dmap = {int(p): f for p, f in zip(af["dr_id"], af["dr_features"])}
umap = {int(p): f for p, f in zip(af["us_id"], af["us_features"])}


def score_dr(src, tgt):
    return mm_eval.compute_metrics(
        y_t[an_dr], probe(src, y_s_dr, tgt))["macro_auroc"]


def score_us(src, tgt):
    return mm_eval.compute_metrics(
        y_t[an_us], probe(src, y_s_us, tgt))["macro_auroc"]


rows = [
    ("原始 · 完整",
     score_dr(SRC_DR, np.stack([dmap[int(an[i]["patient_id"])] for i in an_dr])),
     score_us(SRC_US, np.stack([umap[int(an[i]["patient_id"])] for i in an_us]))),
    ("原始 · 裁剪", score_dr(zc["wn_dr"], zc["an_dr"]),
     score_us(zc["wn_us"][~dr_mask], zc["an_us"][an_us])),
    ("CLAHE · 完整", score_dr(SRC_DR, zcl["dr_full"]),
     score_us(SRC_US, zcl["us_full"][an_us])),
    ("CLAHE · 裁剪", score_dr(zc["wn_dr"], zcl["dr_crop"]),
     score_us(zc["wn_us"][~dr_mask], zcl["us_crop"][an_us])),
    ("直方图匹配 · 完整", score_dr(SRC_DR, zm["dr_match_full"]),
     score_us(SRC_US, zm["us_match_full"][an_us])),
    ("直方图匹配 · 裁剪", score_dr(zc["wn_dr"], zm["dr_match_crop"]),
     score_us(zc["wn_us"][~dr_mask], zm["us_match_crop"][an_us])),
    ("CLAHE+匹配 · 完整",
     score_dr(SRC_DR, zm["dr_clahematch_full"]),
     score_us(SRC_US, zm["us_clahematch_full"][an_us])),
    ("CLAHE+匹配 · 裁剪",
     score_dr(zc["wn_dr"], zm["dr_clahematch_crop"]),
     score_us(zc["wn_us"][~dr_mask], zm["us_clahematch_crop"][an_us])),
]

print("=" * 78)
print("外部验证 · 预处理变体全对比（威宁训练 → 安医测试）")
print("=" * 78)
print(f"{'预处理':<24}{'DR (104条)':>14}{'超声 (129条)':>16}")
print("-" * 78)
for tag, a, b in rows:
    print(f"{tag:<24}{a:>14.3f}{b:>16.3f}")

best_dr = max(rows, key=lambda r: r[1])
best_us = max(rows, key=lambda r: r[2])
print()
print(f"  DR 最优: {best_dr[0]}  {best_dr[1]:.3f}")
print(f"  超声最优: {best_us[0]}  {best_us[2]:.3f}")
