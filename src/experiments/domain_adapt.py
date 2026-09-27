"""Cheap domain adaptation on the features we already have.

The external test collapsed because 安医 images carry different acquisition
statistics.  Before paying for new extractions, this checks whether simply
removing the centre-level mean/scale shift (and optionally the covariance
shift, i.e. CORAL) recovers any of the lost performance.

Nothing here is retrained on 安医 labels -- only unlabelled feature statistics
are used, which is legitimate for a cross-centre deployment setting.
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

FEATURE_DIR = r"E:\模型\mm4class\features"
CLASSES = ["阴性", "ARDS", "湿肺", "肺炎"]


def load_source():
    cohort = mm_data.load_cohort()
    path = os.path.join(FEATURE_DIR, "label_conflicts_text.csv")
    fix = {int(r["patient"]) for r in
           csv.DictReader(open(path, encoding="utf-8-sig"))
           if r["starts_with_ARDS"] == "True"}
    y = cohort.label.copy()
    for i, pid in enumerate(cohort.patient_id):
        if int(pid) in fix and y[i] != 1:
            y[i] = 1
    return cohort, y


def load_target():
    rows = list(csv.DictReader(
        open(os.path.join(FEATURE_DIR, "manifest_ani.csv"), encoding="utf-8-sig")))
    tab = np.load(os.path.join(FEATURE_DIR, "tabular_ani.npz"), allow_pickle=True)
    feat = np.load(os.path.join(FEATURE_DIR, "features_ani.npz"), allow_pickle=True)
    pos = {int(p): i for i, p in enumerate(tab["patient_id"])}
    order = np.array([pos[int(r["patient_id"])] for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    X_tab = tab["features"][order]
    dr_map = {int(p): f for p, f in zip(feat["dr_id"], feat["dr_features"])}
    us_map = {int(p): f for p, f in zip(feat["us_id"], feat["us_features"])}
    imgs, mods = [], []
    for r in rows:
        pid = int(r["patient_id"])
        if int(r["has_dr"]) and pid in dr_map:
            imgs.append(dr_map[pid]); mods.append("DR")
        elif pid in us_map:
            imgs.append(us_map[pid]); mods.append("US")
        else:
            imgs.append(np.zeros(1536, dtype=np.float32)); mods.append("none")
    return rows, y, X_tab, np.stack(imgs), mods


def zscore_fit(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu, sd


def coral(Xs, Xt, lam=1.0):
    """Align target covariance to source covariance (CORAL, Sun et al. 2017)."""
    def cov(X):
        Xc = X - X.mean(0, keepdims=True)
        return Xc.T @ Xc / max(len(X) - 1, 1)

    Cs, Ct = cov(Xs), cov(Xt)
    ds, Vs = np.linalg.eigh(Cs)
    dt, Vt = np.linalg.eigh(Ct)
    ds = np.clip(ds, 1e-8, None)
    dt = np.clip(dt, 1e-8, None)
    Cs_half = Vs @ np.diag(np.sqrt(ds)) @ Vs.T
    Ct_inv_half = Vt @ np.diag(1.0 / np.sqrt(dt)) @ Vt.T
    A = Cs_half @ Ct_inv_half
    A = (1 - lam) * np.eye(A.shape[0]) + lam * A
    return (Xt - Xt.mean(0, keepdims=True)) @ A + Xs.mean(0, keepdims=True)


def probe_source_then_target(Xs, ys, Xt, yt, tag):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    med = np.nanmedian(Xs, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xs = np.where(np.isnan(Xs), med, Xs)
    Xt = np.where(np.isnan(Xt), med, Xt)
    sc = StandardScaler()
    clf = LogisticRegression(max_iter=800, C=0.05, class_weight="balanced",
                             multi_class="multinomial", random_state=0)
    clf.fit(sc.fit_transform(Xs), ys)
    p = clf.predict_proba(sc.transform(Xt))
    m = mm_eval.compute_metrics(yt, p)
    print(f"  {tag:<34} macro-AUROC {m['macro_auroc']:.3f}   "
          f"ARDS AUROC {m['auroc_ARDS']:.3f}")
    return m["macro_auroc"]


cohort, y_s = load_source()
rows_t, y_t, tab_t, img_t, mod_t = load_target()
tab_s = cohort.tabular
img_s = np.stack(cohort.pooled)

print("=" * 76)
print("只对图像特征做域自适应（表格有编码差异，单独看）")
print("=" * 76)
probe_source_then_target(img_s, y_s, img_t, y_t, "1. 原始（基线）")

mu_s, sd_s = zscore_fit(img_s)
mu_t, sd_t = zscore_fit(img_t)
probe_source_then_target((img_s - mu_s) / sd_s, y_s,
                         (img_t - mu_t) / sd_t, y_t,
                         "2. 中心内 z-score")

img_t_coral = coral(img_s, img_t, lam=0.5)
probe_source_then_target((img_s - mu_s) / sd_s, y_s,
                         (img_t_coral - mu_s) / sd_s, y_t,
                         "3. CORAL 对齐（lam=0.5）")

img_t_coral1 = coral(img_s, img_t, lam=1.0)
probe_source_then_target((img_s - mu_s) / sd_s, y_s,
                         (img_t_coral1 - mu_s) / sd_s, y_t,
                         "4. CORAL 对齐（lam=1.0）")

print()
print("=" * 76)
print("表格特征同样处理（已知有编码分布差异）")
print("=" * 76)
probe_source_then_target(tab_s, y_s, tab_t, y_t, "1. 原始（基线）")
mu_s2, sd_s2 = zscore_fit(tab_s)
mu_t2, sd_t2 = zscore_fit(tab_t)
probe_source_then_target((tab_s - mu_s2) / sd_s2, y_s,
                         (tab_t - mu_t2) / sd_t2, y_t,
                         "2. 中心内 z-score")

print()
print("=" * 76)
print("表格 + 图像（都在中心内标准化后拼接）")
print("=" * 76)
is_ = (img_s - mu_s) / sd_s
it_ = (img_t - mu_t) / sd_t
ts_ = (tab_s - mu_s2) / sd_s2
tt_ = (tab_t - mu_t2) / sd_t2
probe_source_then_target(np.hstack([ts_, is_]), y_s,
                         np.hstack([tt_, it_]), y_t, "拼接（中心内标准化）")
