"""External validation, before and after the centre-crop fix."""

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
TABPFN = r"E:\模型\torch_cache\tabpfn\tabpfn-v2-classifier.ckpt"
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


def tabpfn_pred(Xs, ys, Xt):
    from tabpfn import TabPFNClassifier

    med = np.nanmedian(Xs, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    m = TabPFNClassifier(device="cpu", n_estimators=4,
                         ignore_pretraining_limits=True, model_path=TABPFN)
    m.fit(np.where(np.isnan(Xs), med, Xs), ys)
    return m.predict_proba(np.where(np.isnan(Xt), med, Xt))


def show(tag, y, p):
    m = mm_eval.compute_metrics(y, p)
    print(f"  {tag:<22} macro-AUROC {m['macro_auroc']:.3f}   "
          f"平衡正确率 {m['balanced_accuracy']:.3f}   "
          + " ".join(f"{c[:2]}{m['auroc_' + c]:.2f}" for c in CLASSES))
    return m["macro_auroc"]


cohort = mm_data.load_cohort()
y_s = corrected(cohort)
wn_dr_mask = cohort.modality == "DR"
wn_dr_idx = np.flatnonzero(wn_dr_mask)
y_s_dr = y_s[wn_dr_idx]
y_s_us = y_s[~wn_dr_mask]

z = np.load(os.path.join(FDIR, "features_crop.npz"))
af = np.load(os.path.join(FDIR, "features_ani.npz"))
an_rows = list(csv.DictReader(
    open(os.path.join(FDIR, "manifest_ani.csv"), encoding="utf-8-sig")))
y_t = np.array([int(r["label"]) for r in an_rows])

an_dr_rows = [i for i, r in enumerate(an_rows) if int(r["has_dr"]) == 1]
an_us_rows = [i for i, r in enumerate(an_rows) if int(r["has_dr"]) == 0]
print(f"安医: 有 DR {len(an_dr_rows)} 条, 只有超声 {len(an_us_rows)} 条")

dr_an_map = {int(p): f for p, f in zip(af["dr_id"], af["dr_features"])}
us_an_map = {int(p): f for p, f in zip(af["us_id"], af["us_features"])}
X_dr_t_full = np.stack([dr_an_map[int(an_rows[i]["patient_id"])]
                        for i in an_dr_rows])
X_us_t_full = np.stack([us_an_map[int(an_rows[i]["patient_id"])]
                        for i in an_us_rows])

print()
print("=" * 78)
print("外部验证：训练集威宁 → 测试集安医附院")
print("=" * 78)

print("\n[1] 图像分支 · 完整图像（当前基线）")
a = []
a.append(show("DR 子集", y_t[an_dr_rows],
              probe(np.stack(cohort.pooled)[wn_dr_idx], y_s_dr, X_dr_t_full)))
a.append(show("超声子集", y_t[an_us_rows],
              probe(np.stack(cohort.pooled)[~wn_dr_mask], y_s_us, X_us_t_full)))

print("\n[2] 图像分支 · 中心裁剪 70%（本轮修复）")
b = []
b.append(show("DR 子集", y_t[an_dr_rows], probe(z["wn_dr"], y_s_dr, z["an_dr"])))
# z["wn_us"] holds all 879 rows in cohort order (every sample has an
# ultrasound); z["an_us"] holds all 233 in manifest order.
b.append(show("超声子集", y_t[an_us_rows],
              probe(z["wn_us"][~wn_dr_mask], y_s_us,
                    z["an_us"][an_us_rows])))

print(f"\n  裁剪带来的变化:  DR {b[0]-a[0]:+.3f}   超声 {b[1]-a[1]:+.3f}")

print("\n[3] 表格分支 · TabPFN（两院编码分布差异大，仅供参考）")
tab_t = np.load(os.path.join(FDIR, "tabular_ani.npz"), allow_pickle=True)
pos = {int(p): i for i, p in enumerate(tab_t["patient_id"])}
order = np.array([pos[int(r["patient_id"])] for r in an_rows])
p_tab = tabpfn_pred(cohort.tabular, y_s, tab_t["features"][order])
c = show("全部 233 条", y_t, p_tab)

print("\n[4] 融合 · 表格 + 裁剪图像")
from sklearn.linear_model import LogisticRegression
tab_s = tabpfn_pred(cohort.tabular, y_s, cohort.tabular)

# "one image per sample" assembled from the CROPPED features, matching the
# operational rule used everywhere else (DR if present, else ultrasound).
src_img = np.zeros((len(y_s), 1536), dtype=np.float32)
src_img[wn_dr_idx] = z["wn_dr"]
src_img[~wn_dr_mask] = z["wn_us"][~wn_dr_mask]
tgt_img = np.zeros((len(an_rows), 1536), dtype=np.float32)
tgt_img[an_dr_rows] = z["an_dr"]
tgt_img[an_us_rows] = z["an_us"][an_us_rows]

img_s_p = probe(src_img, y_s, src_img)
comb = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                          multi_class="multinomial", random_state=0)
comb.fit(np.hstack([tab_s, img_s_p]), y_s)
fuse = comb.predict_proba(np.hstack([p_tab, probe(src_img, y_s, tgt_img)]))
show("表格 + 裁剪图像", y_t, fuse)

np.savez_compressed(os.path.join(FDIR, "..", "results", "external_final.npz"),
                    y=y_t, tab=p_tab, fuse=fuse)
print("\n已写出 results/external_final.npz")
