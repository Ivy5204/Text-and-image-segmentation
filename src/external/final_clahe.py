"""Best external configuration after the CLAHE + crop fixes, incl. fusion."""

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
    return m


cohort = mm_data.load_cohort()
y_s = corrected(cohort)
dr_mask = cohort.modality == "DR"
dr_idx = np.flatnonzero(dr_mask)
us_idx = np.flatnonzero(~dr_mask)

zc = np.load(os.path.join(FDIR, "features_crop.npz"))
zall = np.load(os.path.join(FDIR, "features_us_all.npz"))
zcl = np.load(os.path.join(FDIR, "features_ani_clahe.npz"))

an_rows = list(csv.DictReader(
    open(os.path.join(FDIR, "manifest_ani.csv"), encoding="utf-8-sig")))
y_t = np.array([int(r["label"]) for r in an_rows])
an_dr = [i for i, r in enumerate(an_rows) if int(r["has_dr"]) == 1]
an_us = [i for i, r in enumerate(an_rows) if int(r["has_dr"]) == 0]

# best per modality: DR = CLAHE + crop, US = CLAHE + full
src = np.zeros((len(y_s), 1536), dtype=np.float32)
src[dr_idx] = zc["wn_dr"]
src[us_idx] = zall["features"][us_idx]
tgt = np.zeros((len(an_rows), 1536), dtype=np.float32)
tgt[an_dr] = zcl["dr_crop"]
tgt[an_us] = zcl["us_full"][an_us]

print("=" * 88)
print("外部验证 · 最优配置（DR: CLAHE+裁剪, 超声: CLAHE+完整）")
print("=" * 88)
p_img = probe(src, y_s, tgt)
m_img = show("仅图像", y_t, p_img)

p_img_dr = probe(zc["wn_dr"], y_s[dr_idx], zcl["dr_crop"])
p_img_us = probe(zall["features"][us_idx], y_s[us_idx], zcl["us_full"][an_us])
print(f"    DR 子集 {mm_eval.compute_metrics(y_t[an_dr], p_img_dr)['macro_auroc']:.3f}"
      f"   超声子集 {mm_eval.compute_metrics(y_t[an_us], p_img_us)['macro_auroc']:.3f}")

tab_t = np.load(os.path.join(FDIR, "tabular_ani.npz"), allow_pickle=True)
pos = {int(p): i for i, p in enumerate(tab_t["patient_id"])}
order = np.array([pos[int(r["patient_id"])] for r in an_rows])
p_tab = tabpfn_pred(cohort.tabular, y_s, tab_t["features"][order])
show("仅表格", y_t, p_tab)

from sklearn.linear_model import LogisticRegression
tab_s = tabpfn_pred(cohort.tabular, y_s, cohort.tabular)
img_s = probe(src, y_s, src)
comb = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                          multi_class="multinomial", random_state=0)
comb.fit(np.hstack([tab_s, img_s]), y_s)
p_fuse = comb.predict_proba(np.hstack([p_tab, p_img]))
show("表格 + 图像", y_t, p_fuse)

print()
print("=" * 88)
print("外部验证演进")
print("=" * 88)
print(f"  {'配置':<34}{'仅图像':>10}{'仅表格':>10}{'融合':>10}")
print(f"  {'原始（未做任何处理）':<32}{0.551:>10.3f}{0.662:>10.3f}{0.625:>10.3f}")
print(f"  {'+ 中心裁剪':<33}{0.583:>10.3f}{'—':>10}{'—':>10}")
print(f"  {'+ CLAHE 预处理对齐（本轮）':<30}"
      f"{m_img['macro_auroc']:>10.3f}"
      f"{mm_eval.compute_metrics(y_t, p_tab)['macro_auroc']:>10.3f}"
      f"{mm_eval.compute_metrics(y_t, p_fuse)['macro_auroc']:>10.3f}")
