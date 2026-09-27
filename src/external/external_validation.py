"""Cross-centre validation: train on 威宁, test on 安医附院.

This is the deliverable the course plan schedules for week 4, and the single
most persuasive piece of evidence in the whole project.

Two caveats that must be reported alongside the numbers:

  * The seven pregnancy one-hot columns are near-constant in 威宁 (99% are
    "无", every condition 0.1-0.3%) but very different in 安医 (27% gestational
    diabetes, 19.7% hypothyroidism).  The tabular branch therefore faces a real
    distribution shift, and its external numbers should be read with that in
    mind.  The IMAGE branch validates cleanly.
  * The class balance differs wildly: ARDS is 4.4% in 威宁 but 35% in 安医,
    so AUPRC is not comparable across centres; macro-AUROC is.
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
TABPFN_CKPT = r"E:\模型\torch_cache\tabpfn\tabpfn-v2-classifier.ckpt"
CLASSES = ["阴性", "ARDS", "湿肺", "肺炎"]


def corrected_labels(cohort):
    """Apply the ARDS relabelling justified by the free-text diagnosis."""
    path = os.path.join(FEATURE_DIR, "label_conflicts_text.csv")
    if not os.path.exists(path):
        return cohort.label.copy()
    fix = {int(r["patient"]) for r in
           csv.DictReader(open(path, encoding="utf-8-sig"))
           if r["starts_with_ARDS"] == "True"}
    y = cohort.label.copy()
    for i, pid in enumerate(cohort.patient_id):
        if int(pid) in fix and y[i] != 1:
            y[i] = 1
    return y


def impute(X, med):
    return np.where(np.isnan(X), med, X)


def fit_predict_tab(X_tr, y_tr, X_te):
    from tabpfn import TabPFNClassifier

    med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    m = TabPFNClassifier(device="cpu", n_estimators=4,
                         ignore_pretraining_limits=True, model_path=TABPFN_CKPT)
    m.fit(impute(X_tr, med), y_tr)
    return m.predict_proba(impute(X_te, med))


def fit_predict_img(feats_tr, y_tr, mod_tr, feats_te, mod_te):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    out = np.zeros((len(feats_te), 4))
    for mod in sorted(set(mod_tr)):
        a = [i for i in range(len(feats_tr)) if mod_tr[i] == mod]
        b = [i for i in range(len(feats_te)) if mod_te[i] == mod]
        if len(a) < 20 or not b:
            continue
        sc = StandardScaler()
        clf = LogisticRegression(max_iter=600, C=0.05, class_weight="balanced",
                                 multi_class="multinomial", random_state=0)
        clf.fit(sc.fit_transform(np.stack([feats_tr[i] for i in a])), y_tr[a])
        out[b] = clf.predict_proba(
            sc.transform(np.stack([feats_te[i] for i in b])))
    return out


def report(tag, y, proba):
    m = mm_eval.compute_metrics(y, proba)
    print(f"  {tag:<18} macro-AUROC {m['macro_auroc']:.3f}   "
          f"balanced acc {m['balanced_accuracy']:.3f}   "
          f"accuracy {m['accuracy']:.3f}")
    print(f"  {'':<18} 各类 AUROC: " + "  ".join(
        f"{c} {m['auroc_' + c]:.3f}" for c in CLASSES))
    return m


print("=" * 74)
print("训练集：威宁妇幼   |   外部测试集：安医附院")
print("=" * 74)

cohort = mm_data.load_cohort()
y_wn = corrected_labels(cohort)
print(f"威宁 {len(cohort)} 条 (修正标签后): "
      f"{dict(zip(CLASSES, np.bincount(y_wn, minlength=4)))}")

ani_rows = list(csv.DictReader(
    open(os.path.join(FEATURE_DIR, "manifest_ani.csv"), encoding="utf-8-sig")))
ani_tab = np.load(os.path.join(FEATURE_DIR, "tabular_ani.npz"), allow_pickle=True)
ani_feat = np.load(os.path.join(FEATURE_DIR, "features_ani.npz"), allow_pickle=True)

ids = list(ani_tab["patient_id"])
pos = {int(p): i for i, p in enumerate(ids)}
order = np.array([pos[int(r["patient_id"])] for r in ani_rows])
y_an = np.array([int(r["label"]) for r in ani_rows])
X_an_tab = ani_tab["features"][order]
print(f"安医 {len(ani_rows)} 条: {dict(zip(CLASSES, np.bincount(y_an, minlength=4)))}")

dr_map = {int(p): f for p, f in zip(ani_feat["dr_id"], ani_feat["dr_features"])}
us_map = {int(p): f for p, f in zip(ani_feat["us_id"], ani_feat["us_features"])}
img_an, mod_an = [], []
for r in ani_rows:
    pid = int(r["patient_id"])
    if int(r["has_dr"]) and pid in dr_map:
        img_an.append(dr_map[pid])
        mod_an.append("DR")
    elif pid in us_map:
        img_an.append(us_map[pid])
        mod_an.append("US")
    else:
        img_an.append(np.zeros(1536, dtype=np.float32))
        mod_an.append("none")
print(f"  其中用 DR 的 {mod_an.count('DR')} 条, 用超声的 {mod_an.count('US')} 条")

# ---- train on 威宁, predict 安医
print("\n训练 ...")
tab_an = fit_predict_tab(cohort.tabular, y_wn, X_an_tab)
img_an_p = fit_predict_img(cohort.pooled, y_wn, list(cohort.modality),
                           img_an, mod_an)

from sklearn.linear_model import LogisticRegression

# combiner trained on 威宁 in-sample predictions (a deliberately simple,
# slightly optimistic combination rule; the point here is the image branch)
tab_wn = fit_predict_tab(cohort.tabular, y_wn, cohort.tabular)
img_wn = fit_predict_img(cohort.pooled, y_wn, list(cohort.modality),
                         cohort.pooled, list(cohort.modality))
comb = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                          multi_class="multinomial", random_state=0)
comb.fit(np.hstack([tab_wn, img_wn]), y_wn)
fuse_an = comb.predict_proba(np.hstack([tab_an, img_an_p]))

print("\n=== 外部验证结果（安医，全部 233 条）===")
report("仅临床表格", y_an, tab_an)
report("仅图像", y_an, img_an_p)
report("表格+图像", y_an, fuse_an)

dr_mask = np.array([m == "DR" for m in mod_an])
print(f"\n=== 只用有 DR 的 {int(dr_mask.sum())} 条 ===")
report("仅图像", y_an[dr_mask], img_an_p[dr_mask])
report("表格+图像", y_an[dr_mask], fuse_an[dr_mask])

print("\n=== 只用超声的 {} 条 ===".format(int((~dr_mask).sum())))
report("仅图像", y_an[~dr_mask], img_an_p[~dr_mask])
report("表格+图像", y_an[~dr_mask], fuse_an[~dr_mask])

np.savez_compressed(
    os.path.join(FEATURE_DIR, "..", "results", "external_ani.npz"),
    y=y_an, tab=tab_an, img=img_an_p, fuse=fuse_an,
)
print("\n已写出 results/external_ani.npz")
