"""Does correcting the 18 mislabelled records improve the model?

Evidence for the correction: 23 of the 25 label-conflict records carry a
free-text 临床诊断 that LEADS with "ARDS" (e.g. "ARDS、新生儿肺炎"), so ARDS is
the primary diagnosis and the folder placement was right while the CSV `sheet`
column picked up a secondary diagnosis.

This runs the best single-image stack twice on the same folds: once with the
CSV labels (ARDS = 39) and once with the corrected labels (ARDS = 57).
"""

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import mm_data
import mm_eval

FEATURE_DIR = r"E:\模型\mm4class\features"
RESULTS_DIR = r"E:\模型\mm4class\results"
TABPFN_CKPT = r"E:\模型\torch_cache\tabpfn\tabpfn-v2-classifier.ckpt"
N_REPEATS = 3
N_INNER = 3


def safe(name):
    return "".join(c if c.isalnum() else "_" for c in name)


def make_tabpfn():
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(device="cpu", n_estimators=4,
                            ignore_pretraining_limits=True,
                            model_path=TABPFN_CKPT)


def impute(X, med):
    return np.where(np.isnan(X), med, X)


def tab_oof(X, y, tr, seed=0):
    from sklearn.model_selection import StratifiedGroupKFold

    oof = np.zeros((len(tr), 4))
    inner = StratifiedGroupKFold(n_splits=N_INNER, shuffle=True, random_state=seed)
    for a, b in inner.split(np.zeros(len(tr)), y[tr], np.arange(len(tr))):
        sub = tr[a]
        med = np.nanmedian(X[sub], axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        m = make_tabpfn()
        m.fit(impute(X[sub], med), y[sub])
        oof[b] = m.predict_proba(impute(X[tr[b]], med))
    return oof


def img_oof(feats, y, tr, modalities, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    oof = np.zeros((len(tr), 4))
    inner = StratifiedGroupKFold(n_splits=N_INNER, shuffle=True, random_state=seed)
    for a, b in inner.split(np.zeros(len(tr)), y[tr], np.arange(len(tr))):
        sub = tr[a]
        tst = tr[b]
        for mod in sorted(set(modalities)):
            aa = [i for i in sub if modalities[i] == mod]
            mask = np.array([modalities[i] == mod for i in tst])
            if len(aa) < 10 or not mask.any():
                continue
            sc = StandardScaler()
            clf = LogisticRegression(max_iter=400, C=0.05, class_weight="balanced",
                                     multi_class="multinomial", random_state=0)
            clf.fit(sc.fit_transform(np.stack([feats[i] for i in aa])), y[aa])
            oof[b[mask]] = clf.predict_proba(
                sc.transform(np.stack([feats[i] for i in tst[mask]])))
    return oof


def img_predict(feats, y, tr, te, modalities):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    out = np.zeros((len(te), 4))
    for mod in sorted(set(modalities)):
        aa = [i for i in tr if modalities[i] == mod]
        mask = np.array([modalities[i] == mod for i in te])
        if len(aa) < 10 or not mask.any():
            continue
        sc = StandardScaler()
        clf = LogisticRegression(max_iter=400, C=0.05, class_weight="balanced",
                                 multi_class="multinomial", random_state=0)
        clf.fit(sc.fit_transform(np.stack([feats[i] for i in aa])), y[aa])
        out[mask] = clf.predict_proba(
            sc.transform(np.stack([feats[i] for i in te[mask]])))
    return out


def run(tag, y, cohort, folds):
    from sklearn.linear_model import LogisticRegression

    tab = cohort.tabular
    feats = cohort.pooled
    modalities = cohort.modality
    store = {"tab": np.zeros((len(y), 4)), "stack": np.zeros((len(y), 4))}
    scores = {"tab": [], "stack": []}
    t0 = time.time()
    for n, (rep, fold, tr, te) in enumerate(folds, 1):
        tr = np.asarray(tr)
        te = np.asarray(te)
        t_tr = tab_oof(tab, y, tr)
        med = np.nanmedian(tab[tr], axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        m = make_tabpfn()
        m.fit(impute(tab[tr], med), y[tr])
        t_te = m.predict_proba(impute(tab[te], med))
        i_tr = img_oof(feats, y, tr, modalities)
        i_te = img_predict(feats, y, tr, te, modalities)
        comb = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                                  multi_class="multinomial", random_state=0)
        comb.fit(np.hstack([t_tr, i_tr]), y[tr])
        s_te = comb.predict_proba(np.hstack([t_te, i_te]))
        store["tab"][te] = t_te
        store["stack"][te] = s_te
        for k, p in (("tab", t_te), ("stack", s_te)):
            scores[k].append(mm_eval.compute_metrics(y[te], p)["macro_auroc"])
        print(f"  [{tag}] fold {n}/{len(folds)} ({time.time()-t0:.0f}s)",
              end="\r", flush=True)
    print()
    out = {}
    for k, name in (("tab", "TabPFN"), ("stack", "Stacking")):
        arr = np.array([v for v in scores[k] if np.isfinite(v)])
        met = mm_eval.compute_metrics(y, store[k])
        out[k] = (arr.mean(), arr.std(), met)
        print(f"  [{tag}] {name:<9} macro-AUROC {arr.mean():.3f} ± {arr.std():.3f}"
              f"   ARDS AUPRC {met['ards_auprc']:.3f}")
        np.savez_compressed(f"{RESULTS_DIR}/oof_relabel_{safe(tag)}_{k}.npz",
                            y=y, oof=store[k], patient_id=cohort.patient_id)
    return out, store


cohort = mm_data.load_cohort()
base_y = cohort.label.copy()

# corrected labels: 23 records whose diagnosis text leads with ARDS
rows = list(csv.DictReader(
    open(os.path.join(FEATURE_DIR, "label_conflicts_text.csv"),
         encoding="utf-8-sig")))
fix = {int(r["patient"]): r for r in rows if r["starts_with_ARDS"] == "True"}
manifest = list(csv.DictReader(
    open(os.path.join(FEATURE_DIR, "manifest.csv"), encoding="utf-8-sig")))
new_y = base_y.copy()
changed = 0
for i, m in enumerate(manifest):
    pid = int(m["patient_id"])
    if pid in fix and new_y[i] != 1:
        new_y[i] = 1
        changed += 1
print(f"重标 {changed} 条记录为 ARDS")
print(f"  重标前: {dict(zip(mm_data.CLASSES, np.bincount(base_y)))}")
print(f"  重标后: {dict(zip(mm_data.CLASSES, np.bincount(new_y)))}")

folds = mm_eval.make_folds(base_y, cohort.groups, n_repeats=N_REPEATS)
print(f"\n折数 {len(folds)}（两套标签用同一套折）\n")

print("=== 用原 CSV 标签 (ARDS=39) ===")
old, _ = run("原始标签", base_y, cohort, folds)
print()
print("=== 用修正标签 (ARDS=57) ===")
new, _ = run("修正标签", new_y, cohort, folds)

print()
print("=" * 74)
print(f"{'model':<12}{'原标签':>10}{'修正标签':>12}{'差值':>10}")
for k, name in (("tab", "TabPFN"), ("stack", "Stacking")):
    print(f"{name:<12}{old[k][0]:>10.3f}{new[k][0]:>12.3f}"
          f"{new[k][0]-old[k][0]:>+10.3f}")
