"""Dual-image stacking: table + DR + ultrasound.

Every one of the 879 samples has an ultrasound; 483 of them also have a DR.
The single-image pipeline used only 879 images; this uses 1362.

Three branches, each with the model that suits it:
    table  -> TabPFN v2
    DR     -> logistic probe on RAD-DINO features   (present for 483)
    US     -> logistic probe on USF-MAE features    (present for all)

The combiner sees [tab(4), us(4), dr(4), has_dr(1)].  When a sample has no DR
its DR vector is filled with the training-fold class prior and the indicator
is 0, so the combiner learns to discount it instead of being fed a zero vector.

Leakage control matches stacking.py: every outer fold's training part goes
through its own inner CV to produce the probabilities the combiner trains on.
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


def impute(X, median):
    return np.where(np.isnan(X), median, X)


def fit_probe(X, y, tr, te):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    clf = LogisticRegression(max_iter=600, C=0.05, class_weight="balanced",
                             multi_class="multinomial", random_state=0)
    clf.fit(sc.fit_transform(X[tr]), y[tr])
    return clf.predict_proba(sc.transform(X[te]))


def tab_oof(X, y, tr, seed=0):
    from sklearn.model_selection import StratifiedGroupKFold

    oof = np.zeros((len(tr), 4))
    inner = StratifiedGroupKFold(n_splits=N_INNER, shuffle=True, random_state=seed)
    for a, b in inner.split(np.zeros(len(tr)), y[tr], np.arange(len(tr))):
        sub = tr[a]
        med = np.nanmedian(X[sub], axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        model = make_tabpfn()
        model.fit(impute(X[sub], med), y[sub])
        oof[b] = model.predict_proba(impute(X[tr[b]], med))
    return oof


def img_oof(X, y, tr, avail, seed=0):
    """OOF probabilities for a branch that only some samples have."""
    from sklearn.model_selection import StratifiedGroupKFold

    oof = np.full((len(tr), 4), np.nan)
    inner = StratifiedGroupKFold(n_splits=N_INNER, shuffle=True, random_state=seed)
    if DEBUG_ONCE[0]:
        print(f"    [img_oof] len(tr)={len(tr)} n_avail={int(avail[tr].sum())}")
    for a, b in inner.split(np.zeros(len(tr)), y[tr], np.arange(len(tr))):
        # b holds LOCAL positions inside tr, so the write-back index must be
        # b[mask] -- not flatnonzero(mask), which is only the offset within b.
        # Getting this wrong made every inner fold overwrite oof[0:len(b)].
        mask = avail[tr[b]]
        pos = b[mask]
        sub = tr[a][avail[tr[a]]]
        tst = tr[b][mask]
        if DEBUG_ONCE[0]:
            print(f"      fold: |a|={len(a)} |b|={len(b)} "
                  f"|sub|={len(sub)} |tst|={len(tst)}")
        if len(sub) < 20 or len(tst) == 0:
            if DEBUG_ONCE[0]:
                print("      -> skipped")
            continue
        oof[pos] = fit_probe(X, y, sub, tst)
    return oof


DEBUG_ONCE = [False]


def fill_missing(oof, avail, y, rows):
    """Replace NaN rows with the class prior of the labelled rows."""
    out = oof.copy()
    prior = np.bincount(y[rows], minlength=4).astype(float)
    prior /= prior.sum()
    mask = np.isnan(oof).any(axis=1)
    out[mask] = prior
    return out


def main():
    cohort = mm_data.load_cohort()
    y, groups = cohort.label, cohort.groups
    tab = cohort.tabular

    dual = list(csv.DictReader(
        open(os.path.join(FEATURE_DIR, "manifest_dual.csv"), encoding="utf-8-sig")))
    if len(dual) != len(cohort):
        raise RuntimeError("manifest_dual 与 cohort 长度不一致")
    avail_dr = np.array([int(r["has_dr"]) == 1 for r in dual])
    print(f"样本 {len(y)}, 其中 {avail_dr.sum()} 条有 DR (占 {avail_dr.mean():.1%})")

    rd = np.load(os.path.join(FEATURE_DIR, "features_dr_raddino.npz"))
    us = np.load(os.path.join(FEATURE_DIR, "features_us_all.npz"))
    print(f"RAD-DINO {rd['features'].shape}, USF-MAE {us['features'].shape}")

    feats = np.zeros((len(dual), 1536), dtype=np.float32)   # both are 1536 wide
    dr_map = {int(p): f for p, f in zip(rd["patient_id"], rd["features"])}
    us_map = {int(p): f for p, f in zip(us["patient_id"], us["features"])}
    for i, r in enumerate(dual):
        pid = int(r["patient_id"])
        feats[i] = dr_map[pid] if int(r["has_dr"]) else us_map[pid]
    dr_feats = np.stack([dr_map[int(r["patient_id"])] if int(r["has_dr"])
                         else np.zeros(1536, dtype=np.float32) for r in dual])
    us_feats = np.stack([us_map[int(r["patient_id"])] for r in dual])

    folds = mm_eval.make_folds(y, groups, n_repeats=N_REPEATS)
    print(f"折数 {len(folds)}\n")

    from sklearn.linear_model import LogisticRegression

    store = {k: np.zeros((len(y), 4)) for k in
             ("tab", "us", "dr", "single", "dual", "dual_tabdr")}
    fold_scores = {k: [] for k in store}
    t0 = time.time()

    for n, (rep, fold, tr, te) in enumerate(folds, 1):
        tr = np.asarray(tr)
        te = np.asarray(te)
        DEBUG_ONCE[0] = (n == 1)

        # ---- table branch
        tab_tr = tab_oof(tab, y, tr)
        med = np.nanmedian(tab[tr], axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        m = make_tabpfn()
        m.fit(impute(tab[tr], med), y[tr])
        tab_te = m.predict_proba(impute(tab[te], med))

        # ---- ultrasound branch (everyone has it)
        us_tr = img_oof(us_feats, y, tr, np.ones(len(y), bool))
        us_te = fit_probe(us_feats, y, tr, te)

        # ---- DR branch (only where available)
        dr_tr = img_oof(dr_feats, y, tr, avail_dr)
        tr_dr = tr[avail_dr[tr]]
        te_dr_mask = avail_dr[te]
        # NaN marks "no DR", so fill_missing can drop in the class prior;
        # a zero vector would be silently interpreted as a strong prediction.
        dr_te = np.full((len(te), 4), np.nan)
        pos = np.flatnonzero(te_dr_mask)
        if len(pos):
            dr_te[pos] = fit_probe(dr_feats, y, tr_dr, te[pos])
        dr_te_full = fill_missing(dr_te, avail_dr[te], y, tr)
        dr_tr_full = fill_missing(dr_tr, avail_dr[tr], y, tr)

        ind_tr = avail_dr[tr].astype(float)[:, None]
        ind_te = avail_dr[te].astype(float)[:, None]

        Z3_tr = np.hstack([tab_tr, us_tr, dr_tr_full, ind_tr])
        Z3_te = np.hstack([tab_te, us_te, dr_te_full, ind_te])
        # single-image baseline for reference: tab + whichever image it used
        Zs_tr = np.hstack([tab_tr, img_oof(feats, y, tr, np.ones(len(y), bool))])
        Zs_te = np.hstack([tab_te, fit_probe(feats, y, tr, te)])
        # table + DR only (no ultrasound for anyone)
        Z2_tr = np.hstack([tab_tr, dr_tr_full, ind_tr])
        Z2_te = np.hstack([tab_te, dr_te_full, ind_te])

        if n == 1:
            for tag, block in (("tab_tr", tab_tr), ("us_tr", us_tr),
                               ("dr_tr", dr_tr), ("dr_tr_full", dr_tr_full),
                               ("dr_te", dr_te), ("dr_te_full", dr_te_full),
                               ("Z3_tr", Z3_tr), ("Z2_tr", Z2_tr),
                               ("Zs_tr", Zs_tr)):
                bad = int(np.isnan(block).sum()) + int(np.isinf(block).sum())
                print(f"    [debug] {tag:<12} shape {str(block.shape):<12} "
                      f"坏值 {bad}")

        def comb(Ztr, Zte):
            c = LogisticRegression(max_iter=1000, C=1.0,
                                   class_weight="balanced",
                                   multi_class="multinomial", random_state=0)
            c.fit(Ztr, y[tr])
            return c.predict_proba(Zte)

        store["tab"][te] = tab_te
        store["us"][te] = us_te
        store["dr"][te] = dr_te_full
        store["single"][te] = comb(Zs_tr, Zs_te)
        store["dual"][te] = comb(Z3_tr, Z3_te)
        store["dual_tabdr"][te] = comb(Z2_tr, Z2_te)

        for key in store:
            fold_scores[key].append(
                mm_eval.compute_metrics(y[te], store[key][te])["macro_auroc"]
            )
        print(f"  fold {n}/{len(folds)}  ({time.time()-t0:.0f}s)",
              end="\r", flush=True)
    print()

    LABELS = {
        "tab": "TabPFN 表格",
        "us": "超声单模态",
        "dr": "DR 单模态",
        "single": "单图 stacking (基线)",
        "dual_tabdr": "表格+DR (无超声)",
        "dual": "双图 stacking (表格+DR+超声)",
    }
    print()
    for key, name in LABELS.items():
        arr = np.array([v for v in fold_scores[key] if np.isfinite(v)])
        m = mm_eval.compute_metrics(y, store[key])
        print(f"  {name:<26} macro-AUROC {arr.mean():.3f} ± {arr.std():.3f}   "
              f"ARDS AUPRC {m['ards_auprc']:.3f}")
        np.savez_compressed(f"{RESULTS_DIR}/oof_dual_{safe(name)}.npz",
                            y=y, oof=store[key], patient_id=groups)


if __name__ == "__main__":
    main()
