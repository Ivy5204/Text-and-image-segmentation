"""Late fusion by stacking the two branches' out-of-fold probabilities.

The end-to-end gated fusion tops out at 0.802 and every architectural change
we tried failed.  This is a different fusion mechanism: each branch uses the
model that suits it best (TabPFN for the 20 clinical variables, a linear probe
for the frozen image features) and a small combiner learns how much to trust
each one.

Leakage control: inside every outer fold the training part goes through its
own inner CV to produce the out-of-fold probabilities that the combiner is
trained on.  The test part always gets probabilities from a model that never
saw it.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import mm_data
import mm_eval

RESULTS_DIR = r"E:\模型\mm4class\results"
TABPFN_CKPT = r"E:\模型\torch_cache\tabpfn\tabpfn-v2-classifier.ckpt"
N_REPEATS = 3
N_INNER = 3


def make_tabpfn():
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(device="cpu", n_estimators=4,
                            ignore_pretraining_limits=True,
                            model_path=TABPFN_CKPT)


def impute(X, median):
    return np.where(np.isnan(X), median, X)


def branch_oof(X, y, train_idx, n_inner=N_INNER, seed=0):
    """Inner-CV out-of-fold probabilities for the combiner's training rows."""
    from sklearn.model_selection import StratifiedGroupKFold

    oof = np.zeros((len(train_idx), 4))
    inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    local_groups = np.arange(len(train_idx))
    for tr, te in inner.split(np.zeros(len(train_idx)), y[train_idx], local_groups):
        sub = train_idx[tr]
        median = np.nanmedian(X[sub], axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        model = make_tabpfn()
        model.fit(impute(X[sub], median), y[sub])
        oof[te] = model.predict_proba(impute(X[train_idx[te]], median))
    return oof


def branch_fit_predict(X, y, train_idx, test_idx):
    median = np.nanmedian(X[train_idx], axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    model = make_tabpfn()
    model.fit(impute(X[train_idx], median), y[train_idx])
    return model.predict_proba(impute(X[test_idx], median))


def image_branch_oof(feats, y, train_idx, modalities, n_inner=N_INNER, seed=0):
    """Per-modality linear probe, since the two encoders have different widths."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    oof = np.zeros((len(train_idx), 4))
    inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    local_groups = np.arange(len(train_idx))
    for tr, te in inner.split(np.zeros(len(train_idx)), y[train_idx], local_groups):
        sub = train_idx[tr]
        tst = train_idx[te]
        for mod in sorted(set(modalities)):
            a = [i for i in sub if modalities[i] == mod]
            mask = np.array([modalities[i] == mod for i in tst])
            if len(a) < 10 or not mask.any():
                continue
            # te holds LOCAL positions inside train_idx; write back at those
            # positions, not at the offset inside the fold.
            local_pos = te[mask]
            sc = StandardScaler()
            clf = LogisticRegression(max_iter=400, C=0.05,
                                     class_weight="balanced",
                                     multi_class="multinomial", random_state=0)
            clf.fit(sc.fit_transform(np.stack([feats[i] for i in a])), y[a])
            oof[local_pos] = clf.predict_proba(
                sc.transform(np.stack([feats[i] for i in tst[mask]]))
            )
    return oof


def image_branch_predict(feats, y, train_idx, test_idx, modalities):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    out = np.zeros((len(test_idx), 4))
    for mod in sorted(set(modalities)):
        a = [i for i in train_idx if modalities[i] == mod]
        b = [k for k, i in enumerate(test_idx) if modalities[i] == mod]
        if len(a) < 10 or not b:
            continue
        sc = StandardScaler()
        clf = LogisticRegression(max_iter=400, C=0.05, class_weight="balanced",
                                 multi_class="multinomial", random_state=0)
        clf.fit(sc.fit_transform(np.stack([feats[i] for i in a])), y[a])
        out[b] = clf.predict_proba(sc.transform(np.stack([feats[tst] for tst in
                                                          [test_idx[k] for k in b]])))
    return out


def main():
    cohort = mm_data.load_cohort()
    y, groups = cohort.label, cohort.groups
    tab = cohort.tabular
    feats = cohort.pooled
    modalities = cohort.modality
    folds = mm_eval.make_folds(y, groups, n_repeats=N_REPEATS)
    print(f"样本 {len(y)}, 折数 {len(folds)}\n")

    from sklearn.linear_model import LogisticRegression

    oof_tab = np.zeros((len(y), 4))
    oof_img = np.zeros((len(y), 4))
    oof_stack = np.zeros((len(y), 4))
    fold_scores = {"tab": [], "img": [], "stack": []}

    t0 = time.time()
    for n, (rep, fold, tr, te) in enumerate(folds, 1):
        tr = np.asarray(tr)
        te = np.asarray(te)
        tab_tr = branch_oof(tab, y, tr)
        tab_te = branch_fit_predict(tab, y, tr, te)
        img_tr = image_branch_oof(feats, y, tr, modalities)
        img_te = image_branch_predict(feats, y, tr, te, modalities)

        Z_tr = np.hstack([tab_tr, img_tr])
        Z_te = np.hstack([tab_te, img_te])
        comb = LogisticRegression(max_iter=800, C=1.0, class_weight="balanced",
                                  multi_class="multinomial", random_state=0)
        comb.fit(Z_tr, y[tr])

        oof_tab[te] = tab_te
        oof_img[te] = img_te
        oof_stack[te] = comb.predict_proba(Z_te)

        # per-fold scores, so the headline number follows the same protocol as
        # every other experiment in this project (mean over folds)
        for key, pred in (("tab", tab_te), ("img", img_te),
                          ("stack", oof_stack[te])):
            fold_scores[key].append(
                mm_eval.compute_metrics(y[te], pred)["macro_auroc"]
            )
        print(f"  fold {n}/{len(folds)} 完成  ({time.time()-t0:.0f}s)",
              end="\r", flush=True)
    print()

    for name, oof, key in (("TabPFN 表格", oof_tab, "tab"),
                           ("图像线性探针", oof_img, "img"),
                           ("Stacking 融合", oof_stack, "stack")):
        m = mm_eval.compute_metrics(y, oof)
        arr = np.array([v for v in fold_scores[key] if np.isfinite(v)])
        print(f"  {name:<16} macro-AUROC 逐折 {arr.mean():.3f} ± {arr.std():.3f}"
              f"  (合并 OOF {m['macro_auroc']:.3f})   "
              f"ARDS AUPRC {m['ards_auprc']:.3f}   "
              f"平衡正确率 {m['balanced_accuracy']:.3f}")
        safe = name.replace(" ", "_")
        np.savez_compressed(f"{RESULTS_DIR}/oof_stack_{safe}.npz",
                            y=y, oof=oof, patient_id=groups)


if __name__ == "__main__":
    main()
