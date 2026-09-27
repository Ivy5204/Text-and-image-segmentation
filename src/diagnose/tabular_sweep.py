"""Where is the table branch losing?  Sweep its knobs on the shared folds.

Trigger for this script: T-only Logistic (0.778) beat T-only LightGBM (0.770).
A linear model beating a GBDT on 20 features means the GBDT is mis-tuned, not
that the problem is linear.

Also tests a few clinically motivated engineered features, because at 39 ARDS
events the win comes from reducing effective dimensionality, not adding to it.
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

N_REPEATS = 3


def engineered(X, cols):
    """Add a handful of clinically motivated interactions."""
    idx = {c: i for i, c in enumerate(cols)}
    out = [X]
    names = list(cols)

    def col(name):
        return X[:, idx[name]] if name in idx else None

    gest = col("孕周_天")
    weight = col("新生儿体重g")
    ap1 = col("Apgar评分1分钟")
    ap5 = col("Apgar评分5分钟")

    if gest is not None:
        pre = (gest < 259).astype(np.float32)          # < 37 weeks
        out.append(pre[:, None])
        names.append("早产37周")
    if gest is not None and weight is not None:
        ratio = weight / np.maximum(gest, 1.0)          # birth weight per day
        out.append(ratio[:, None])
        names.append("体重每孕日")
    if ap1 is not None and ap5 is not None:
        out.append((ap5 - ap1)[:, None])
        names.append("Apgar差值")
        out.append((ap5 < 7).astype(np.float32)[:, None])
        names.append("Apgar5低")
    if weight is not None:
        out.append((weight < 2500).astype(np.float32)[:, None])
        names.append("低出生体重")

    return np.concatenate(out, axis=1).astype(np.float32), names


def add_missing_indicators(X):
    miss = np.isnan(X).astype(np.float32)
    keep = miss.sum(axis=0) > 0
    return np.concatenate([X, miss[:, keep]], axis=1)


def run(name, X, y, groups, folds, factory):
    t0 = time.time()
    res = mm_eval.run_cv(
        name, y, groups,
        lambda tr, te: factory().fit(X[tr], y[tr]).predict_proba(X[te]),
        folds=folds,
    )
    s = res.summary()
    print(f"  {name:<44} {s['macro_auroc_mean']:.3f} ± {s['macro_auroc_std']:.3f}"
          f"   ARDS AUPRC {s['ards_auprc_mean']:.3f}   ({time.time()-t0:.0f}s)")
    return s["macro_auroc_mean"]


class LocalLGBM:
    """Same idea as mm_models.LightGBMHead but with every knob exposed,
    so the sweep does not need to touch the shared project file."""

    def __init__(self, num_leaves=12, n_estimators=400, learning_rate=0.03,
                 min_child_samples=12, class_weight="balanced",
                 colsample_bytree=0.7, reg_lambda=1.0, seed=0):
        import lightgbm as lgb

        self.model = lgb.LGBMClassifier(
            objective="multiclass", num_class=4,
            num_leaves=num_leaves,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_child_samples=min_child_samples,
            subsample=0.8, subsample_freq=1,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            class_weight=class_weight,
            random_state=seed, n_jobs=4, verbose=-1,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def lgbm(**kw):
    return LocalLGBM(**kw)


class LocalLogistic:
    """Median-impute + standardise + multinomial logistic.

    LogisticRegression cannot consume NaN, so the imputer has to be fitted on
    the training fold (and only there).
    """

    def __init__(self, C=0.05, add_missing=False):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.add_missing = add_missing
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1500, C=C, class_weight="balanced",
                               multi_class="multinomial", random_state=0),
        )

    def fit(self, X, y):
        self.median = np.nanmedian(X, axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        self.model.fit(self._prep(X), y)
        return self

    def _prep(self, X):
        filled = np.where(np.isnan(X), self.median, X)
        if self.add_missing:
            return np.concatenate(
                [filled, np.isnan(X).astype(np.float32)], axis=1
            )
        return filled

    def predict_proba(self, X):
        return self.model.predict_proba(self._prep(X))


def logreg(C=0.05):
    return LocalLogistic(C=C)


cohort = mm_data.load_cohort()
y, groups = cohort.label, cohort.groups
folds = mm_eval.make_folds(y, groups, n_repeats=N_REPEATS)
X, cols = cohort.tabular, cohort.tabular_columns
X_eng, cols_eng = engineered(X, cols)
X_miss = add_missing_indicators(X)
X_eng_miss = add_missing_indicators(X_eng)

print(f"原始 {X.shape[1]} 维 | 工程特征后 {X_eng.shape[1]} 维 | "
      f"加缺失指示 {X_miss.shape[1]} 维 | 两者都加 {X_eng_miss.shape[1]} 维\n")

print("--- 线性基线 ---")
for C in (0.01, 0.05, 0.2, 1.0):
    run(f"Logistic C={C}", X, y, groups, folds, lambda C=C: logreg(C))

print("\n--- LightGBM 超参 ---")
run("LGBM 默认(balanced,leaves12,400)", X, y, groups, folds, lambda: lgbm())
run("LGBM 不用类别权重", X, y, groups, folds,
    lambda: lgbm(class_weight=None))
run("LGBM leaves4,300", X, y, groups, folds,
    lambda: lgbm(num_leaves=4))
run("LGBM leaves4,150,min25", X, y, groups, folds,
    lambda: lgbm(num_leaves=4, n_estimators=150, min_child_samples=25))

print("\n--- 特征工程 ---")
run("Logistic C=0.05 + 工程特征", X_eng, y, groups, folds, lambda: logreg(0.05))
run("Logistic C=0.05 + 缺失指示", X_miss, y, groups, folds, lambda: logreg(0.05))
run("Logistic C=0.05 + 工程 + 缺失", X_eng_miss, y, groups, folds, lambda: logreg(0.05))
run("LGBM leaves4,150 + 工程 + 缺失", X_eng_miss, y, groups, folds,
    lambda: lgbm(num_leaves=4, n_estimators=150, min_child_samples=25))
