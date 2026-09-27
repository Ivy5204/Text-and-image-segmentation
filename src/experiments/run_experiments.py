"""Full experiment matrix, driven by the ablation list in the decision doc.

Usage
  python run_experiments.py --quick          # 3 repeats, 18 epochs, for a read
  python run_experiments.py --full           # 5 repeats, 40 epochs, for the write-up
  python run_experiments.py --full --only full

Every model runs on the *same* folds, and the fusion model is compared against
the full variant with a corrected resampled t-test, because with 39 ARDS cases
a raw score difference of 0.05 means nothing on its own.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable, Dict, List

import numpy as np
import torch

import mm_data
import mm_eval
import mm_train
from mm_models import FusionConfig, LightGBMHead, LogisticHead

RESULTS_DIR = r"E:\模型\mm4class\results"


def _per_modality_head(feats, modalities, head_factory) -> Callable:
    """Train one head per modality and merge predictions.

    The two encoders emit different widths, so a single head cannot span both.
    """
    def fit_predict(tr, te):
        preds = np.zeros((len(te), 4))
        for mod in sorted(set(modalities)):
            # tr/te hold *global* indices; te_m holds positions within te.
            tr_m = [i for i in tr if modalities[i] == mod]
            te_m = [k for k, i in enumerate(te) if modalities[i] == mod]
            if not tr_m or not te_m:
                continue
            head = head_factory()
            head.fit(np.stack([feats[i] for i in tr_m]), CURRENT_Y[tr_m])
            preds[te_m] = head.predict_proba(
                np.stack([feats[te[k]] for k in te_m])
            )
        return preds
    return fit_predict


CURRENT_Y = None


def build_experiments(cohort, arrays, cfg, only=None):
    global CURRENT_Y
    pooled_arr, tokens_arr, mod_arr = arrays
    pooled_list = [r for r in cohort.pooled]
    tab = cohort.tabular
    modalities = cohort.modality
    y = cohort.label
    CURRENT_Y = y

    experiments = {}

    # ---------------- table only ----------------
    def t_lgbm(tr, te):
        model = LightGBMHead(seed=0).fit(tab[tr], y[tr])
        return model.predict_proba(tab[te])

    def t_logreg(tr, te):
        from mm_models import TabularPreprocessor
        prep = TabularPreprocessor().fit(tab[tr])
        model = LogisticHead().fit(prep.transform(tab[tr]), y[tr])
        return model.predict_proba(prep.transform(tab[te]))

    experiments["T-only LightGBM"] = t_lgbm
    experiments["T-only Logistic"] = t_logreg

    # ---------------- image only ----------------
    experiments["I-only LightGBM"] = _per_modality_head(
        pooled_list, modalities, lambda: LightGBMHead(seed=0)
    )
    experiments["I-only Logistic"] = _per_modality_head(
        pooled_list, modalities, lambda: LogisticHead()
    )

    # ---------------- fusion ----------------
    def make_fusion(overrides):
        c = FusionConfig(**{**cfg.__dict__, **overrides})

        def fit_predict(tr, te):
            tr_arrays = (
                pooled_arr[tr], tokens_arr[tr],
                tab[tr], mod_arr[tr],
            )
            te_arrays = (
                pooled_arr[te], tokens_arr[te],
                tab[te], mod_arr[te],
            )
            out = mm_train.train_fusion(
                tr_arrays, te_arrays, y[tr], y[te], c, seed=0
            )
            GATES.append({"fold": len(GATES), "gate": out["gate"]})
            return out["proba"]
        return fit_predict

    GATES.clear()
    # Headline model: gated fusion only.  Cross-attention is no longer part of
    # the main line -- now that it is implemented correctly it gets its own
    # ablation instead of being assumed to help.
    experiments["MAIN: gated fusion"] = make_fusion(
        {"fusion_mode": "gate", "use_cross_attn": False}
    )

    # ---------------- ablations ----------------
    # The old entry named "concat fusion" actually computed z_img + query (a
    # sum, not a concatenation).  It is now two honestly-named variants.
    experiments["abl: + cross-attn"] = make_fusion(
        {"fusion_mode": "gate", "use_cross_attn": True}
    )
    experiments["abl: concat fusion"] = make_fusion(
        {"fusion_mode": "concat", "use_cross_attn": False}
    )
    experiments["abl: sum fusion"] = make_fusion(
        {"fusion_mode": "sum", "use_cross_attn": False}
    )
    experiments["abl: no modality dropout"] = make_fusion(
        {"fusion_mode": "gate", "use_cross_attn": False, "modality_dropout": 0.0}
    )
    experiments["abl: no aux heads"] = make_fusion(
        {"fusion_mode": "gate", "use_cross_attn": False, "use_aux": False}
    )
    experiments["abl: no InfoNCE"] = make_fusion(
        {"fusion_mode": "gate", "use_cross_attn": False, "use_infonce": False}
    )

    if only:
        wanted = [k for k in experiments if only.lower() in k.lower()]
        return {k: experiments[k] for k in wanted}
    return experiments


GATES: List[Dict] = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="3 repeats, 18 epochs")
    parser.add_argument("--full", action="store_true",
                        help="5 repeats, 40 epochs")
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()

    if args.quick:
        repeats, epochs = 3, 18
    else:
        repeats, epochs = mm_eval.N_REPEATS, 40

    # This box is CPU-only and memory tight.  Extra inter-op threads just add
    # contention; denormal flushing avoids the 10-100x convolution penalty.
    torch.set_flush_denormal(True)
    torch.set_num_interop_threads(1)

    cfg = FusionConfig(epochs=epochs)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("载入数据 ...")
    cohort = mm_data.load_cohort()
    print(mm_data.describe(cohort))

    print("\n组装数组 ...")
    arrays = mm_train.build_arrays(cohort)
    print(f"  pooled {arrays[0].shape}  tokens {arrays[1].shape}")

    folds = mm_eval.make_folds(cohort.label, cohort.groups, n_repeats=repeats)
    print(f"  折数 {len(folds)} (repeats={repeats})")

    experiments = build_experiments(cohort, arrays, cfg, only=args.only)
    print(f"  实验数 {len(experiments)}")

    results = []
    tag = "quick" if args.quick else "full"
    # A focused run (--only) must not clobber the full-matrix summary.
    suffix = ""
    if args.only:
        safe = "".join(c if c.isalnum() else "_" for c in args.only)
        suffix = f"_only_{safe}"
    summary_path = os.path.join(RESULTS_DIR, f"summary_{tag}{suffix}.json")

    def dump():
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "settings": {
                        "repeats": repeats,
                        "epochs": epochs,
                        "n_samples": len(cohort),
                        "n_folds": len(folds),
                        "token_grid": mm_train.TOKEN_GRID,
                        "completed": len(results),
                        "planned": len(experiments),
                    },
                    "results": [{"name": r.name, **r.summary()} for r in results],
                },
                fh, ensure_ascii=False, indent=2,
            )

    for name, fit_predict in experiments.items():
        t0 = time.time()
        res = mm_eval.run_cv(name, cohort.label, cohort.groups, fit_predict,
                             folds=folds)
        results.append(res)
        # Keep the out-of-fold probabilities: threshold tuning needs them and
        # re-running a fusion model just to re-derive a decision rule is waste.
        if res.pooled_oof is not None:
            safe = "".join(c if c.isalnum() else "_" for c in name)
            np.savez_compressed(
                os.path.join(RESULTS_DIR, f"oof_{tag}{suffix}_{safe}.npz"),
                y=cohort.label,
                oof=res.pooled_oof,
                patient_id=cohort.patient_id,
            )
        s = res.summary()
        print(f"  [{time.time()-t0:6.1f}s] {name:<28} "
              f"macro-AUROC {s['macro_auroc_mean']:.3f} ± {s['macro_auroc_std']:.3f}"
              f"   ARDS AUPRC {s['ards_auprc_mean']:.3f}")
        dump()   # write after every model so a long run is never lost

    print("\n" + mm_eval.format_table(results))

    # ---------------- paired comparisons vs the full model ----------------
    full = next((r for r in results if r.name.startswith("MAIN")), None)
    comparison_rows = []
    if full is not None:
        for r in results:
            if r is full:
                continue
            stat = mm_eval.paired_resampled_ttest(
                full.fold_scores["macro_auroc"],
                r.fold_scores["macro_auroc"],
                n_train=int(len(cohort) * 0.8) if False else 700,
                n_test=180,
            )
            comparison_rows.append({
                "model": r.name,
                "delta_macro_auroc": stat["diff"],
                "t": stat["t"],
                "p_value": stat["p"],
                "significant_at_0.05": bool(stat["p"] < 0.05)
                if np.isfinite(stat["p"]) else False,
            })

    dump()
    print(f"\n已写出 {summary_path}")

    if comparison_rows:
        print("\n===== 与完整模型的配对比较 (macro-AUROC) =====")
        print(f"{'model':<30}{'delta':>9}{'p':>10}  sig")
        for row in comparison_rows:
            print(f"{row['model']:<30}{row['delta_macro_auroc']:>9.4f}"
                  f"{row['p_value']:>10.4f}  {'YES' if row['significant_at_0.05'] else 'no'}")
        print("\n提示: p 值基于配对的 25 折分数; ARDS 仅 39 例, 差异 <0.08 时请谨慎解读")


if __name__ == "__main__":
    main()
