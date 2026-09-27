"""Unified data access for the newborn DR/US + tabular four-class task.

Everything downstream (baselines, fusion, ablations) reads through
``load_cohort`` so the sample order, labels and group ids are defined once.

Sample order is guaranteed to match the concatenation of train/val/test CSVs,
because that is exactly how the manifest was built.  We assert on the length
rather than merging on 住院号, which would be ambiguous: 24 patients have more
than one record.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

CLASSES = ["阴性", "ARDS", "湿肺", "肺炎"]
LABEL_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

# Identifier / free-text columns that must never become features.
# 孕周 is a free-text string like "33w+5d"; the numeric 孕周_天 column below
# already carries the same information.
DROP_COLUMNS = {"序号", "住院号", "孕妇姓名", "临床诊断", "label", "sheet", "孕周"}

DEFAULT_FEATURE_DIR = r"E:\模型\mm4class\features"
DEFAULT_CSV_DIR = r"E:\模型\PythonProject"


@dataclass
class Cohort:
    patient_id: np.ndarray          # (N,) int64
    label: np.ndarray               # (N,) int64, 0..3
    split: np.ndarray               # (N,) object, train/val/test
    modality: np.ndarray            # (N,) object, DR/US
    tabular: np.ndarray             # (N, D) float32, NaN allowed
    tabular_columns: List[str]
    pooled: List[np.ndarray]        # per-sample (D_img,) float32
    tokens: List[np.ndarray]        # per-sample (49, C) float32
    labels_match_folder: np.ndarray # (N,) bool

    def __len__(self) -> int:
        return len(self.label)

    @property
    def groups(self) -> np.ndarray:
        return self.patient_id

    def subset(self, mask: np.ndarray) -> "Cohort":
        idx = np.flatnonzero(mask)
        return Cohort(
            patient_id=self.patient_id[idx],
            label=self.label[idx],
            split=self.split[idx],
            modality=self.modality[idx],
            tabular=self.tabular[idx],
            tabular_columns=self.tabular_columns,
            pooled=[self.pooled[i] for i in idx],
            tokens=[self.tokens[i] for i in idx],
            labels_match_folder=self.labels_match_folder[idx],
        )


def _read_tabular(csv_dir: str) -> pd.DataFrame:
    frames = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(os.path.join(csv_dir, f"{split}.csv"))
        df["__split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _encode_tabular(df: pd.DataFrame):
    """Return (matrix, column names).  NaNs are preserved on purpose.

    LightGBM handles missing values natively, so imputing here would throw
    away information for no reason.  The neural branch imputes separately.
    """
    feature_cols = [c for c in df.columns
                    if c not in DROP_COLUMNS and not c.startswith("__")]

    encoded, names = [], []
    dropped = []
    for col in feature_cols:
        series = df[col]
        if series.dtype == bool:
            encoded.append(series.astype(np.float32).to_numpy())
            names.append(col)
            continue
        if pd.api.types.is_numeric_dtype(series):
            encoded.append(series.astype(np.float32).to_numpy())
            names.append(col)
            continue
        # Mixed / textual column: try a numeric coercion, otherwise one-hot
        # the small-cardinality ones and drop the rest.
        coerced = pd.to_numeric(series, errors="coerce")
        if coerced.notna().mean() > 0.95:
            encoded.append(coerced.astype(np.float32).to_numpy())
            names.append(col)
            continue
        nunique = series.nunique(dropna=True)
        if nunique <= 12:
            dummies = pd.get_dummies(series, prefix=col, dummy_na=False, dtype=np.float32)
            for sub in dummies.columns:
                encoded.append(dummies[sub].to_numpy())
                names.append(str(sub))
        else:
            dropped.append((col, nunique))

    if dropped:
        print("  丢弃的高基数文本列:", dropped)

    matrix = np.stack(encoded, axis=1).astype(np.float32)
    return matrix, names


def load_cohort(
    feature_dir: str = DEFAULT_FEATURE_DIR,
    csv_dir: str = DEFAULT_CSV_DIR,
) -> Cohort:
    manifest_path = os.path.join(feature_dir, "manifest.csv")
    with open(manifest_path, encoding="utf-8-sig") as fh:
        manifest = list(csv.DictReader(fh))

    df = _read_tabular(csv_dir)
    if len(df) != len(manifest):
        raise RuntimeError(
            f"CSV 行数 {len(df)} 与 manifest 行数 {len(manifest)} 不一致；"
            "两者必须按 train/val/test 顺序一一对应"
        )

    pooled_by_mod, tokens_by_mod = {}, {}
    for tag, fname in (("DR", "features_dr.npz"), ("US", "features_us.npz")):
        path = os.path.join(feature_dir, fname)
        z = np.load(path)
        pooled_by_mod[tag] = {int(i): f for i, f in zip(z["patient_id"], z["features"])}
        tokens_by_mod[tag] = {int(i): t.astype(np.float32)
                              for i, t in zip(z["patient_id"], z["tokens"])}

    tabular, tabular_columns = _encode_tabular(df)

    patient_id = np.array([int(r["patient_id"]) for r in manifest], dtype=np.int64)
    label = np.array([int(r["label"]) for r in manifest], dtype=np.int64)
    split = np.array([r["split"] for r in manifest], dtype=object)
    modality = np.array([r["modality"] for r in manifest], dtype=object)
    label_ok = np.array([int(r["label_matches_folder"]) == 1 for r in manifest])

    pooled, tokens = [], []
    for row in manifest:
        pid, mod = int(row["patient_id"]), row["modality"]
        pooled.append(pooled_by_mod[mod][pid])
        tokens.append(tokens_by_mod[mod][pid])

    return Cohort(
        patient_id=patient_id,
        label=label,
        split=split,
        modality=modality,
        tabular=tabular,
        tabular_columns=tabular_columns,
        pooled=pooled,
        tokens=tokens,
        labels_match_folder=label_ok,
    )


def describe(cohort: Cohort) -> str:
    lines = [f"样本数 {len(cohort)}  特征维度 {cohort.tabular.shape[1]}"]
    for mod in sorted(set(cohort.modality)):
        m = cohort.modality == mod
        lines.append(f"  {mod:<3} n={m.sum():4d}  缺失特征比例 "
                     f"{np.isnan(cohort.tabular[m]).mean():.1%}")
    lines.append("  类别分布: " + ", ".join(
        f"{LABEL_TO_CLASS[k]}={int((cohort.label == k).sum())}" for k in range(4)))
    lines.append(f"  标签与文件夹不一致: {int((~cohort.labels_match_folder).sum())} 条")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe(load_cohort()))
