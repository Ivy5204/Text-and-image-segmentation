"""Build the 安医附院 cohort in the same layout as 威宁.

Differences that need bridging:
  * 孕期情况 is free text here ("妊娠合并甲状腺功能减退、妊娠期糖尿病")
    rather than the seven one-hot columns 威宁 already has.  Keywords are
    mapped onto those seven; anything unrecognised becomes all-zero, which is
    a real limitation of the external test and is reported as such.
  * 孕妇MBI is a typo for 孕妇BMI.
  * 孕周 is "39w+6d" and has to become days.
  * Sheet names differ (新生儿呼吸窘迫综合征 -> ARDS, 新生儿湿肺 -> 湿肺).

Outputs (into the feature directory):
  manifest_ani.csv   patient id, label, image paths per modality
  tabular_ani.npz    the 20-column matrix, identical column order to 威宁
"""

from __future__ import annotations

import csv
import os
import re

import numpy as np
import pandas as pd

ROOT = r"E:\学习资料\第一个项目\胎肺数据\安医附院数据"
XLSX = os.path.join(ROOT, "安医附院新生儿数据资料.xlsx")
FEATURE_DIR = r"E:\模型\mm4class\features"

SHEET_TO_CLASS = {
    "阴性": "阴性",
    "新生儿呼吸窘迫综合征": "ARDS",
    "新生儿湿肺": "湿肺",
    "肺炎": "肺炎",
}
CLASSES = ["阴性", "ARDS", "湿肺", "肺炎"]

# 威宁's seven one-hot columns and the keywords that should set them in 安医
PREG_KEYWORDS = {
    "孕期_乙肝携带者": ["乙肝"],
    "孕期_妊娠期糖尿病": ["妊娠期糖尿病", "妊娠期发生的糖尿病"],
    "孕期_孕期糖尿病未处理": [],
    "孕期_无": ["无"],
    "孕期_甲减": ["甲状腺功能减退", "甲减"],
    "孕期_糖尿病": ["糖尿病"],
    "孕期_高血糖": ["高血糖"],
}

FOLDER_BY_CLASS = {
    ("DR", "阴性"): "DR  阴性",
    ("DR", "ARDS"): "DR ARDS",
    ("DR", "湿肺"): "DR 湿肺",
    ("DR", "肺炎"): "DR  肺炎",
    ("US", "阴性"): "四腔心  阴性",
    ("US", "ARDS"): "四腔心  呼吸窘迫综合征",
    ("US", "湿肺"): "四腔心  新生儿湿肺",
    ("US", "肺炎"): "四腔心  肺炎",
}


def gest_days(text):
    if pd.isna(text):
        return np.nan
    m = re.match(r"(\d+)\s*[wW]\s*\+?\s*(\d+)?", str(text))
    if not m:
        return np.nan
    weeks = int(m.group(1))
    days = int(m.group(2)) if m.group(2) else 0
    return weeks * 7 + days


def main():
    frames = []
    for sheet, cls in SHEET_TO_CLASS.items():
        df = pd.read_excel(XLSX, sheet_name=sheet, header=1)
        df = df.dropna(subset=["住院号"]).copy()
        df["__class"] = cls
        df["__label"] = CLASSES.index(cls)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    print(f"安医记录 {len(data)} 条, 唯一住院号 {data['住院号'].nunique()}")

    # --- build the 20 columns in exactly the 威宁 order
    cols_numeric = ["孕妇年龄", "孕妇身高", "孕妇体重", "孕妇BMI",
                    "孕期增加体重", "剖宫产", "胎膜早破", "产时羊水性质",
                    "新生儿性别", "新生儿体重g", "Apgar评分1分钟",
                    "Apgar评分5分钟", "孕周_天"]

    out = pd.DataFrame(index=data.index)
    out["住院号"] = data["住院号"].astype(int)
    out["label"] = data["__label"].astype(int)
    out["class"] = data["__class"].values
    out["孕妇年龄"] = pd.to_numeric(data["孕妇年龄"], errors="coerce")
    out["孕妇身高"] = pd.to_numeric(data["孕妇身高"], errors="coerce")
    out["孕妇体重"] = pd.to_numeric(data["孕妇体重"], errors="coerce")
    out["孕妇BMI"] = pd.to_numeric(data.get("孕妇MBI"), errors="coerce")
    out["孕期增加体重"] = pd.to_numeric(data["孕期增加体重"], errors="coerce")
    out["剖宫产"] = pd.to_numeric(data.get("剖宫产（1）"), errors="coerce")
    out["胎膜早破"] = pd.to_numeric(data.get("胎膜早破（1）"), errors="coerce")
    out["产时羊水性质"] = pd.to_numeric(data.get("产时羊水性质（清=0）"),
                                   errors="coerce")
    out["新生儿性别"] = pd.to_numeric(
        data.get("新生儿性别（男=1，女=0）"), errors="coerce")
    out["新生儿体重g"] = pd.to_numeric(data["新生儿体重g"], errors="coerce")
    out["Apgar评分1分钟"] = pd.to_numeric(data["Apgar评分1分钟"], errors="coerce")
    out["Apgar评分5分钟"] = pd.to_numeric(data["Apgar评分5分钟"], errors="coerce")
    out["孕周_天"] = data["孕周"].map(gest_days)

    text = data["孕期情况"].fillna("").astype(str)
    # 威宁 separates pre-existing 糖尿病 from 妊娠期糖尿病, so the plain
    # keyword would double-count every gestational case.  Strip those first.
    text_pre = text.str.replace("妊娠期糖尿病", "", regex=False)
    text_pre = text_pre.str.replace("妊娠期发生的糖尿病", "", regex=False)
    for col, kws in PREG_KEYWORDS.items():
        if not kws:
            out[col] = False
            continue
        source = text_pre if col == "孕期_糖尿病" else text
        hit = np.zeros(len(text), dtype=bool)
        for kw in kws:
            hit |= source.str.contains(kw, regex=False).to_numpy()
        out[col] = hit
    # "无" should not fire when another condition is also named
    other = np.zeros(len(text), dtype=bool)
    for col, kws in PREG_KEYWORDS.items():
        if col == "孕期_无":
            continue
        for kw in kws:
            other |= text.str.contains(kw, regex=False).to_numpy()
    out["孕期_无"] = out["孕期_无"].to_numpy() & ~other

    unordered = ["孕妇年龄", "孕妇身高", "孕妇体重", "孕妇BMI", "孕期增加体重",
                 "剖宫产", "胎膜早破", "产时羊水性质", "新生儿性别",
                 "新生儿体重g", "Apgar评分1分钟", "Apgar评分5分钟", "孕周_天"] \
        + list(PREG_KEYWORDS)
    missing = [c for c in cols_numeric if c not in out.columns]
    assert not missing, missing

    print("\n孕期情况 关键词命中率:")
    for col, kws in PREG_KEYWORDS.items():
        if kws:
            print(f"  {col:<22} {int(out[col].sum()):>3} 条命中")
    unknown = (~out[list(PREG_KEYWORDS)].any(axis=1)).sum()
    print(f"  七个标志全为 0 的记录: {unknown} 条 "
          f"({unknown/len(out):.0%}) —— 这些的孕期信息在威宁的编码体系里丢失了")

    # --- image paths
    rows = []
    for _, r in out.iterrows():
        pid = int(r["住院号"])
        entry = {"patient_id": pid, "split": "external",
                 "label": int(r["label"]), "class": r["class"],
                 "has_dr": 0, "has_us": 0, "dr_path": "", "us_path": ""}
        for modality, key in (("DR", "dr_path"), ("US", "us_path")):
            folder = FOLDER_BY_CLASS.get((modality, r["class"]))
            if not folder:
                continue
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                p = os.path.join(ROOT, folder, f"{pid}{ext}")
                if os.path.exists(p):
                    entry[key] = p
                    entry["has_dr" if modality == "DR" else "has_us"] = 1
                    break
        rows.append(entry)

    n_dr = sum(r["has_dr"] for r in rows)
    n_us = sum(r["has_us"] for r in rows)
    print(f"\n图像匹配: 有 DR {n_dr} 条, 有超声 {n_us} 条, "
          f"两者都有 {sum(1 for r in rows if r['has_dr'] and r['has_us'])} 条")

    mpath = os.path.join(FEATURE_DIR, "manifest_ani.csv")
    with open(mpath, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"已写出 {mpath}")

    matrix = out[unordered].to_numpy(dtype=np.float32)
    np.savez_compressed(
        os.path.join(FEATURE_DIR, "tabular_ani.npz"),
        patient_id=out["住院号"].to_numpy(dtype=np.int64),
        label=out["label"].to_numpy(dtype=np.int64),
        features=matrix,
        columns=np.array(unordered, dtype=object),
    )
    print(f"已写出 tabular_ani.npz  形状 {matrix.shape}")
    print(f"列顺序: {unordered}")


if __name__ == "__main__":
    main()
