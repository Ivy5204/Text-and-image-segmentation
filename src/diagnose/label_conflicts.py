"""Dump the 25 label-conflict records with their free-text clinical diagnosis.

The manifest's `class` comes from the image folder; the pipeline's label comes
from the CSV `sheet` column.  For 25 records they disagree.  The CSV also
carries a free-text `临床诊断` column, which is the closest thing to a source
of truth we have without opening the medical records.
"""

import csv
import os
import sys

import numpy as np
import pandas as pd

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

CSV_DIR = r"E:\模型\PythonProject"
FEATURE_DIR = r"E:\模型\mm4class\features"
CLASSES = ["阴性", "ARDS", "湿肺", "肺炎"]

frames = []
for split in ("train", "val", "test"):
    df = pd.read_csv(os.path.join(CSV_DIR, f"{split}.csv"))
    df["__split"] = split
    frames.append(df)
allc = pd.concat(frames, ignore_index=True)

manifest = list(csv.DictReader(
    open(os.path.join(FEATURE_DIR, "manifest.csv"), encoding="utf-8-sig")))
if len(allc) != len(manifest):
    raise SystemExit("CSV 与 manifest 行数不一致")

rows = []
for i, m in enumerate(manifest):
    csv_label = CLASSES[int(m["label"])]
    folder_label = m["class"]
    if csv_label == folder_label:
        continue
    rec = allc.iloc[i]
    rows.append({
        "住院号": int(m["patient_id"]),
        "划分": m["split"],
        "CSV标签": csv_label,
        "文件夹": folder_label,
        "模态": m["modality"],
        "临床诊断": str(rec.get("临床诊断", ""))[:60],
        "孕周": rec.get("孕周", ""),
        "新生儿体重g": rec.get("新生儿体重g", ""),
        "Apgar1": rec.get("Apgar评分1分钟", ""),
        "Apgar5": rec.get("Apgar评分5分钟", ""),
    })

print(f"冲突总数 {len(rows)}\n")
print("按冲突类型分组:")
groups = {}
for r in rows:
    groups.setdefault((r["CSV标签"], r["文件夹"]), []).append(r)
for (a, b), items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    mods = {i["模态"] for i in items}
    print(f"\n{'='*100}")
    print(f"CSV={a}  ->  文件夹={b}   共 {len(items)} 条   模态 {sorted(mods)}")
    print(f"{'='*100}")
    print(f"{'住院号':>8} {'划分':<6} {'孕周':<10} {'体重g':>7} "
          f"{'Ap1':>4} {'Ap5':>4}  临床诊断")
    for r in items:
        print(f"{r['住院号']:>8} {r['划分']:<6} {str(r['孕周']):<10} "
              f"{str(r['新生儿体重g']):>7} {str(r['Apgar1']):>4} "
              f"{str(r['Apgar5']):>4}  {r['临床诊断']}")

out = os.path.join(FEATURE_DIR, "label_conflicts.csv")
with open(out, "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\n已写出 {out}")

# How much would the ARDS class grow if the folders were authoritative?
csv_ards = sum(1 for m in manifest if int(m["label"]) == 1)
folder_ards = sum(1 for m in manifest if m["class"] == "ARDS")
print(f"\nARDS 例数: 按 CSV {csv_ards} 例, 按文件夹 {folder_ards} 例")
