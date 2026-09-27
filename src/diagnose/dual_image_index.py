"""How many images does each sample actually have?

The current pipeline assigns exactly one image per sample (DR if present,
ultrasound otherwise).  If the patients who got a DR also have an ultrasound,
that second image is being thrown away -- 483 of 879 samples.
"""

import csv
import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from extract_features import scan_image_index  # noqa: E402

IMG_ROOT = r"E:\学习资料\第一个项目\胎肺数据\威宁妇幼数据_clahe"
FEATURE_DIR = r"E:\模型\mm4class\features"

index = scan_image_index(IMG_ROOT)
print(f"图像索引: {len(index)} 条 (模态, 住院号) 记录")
for mod in ("DR", "US"):
    n = sum(1 for k in index if k[0] == mod)
    print(f"  {mod}: {n} 张")

manifest = list(csv.DictReader(
    open(os.path.join(FEATURE_DIR, "manifest.csv"), encoding="utf-8-sig")))
print(f"\n样本 {len(manifest)} 条")

both = dr_only = us_only = neither = 0
rows = []
for r in manifest:
    pid = int(r["patient_id"])
    has_dr = ("DR", pid) in index
    has_us = ("US", pid) in index
    if has_dr and has_us:
        both += 1
    elif has_dr:
        dr_only += 1
    elif has_us:
        us_only += 1
    else:
        neither += 1
    rows.append({
        "patient_id": pid,
        "split": r["split"],
        "label": int(r["label"]),
        "current_modality": r["modality"],
        "has_dr": int(has_dr),
        "has_us": int(has_us),
        "dr_path": index.get(("DR", pid), {}).get("path", ""),
        "us_path": index.get(("US", pid), {}).get("path", ""),
    })

print("\n按图片可得性统计:")
print(f"  两种都有 : {both:4d}  ({both/len(manifest):.1%})")
print(f"  只有 DR  : {dr_only:4d}")
print(f"  只有超声 : {us_only:4d}  ({us_only/len(manifest):.1%})")
print(f"  都没有   : {neither:4d}")

print("\n当前策略 vs 可用图:")
cur_dr = sum(1 for r in rows if r["current_modality"] == "DR")
print(f"  当前用 DR 的样本 {cur_dr} 条, 其中能同时拿到超声的 "
      f"{sum(1 for r in rows if r['current_modality'] == 'DR' and r['has_us'])} 条")
print(f"  全部样本中, 有 DR 的 {sum(r['has_dr'] for r in rows)} 条, "
      f"有超声的 {sum(r['has_us'] for r in rows)} 条")
print(f"  单图方案用掉 {len(manifest)} 张; 双图方案可用 "
      f"{sum(r['has_dr'] for r in rows) + sum(r['has_us'] for r in rows)} 张")

out = os.path.join(FEATURE_DIR, "manifest_dual.csv")
with open(out, "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\n已写出 {out}")
