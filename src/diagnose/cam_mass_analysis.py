"""Where does the Grad-CAM energy actually sit?

Eyeballing heatmaps is subjective.  This splits each CAM map into concentric
zones and reports the share of energy in each -- the shortcut we are checking
for (hospital name, date, device label) lives in the outer border, while lung
pathology lives in the central field.
"""

import os

import numpy as np

OUT = r"E:\模型\mm4class\results\gradcam"


def zones(cam: np.ndarray):
    g = cam.shape[0]
    yy, xx = np.mgrid[0:g, 0:g] / (g - 1)          # 0..1
    r = np.maximum(np.abs(yy - 0.5), np.abs(xx - 0.5)) * 2   # 0 centre -> 1 edge
    total = cam.sum()
    if total <= 0:
        return None
    return {
        "中心 50%": cam[r <= 0.5].sum() / total,
        "中间 50-80%": cam[(r > 0.5) & (r <= 0.8)].sum() / total,
        "外圈 80-100%": cam[r > 0.8].sum() / total,
    }


for name, path in (("DR (RAD-DINO, 37x37)", "cams_dr.npz"),
                   ("超声 (USF-MAE, 14x14)", "cams_us.npz")):
    full = os.path.join(OUT, path)
    if not os.path.exists(full):
        print(f"{name}: 缺少 {path}")
        continue
    z = np.load(full)
    print("=" * 70)
    print(name)
    print("=" * 70)
    per_zone = {"中心 50%": [], "中间 50-80%": [], "外圈 80-100%": []}
    per_class = {}
    for key in z.files:
        cam = z[key]
        res = zones(cam)
        if res is None:
            continue
        for k, v in res.items():
            per_zone[k].append(v)
        cls = key.rsplit("_", 1)[0]
        per_class.setdefault(cls, {"中心 50%": [], "外圈 80-100%": []})
        per_class[cls]["中心 50%"].append(res["中心 50%"])
        per_class[cls]["外圈 80-100%"].append(res["外圈 80-100%"])

    print(f"{'区域':<16}{'能量占比':>10}{'面积占比':>10}   结论")
    area = {"中心 50%": 0.25, "中间 50-80%": 0.39, "外圈 80-100%": 0.36}
    for k in per_zone:
        v = float(np.mean(per_zone[k]))
        ratio = v / area[k]
        verdict = ("集中" if ratio > 1.3 else
                   "均匀" if ratio > 0.7 else "偏少")
        print(f"{k:<16}{v:>9.1%}{area[k]:>10.0%}   {verdict}")

    print()
    print("按类别看中心区域占比:")
    for cls in ("阴性", "ARDS", "湿肺", "肺炎"):
        if cls in per_class:
            c = np.mean(per_class[cls]["中心 50%"])
            e = np.mean(per_class[cls]["外圈 80-100%"])
            print(f"  {cls:<6} 中心 {c:.1%}   外圈 {e:.1%}")
    print()
