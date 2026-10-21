# -*- coding: utf-8 -*-
"""检查 pkl 中 603353 的 date 数组/d2 数组与 si 映射"""
import pickle, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
d = pickle.load(open(os.path.join(HERE, "bp1_features_poolref84.pkl"), "rb"))
feats = d["feats"]
ft = feats["603353"]
dt = ft["date"]
print("date dtype:", dt.dtype, "len:", len(dt))
print("first 3:", dt[:3], "last 3:", dt[-3:])
SD = "20240603"
si = [i for i, x in enumerate([str(vv) for vv in dt]) if x == SD]
print("si candidates:", si)
if si:
    si = si[0]
    print("date[si] =", repr(dt[si]))
    print("d2[si-2:si+3] =", ft["d2"][max(0,si-2):si+3])
    print("d1[si-2:si+3] =", ft["d1"][max(0,si-2):si+3])
    print("n_raw[si-2:si+3] =", ft["n_raw"][max(0,si-2):si+3])
    print("d3[si-2:si+3] =", ft["d3"][max(0,si-2):si+3])
    # 看看 d2=12 出现在哪些日期
    idx12 = np.where(ft["d2"] == 12)[0]
    print("d2==12 的行号:", idx12[:20], "... 共", len(idx12))
    for ii in idx12[:10]:
        print("   d2=12 at", dt[ii])
# pkl 的 meta
print("meta keys:", list(d.keys()))
for k in d.keys():
    if k != "feats":
        print("  ", k, "=", str(d[k])[:200])
