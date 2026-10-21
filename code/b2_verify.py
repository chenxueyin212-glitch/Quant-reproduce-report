# -*- coding: utf-8 -*-
"""校准验证: b2_features.pkl 重放 vs all_b2_signals.csv"""
import pickle, csv, os, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:\Users\Lenovo\WorkBuddy\2026-07-17-00-40-38"
d = pickle.load(open(os.path.join(HERE, "b2_features.pkl"), "rb"))
feats = d["feats"]

orig = collections.defaultdict(dict)
with open(os.path.join(SRC, "all_b2_signals.csv"), encoding="utf-8") as f:
    for x in csv.DictReader(f):
        if x["code"] in feats:
            orig[x["code"]][x["signal_date"].replace("-", "")] = (x["b2_score"], x["b2_status"])

new = collections.defaultdict(dict)
for code, ft in feats.items():
    dates = ft["date"]
    for i in np.where(ft["b2_sig"])[0]:
        sc = int(ft["b2_score"][i])
        new[code][dates[i].replace("-", "")] = (str(sc), "B2强信号" if sc >= 7 else "B2信号")

codes = set(orig) | set(new)
miss = extra = typ = 0
miss_ex, extra_ex = [], []
for c in codes:
    o, n_ = orig.get(c, {}), new.get(c, {})
    for k in set(o) | set(n_):
        if k not in n_: miss += 1; miss_ex.append((c, k))
        elif k not in o: extra += 1; extra_ex.append((c, k))
        elif o[k][0] != n_[k][0]: typ += 1
print("orig(in-universe): %d | replay: %d" % (sum(len(v) for v in orig.values()), sum(len(v) for v in new.values())))
print("missing: %d | extra: %d | score-mismatch: %d" % (miss, extra, typ))
print("miss examples:", miss_ex[:5])
print("extra examples:", extra_ex[:5])
# 类型分布对比
from collections import Counter
co = Counter(v[1] for d_ in orig.values() for v in d_.values())
cn = Counter(v[1] for d_ in new.values() for v in d_.values())
print("orig status:", dict(co))
print("replay status:", dict(cn))
print("VERDICT:", "PASS" if (miss == 0 and typ == 0) else "CHECK")
