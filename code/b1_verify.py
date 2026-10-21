# -*- coding: utf-8 -*-
"""校准验证: b1_features.pkl 重放信号 vs all_b1_signals.csv 原始信号, 必须逐条一致"""
import pickle, csv, os, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:\Users\Lenovo\WorkBuddy\2026-07-17-00-40-38"
d = pickle.load(open(os.path.join(HERE, "b1_features.pkl"), "rb"))
feats = d["feats"]

# 原始CSV信号集 (code, yyyymmdd, type)
orig = collections.defaultdict(set)
orig_n = 0
with open(os.path.join(SRC, "all_b1_signals.csv"), encoding="utf-8") as f:
    for x in csv.DictReader(f):
        if x["code"] in feats:
            orig[x["code"]].add((x["signal_date"].replace("-", ""), x["bp"]))
            orig_n += 1

# 新管道重放 (与原引擎一致: 从第59根K线起检测, 即 MIN_KLINES-1)
new = collections.defaultdict(set)
new_n = 0
for code, ft in feats.items():
    dates = ft["date"]; sb = ft["sig_base"]; sdc = ft["sig_dc"]; sdk = ft["sig_dk"]
    for i in np.where(sb > 0)[0]:
        if i < 59: continue
        t = "b1_dc" if sdc[i] else "b1_dk"
        new[code].add((dates[i].replace("-", ""), t))
        new_n += 1

import numpy as np  # noqa
codes = set(orig) | set(new)
miss = extra = 0
miss_ex, extra_ex = [], []
for c in codes:
    o, n_ = orig.get(c, set()), new.get(c, set())
    if o != n_:
        miss += len(o - n_); extra += len(n_ - o)
        if len(miss_ex) < 5 and (o - n_): miss_ex.append((c, sorted(o - n_)[:3]))
        if len(extra_ex) < 5 and (n_ - o): extra_ex.append((c, sorted(n_ - o)[:3]))
print("orig signals(in-universe): %d | replay: %d" % (orig_n, new_n))
print("codes mismatch: %d | missing(in orig not replay): %d | extra(in replay not orig): %d" % (
    sum(1 for c in codes if orig.get(c, set()) != new.get(c, set())), miss, extra))
print("miss examples:", miss_ex)
print("extra examples:", extra_ex)
# 类型级一致性
print("VERDICT:", "PASS" if (miss == 0 and extra == 0) else "FAIL")
