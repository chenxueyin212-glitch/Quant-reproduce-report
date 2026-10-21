# -*- coding: utf-8 -*-
"""bp1金标准交叉验证: 原引擎(DB float64输入) vs 我的向量化实现"""
import pickle, os, json, sys, bisect, random, sqlite3
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _niigu_judge_ref import compute_niugu_score
from _shunshi_buy_points_ref import _bp1_platform_breakout, _check_l2_daily_gate

SRC = r"C:/Users/Lenovo/WorkBuddy/2026-07-17-00-40-38"
DB = os.path.join(SRC, "kline_cache_all.db")
con = sqlite3.connect("file:%s?mode=ro" % DB.replace("\\", "/"), uri=True)
cur = con.cursor()

d = pickle.load(open(os.path.join(HERE, "bp1_features_poolref84.pkl"), "rb"))
feats = d["feats"]
POOL_FILE = os.path.join(HERE, "_pool_20260804.json")
pool_codes = set(s["code"] for s in json.load(open(POOL_FILE, encoding="utf-8"))["stocks"])
ref_codes = [c for c in feats if c in pool_codes]
print("参考系 %d 只" % len(ref_codes))

# 从DB读float64原始K线(原引擎真实输入)
def db_klines(code):
    rows = cur.execute("SELECT date,open,high,low,close,volume FROM klines WHERE code=? ORDER BY date", (code,)).fetchall()
    return [(r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])) for r in rows]

ref_data = {rc: db_klines(rc) for rc in ref_codes}

random.seed(7)
sample_codes = random.sample(ref_codes, 25)
all_dates = sorted({dt for c in sample_codes for dt, *_ in ref_data[c]})
month_first = {}
for dt in all_dates:
    ym = dt[:6]
    if ym not in month_first: month_first[ym] = dt
sample_dates = [v for ym, v in sorted(month_first.items()) if "202406" <= ym <= "202606"]

def klines_of(kls, upto):
    out = []
    for d_, o_, h_, l_, c_, v_ in kls:
        if d_ > upto: break
        out.append({"date": d_, "open": o_, "high": h_, "low": l_, "close": c_, "volume": v_})
    return out

n_cmp = 0
n_ok = {"d1": 0, "d2": 0, "d3": 0, "d4": 0, "d6": 0, "n_raw": 0, "tech": 0, "l2": 0}
mismatch = []
for code in sample_codes:
    kls = ref_data[code]
    ft = feats[code]
    dates = [str(x) for x in ft["date"]]
    for sd in sample_dates:
        sub = klines_of(kls, sd)
        if len(sub) < 60: continue
        all_closes_map = {}
        for rc in ref_codes:
            rsub = klines_of(ref_data[rc], sd)
            if len(rsub) >= 20:
                all_closes_map[rc] = np.array([k["close"] for k in rsub])
        nr = compute_niugu_score(sub, all_closes_map=all_closes_map)
        si = None
        for i, ds in enumerate(dates):
            if ds == sd: si = i; break
        if si is None: continue
        bp = _bp1_platform_breakout(sub)
        l2 = _check_l2_daily_gate(sub)
        mine = {"d1": float(ft["d1"][si]), "d2": float(ft["d2"][si]), "d3": float(ft["d3"][si]),
                "d4": float(ft["d4"][si]), "d6": float(ft["d6"][si]),
                "n_raw": int(ft["n_raw"][si]), "tech": bool(ft["bp1_tech"][si]), "l2": bool(ft["l2_pass"][si])}
        theirs = {"d1": nr["dimensions"]["趋势结构"]["score"], "d2": nr["dimensions"]["量价关系"]["score"],
                  "d3": nr["dimensions"]["相对强弱"]["score"], "d4": nr["dimensions"]["波动率压缩"]["score"],
                  "d6": nr["dimensions"]["筹码结构"]["score"], "n_raw": nr["n_raw_score"],
                  "tech": bp is not None, "l2": l2[0]}
        n_cmp += 1
        bad = []
        for k in ["d1", "d2", "d3", "d4", "d6", "n_raw", "tech", "l2"]:
            if mine[k] != theirs[k]:
                bad.append("%s: mine=%s orig=%s" % (k, mine[k], theirs[k]))
        if bad:
            mismatch.append((code, sd, bad))
        else:
            for k in n_ok: n_ok[k] += 1

print("比对样本: %d" % n_cmp)
for k in n_ok:
    print("  %-6s 一致 %d/%d (%.1f%%)" % (k, n_ok[k], n_cmp, 100*n_ok[k]/n_cmp))
print("全维度一致: %d (%.1f%%)" % (n_cmp-len(mismatch), 100*(n_cmp-len(mismatch))/n_cmp))
if mismatch:
    print("\n不一致明细(前30条):")
    for code, sd, bad in mismatch[:30]:
        print("  %s %s: %s" % (code, sd, " | ".join(bad)))
json.dump({"n_cmp": n_cmp, "n_ok": n_cmp-len(mismatch),
           "mismatch": [{"code": c, "date": s, "bad": b} for c, s, b in mismatch]},
          open(os.path.join(HERE, "bp1_crosscheck.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved -> bp1_crosscheck.json")
