# -*- coding: utf-8 -*-
"""诊断 300671/002463 的 d4 与 stage_mult 差异"""
import pickle, os, json, sys, sqlite3
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _niigu_judge_ref import compute_niugu_score, _ma
from _shunshi_buy_points_ref import _compute_atr
from _shunshi_buy_points_ref import _bp1_platform_breakout

SRC = r"C:/Users/Lenovo/WorkBuddy/2026-07-17-00-40-38"
DB = os.path.join(SRC, "kline_cache_all.db")
con = sqlite3.connect("file:%s?mode=ro" % DB.replace("\\", "/"), uri=True)
cur = con.cursor()
d = pickle.load(open(os.path.join(HERE, "bp1_features_poolref84.pkl"), "rb"))
feats = d["feats"]
POOL_FILE = os.path.join(HERE, "_pool_20260804.json")
pool_codes = set(s["code"] for s in json.load(open(POOL_FILE, encoding="utf-8"))["stocks"])
ref_codes = [c for c in feats if c in pool_codes]

def db_klines(code):
    rows = cur.execute("SELECT date,open,high,low,close,volume FROM klines WHERE code=? ORDER BY date", (code,)).fetchall()
    return [(r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])) for r in rows]

for code, sd in [("300671", "20240701"), ("300671", "20240902"), ("002463", "20240701"), ("002463", "20241101")]:
    kls_all = db_klines(code)
    sub = [k for k in kls_all if k[0] <= sd]
    n = len(sub)
    c = np.array([k[4] for k in sub]); h = np.array([k[2] for k in sub]); l = np.array([k[3] for k in sub])
    # 原引擎 stage 细节
    all_closes_map = {}
    for rc in ref_codes:
        rsub = [k for k in db_klines(rc) if k[0] <= sd]
        if len(rsub) >= 20: all_closes_map[rc] = np.array([k[4] for k in rsub])
    nr = compute_niugu_score([{"date": k[0], "open": k[1], "high": k[2], "low": k[3], "close": k[4], "volume": k[5]} for k in sub], all_closes_map=all_closes_map)
    sd_detail = nr["dimensions"]["阶段新鲜度"]["details"]
    ft = feats[code]
    si = [i for i, x in enumerate([str(v) for v in ft["date"]]) if x == sd][0]
    print("%s %s (n=%d):" % (code, sd, n))
    print("  orig stage: mult=%s ext=%s trend_age=%s | n_raw=%s d1=%s d2=%s d3=%s d4=%s d6=%s" % (
        sd_detail.get("mult"), sd_detail.get("ext_pct"), sd_detail.get("trend_age"), nr["n_raw_score"],
        nr["dimensions"]["趋势结构"]["score"], nr["dimensions"]["量价关系"]["score"],
        nr["dimensions"]["相对强弱"]["score"], nr["dimensions"]["波动率压缩"]["score"],
        nr["dimensions"]["筹码结构"]["score"]))
    print("  mine stage: mult=%.4f ext=%.4f trend_age=%s | n_raw=%s d1=%s d2=%s d3=%s d4=%s d6=%s" % (
        ft["stage_mult"][si], ft["ext"][si]*100 if not np.isnan(ft["ext"][si]) else -999,
        ft["trend_age"][si], ft["n_raw"][si], ft["d1"][si], ft["d2"][si], ft["d3"][si], ft["d4"][si], ft["d6"][si]))
    # 原引擎 d4 细节
    v4 = nr["dimensions"]["波动率压缩"]["details"]
    print("  orig d4 detail: BBW=%s ATR=%s" % (v4.get("BBW分位"), v4.get("ATR分位")))
    # 我的 bbw/atr 分位重算
    ma20 = _ma(c, 20); std20 = np.full(n, np.nan)
    for i in range(19, n): std20[i] = np.std(c[i-19:i+1])
    bbw = np.full(n, np.nan)
    for i in range(19, n):
        if ma20[i] > 0: bbw[i] = 4*std20[i]/ma20[i]
    atr = _compute_atr(h, l, c, 14)
    atr_ratio = atr/c
    lookback = min(120, n)
    bbw_valid = bbw[-lookback:]; bbw_valid = bbw_valid[~np.isnan(bbw_valid)]
    atr_valid = atr_ratio[-lookback:]; atr_valid = atr_valid[~np.isnan(atr_valid)]
    bbw_pct = np.sum(bbw_valid <= bbw[-1])/len(bbw_valid)*100
    atr_pct = np.sum(atr_valid <= atr_ratio[-1])/len(atr_valid)*100
    print("  mine bbw_pct=%.1f (cur=%.4f) atr_pct=%.1f (cur=%.4f) valid_n=%d/%d" % (
        bbw_pct, bbw[-1], atr_pct, atr_ratio[-1], len(bbw_valid), len(atr_valid)))
