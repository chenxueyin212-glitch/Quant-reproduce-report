# -*- coding: utf-8 -*-
"""诊断剩余临界不一致: 001914 d1/l2, 002463/600449/001270 n_raw±1"""
import os, sys, sqlite3
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _niigu_judge_ref import compute_niugu_score, _ma, _safe_get

SRC = r"C:/Users/Lenovo/WorkBuddy/2026-07-17-00-40-38"
DB = os.path.join(SRC, "kline_cache_all.db")
con = sqlite3.connect("file:%s?mode=ro" % DB.replace("\\", "/"), uri=True)
cur = con.cursor()

def db_klines(code, upto=None):
    q = "SELECT date,open,high,low,close,volume FROM klines WHERE code=?"
    args = [code]
    if upto: q += " AND date<=?"; args.append(upto)
    q += " ORDER BY date"
    rows = cur.execute(q, args).fetchall()
    return [(r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])) for r in rows]

CASES = [("001914", "20241202"), ("002463", "20251103"), ("600449", "20250102"), ("001270", "20251009")]
for code, sd in CASES:
    kls = db_klines(code, sd)
    n = len(kls)
    c = np.array([k[4] for k in kls]); h = np.array([k[2] for k in kls]); l = np.array([k[3] for k in kls])
    v = np.array([k[5] for k in kls])
    # 原引擎 MA (np.mean 口径)
    ma20_o = _ma(c, 20); ma60_o = _ma(c, 60); ma5_o = _ma(c, 5); ma10_o = _ma(c, 10)
    # 我的 MA (cumsum 口径)
    def ma_arr(arr, period):
        nn = len(arr); r = np.full(nn, np.nan)
        if nn < period: return r
        cs = np.concatenate([[0.0], np.cumsum(arr)])
        r[period-1:] = (cs[period:] - cs[:-period]) / period
        return r
    ma20_m = ma_arr(c, 20); ma60_m = ma_arr(c, 60)
    i = n-1
    print("=== %s %s (n=%d) ===" % (code, sd, n))
    print("  c      = %.6f" % c[i])
    print("  ma20: orig=%.10f mine=%.10f diff=%.2e" % (ma20_o[i], ma20_m[i], ma20_m[i]-ma20_o[i]))
    print("  ma60: orig=%.10f mine=%.10f diff=%.2e" % (ma60_o[i], ma60_m[i], ma60_m[i]-ma60_o[i]))
    print("  orig: c>ma20=%s ma20>ma60=%s | mine: c>ma20=%s ma20>ma60=%s" % (
        c[i] > ma20_o[i], ma20_o[i] > ma60_o[i], c[i] > ma20_m[i], ma20_m[i] > ma60_m[i]))
    print("  ma5=%.6f ma10=%.6f (orig) | ma5>ma20>ma60: orig=%s mine=%s" % (
        ma5_o[i], ma10_o[i], ma5_o[i] > ma20_o[i] > ma60_o[i], ma5_o[i] > ma20_m[i] > ma60_m[i]))
    # stage_mult 边界
    ext = c[i]/ma60_o[i]-1.0
    print("  ext=%.6f (ext_mult 边界 0.20/0.50)" % ext)
    # trend_age
    up = (ma20_o > ma60_o)
    valid_prev = (~np.isnan(np.concatenate([[np.nan], ma20_o[:-1]]))) & (~np.isnan(np.concatenate([[np.nan], ma60_o[:-1]])))
    cross = np.zeros(n, bool); cross[1:] = up[1:] & (~up[:-1]) & valid_prev[1:]
    last_cross = np.maximum.accumulate(np.where(cross, np.arange(n), -1))
    age = (n-1) - last_cross[-1] if last_cross[-1] >= 0 else None
    print("  trend_age=%s (age_mult 边界 60/120), up=%s" % (age, up[i]))
    # 原引擎总分与维度
    nr = compute_niugu_score([{"date": k[0], "open": k[1], "high": k[2], "low": k[3], "close": k[4], "volume": k[5]} for k in kls], all_closes_map={})
    dims = nr["dimensions"]
    s = sum(dims[k]["score"] for k in ["趋势结构", "量价关系", "相对强弱", "波动率压缩", "筹码结构"])
    print("  orig: d1=%s d2=%s d3=%s d4=%s d6=%s sum=%s n_raw=%s stage_mult=%s" % (
        dims["趋势结构"]["score"], dims["量价关系"]["score"], dims["相对强弱"]["score"],
        dims["波动率压缩"]["score"], dims["筹码结构"]["score"], s, nr["n_raw_score"],
        nr["dimensions"]["阶段新鲜度"]["details"].get("mult")))
    print("  orig sum*mult = %.6f -> floor=%d" % (s*nr["dimensions"]["阶段新鲜度"]["details"].get("mult", 1.0), int(s*nr["dimensions"]["阶段新鲜度"]["details"].get("mult", 1.0))))
    print()
