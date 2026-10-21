# -*- coding: utf-8 -*-
"""诊断 000571 20240603 d4 差异: bbw_pct / atr_pct 两边精算"""
import os, sys, sqlite3
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _niigu_judge_ref import _ma, _percentile_rank

SRC = r"C:/Users/Lenovo/WorkBuddy/2026-07-17-00-40-38"
DB = os.path.join(SRC, "kline_cache_all.db")
con = sqlite3.connect("file:%s?mode=ro" % DB.replace("\\", "/"), uri=True)
cur = con.cursor()

CODE, SD = "000571", "20240603"
rows = cur.execute("SELECT date,open,high,low,close,volume FROM klines WHERE code=? AND date<=? ORDER BY date", (CODE, SD)).fetchall()
kls = [(r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])) for r in rows]
n = len(kls)
c = np.array([k[4] for k in kls]); h = np.array([k[2] for k in kls]); l = np.array([k[3] for k in kls])
lookback = min(120, n)

def ma_arr(arr, period):
    nn = len(arr); r = np.full(nn, np.nan)
    if nn < period: return r
    cs = np.concatenate([[0.0], np.cumsum(arr)])
    r[period-1:] = (cs[period:] - cs[:-period]) / period
    return r

# ---- BBW ----
sma20_o = _ma(c, 20); sma20_m = ma_arr(c, 20)
std20_o = np.full(n, np.nan)
for i in range(19, n): std20_o[i] = np.std(c[i-19:i+1])
std20_m = np.full(n, np.nan)
std20_m[19:] = np.lib.stride_tricks.sliding_window_view(c, 20).std(axis=1)
bbw_o = 4*std20_o/np.where(sma20_o > 0, sma20_o, np.nan)
bbw_m = 4*std20_m/np.where(sma20_m > 0, sma20_m, np.nan)
bbw_o_v = bbw_o[-lookback:]; bbw_o_v = bbw_o_v[~np.isnan(bbw_o_v)]
bbw_m_v = bbw_m[-lookback:]; bbw_m_v = bbw_m_v[~np.isnan(bbw_m_v)]
bbw_pct_o = _percentile_rank(bbw_o_v.tolist(), bbw_o[-1])
bbw_pct_m = 100*np.mean(bbw_m_v <= bbw_m[-1])
print("BBW: cur=%.10f | orig_pct=%.4f (valid=%d) mine_pct=%.4f (valid=%d)" % (
    bbw_o[-1], bbw_pct_o, len(bbw_o_v), bbw_pct_m, len(bbw_m_v)))
print("  sma20 diff max=%.2e std20 diff max=%.2e" % (np.nanmax(np.abs(sma20_o-sma20_m)), np.nanmax(np.abs(std20_o-std20_m))))

# ---- ATR ----
tr_o = np.full(n, np.nan)
for i in range(1, n):
    tr_o[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
atr_o = _ma(tr_o, 14)
tr_m = np.zeros(n); tr_m[0] = h[0]-l[0]
for i in range(1, n):
    tr_m[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
atr_m = ma_arr(tr_m, 14)
ar_o = atr_o/c; ar_m = atr_m/c
ar_o_v = ar_o[-lookback:]; ar_o_v = ar_o_v[~np.isnan(ar_o_v)]
ar_m_v = ar_m[-lookback:]; ar_m_v = ar_m_v[~np.isnan(ar_m_v)]
atr_pct_o = _percentile_rank(ar_o_v.tolist(), ar_o[-1])
atr_pct_m = 100*np.mean(ar_m_v <= ar_m[-1])
print("ATR: cur=%.10f | orig_pct=%.4f (valid=%d) mine_pct=%.4f (valid=%d)" % (
    ar_o[-1], atr_pct_o, len(ar_o_v), atr_pct_m, len(ar_m_v)))
print("  atr[-1] diff=%.2e | atr序列 NaN位置: orig=%s mine=%s" % (
    abs(atr_o[-1]-atr_m[-1]), np.where(np.isnan(atr_o))[0][:5], np.where(np.isnan(atr_m))[0][:5]))
# 边界附近的值
cur_o = ar_o[-1]
near_o = np.sort(ar_o_v)
print("  orig窗口内 cur 的排名位置: %d/%d" % ((ar_o_v <= cur_o).sum(), len(ar_o_v)))
near_vals = ar_o_v[np.argsort(np.abs(ar_o_v-cur_o))[:5]]
print("  orig窗口内最接近cur的5个值:", ["%.10f" % x for x in near_vals])
print("  cur=%.10f" % cur_o)
