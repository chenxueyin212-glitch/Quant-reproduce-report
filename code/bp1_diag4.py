# -*- coding: utf-8 -*-
"""决定性测试: 用 prep 的精确代码块在完整序列上复算 603353 d2[447]"""
import pickle, os, sqlite3
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:/Users/Lenovo/WorkBuddy/2026-07-17-00-40-38"
DB = os.path.join(SRC, "kline_cache_all.db")
con = sqlite3.connect("file:%s?mode=ro" % DB.replace("\\", "/"), uri=True)
cur = con.cursor()

# ---- 从 bp1_prep.py 原样复制的工具与 d2 块 ----
def ma_arr(arr, period):
    n = len(arr); r = np.full(n, np.nan)
    if n < period: return r
    cs = np.concatenate([[0.0], np.cumsum(arr)])
    r[period-1:] = (cs[period:] - cs[:-period]) / period
    return r

def rolling_any(a, w):
    n = len(a); r = np.zeros(n, bool)
    if n < w: return r
    cs = np.concatenate([[0], np.cumsum(a.astype(np.int32))])
    r[w-1:] = (cs[w:] - cs[:-w]) > 0
    return r

def win_corr_upto(x, y, w):
    n = len(x); r = np.full(n, np.nan)
    if n < w: return r
    svx = np.lib.stride_tricks.sliding_window_view(x, w)
    svy = np.lib.stride_tricks.sliding_window_view(y, w)
    mx = svx.mean(axis=1); my = svy.mean(axis=1)
    cx = svx-mx[:, None]; cy = svy-my[:, None]
    cov = (cx*cy).sum(axis=1)
    den = np.sqrt((cx**2).sum(axis=1)*(cy**2).sum(axis=1))
    ok = den > 0
    r[w-1:] = np.where(ok, cov/np.where(ok, den, 1), np.nan)
    return r

CODE = "603353"
rows = cur.execute("SELECT date,open,high,low,close,volume FROM klines WHERE code=? ORDER BY date", (CODE,)).fetchall()
print("DB rows for %s: %d" % (CODE, len(rows)))
dates = np.array([r[0] for r in rows])
o = np.array([float(r[1]) for r in rows]); h = np.array([float(r[2]) for r in rows])
l = np.array([float(r[3]) for r in rows]); c = np.array([float(r[4]) for r in rows])
v = np.array([float(r[5]) for r in rows]); n = len(c)

vma20 = ma_arr(v, 20)
vr5_today = v/np.where(ma_arr(v, 5) > 0, ma_arr(v, 5), np.nan)

# ---- dim2 精确复制 ----
d2 = np.zeros(n)
shrink = (v < vma20*0.6) & (~np.isnan(vma20))
has_shrink = rolling_any(shrink, 20)
d2 += np.where(has_shrink & (vr5_today > 1.5), 10, np.where(has_shrink & (vr5_today > 1.2), 7, np.where(has_shrink, 4, 0)))
dv = np.diff(v, prepend=np.nan); dc = np.diff(c, prepend=np.nan)
vp_corr = win_corr_upto(dv, dc, 5)
d2 += np.where(vp_corr > 0.5, 5, np.where(vp_corr > 0, 3, 0))
bullish = c > o
lvr = v/np.where(vma20 > 0, vma20, np.nan)
is_b5 = bullish & (lvr > 2.0) & (~np.isnan(lvr))
is_b15 = bullish & (lvr > 1.5) & (~np.isnan(lvr))
bottom_score = np.zeros(n)
if n >= 60:
    am = np.lib.stride_tricks.sliding_window_view(l, 60).argmin(axis=1)
    gidx = np.arange(59, n)-59+am
    found = np.zeros(n-59, bool); score = np.zeros(n-59)
    for off in (-1, 0, 1, 2, 3):
        pos = np.clip(gidx+off, 0, n-1)
        hit5 = is_b5[pos] & (~found)
        score[hit5] = 5; found |= hit5
        hit15 = is_b15[pos] & (~found)
        score[hit15] = 3; found |= hit15
    bottom_score[59:] = score
d2 += bottom_score
d2 = np.minimum(d2, 20)

si = 447
print("full-series recompute: date[447]=%s d2=%s A=%s B=%s C=%s" % (
    dates[si], d2[si],
    np.where(has_shrink[si] & (vr5_today[si] > 1.5), 10, np.where(has_shrink[si] & (vr5_today[si] > 1.2), 7, np.where(has_shrink[si], 4, 0))),
    np.where(vp_corr[si] > 0.5, 5, np.where(vp_corr[si] > 0, 3, 0)),
    bottom_score[si]))
print("has_shrink[447]=%s vr5_today[447]=%.3f vp_corr[447]=%.3f" % (has_shrink[si], vr5_today[si], vp_corr[si]))

# 与 pkl 对比
d = pickle.load(open(os.path.join(HERE, "bp1_features_poolref84.pkl"), "rb"))
ft = d["feats"][CODE]
print("pkl: len(date)=%d d2[447]=%s" % (len(ft["date"]), ft["d2"][si]))
# 日期序列是否一致
mism = [i for i in range(min(len(dates), len(ft["date"]))) if str(dates[i]) != str(ft["date"][i])]
print("date 序列不一致行数:", len(mism), mism[:10])
# d2 全序列对比
diff = np.where(d2 != np.asarray(ft["d2"], dtype=np.float64))[0]
print("d2 全序列不一致行数: %d / %d" % (len(diff), n))
for i in diff[:20]:
    print("   row %d date=%s recompute=%s pkl=%s" % (i, dates[i], d2[i], ft["d2"][i]))
