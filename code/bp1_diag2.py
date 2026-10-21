# -*- coding: utf-8 -*-
"""调试 603353 20240603 d2 差异: 原引擎 vs 向量化, 逐部分(A/B/C)对比"""
import pickle, os, sys, sqlite3
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _niigu_judge_ref import compute_niugu_score, _ma, _safe_get

SRC = r"C:/Users/Lenovo/WorkBuddy/2026-07-17-00-40-38"
DB = os.path.join(SRC, "kline_cache_all.db")
con = sqlite3.connect("file:%s?mode=ro" % DB.replace("\\", "/"), uri=True)
cur = con.cursor()
d = pickle.load(open(os.path.join(HERE, "bp1_features_poolref84.pkl"), "rb"))
feats = d["feats"]

CODE, SD = "603353", "20240603"
rows = cur.execute("SELECT date,open,high,low,close,volume FROM klines WHERE code=? AND date<=? ORDER BY date", (CODE, SD)).fetchall()
kls = [(r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])) for r in rows]
n = len(kls)
c = np.array([k[4] for k in kls]); o = np.array([k[1] for k in kls])
h = np.array([k[2] for k in kls]); l = np.array([k[3] for k in kls])
v = np.array([k[5] for k in kls])
print("n =", n, "last date =", kls[-1][0])

# ---------- 向量化复算(与bp1_prep相同公式, 在同一数据上) ----------
def ma_arr(a, p):
    r = np.full(len(a), np.nan)
    if len(a) >= p: r[p-1:] = np.lib.stride_tricks.sliding_window_view(a, p).mean(axis=1)
    return r
vma20 = ma_arr(v, 20); vma5 = ma_arr(v, 5)
vr5 = v[-1]/vma5[-1]
shrink = (v < vma20*0.6) & (~np.isnan(vma20))
has_sh = bool(shrink[-20:].any())
A2 = 10 if (has_sh and vr5 > 1.5) else (7 if (has_sh and vr5 > 1.2) else (4 if has_sh else 0))
dv = np.diff(v, prepend=np.nan); dc = np.diff(c, prepend=np.nan)
svx = np.lib.stride_tricks.sliding_window_view(dv, 5); svy = np.lib.stride_tricks.sliding_window_view(dc, 5)
mx = svx.mean(axis=1); my = svy.mean(axis=1)
cx = svx-mx[:, None]; cy = svy-my[:, None]
cov = (cx*cy).sum(axis=1); den = np.sqrt((cx**2).sum(axis=1)*(cy**2).sum(axis=1))
corr2 = cov[-1]/den[-1] if den[-1] > 0 else np.nan
B2 = 5 if (corr2 > 0.5) else (3 if (corr2 > 0) else 0)
bullish = c > o
lvr = v/np.where(vma20 > 0, vma20, np.nan)
is_b5 = bullish & (lvr > 2.0) & (~np.isnan(lvr))
is_b15 = bullish & (lvr > 1.5) & (~np.isnan(lvr))
am = np.lib.stride_tricks.sliding_window_view(l, 60).argmin(axis=1)
gidx_v = (np.arange(59, n)-59)+am
gv = int(gidx_v[-1])
print("向量化复算: gidx_v=%d min_low=%.2f date=%s (n=%d)" % (gv, l[gv], kls[gv][0], n))
C2 = 0
found = False
for off in (-1, 0, 1, 2, 3):
    pos = min(max(gv+off, 0), n-1)
    print("   off=%d pos=%d date=%s lvr=%.3f bull=%s b5=%s b15=%s found=%s" % (
        off, pos, kls[pos][0], lvr[pos], bool(bullish[pos]), bool(is_b5[pos]), bool(is_b15[pos]), found))
    if not found and is_b5[pos]:
        C2 = 5; found = True
    elif not found and is_b15[pos]:
        C2 = 3; found = True
print("向量化复算: A=%d B=%d(corr=%.3f) C=%d -> d2=%d" % (A2, B2, corr2, C2, A2+B2+C2))

# ---------- 原引擎 d2 手工复算 ----------
vol_ma20 = _ma(v, 20)
last_vol = v[-1]; last_vol_ma5 = _ma(v, 5)[-1]
vol_ratio_5 = last_vol/last_vol_ma5
has_shrink = False; shrink_days = 0
for i in range(-20, 0):
    v_ma = _safe_get(vol_ma20, i)
    if v_ma and v_ma > 0 and v[i] < v_ma*0.6:
        has_shrink = True; shrink_days += 1
if has_shrink and vol_ratio_5 > 1.5: A = 10
elif has_shrink and vol_ratio_5 > 1.2: A = 7
elif has_shrink: A = 4
else: A = 0
# B
recent_vol = v[-6:]; recent_close = c[-6:]
vol_chg = np.diff(recent_vol); close_chg = np.diff(recent_close)
corr = np.corrcoef(vol_chg, close_chg)[0, 1] if (np.std(vol_chg) > 0 and np.std(close_chg) > 0) else np.nan
B = 5 if (not np.isnan(corr) and corr > 0.5) else (3 if (not np.isnan(corr) and corr > 0) else 0)
# C
lookback = min(60, n)
recent_lows = l[-lookback:] if n >= 60 else l
min_low_idx = np.argmin(recent_lows)
gidx = n - lookback + min_low_idx
C = 0; hit_detail = None
for offset in range(-1, 4):
    ci = gidx + offset
    if 0 <= ci < n:
        lvm = _safe_get(vol_ma20, ci)
        lvr = v[ci]/lvm if (lvm and lvm > 0) else 1.0
        is_bull = c[ci] > o[ci]
        print("  offset=%d ci=%d date=%s lvr=%.3f bull=%s" % (offset, ci, kls[ci][0], lvr, is_bull))
        if is_bull and lvr > 2.0:
            C = 5; hit_detail = (offset, ci, lvr); break
        elif is_bull and lvr > 1.5:
            C = 3; hit_detail = (offset, ci, lvr); break
print("原引擎复算: A=%d B=%d(corr=%.3f) C=%d -> d2=%d  (gidx=%d min_low=%.2f date=%s, shrink_days=%d vr5=%.2f)" % (
    A, B, corr if not np.isnan(corr) else -999, C, A+B+C, gidx, l[gidx], kls[gidx][0], shrink_days, vol_ratio_5))

# ---------- 我的向量化特征 ----------
ft = feats[CODE]
si = [i for i, x in enumerate([str(vv) for vv in ft["date"]]) if x == SD][0]
print("向量化:   d1=%s d2=%s d3=%s d4=%s d6=%s n_raw=%s" % (ft["d1"][si], ft["d2"][si], ft["d3"][si], ft["d4"][si], ft["d6"][si], ft["n_raw"][si]))

# 原引擎 compute_niugu_score 完整结果
nr = compute_niugu_score([{"date": k[0], "open": k[1], "high": k[2], "low": k[3], "close": k[4], "volume": k[5]} for k in kls], all_closes_map={})
print("原引擎完整: d1=%s d2=%s d3=%s d4=%s d6=%s n_raw=%s" % (
    nr["dimensions"]["趋势结构"]["score"], nr["dimensions"]["量价关系"]["score"],
    nr["dimensions"]["相对强弱"]["score"], nr["dimensions"]["波动率压缩"]["score"],
    nr["dimensions"]["筹码结构"]["score"], nr["n_raw_score"]))
vp = nr["dimensions"]["量价关系"]["details"]
for kk, vv in vp.items():
    print("   d2 detail %s: %s" % (kk, vv))
