# -*- coding: utf-8 -*-
"""
补票（单针下20）特征预处理 — 复刻 zge_compute.calc_zge_indicators + check_bupiao
定义（check_bupiao）:
  单针下20_短 ds = 100*(C-LLV(L,3))/(HHV(C,3)-LLV(L,3))   # 注意 HHV 用收盘价!
  单针下20_长 dl = 100*(C-LLV(L,21))/(HHV(C,21)-LLV(L,21))
  标准型: dl>=85 & ds<30 → +6分
  激进型: dl>=80 & ds<20 → +5分
  趋势确认: C>MA60 → +1分
  近20日涨>5% → +1分; 跌>5% 仅警告不加分
  生产信号(回测口径): score>=6
口径: T日收盘后可知 → T+1开盘入场 → T+20收盘出场(t1_20) — 与B1/B2/bp1一致
复用 bp1_features.pkl 的 o/h/l/c/v 与路径指标(t1_20/mae20/...)。
"""
import pickle, os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = r"C:\Users\Lenovo\WorkBuddy\2026-06-07-15-29-35\screener\stock_pool.json"
D0, D1 = "20231101", "20260821"

src = pickle.load(open(os.path.join(HERE, "bp1_features.pkl"), "rb"))
feats_in = src["feats"]
pool_codes = set(s["code"] for s in json.load(open(POOL, encoding="utf-8"))["stocks"])
print("宇宙 %d 只, 池 %d 只" % (len(feats_in), len(pool_codes)))

feats_out = {}
n_sig = 0
for code, ft in feats_in.items():
    o = ft["o"].astype(np.float64); h = ft["h"].astype(np.float64)
    l = ft["l"].astype(np.float64); c = ft["c"].astype(np.float64)
    n = len(c)
    if n < 3:
        continue

    # ---- 单针下20_短: HHV(C,3)/LLV(L,3) — 与 zge_compute 逐位一致 ----
    llv3 = np.full(n, np.nan); hhv3 = np.full(n, np.nan)
    for i in range(2, n):
        llv3[i] = np.min(l[i-2:i+1])
        hhv3[i] = np.max(c[i-2:i+1])          # HHV 用收盘价
    d3 = hhv3 - llv3
    ds = np.where(d3 > 0, 100.0 * (c - llv3) / d3, 50.0)

    # ---- 单针下20_长: HHV(C,21)/LLV(L,21) ----
    llv21 = np.full(n, np.nan); hhv21 = np.full(n, np.nan)
    for i in range(20, n):
        llv21[i] = np.min(l[i-20:i+1])
        hhv21[i] = np.max(c[i-20:i+1])
    d21 = hhv21 - llv21
    dl = np.where(d21 > 0, 100.0 * (c - llv21) / d21, 50.0)

    # ---- MA60 ----
    ma60 = np.full(n, np.nan)
    if n >= 60:
        from numpy.lib.stride_tricks import sliding_window_view
        ma60[59:] = sliding_window_view(c, 60).mean(axis=1)

    # ---- chg_20d = (C/C[-20]-1)*100 — 复刻 closes[-1]/closes[-20]（19期，i>=20 才有值）----
    chg20d = np.zeros(n)
    if n > 20:
        chg20d[20:] = (c[20:] / c[1:n-19] - 1.0) * 100.0

    # ---- 评分（与 check_bupiao 逐条对应）----
    score = np.zeros(n, dtype=np.int32)
    # 标准型/激进型互斥
    std_type = (dl >= 85) & (ds < 30)
    agg_type = (dl >= 80) & (ds < 20) & (~std_type)
    score += np.where(std_type, 6, 0)
    score += np.where(agg_type, 5, 0)
    # 趋势确认 C>MA60（ma60 NaN 时 = False，与 if ma60_v 一致）
    score += np.where((~np.isnan(ma60)) & (c > ma60), 1, 0)
    # 近20日涨>5%
    score += np.where(chg20d > 5, 1, 0)

    sig6 = score >= 6      # 生产口径（backtest_bupiao_banzhuan.py）
    sig5 = score >= 5      # 弱信号口径（check_bupiao status）

    # 复用路径指标
    feats_out[code] = {
        "date": ft["date"],
        "ds": ds.astype(np.float32), "dl": dl.astype(np.float32),
        "ma60": ma60.astype(np.float32), "chg20d": chg20d.astype(np.float32),
        "score": score, "sig": sig6, "sig5": sig5,
        "bupiao_type": np.where(std_type, 1, np.where(agg_type, 2, 0)).astype(np.int8),
        "t1_20": ft["t1_20"], "fwd20": ft["fwd20"],
        "mae20": ft["mae20"], "mfe20": ft["mfe20"],
        "pre20": ft["pre20"], "pre60": ft["pre60"],
        "dd20": ft["dd20"], "vol20": ft["vol20"],
        "below_days": ft["below_days"],
        "first5_min": ft["first5_min"], "last15_max": ft["last15_max"],
    }
    n_sig += int(sig6.sum())

print("补票信号(score>=6) %d 条" % n_sig)
pickle.dump({"feats": feats_out, "pool_codes": pool_codes},
            open(os.path.join(HERE, "bupiao_features.pkl"), "wb"))
print("saved -> bupiao_features.pkl")
