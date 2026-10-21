# -*- coding: utf-8 -*-
"""
bp2（均线回踩买点）特征预处理 — 复刻 shunshi_buy_points._bp2_ma20_pullback + 门控链
定义（_bp2_ma20_pullback，Kimi V1.1 P0+P1+P2 全量化）:
  ① 趋势向上: MA20>MA50 且 MA20>REF(MA20,5)（MA50 无效即否决）
  ② 中期均线: n>=60 时要求 MA50>REF(MA50,10)（n<60 跳过）
  ③ 触及均线: 近5日(t-4..t)存在 L<=MA20*1.02 且 C>=MA20*0.98（从旧到新取首个命中日 touch_idx）
  ④ 缩量回调: mean(V[t-5..t-1]) / MA20(V)[t] <= 0.8（比值>0.8 否决）
  ⑤ 回撤幅度: (HHV20-L[touch])/(HHV40-LLV40) < 0.5（HHV20/HHV40/LLV40 均不含今日；n=40 边界含今日）
  ⑥ 止跌K线(touch当日): |C-O|/O<0.03 且 (min(C,O)-L)/(H-L)>0.3（H==L 跳过）
门控链（detect_shunshi_buy_point，与 backtest_kimibp_daily.py 一致）:
  L0: n_raw_score >= 40（复用 bp1_features 的 n_raw/l0_pass）
  L1: n_raw>=40 自动通过（只降置信度不否决）
  L2/L4: 只影响置信度，不否决（a_score=d_score=0）
  L3 互斥: bp1 优先 —— bp2 = l0_pass & ~bp1_tech & bp2_tech
口径: T日收盘后可知 → T+1开盘入场 → T+20收盘出场(t1_20) — 与前五章一致
复用 bp1_features.pkl 的 o/h/l/c/v、n_raw、bp1_tech 与路径指标。
末尾附校准（截断模式 vs 原引擎 detect_shunshi_buy_point）。
"""
import pickle, os, sys, json
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = r"C:\Users\Lenovo\WorkBuddy\2026-06-07-15-29-35\screener"
POOL = r"C:\Users\Lenovo\WorkBuddy\2026-06-07-15-29-35\screener\stock_pool.json"
D0, D1 = "20231101", "20260821"

src = pickle.load(open(os.path.join(HERE, "bp1_features.pkl"), "rb"))
feats_in = src["feats"]
pool_codes = set(s["code"] for s in json.load(open(POOL, encoding="utf-8"))["stocks"])
print("宇宙 %d 只, 池 %d 只" % (len(feats_in), len(pool_codes)))


def ma_arr(arr, period):
    n = len(arr); r = np.full(n, np.nan)
    if n < period: return r
    r[period-1:] = sliding_window_view(arr, period).mean(axis=1)
    return r


def roll_max_past(arr, w):
    """max(arr[t-w:t]) 不含今日，t>=w；t<w 处 NaN"""
    n = len(arr); r = np.full(n, np.nan)
    if n > w:
        r[w:] = sliding_window_view(arr, w).max(axis=1)[:-1]  # 窗口 [t-w, t-1]
    return r


def roll_min_past(arr, w):
    n = len(arr); r = np.full(n, np.nan)
    if n > w:
        r[w:] = sliding_window_view(arr, w).min(axis=1)[:-1]
    return r


def bp2_eval(o, h, l, c, v):
    """逐日评估六条件 → (bp2_tech, touch_idx, vol_shrink, pullback, body, shadow, dist_ma20)"""
    n = len(c)
    ma20 = ma_arr(c, 20); ma50 = ma_arr(c, 50); vol_ma20 = ma_arr(v, 20)
    hhv20 = roll_max_past(h, 20)      # t>=20: max(h[t-20:t])
    hhv40 = roll_max_past(h, 40)      # t>=40
    llv40 = roll_min_past(l, 40)
    vol5 = np.full(n, np.nan)
    if n > 5:
        vol5[5:] = sliding_window_view(v, 5).mean(axis=1)[:-1]  # mean(v[t-5:t])

    tech = np.zeros(n, dtype=bool)
    touch_idx = np.full(n, -1, dtype=np.int32)
    vol_shrink = np.full(n, np.nan); pullback = np.full(n, np.nan)
    body_pct = np.full(n, np.nan); shadow_pct = np.full(n, np.nan)

    for t in range(n):
        # ① 趋势: MA20>MA50 且 MA20>REF(MA20,5)
        if np.isnan(ma20[t]) or ma20[t] <= 0:
            continue
        if np.isnan(ma50[t]) or ma20[t] <= ma50[t]:
            continue
        if t < 5 or np.isnan(ma20[t-5]) or ma20[t] <= ma20[t-5]:
            continue
        # ② 中期: n_eff>=60 时 MA50>REF(MA50,10)
        n_eff = t + 1
        if n_eff >= 60 and (not np.isnan(ma50[t])) and (not np.isnan(ma50[t-10])):
            if ma50[t] <= ma50[t-10]:
                continue
        # ③ 触及: 近5日 t-4..t，从旧到新首个命中
        ti = -1
        for j in range(max(t-4, 0), t+1):
            if not np.isnan(ma20[j]) and l[j] <= ma20[j]*1.02 and c[j] >= ma20[j]*0.98:
                ti = j
                break
        if ti < 0:
            continue
        # ④ 缩量: mean(V[t-5:t])/vol_ma20[t] <= 0.8
        shr = 1.0
        if not np.isnan(vol_ma20[t]) and vol_ma20[t] > 0:
            if t < 5:
                continue
            shr = float(vol5[t] / vol_ma20[t])
            if shr > 0.8:
                continue
        # ⑤ 回撤: (HHV20-L[ti])/(HHV40-LLV40) < 0.5
        pb = np.nan
        if n_eff >= 40:
            if n_eff >= 41:
                h40 = hhv40[t]; l40 = llv40[t]
            else:  # n==40 边界: 含今日全窗（复刻源码 fallback）
                h40 = float(np.max(h[:t+1])); l40 = float(np.min(l[:t+1]))
            h20 = hhv20[t] if not np.isnan(hhv20[t]) else float(np.max(h[:t+1]))
            if h40 > l40:
                pb = float((h20 - l[ti]) / (h40 - l40))
                if pb >= 0.5:
                    continue
        # ⑥ 止跌K线(touch当日)
        if o[ti] > 0:
            rng = h[ti] - l[ti]
            if rng > 0:
                b = abs(c[ti] - o[ti]) / o[ti]
                sh = (min(c[ti], o[ti]) - l[ti]) / rng
                if not (b < 0.03 and sh > 0.3):
                    continue
                body_pct[t] = b; shadow_pct[t] = sh
        tech[t] = True
        touch_idx[t] = ti
        vol_shrink[t] = shr; pullback[t] = pb

    return tech, touch_idx, vol_shrink, pullback, body_pct, shadow_pct, ma20


feats_out = {}
n_sig = 0
for code, ft in feats_in.items():
    o = ft["o"].astype(np.float64); h = ft["h"].astype(np.float64)
    l = ft["l"].astype(np.float64); c = ft["c"].astype(np.float64)
    v = ft["v"].astype(np.float64)
    n = len(c)
    if n < 50:
        continue
    tech, tidx, shr, pb, body, shadow, ma20 = bp2_eval(o, h, l, c, v)
    l0 = ft["l0_pass"].astype(bool)
    bp1_fire = ft["bp1_tech"].astype(bool) & l0      # L3 互斥: bp1 优先
    sig = l0 & (~ft["bp1_tech"].astype(bool)) & tech # 生产口径: 门控过 + bp1 未占 + bp2 六条件

    with np.errstate(invalid="ignore", divide="ignore"):
        dist_ma20 = np.where(~np.isnan(ma20) & (ma20 != 0), (c - ma20) / ma20 * 100, np.nan)

    feats_out[code] = {
        "date": np.array([str(x).replace("-", "") for x in ft["date"]]),  # D29 归一
        "sig": sig, "bp2_tech": tech, "l0_pass": l0, "bp1_fire": bp1_fire,
        "touch_idx": tidx, "touch_fresh": (tidx == np.arange(n)).astype(np.int8),
        "vol_shrink": shr.astype(np.float32), "pullback": pb.astype(np.float32),
        "body": body.astype(np.float32), "shadow": shadow.astype(np.float32),
        "dist_ma20": dist_ma20.astype(np.float32),
        "n_raw": ft["n_raw"], "ma20": ma20.astype(np.float32),
        "t1_20": ft["t1_20"], "fwd20": ft["fwd20"],
        "mae20": ft["mae20"], "mfe20": ft["mfe20"],
        "pre20": ft["pre20"], "pre60": ft["pre60"],
        "dd20": ft["dd20"], "vol20": ft["vol20"],
        "below_days": ft["below_days"],
        "first5_min": ft["first5_min"], "last15_max": ft["last15_max"],
    }
    n_sig += int(sig.sum())

print("bp2信号(l0 & ~bp1 & tech) %d 条 (纯tech %d)" % (
    n_sig, sum(int(x["bp2_tech"].sum()) for x in feats_out.values())))

# ================= 校准（截断模式 vs 原引擎 detect_shunshi_buy_point） =================
sys.path.insert(0, ENGINE_DIR)
import shunshi_buy_points as sbp  # noqa

rng = np.random.default_rng(20261020)
codes = sorted(feats_in.keys())
sample_codes = [codes[i] for i in rng.choice(len(codes), 30, replace=False)]
alld = sorted({str(x).replace("-", "") for ft in feats_in.values() for x in ft["date"]})
months = {}
for ds in alld:
    months[ds[:6]] = ds
sample_dates = sorted(months.values())[-20:]

def to_klines(ft, upto):
    ds = ft["date"]; ks = []
    for i in range(len(ds)):
        if str(ds[i]).replace("-", "") > upto:
            break
        ks.append({"date": str(ds[i]), "open": float(ft["o"][i]), "high": float(ft["h"][i]),
                   "low": float(ft["l"][i]), "close": float(ft["c"][i]), "volume": float(ft["v"][i])})
    return ks

n_cmp = 0; n_ok = 0; mismatch = []
n_cmp_bp1 = 0; n_ok_bp1 = 0
for code in sample_codes:
    ft = feats_in[code]
    dates_n = np.array([str(x).replace("-", "") for x in ft["date"]])
    my_sig = feats_out[code]["sig"]; my_bp1 = feats_out[code]["bp1_fire"]
    n_raw = ft["n_raw"]
    for ds in sample_dates:
        idxs = np.where(dates_n == ds)[0]
        if len(idxs) == 0:
            continue
        i = int(idxs[-1])
        ks = to_klines(ft, ds)
        if len(ks) < 50:
            continue
        ns = int(n_raw[i])
        r = sbp.detect_shunshi_buy_point(code, ks, {"n_raw_score": ns}, 0, 0)
        eng_bp2 = bool(r.get("signal") and r.get("bp_type") == "买点二")
        eng_bp1 = bool(r.get("signal") and r.get("bp_type") == "买点一")
        my_bp2 = bool(my_sig[i]); my_b1 = bool(my_bp1[i])
        n_cmp += 1
        if eng_bp2 == my_bp2:
            n_ok += 1
        else:
            mismatch.append({"code": code, "date": ds, "eng_bp2": eng_bp2, "my_bp2": my_bp2,
                             "l0": ns, "eng_type": r.get("bp_type")})
        n_cmp_bp1 += 1
        if eng_bp1 == my_b1:
            n_ok_bp1 += 1

print("校准(bp2): %d/%d = %.2f%% | (bp1互斥对照): %d/%d = %.2f%%" % (
    n_ok, n_cmp, 100.0*n_ok/max(n_cmp, 1), n_ok_bp1, n_cmp_bp1, 100.0*n_ok_bp1/max(n_cmp_bp1, 1)))
if mismatch[:5]:
    print("前5条不一致:", mismatch[:5])

pickle.dump({"feats": feats_out, "pool_codes": pool_codes},
            open(os.path.join(HERE, "bp2_features.pkl"), "wb"))
json.dump({"n_cmp": n_cmp, "n_ok": n_ok, "mismatch": mismatch[:20],
           "n_cmp_bp1": n_cmp_bp1, "n_ok_bp1": n_ok_bp1},
          open(os.path.join(HERE, "bp2_verify.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("saved -> bp2_features.pkl / bp2_verify.json")
