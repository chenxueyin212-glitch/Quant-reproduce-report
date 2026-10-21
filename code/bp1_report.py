# -*- coding: utf-8 -*-
"""bp1（Kimi买点一·平台突破）因子重构评估报告生成器 — 输出 standalone HTML"""
import json, os, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "bp1_signal_results.json"), encoding="utf-8"))
P = json.load(open(os.path.join(HERE, "bp1_port_results.json"), encoding="utf-8"))
N = json.load(open(os.path.join(HERE, "bp1_nav_results.json"), encoding="utf-8"))
EQ = json.load(open(os.path.join(HERE, "bp1_nav_equity.json"), encoding="utf-8"))
R = json.load(open(os.path.join(HERE, "bp1_robust_results.json"), encoding="utf-8"))
M = json.load(open(os.path.join(HERE, "bp1_micro_results.json"), encoding="utf-8"))

def f2(x): return ("%+.2f" % x) if x is not None else "—"
def f1(x): return ("%+.1f" % x) if x is not None else "—"
def cls(x): return "up" if (x or 0) > 0 else "down"

# ---------------- SVG helpers ----------------
def line_chart(series, width=940, height=300, pad=(56, 18, 34, 60), ymin=None, ymax=None, vline=None, vlabel=None):
    pl, pr, pt, pb = pad
    iw, ih = width-pl-pr, height-pt-pb
    ys = [v for _, _, d in series for v in d.values()]
    lo = ymin if ymin is not None else min(ys)*0.95
    hi = ymax if ymax is not None else max(ys)*1.03
    xs_all = sorted({k for _, _, d in series for k in d})
    def X(i): return pl + iw*i/(len(xs_all)-1)
    def Y(v): return pt + ih*(hi-v)/(hi-lo)
    out = ['<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" style="width:100%%;height:auto">' % (width, height)]
    for g in range(5):
        v = lo+(hi-lo)*g/4; y = Y(v)
        out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#e5e7eb" stroke-width="1"/>' % (pl, y, pl+iw, y))
        out.append('<text x="%d" y="%.1f" font-size="11" fill="#6b7280" text-anchor="end">%.1f</text>' % (pl-6, y+4, v))
    for i in range(0, len(xs_all), max(1, len(xs_all)//12)):
        out.append('<text x="%.1f" y="%d" font-size="10.5" fill="#6b7280" text-anchor="middle">%s</text>' % (X(i), height-10, xs_all[i]))
    if vline and vline in xs_all:
        i = xs_all.index(vline)
        out.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#d97706" stroke-width="1.4" stroke-dasharray="5,4"/>' % (X(i), pt, X(i), pt+ih))
        out.append('<text x="%.1f" y="%d" font-size="11" fill="#d97706" text-anchor="middle">%s</text>' % (X(i), pt-6, vlabel or ""))
    for name, color, d in series:
        pts = " ".join("%.1f,%.1f" % (X(xs_all.index(k)), Y(v)) for k, v in sorted(d.items()) if k in xs_all)
        out.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2.2"/>' % (pts, color))
        lastk = sorted(d.keys())[-1]
        out.append('<circle cx="%.1f" cy="%.1f" r="3" fill="%s"/>' % (X(xs_all.index(lastk)), Y(d[lastk]), color))
        out.append('<text x="%.1f" y="%.1f" font-size="11.5" fill="%s" font-weight="600">%s</text>' % (X(xs_all.index(lastk))+5, Y(d[lastk])+4, color, name))
    out.append("</svg>")
    return "".join(out)

def bar_chart(groups, width=940, height=None, pad=(56, 14, 30, 60), fmt="%+.2f"):
    ns = max(len(g[1]) for g in groups)
    bh = 22; gap = 7; gh = ns*bh+(ns-1)*2+gap
    height = height or (pad[2]+len(groups)*gh+pad[3])
    pl, pr, pt, pb = pad
    iw, ih = width-pl-pr, height-pt-pb
    vals = [v for _, ss in groups for _, _, v in ss]
    lo, hi = min(vals+[0]), max(vals+[0])
    rng = (hi-lo) or 1
    lo -= rng*0.12; hi += rng*0.18
    def Y(v): return pt+ih*(hi-v)/(hi-lo)
    out = ['<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" style="width:100%%;height:auto">' % (width, height)]
    for g in range(5):
        v = lo+(hi-lo)*g/4; y = Y(v)
        out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#e5e7eb"/>' % (pl, y, pl+iw, y))
        out.append('<text x="%d" y="%.1f" font-size="11" fill="#6b7280" text-anchor="end">%s</text>' % (pl-6, y+4, fmt % v))
    y0 = Y(0)
    out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#9ca3af" stroke-width="1.2"/>' % (pl, y0, pl+iw, y0))
    bw = 96
    for gi, (label, ss) in enumerate(groups):
        gx = pl+iw*(gi+0.5)/len(groups)
        total_w = len(ss)*bw+(len(ss)-1)*2
        for si, (sn, color, v) in enumerate(ss):
            bx = gx-total_w/2+si*(bw+2)
            yv = Y(v)
            out.append('<rect x="%.1f" y="%.1f" width="%d" height="%.1f" fill="%s" rx="2"/>' % (bx, min(yv, y0), bw, abs(y0-yv), color))
            ty = yv-5 if v >= 0 else yv+13
            out.append('<text x="%.1f" y="%.1f" font-size="10.5" fill="%s" text-anchor="middle" font-weight="600">%s</text>' % (bx+bw/2, ty, color, fmt % v))
        out.append('<text x="%.1f" y="%d" font-size="11.5" fill="#374151" text-anchor="middle">%s</text>' % (gx, height-8, label))
    lx = pl; seen = []
    for _, ss in groups:
        for sn, color, _ in ss:
            if (sn, color) not in seen: seen.append((sn, color))
    for sn, color in seen:
        out.append('<rect x="%d" y="6" width="10" height="10" fill="%s" rx="2"/>' % (lx, color))
        out.append('<text x="%d" y="15" font-size="11" fill="#374151">%s</text>' % (lx+14, sn))
        lx += 20+len(sn)*12
    out.append("</svg>")
    return "".join(out)

def hist_chart(edges, counts, width=940, height=260):
    pl, pr, pt, pb = 56, 14, 26, 40
    iw, ih = width-pl-pr, height-pt-pb
    hi = max(counts)*1.1
    bw = iw/len(counts)
    out = ['<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" style="width:100%%;height:auto">' % (width, height)]
    for g in range(5):
        v = hi*g/4; y = pt+ih*(1-v/hi)
        out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#e5e7eb"/>' % (pl, y, pl+iw, y))
        out.append('<text x="%d" y="%.1f" font-size="11" fill="#6b7280" text-anchor="end">%d</text>' % (pl-6, y+4, v))
    for i, cnt in enumerate(counts):
        x = pl+i*bw; h = ih*cnt/hi
        col = "#27ae60" if edges[i] >= 0 else "#e74c3c"
        op = 0.92 if edges[i] >= 0 else 0.75
        out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s" opacity="%.2f"/>' % (x+0.5, pt+ih-h, bw-1, h, col, op))
        if i % 3 == 0:
            out.append('<text x="%.1f" y="%d" font-size="10" fill="#6b7280" text-anchor="middle">%d</text>' % (x+bw/2, height-22, edges[i]))
    out.append('<text x="%d" y="%d" font-size="11" fill="#6b7280">收益(4%%桶) →</text>' % (pl, height-6))
    out.append("</svg>")
    return "".join(out)

# ---------------- 直方图数据(从flat CSV) ----------------
t1v = []
with open(os.path.join(HERE, "bp1_signals_flat.csv"), encoding="utf-8") as f:
    for r_ in csv.DictReader(f):
        t1v.append(float(r_["t1_20"]))
t1v = np.array(t1v)
edges = list(range(-56, 280, 4))
counts, _ = np.histogram(t1v, bins=edges)

# ---------------- 数据装配 ----------------
g = S["groups"]; ic = S["ic"]; nv = S["naive"]; dc = S["deciles"]
months = sorted(EQ["bp1_全A"].keys())
te = "202509"

def row(label, d, bold=False):
    if not d: return ""
    st = ' style="font-weight:700"' if bold else ""
    t5 = f1(d["top5_share"]) + "%" if d.get("top5_share") is not None else "—"
    return ("<tr%s><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td><td>%s%%</td>"
            "<td>%s%%</td><td>%s%%</td><td>%s</td></tr>") % (
        st, label, "{:,}".format(d["n"]), cls(d["mean"]), f2(d["mean"]), cls(d["median"]), f2(d["median"]),
        f1(d["win"]), f1(d["p10"]), f1(d["p90"]), t5)

rows_groups = "".join([
    row("全体", g["全体"], True),
    row("训练期（<20250908）", g["训练期"]),
    row("验证期（≥20250908）", g["验证期"]),
    row("池内（选股池 848 只）", g["池内"]),
    row("池外", g["池外"]),
    row("池内-训练", g["池内-训练"]),
    row("池内-验证", g["池内-验证"]),
    row("池外-训练", g["池外-训练"]),
    row("池外-验证", g["池外-验证"]),
    row("n_raw 40-49", g["n_raw 40-49"]),
    row("n_raw 50-59", g["n_raw 50-59"]),
    row("n_raw >=60", g["n_raw >=60"]),
    row("L2通过", g["L2通过"]),
    row("L2未过（7条）", g["L2未过"]),
])

rows_decile = "".join(
    "<tr><td>%s</td><td>%s</td><td>%s%%</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td></tr>" % (
        k, k.split("(")[1].rstrip(")"), f2(v["share"]), cls(v["mean"]), f2(v["mean"]), cls(v["pnl_share"]), f2(v["pnl_share"]))
    for k, v in dc.items())

# 止损表
stop_rows = ""
for k in ["不止损", "止损8%", "止损12%", "止损15%", "止损20%"]:
    v = P["results"][k]; a = v["portfolio_all"]; per = v["portfolio_period"]; co = P["costed"][k]
    stop_rows += ("<tr><td>%s</td><td class='%s'>%s%%</td><td>%s%%</td><td>%s</td><td>%s%%</td>"
                  "<td class='%s'>%s%%</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td></tr>") % (
        k, cls(a["total"]), f1(a["total"]), f1(a["ann"]), f2(a["sharpe"]), f1(a["maxdd"]),
        cls(per["训练期"]["total"]), f1(per["训练期"]["total"]),
        cls(per["验证期"]["total"]), f1(per["验证期"]["total"]),
        cls(co["total"]), f1(co["total"]))

# 池内表
pool_rows = ""
for k in ["不止损", "止损8%", "止损12%", "止损15%", "止损20%"]:
    v = P["pool"][k]; a = v["all"]; per = v["period"]; co = v["costed"]
    pool_rows += ("<tr><td>%s</td><td class='%s'>%s%%</td><td>%s</td><td>%s%%</td>"
                  "<td class='%s'>%s%%</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td></tr>") % (
        k, cls(a["total"]), f1(a["total"]), f2(a["sharpe"]), f1(a["maxdd"]),
        cls(per["训练期"]["total"]), f1(per["训练期"]["total"]),
        cls(per["验证期"]["total"]), f1(per["验证期"]["total"]),
        cls(co["total"]), f1(co["total"]))
bp_ = P["bench_port"]["等权池内"]
pool_rows += ("<tr style='background:#fdecea'><td><strong>池内基准(等权持有)</strong></td>"
              "<td class='up'><strong>%s%%</strong></td><td><strong>%s</strong></td><td><strong>%s%%</strong></td>"
              "<td class='up'><strong>%s%%</strong></td><td class='up'><strong>%s%%</strong></td><td>—</td></tr>") % (
    f1(bp_["全部"]["total"]), f2(bp_["全部"]["sharpe"]), f1(bp_["全部"]["maxdd"]),
    f1(bp_["训练期"]["total"]), f1(bp_["验证期"]["total"]))

# NAV表
nav_rows = ""
for k, lbl in [("bp1_全A", "bp1信号（全A，日度等权再平衡）"), ("bp1_池内", "bp1信号（池内）"),
               ("bp1_池外", "bp1信号（池外）"),
               ("基准_等权全A", "等权全A（主基准）"), ("基准_等权池内", "等权池内（含后视偏差）"),
               ("基准_等权池外", "等权池外")]:
    v = N[k]["metrics"]
    nav_rows += ("<tr><td>%s</td><td class='%s'>%s%%</td><td>%s%%</td><td>%s</td><td>%s%%</td>"
                 "<td class='%s'>%s%%</td><td class='%s'>%s%%</td></tr>") % (
        lbl, cls(v["全部"]["total"]), f1(v["全部"]["total"]), f1(v["全部"]["ann"]), f2(v["全部"]["sharpe"]),
        f1(v["全部"]["maxdd"]), cls(v["训练期"]["total"]), f1(v["训练期"]["total"]),
        cls(v["验证期"]["total"]), f1(v["验证期"]["total"]))

# 月度 NAV 图 (从日度净值抽月度)
def to_monthly(eqdict):
    out = {}
    for k in sorted(eqdict.keys()):
        mk = k[:6]
        out[mk] = eqdict[k]
    return out

nav_series = []
for k, lbl, col in [("bp1_全A", "bp1全A", "#e74c3c"), ("基准_等权全A", "等权全A基准", "#2563eb"),
                     ("bp1_池内", "bp1池内", "#d97706"), ("基准_等权池内", "池内基准", "#27ae60")]:
    nav_series.append((lbl, col, to_monthly(EQ[k])))

# OFAT 表
ofat_rows = ""
for k, v in R["ofat"].items():
    bold = ' style="background:#eef4ff;font-weight:700"' if k == "基准" else ""
    ofat_rows += "<tr%s><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td>%s%%</td><td class='%s'>%s</td><td class='%s'>%s</td></tr>" % (
        bold, k, "{:,}".format(v["n_total"]), cls(v["全部"]["mean"]), f2(v["全部"]["mean"]), f1(v["全部"]["win"]),
        cls(v["训练期"]["rel"]), f2(v["训练期"]["rel"]), cls(v["验证期"]["rel"]), f2(v["验证期"]["rel"]))

# Regime 表
regime_rows = ""
regime_groups = []
for k in ["上涨", "下跌", "低波", "高波"]:
    v = R["regime"][k]
    regime_rows += "<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td>%s%%</td><td>%s</td><td class='%s'>%s%%</td><td class='%s'>%s pct</td></tr>" % (
        k, "{:,}".format(v["sig_n"]), cls(v["sig_mean"]), f2(v["sig_mean"]), f1(v["sig_win"]),
        "{:,}".format(v["base_n"]), cls(v["base_mean"]), f2(v["base_mean"]), cls(v["rel"]), f2(v["rel"]))
    regime_groups.append((k, [("bp1信号", "#e74c3c", v["sig_mean"]), ("同regime基准", "#2563eb", v["base_mean"])]))

# L0门控增量
l0 = S["l0_gate"]
l0_rows = ("<tr><td>完整信号（L0 N类门通过）</td><td>%s</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td><td>%s%%</td></tr>"
           "<tr><td>仅技术条件（L0未过）</td><td>%s</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td><td>%s%%</td></tr>"
           "<tr style='background:#f0f3f7'><td><strong>L0增量（差值）</strong></td><td>—</td>"
           "<td><strong>%s pct</strong></td><td><strong>%s pct</strong></td><td><strong>%s pct</strong></td></tr>") % (
    "{:,}".format(l0["完整信号(L0过)"]["n"]), cls(l0["完整信号(L0过)"]["mean"]), f2(l0["完整信号(L0过)"]["mean"]),
    cls(l0["完整信号(L0过)"]["median"]), f2(l0["完整信号(L0过)"]["median"]), f1(l0["完整信号(L0过)"]["win"]),
    "{:,}".format(l0["仅技术条件(L0否)"]["n"]), cls(l0["仅技术条件(L0否)"]["mean"]), f2(l0["仅技术条件(L0否)"]["mean"]),
    cls(l0["仅技术条件(L0否)"]["median"]), f2(l0["仅技术条件(L0否)"]["median"]), f1(l0["仅技术条件(L0否)"]["win"]),
    f2(l0["完整信号(L0过)"]["mean"]-l0["仅技术条件(L0否)"]["mean"]),
    f2(l0["完整信号(L0过)"]["median"]-l0["仅技术条件(L0否)"]["median"]),
    f2(l0["完整信号(L0过)"]["win"]-l0["仅技术条件(L0否)"]["win"]))

# 朴素对照
naive_rows = ""
for k, v in nv.items():
    naive_rows += ("<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td><td>%s%%</td></tr>") % (
        k, "{:,}".format(v["n"]), cls(v["mean"]), f2(v["mean"]), cls(v["median"]), f2(v["median"]), f1(v["win"]))

# 微观: 质量分层
def micro_rows(d, name):
    out = "<tr style='background:#f8fafc'><td colspan='6'><strong>%s</strong></td></tr>" % name
    for lb, e in d.items():
        out += "<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
            e.get("label", lb), "{:,}".format(e["n"]), cls(e["mean"]), f2(e["mean"]),
            f2(e.get("训练期")) if e.get("训练期") is not None else "—",
            f2(e.get("验证期")) if e.get("验证期") is not None else "—",
            "%.0f%%" % e["win"] if e.get("win") is not None else "—")
    return out

q_rows = ""
for k, kn in [("platform_range", "平台振幅 HHV/LLV"), ("vol_atr_ratio", "波动率 ATR/MA"),
              ("breakout_amp", "突破幅度 C/HHV-1"), ("vr5_1", "量比 V/MA5"), ("vr_long", "量比 V/MA60")]:
    if k in M["quality"]:
        q_rows += micro_rows(M["quality"][k], kn)

ns_rows = ""
for k, kn in [("n_raw", "N类总分 n_raw"), ("d1_趋势结构", "d1 趋势结构"), ("d2_量价关系", "d2 量价关系"),
              ("d3_相对强弱", "d3 相对强弱 RPS"), ("d4_波动压缩", "d4 波动率压缩")]:
    if k in M["nscore"]:
        ns_rows += micro_rows(M["nscore"][k], kn)

cp_rows = ""
for k, kn in [("pre20", "前20日涨幅"), ("pre60", "前60日涨幅"), ("dd20", "20日回撤"), ("vol20", "20日波动率")]:
    if k in M["counterparty"]:
        cp_rows += micro_rows(M["counterparty"][k], kn)

# 路径表
pt = M["path"]["bp1信号"]
ptb = M["path"].get("全A基准", {})
path_rows = ("<tr><td>20日最大不利偏移 MAE（中位 / 均值）</td><td class='down'>%s%% / %s%%</td><td>%s%% / %s%%</td></tr>"
             "<tr><td>20日最大有利偏移 MFE（中位 / 均值）</td><td>%s%% / %s%%</td><td>%s%% / %s%%</td></tr>"
             "<tr><td>前 5 日最低（中位）</td><td class='down'>%s%%</td><td>—</td></tr>"
             "<tr><td>后 15 日最低（中位）</td><td class='down'>%s%%</td><td>—</td></tr>"
             "<tr><td>持仓低于成本天数（中位，/20日）</td><td>%s</td><td>—</td></tr>"
             "<tr><td>MAE 发生时点（前5日 / 5-10日 / 10-20日）</td><td colspan='2'>%s%% / %s%% / %s%%（平均第 %.1f 日）</td></tr>") % (
    f2(pt["mae20"]["median"]), f2(pt["mae20"]["mean"]),
    f2(ptb.get("mae20", {}).get("median")), f2(ptb.get("mae20", {}).get("mean")),
    f2(pt["mfe20"]["median"]), f2(pt["mfe20"]["mean"]),
    f2(ptb.get("mfe20", {}).get("median")), f2(ptb.get("mfe20", {}).get("mean")),
    f2(pt["first5_min"]["median"]), f2(pt["mae后15日"]["median"]),
    "%.0f" % pt["below_days"]["median"],
    "%.1f" % pt["mae_day_hist"]["前5日"], "%.1f" % pt["mae_day_hist"]["5-10日"],
    "%.1f" % pt["mae_day_hist"]["10-20日"], pt["mae_day_hist"]["mean_day"])

# vr5 表
v5_rows = ""
for k, v in M["vr5_trade"].items():
    if k == "秩相关": continue
    v5_rows += "<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td>%s%%</td><td class='%s'>%s%%</td><td>%s%%</td><td class='%s'>%s%%</td></tr>" % (
        k, "{:,}".format(v["n"]), cls(v["前5日"]), f2(v["前5日"]), f1(v["前5日_胜率"]),
        cls(v["后15日"]), f2(v["后15日"]), f1(v["后15日_胜率"]), cls(v["合计"]), f2(v["合计"]))

vr_rows = ""
for k, v in M["volratio"].items():
    vr_rows += "<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td></tr>" % (k, "{:,}".format(v["n"]), cls(v["mean"]), f2(v["mean"]))

# 秩相关
rc = {k: M["nscore"].get("rankcorr_"+k) for k in ["n_raw","d1","d2","d3","d4","d6","rps"]}
rc_str = " ｜ ".join("%s: %s" % (k, ("%+.4f" % v) if v is not None else "常数无定义") for k, v in rc.items())

# 入场滑点
gp = M["gap"]
dc_ = M["decompose"]

# MAE 失配
mae = P["mae_mismatch"]

# 图表: N类门槛柱状图
n_groups = [("n_raw 40-49", [("bp1", "#e74c3c", g["n_raw 40-49"]["mean"])]),
            ("n_raw 50-59", [("bp1", "#e74c3c", g["n_raw 50-59"]["mean"])]),
            ("n_raw >=60", [("bp1", "#e74c3c", g["n_raw >=60"]["mean"])])]
# 图表: Regime
# 图表: 平台振幅
plat_groups = [(lb, [("bp1", "#e74c3c", e["mean"])]) for lb, e in M["quality"].get("platform_range", {}).items()]
# 图表: 突破幅度
brk_groups = [(lb, [("bp1", "#e74c3c", e["mean"])]) for lb, e in M["quality"].get("breakout_amp", {}).items()]

equity_svg = line_chart(nav_series, vline="202509", vlabel="训练|验证 切分")

html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>bp1（Kimi买点一·平台突破）因子深度评估报告 · 重构版（2026-10-19）</title>
<style>
:root{--bg:#f7f8fa;--card:#fff;--text:#1a1a1a;--muted:#6b7280;--line:#e5e7eb;--red:#e74c3c;--green:#27ae60;--blue:#2563eb;--amber:#d97706;}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--text);line-height:1.75;padding:24px}
.wrap{max-width:1000px;margin:0 auto}
h1{font-size:25px;margin-bottom:4px}
.sub{color:var(--muted);font-size:13px;margin-bottom:22px}
h2{font-size:20px;margin:36px 0 14px;padding-left:10px;border-left:4px solid var(--blue)}
h3{font-size:16px;margin:20px 0 8px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;margin-bottom:16px}
table{width:100%;border-collapse:collapse;font-size:13px;margin:10px 0}
th{background:#f0f3f7;text-align:left;padding:7px 9px;font-weight:600;border-bottom:2px solid var(--line);white-space:nowrap}
td{padding:7px 9px;border-bottom:1px solid var(--line);white-space:nowrap}
tr:last-child td{border-bottom:none}
.up{color:var(--red);font-weight:600}
.down{color:var(--green);font-weight:600}
.tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600}
.tag-red{background:#fdecea;color:var(--red)}
.tag-green{background:#e8f8f5;color:var(--green)}
.tag-gray{background:#f0f3f7;color:var(--muted)}
.tag-amber{background:#fef3e2;color:var(--amber)}
.note{background:#fef9e7;border-left:4px solid var(--amber);padding:10px 14px;border-radius:6px;font-size:13px;margin:12px 0}
.warn{background:#fdecea;border-left:4px solid var(--red);padding:10px 14px;border-radius:6px;font-size:13px;margin:12px 0}
.key{background:#eef4ff;border-left:4px solid var(--blue);padding:10px 14px;border-radius:6px;font-size:13px;margin:12px 0}
.muted{color:var(--muted);font-size:12px}
code{background:#f0f3f7;padding:1px 5px;border-radius:4px;font-size:12.5px}
.formula{background:#f8fafc;border:1px solid var(--line);border-radius:8px;padding:12px 16px;font-size:13px;margin:8px 0}
@page{margin:11mm 9mm;size:A4}
@media print{
body{padding:0;background:#fff;font-size:12px}
.wrap{max-width:100%}
.card{break-inside:avoid;page-break-inside:avoid;border:1px solid #d1d5db}
table{break-inside:avoid;page-break-inside:avoid;font-size:11px}
th,td{padding:5px 7px}
h2{break-after:avoid;page-break-after:avoid;margin:22px 0 10px}
h3{break-after:avoid;page-break-after:avoid}
svg{break-inside:avoid;page-break-inside:avoid}
.note,.warn,.key{break-inside:avoid;page-break-inside:avoid}
*{-webkit-print-color-adjust:exact;print-color-adjust:exact}
}
</style>
</head>
<body>
<div class="wrap">

<h1>bp1（Kimi买点一 · 平台突破）因子深度评估报告 <span class="tag tag-amber">重构版 · 第三章</span></h1>
<div class="sub">生成日期：2026-10-19 ｜ 方法：A股量化研报复现专家团 Workflow B（因子/信号有效性检验）｜ 数据：kline_cache_all.db（前复权日线，2023-06-01 ~ 2026-08-21，全A 4999 只）｜ 验收：role-07 研报复现验收工程师｜ 配套：第一章《B1》《第二章 B2》</div>

<div class="warn">
<strong>验收结论先行（冷酷版）：</strong>bp1（Kimi 买点一·平台突破）<strong>校准通过（1003/1003 = 100.00%），但因子有效性不通过——这是迄今三章中最惨的信号。</strong>三重证据：① 信号层均值 <span class="down">−0.29%</span>/20日、中位 −3.33%、胜率 39.4%，日度截面 RankIC = <span class="down">−0.0167</span>（t = −14.1）；② 真实可投资口径（日度等权再平衡、含成本）全A 累计 <span class="down">−53.2%</span>（夏普 −1.04）vs 等权全A基准 <span class="up">+31.6%</span>——<strong>比 B1（−3.8%）和 B2（+16.3%）差一个数量级</strong>；③ 14 组参数变体无一能翻正到基准之上。池外信号（12,335 条，占 80%）均值 −0.92%，是拖垮整体的主因；池内信号（3,133 条）均值 +2.15%，alpha 仍只在选股池。
</div>

<div class="key">
<strong>一句话给用户的答案：</strong>bp1 的"平台突破"叙事——横盘缩量后放量突破前高——在数据上<strong>方向完全反了</strong>。突破幅度越大未来越差（3-5% 突破 +1.40% vs &gt;12% 突破 <span class="down">−4.21%</span>），量放得越猛未来越差（vr60 1.5-2 倍 +2.13% vs &gt;5 倍 <span class="down">−2.17%</span>），N 类评分唯一有效的维度是 <strong>d4 波动率压缩</strong>（秩相关 +0.141）——它奖励的是"还没突破"而非"正在突破"。突破那一刻你追进去的，平均是接盘。<strong>bp1 应立即停用为买入信号；它的残值是 d4 维度的波动率压缩逻辑，可单独保留。</strong>
</div>

<!-- ==================== 一、定义 ==================== -->
<h2>一、因子定义与参数（回源码逐行核对）</h2>
<div class="card">
<p>定义来源：<code>detect_shunshi_buy_point</code>（<code>_shunshi_buy_points_ref.py</code>）L3 四大买点之首 + <code>compute_niugu_score</code>（<code>_niigu_judge_ref.py</code>）N 类评分。bp1 = <strong>L0 N类门控通过</strong>（<code>n_raw_score ≥ 40</code>）& <strong>L1 自动通过</strong> & <strong>L2 日度门（仅降 confidence 不否决）</strong> & <strong>技术五条件全满足</strong>。信号日 T 日收盘后可知，<strong>T+1 开盘入场、T+20 收盘出场</strong>（持有 20 个交易日）。</p>
<div class="formula">
<strong>技术五条件（AND，全部须满足）：</strong><br>
① <strong>K线数 ≥ 60</strong>（上市不满 60 交易日不触发）<br>
② <strong>平台振幅</strong> <code>HHV(H,20)/LLV(L,20) &lt; 1.25</code>（近 20 日<b>不含今日</b>的最高/最低 &lt; 25%）<br>
③ <strong>波动率压缩</strong> <code>ATR(14, Wilder)/MA(C,14) &lt; 0.06</code><br>
④ <strong>突破确认</strong> <code>C &gt; HHV(H,20)</code> 且 <code>C/HHV(H,20) &gt; 1.03</code>（今日收盘突破前 20 日高点至少 3%）<br>
⑤ <strong>放量确认</strong> <code>V &gt; MA5(V)×1.5</code> 且 <code>V &gt; MA60(V)×1.5</code>（5 日和 60 日量比均超 1.5 倍，均不含今日）
</div>
<div class="formula">
<strong>N类评分 n_raw（L0 门槛 40 分）：</strong><br>
<code>n_raw = floor((d1 + d2 + d3 + d4 + d6) × stage_mult)</code><br>
· <strong>d1 趋势结构</strong>（满分 25）：MA 多头排列 + 价格位置 + 趋势新鲜度（<code>trend_age</code>）<br>
· <strong>d2 量价关系</strong>（满分 20）：A 突破量能 / B 量价背离 / C 底部放量三子项<br>
· <strong>d3 相对强弱 RPS</strong>（满分 20）：<code>min(250, n_i)</code> 日收益在全体 ≥ 该根数股票中的百分位排名<br>
· <strong>d4 波动率压缩</strong>（满分 15）：BBW 与 ATR/MA 双百分位（<code>lookback=min(120,t+1)</code> 扩展窗口，门禁 n≥30/24）<br>
· <strong>d6 筹码结构</strong>（满分 10）：<code>extra_data=None</code> 时恒定 6 分<br>
· <strong>stage_mult</strong> 阶段新鲜度乘数（0.82~1.0，float64 计算）
</div>
<table>
<tr><th>参数</th><th>值</th><th>参数</th><th>值</th></tr>
<tr><td>L0 门槛</td><td>n_raw ≥ 40</td><td>L2 作用</td><td>仅降 confidence，不否决信号</td></tr>
<tr><td>平台振幅</td><td>HHV/LLV &lt; 25%（20日）</td><td>ATR/MA</td><td>Wilder ATR(14)/MA14 &lt; 0.06</td></tr>
<tr><td>突破幅度</td><td>C/HHV(H,20) &gt; 1.03</td><td>量比</td><td>vr5&gt;1.5 且 vr60&gt;1.5（不含今日）</td></tr>
<tr><td>信号日</td><td>T 日收盘后可知</td><td>入场 / 出场</td><td>T+1 开盘 / T+20 收盘</td></tr>
<tr><td>信号总量</td><td>15,468 条（完整前瞻，去右截断 183 条）</td><td>宇宙</td><td>全A 4,999 只</td></tr>
</table>
</div>

<h2>校准验证：与原引擎金标准逐维度比对</h2>
<div class="card">
<div class="key">
<strong>校准结果：1003/1003 = 100.00% 全维度一致。</strong>方法：截断模式（truncate klines 至采样日）现算 vs 原引擎 <code>compute_niugu_score</code> / <code>_bp1_platform_breakout</code> / <code>_check_l2_daily_gate</code> 逐维度比对。采样 40 只成熟池内股 + 10 只年轻股（60≤row_counts&lt;200）× 25 个月首日（202406~202606）。这是与原引擎数据口径完全一致的检验（全序列 pkl 存在前视，故以截断模式为准）。
</div>
<p><strong>五项修复记录（每项都是实际踩坑）：</strong></p>
<table>
<tr><th>#</th><th>缺陷</th><th>表现</th><th>修复</th></tr>
<tr><td>1</td><td>d2 C 部分（底部放量）前视偏差</td><td>向量化在完整序列上 anchor 用 n-1，offset +1/+2/+3 越过当前日读到未来 K 线（例：603353 20240603 全序列 d2=12 vs 截断 d2=7）</td><td>上锚改 <code>jend = arange(59, n)</code>，offset 越界即跳过</td></tr>
<tr><td>2</td><td>RPS 截面口径错（年轻票）</td><td><code>wk == w</code> 只含同根数票；原引擎截面 = 所有 <code>len ≥ rps_period</code> 的票</td><td>改 <code>wk ≥ w</code> + Cbrow_mat padded gather 向量化</td></tr>
<tr><td>3</td><td>n_raw float32 边界</td><td>floor(50×0.9f) = 44 而非 45（3 条 ±1）</td><td>float64 计算 + stage_mult64 字段</td></tr>
<tr><td>4</td><td>d4 百分位边界噪声</td><td>cumsum MA 的 1e-16 噪声使 tie 翻转 10% 分位边界</td><td>ma_arr 换滑动窗口 mean（与 np.mean 逐位一致）</td></tr>
<tr><td>5</td><td>d4 短历史口径缺失</td><td>原引擎 <code>lookback=min(120,n)</code> 扩展窗口 + n≥30/24 门禁；我方 n&lt;120 直接 NaN</td><td>rolling_pct_of_last 前补 NaN + min_n 门禁</td></tr>
</table>
<p class="muted">另修复：bbw 改 <code>(upper-lower)/sma</code> 显式计算（非 4*std/sma，浮点路径不同）；d4 的 tr[0]=NaN（原引擎如此）。</p>
</div>

<!-- ==================== 二、九项口径 ==================== -->
<h2>二、九项口径声明（与 B1/B2 报告一致）</h2>
<div class="card">
<table>
<tr><th>#</th><th>口径项</th><th>声明值</th></tr>
<tr><td>1</td><td>下期收益定义</td><td>T 日信号收盘后可知 → T+1 开盘入场 → T+20 收盘出场（持有 20 个交易日）</td></tr>
<tr><td>2</td><td>分组数</td><td>信号层十分位 + 14 组分层（全体/训练/验证/池内/池外/交叉/n_raw 三档/L2）</td></tr>
<tr><td>3</td><td>多空方向</td><td>仅多头；对照基准为同宇宙等权持有，不设空头腿</td></tr>
<tr><td>4</td><td>股票池过滤</td><td>剔除指数与名单外代码；<strong>未做</strong> ST / 涨跌停 / 停牌 / 上市日过滤（D2）</td></tr>
<tr><td>5</td><td>极值处理</td><td>不做 MAD 缩尾（收益率为原始值）</td></tr>
<tr><td>6</td><td>中性化</td><td><strong>不做</strong>行业 / 市值中性化（D12）</td></tr>
<tr><td>7</td><td>基准</td><td>等权全A（4999 只，主基准）/ 等权池内（848 只，含后视偏差）/ 等权池外</td></tr>
<tr><td>8</td><td>无风险利率</td><td>0（月度夏普 = 均值/标准差×√12）</td></tr>
<tr><td>9</td><td>费率</td><td>基准口径不计成本；计费口径：佣金 0.00025 双边 + 印花税 0.0005 卖出 + 滑点 0.001 双边 ≈ 单次往返 0.30%</td></tr>
</table>
<p class="muted">训练/验证切分：信号日 &lt; 2025-09-08 为训练期（11,176 条），≥ 为验证期（4,292 条）。参数为既有生产参数直接沿用，本次不重新寻优。原引擎参考系为池内（stock_pool.json），生产评估用全A参考系（与 B1/B2 口径一致，偏离项 D21）。</p>
</div>

<!-- ==================== 三、信号层 ==================== -->
<h2>三、信号层有效性检验</h2>
<div class="card">
<h3>3.1 分组统计（T+1开盘 → T+20收盘，无止损，%）</h3>
<table>
<tr><th>分组</th><th>n</th><th>均值</th><th>中位</th><th>胜率</th><th>P10</th><th>P90</th><th>top5%盈亏贡献</th></tr>
__ROWS_GROUPS__
</table>
<div class="warn"><strong>读法：</strong>比 B1/B2 更纯粹的负期望——均值 −0.29%、中位 −3.33%、胜率 39.4%，<strong>连彩票型结构都算不上</strong>（top5% 贡献 −929%：赢家远不够补亏家）。池内均值 +2.15% 是唯一的亮点但样本仅 3,133 条且中位仍为负（−1.55%），靠右尾撑均值。n_raw ≥60 组均值 +1.20%——N 类门控提上去有单调改善，但仍不及基准。202409 异常峰值 3,084 条（占 20%），该月信号爆炸式触发需单独审视。</div>

<h3>3.2 日度截面 RankIC（对 20 日前向收益）</h3>
<table>
<tr><th>因子形式</th><th>IC 均值</th><th>ICIR</th><th>IC&gt;0 占比</th><th>t 值</th><th>判定</th></tr>
<tr><td>bp1 信号（0/1）</td><td class="down">−0.0167</td><td class="down">−0.561</td><td>25.3%</td><td class="down">−14.11</td><td><span class="tag tag-red">负且极显著</span></td></tr>
<tr><td>n_raw（连续）</td><td class="down">−0.0752</td><td class="down">−0.491</td><td>32.8%</td><td class="down">−12.83</td><td><span class="tag tag-red">负且极显著</span></td></tr>
<tr><td>信号组−非信号组价差</td><td class="down">−1.71 pct/日</td><td>—</td><td>33.8%</td><td class="down">−5.13</td><td><span class="tag tag-red">负且显著</span></td></tr>
</table>
<div class="note"><strong>关键发现：</strong>n_raw 连续 IC = −0.075 意味着在<b>全A截面上</b>，N 类评分越高未来越差。但注意分层表里 n_raw ≥60 组均值为正——这是条件效应（bp1 触发的子集内高分好），不是截面效应（全体股票里高分差）。两者并存说明 N 类评分在"已突破"的股票上无选股力，在全体股票上甚至反向。<strong>L0 门控的增量很小</strong>：完整信号 −0.29% vs 仅技术条件 −0.85%，差 +0.55 pct——大部分信息在技术五条件里，N 类门只挡掉了一小部分更差的。</div>

<h3>3.3 收益分布（n=15,468，4% 等宽桶）</h3>
__HIST__
<table>
<tr><th>十分位</th><th>收益区间</th><th>数量占比</th><th>盈亏贡献</th><th>组均值</th></tr>
__ROWS_DECILE__
</table>

<h3>3.4 N类门控增量（L0 通过 vs 仅技术条件）</h3>
<table>
<tr><th>组</th><th>n</th><th>均值%</th><th>中位%</th><th>胜率%</th></tr>
__L0_ROWS__
</table>

<h3>3.5 朴素对照（同一持有期）</h3>
<table>
<tr><th>组</th><th>n</th><th>均值%</th><th>中位%</th><th>胜率%</th></tr>
__NAIVE_ROWS__
</table>
<div class="warn"><strong>与基准的差距是碾压性的：</strong>bp1 信号均值 −0.29% vs 全股票等权 +1.91%（差 −2.20 pct/20日）；池内信号 +2.15% vs 池内全股票等权 +2.54%（差 −0.39 pct）。bp1 在池内也<b>没有跑赢随机买池内股票</b>。</div>
</div>

<!-- ==================== 四、组合层 ==================== -->
<h2>四、组合层绩效</h2>
<div class="card">
<h3>4.1 真实可投资口径：日度等权再平衡 NAV（含成本，本报告主口径）</h3>
__EQUITY__
<table>
<tr><th>组合（日度等权再平衡）</th><th>累计%</th><th>年化%</th><th>夏普</th><th>最大回撤%</th><th>训练期%</th><th>验证期%</th></tr>
__ROWS_NAV__
</table>
<div class="warn"><strong>bp1 是三章中唯一"稳定大幅亏损"的信号。</strong>全A 累计 <span class="down">−53.2%</span>（夏普 −1.04）vs 等权全A <span class="up">+31.6%</span>——相对基准跑输 <strong>84.8 个百分点</strong>。池内 +3.5% vs 池内基准 +82.2%（跑输 78.7 pct）。池外 <span class="down">−66.4%</span> vs +22.9%（跑输 89.3 pct）。训练期 −45.7%、验证期 −14.9%——两期皆负，不是过拟合能解释的，是机制本身反向。<strong>"平台突破买入"在 A 股 2023-2026 是系统性亏钱策略。</strong></div>

<h3>4.2 月度 cohort 等权口径（止损档对照）</h3>
<table>
<tr><th>止损档</th><th>累计%</th><th>年化%</th><th>夏普</th><th>最大回撤%</th><th>训练期%</th><th>验证期%</th><th>含成本%</th></tr>
__STOP_ROWS__
</table>
<div class="note"><strong>止损 8% 是唯一止血档</strong>（−42.7% vs 不止损 −61.0%），但代价是 61.4% 的信号被止损扫掉（9,582/15,642）。止损机会成本：被止损的 9,582 条若不止损平均还要再亏 −8.12%——<strong>止损在这类信号上是"减亏"而非"保盈"</strong>。含成本后全档恶化 4-6 pct。</div>

<h3>4.3 池内组合（选股池口径）</h3>
<table>
<tr><th>止损档</th><th>累计%</th><th>夏普</th><th>最大回撤%</th><th>训练期%</th><th>验证期%</th><th>含成本%</th></tr>
__POOL_ROWS__
</table>

<h3>4.4 止损与波动率失配（MAE 分布，n=15,468）</h3>
<table>
<tr><th>阈值</th><th>MAE ≤ 阈值占比</th><th>含义</th></tr>
__MAE_ROWS__
</table>
<div class="note">MAE 中位 <span class="down">−10.57%</span>、均值 −11.80%——<strong>bp1 的入场点是"突破加速点"，天然买在局部高波动区</strong>，20 日内回撤超 8% 的占 61.4%。这与全A基准（MAE 中位 −6.74%）差距巨大。8% 止损会被超过六成的信号打掉；若要匹配 bp1 的波动结构，止损需 ≥15%（32.9% 触发）甚至 20%（15.7% 触发），但那又回到不止损的亏损深渊。<strong>止损失配在 bp1 上是无解的：宽止损亏更多，窄止损被扫光。</strong></div>
</div>

<!-- ==================== 五、稳健性 ==================== -->
<h2>五、稳健性检验</h2>
<div class="card">
<h3>5.1 参数敏感性（OFAT，相对基准 alpha，pct/20日）</h3>
<table>
<tr><th>变体</th><th>n</th><th>均值%</th><th>胜率%</th><th>训练期 rel</th><th>验证期 rel</th></tr>
__OFAT_ROWS__
</table>
<div class="note"><strong>能救的旋钮不多：</strong>L0 门槛提到 50 后均值勉强转正（+0.06%）但验证期仍负（−0.29%）；突破门槛放宽到 ≥2% 均值 +0.27%（<strong>放宽=更不追高，反而好</strong>）；持有期拉到 30 日均值 +0.72%（趋势需要更长兑现）。收紧任何条件（更窄平台、更低 ATR、更高突破）在训练期有效但验证期恶化——<strong>没有一个变体能在两个时期同时跑赢基准（rel &gt; 0）</strong>。</div>

<h3>5.2 市场状态条件</h3>
<table>
<tr><th>状态</th><th>信号 n</th><th>信号均值%</th><th>信号胜率</th><th>基准 n</th><th>基准均值%</th><th>相对 pct</th></tr>
__REGIME_ROWS__
</table>
__REGIME_CHART__
<div class="warn"><strong>高波动市场是 bp1 的坟场</strong>：高波 regime 信号均值 <span class="down">−3.94%</span>（胜率 28.2%），相对基准 −7.83 pct。低波 regime 勉强 +1.05%。下跌市相对基准 −4.19 pct。bp1 触发本身就依赖放量突破，天然集中在高波动时刻——<strong>机制与市场状态的耦合是结构性的，不是参数能解的</strong>。</div>

<h3>5.3 数据质量与截断</h3>
<p>右截断：15,651 条原始信号中 183 条（1.2%）因持有期超出数据截止被剔除，影响可忽略。全A宇宙 4,999 只（剔除指数与名单外）。月度信号量 202409 异常峰值 3,084 条，202607 仅 49 条——信号量与市场波动状态强相关。</p>
</div>

<!-- ==================== 六、微观 ==================== -->
<h2>六、微观经济假设检验（"平台突破"机制到底在买什么）</h2>
<div class="card">
<h3>6.1 技术条件质量分层：越"标准"的平台突破越差（H1 证伪）</h3>
<table>
<tr><th>分组</th><th>n</th><th>均值</th><th>训练期</th><th>验证期</th><th>胜率</th></tr>
__Q_ROWS__
</table>
<div class="warn"><strong>假设"标准的平台突破（窄平台、低波动、强突破、猛放量）更可靠"被逐项证伪：</strong>平台振幅 &lt;5% 组 +3.02% vs 20-25% 组 −1.15%（窄的好，符合直觉）；但突破幅度 3-5% 组 +1.40% vs &gt;12% 组 <span class="down">−4.21%</span>（<strong>突破越猛越差</strong>）；量比 vr5 1.5-2 组 +1.85% vs &gt;5 组 −0.75%，vr60 1.5-2 组 +2.13% vs &gt;5 组 <span class="down">−2.17%</span>（<strong>量放得越猛越差</strong>）。"平台突破"叙事的核心——放量突破确认——在数据上是<b>反指</b>。</div>

<h3>6.2 N类评分分层与秩相关</h3>
<table>
<tr><th>分组</th><th>n</th><th>均值</th><th>训练期</th><th>验证期</th><th>胜率</th></tr>
__NS_ROWS__
</table>
<p><strong>各维度秩相关（vs t1_20）：</strong>__RC_STR__</p>
<div class="note"><strong>d4 波动率压缩是唯一有正判别力的维度</strong>（秩相关 +0.141，&gt;12 分组均值 +5.56%、胜率 58.3%）——它奖励的是"波动率被压缩到极致"，即<b>突破之前</b>的状态。其余维度：d1 趋势 −0.016、d3 RPS −0.020、d2 量价 +0.006 均无判别力；d6 因 extra_data=None 恒为 6 分是常数。<strong>N 类评分体系里唯一有价值的信息是"还没动"，不是"正在突破"。</strong></div>

<h3>6.3 对手方分层（前涨幅/回撤/波动率）</h3>
<table>
<tr><th>分组</th><th>n</th><th>均值</th><th>训练期</th><th>验证期</th><th>胜率</th></tr>
__CP_ROWS__
</table>
<div class="warn"><strong>接盘效应清晰：</strong>前 20 日涨幅 &gt;20% 组均值 −1.47%、前 60 日涨幅 50-100% 组 −2.58%——<strong>bp1 触发前涨得越多、未来越差</strong>（20日回撤 −15~-10% 组 −2.56% vs &gt;-5% 组 +1.44%）。波动率 &gt;4% 组 −2.30% vs &lt;2% 组 +2.52%。bp1 突破买入的对手方是趋势末端的卖压，不是"新资金进场"。</div>

<h3>6.4 持仓路径分析</h3>
<table>
<tr><th>指标</th><th>bp1 信号</th><th>全A基准</th></tr>
__PATH_ROWS__
</table>
<div class="note"><strong>持仓体验极差：</strong>中位 15/20 天低于成本、MAE 中位 −10.57%、MAE 平均在第 8.8 日发生（前 5 日占 35.2%）。MFE 中位仅 +9.33%——向上的空间远不够覆盖向下的回撤。<strong>bp1 信号的典型路径是：追高入场 → 立刻回撤 → 横到到期。</strong></div>

<h3>6.5 量能事前性（vr5 五分位分解）</h3>
<table>
<tr><th>vr5 分位</th><th>n</th><th>前5日%</th><th>前5日胜率</th><th>后15日%</th><th>后15日胜率</th><th>合计%</th></tr>
__V5_ROWS__
</table>
<p class="muted">量比与前 5 日收益秩相关 −0.024、与后 15 日 −0.061——<strong>放量对未来收益零预测甚至微负</strong>。vr5 最低分位组（1.5-2.1 倍）后 15 日 +2.63% 是全场最好——恰是"放量刚过线"的组；最高分位（&gt;4.2 倍）后 15 日仅 +0.10%。</p>

<h3>6.6 入场滑点与收益对数分解</h3>
<table>
<tr><th>组</th><th>n</th><th>均值%</th></tr>
__VR_ROWS__
</table>
<div class="note"><strong>入场滑点：</strong>T+1 开盘跳空均值 <span class="up">+2.03%</span>（中位 +0.04%），高开&gt;1% 占 38.8%、低开&gt;1% 占 25.9%。gap 与 t1_20 秩相关 −0.109——<strong>高开越猛未来越差</strong>，追高入场成本自己就吃掉了一段。收益对数分解（n=15,468）：跳空贡献 +290.9、首日 −61.3、余 19 日 −206.9、总 log −268.2——<strong>信号日到入场日的跳空是正的（你没法全吃到），真正持仓期（首日+余19日）合计 −268.2，是实打实的亏损</strong>。盈亏比 &gt;1.6 的量比组均值 +2.79% 是唯一正组。</div>
</div>

<!-- ==================== 七、偏离登记 ==================== -->
<h2>七、偏离登记（继承 B1/B2 D1-D17，bp1 专项新增）</h2>
<div class="card">
<table>
<tr><th>#</th><th>偏离项</th><th>说明</th><th>影响</th><th>等级</th></tr>
<tr><td>D1</td><td>无市值/行业中性化</td><td>数据源不含相关字段</td><td>高</td><td>中</td></tr>
<tr><td>D2</td><td>未过滤 ST/涨跌停/停牌</td><td>实际交易受限</td><td>高</td><td>中</td></tr>
<tr><td>D9</td><td>持有期口径</td><td>bp1 为 T+1 开盘→T+20 收盘，与 B1 相同；B2 为 T+2 开盘（晚一日）</td><td>跨章对比</td><td>低</td></tr>
<tr><td>D12</td><td>幸存者偏差</td><td>退市股缺失，结果偏乐观</td><td>高</td><td>中</td></tr>
<tr><td>D17</td><td>样本内参数</td><td>所有参数为生产既有值，未寻优但也不保证样本外稳定</td><td>中</td><td>低</td></tr>
<tr><td><strong>D21</strong></td><td><strong>RPS 参考系差异</strong></td><td><strong>原引擎 RPS 参考系 = 池内（stock_pool.json 848 只）；本报告用全A（4999 只）建 all_closes_map。d3 分数在两个参考系下不同，可能改变 n_raw 分布与 L0 通过集。已确认原始 47 条 CSV 信号集因池版本缺失（9 只不在池）+ L0 边缘失配导致匹配率 70%；截断模式金标准验证以原引擎函数为准 100% 一致，此偏离仅影响"生产全A复现"与"原引擎池内复现"的可比性</strong></td><td>中</td><td><strong>高</strong></td></tr>
<tr><td><strong>D22</strong></td><td><strong>202409 信号异常峰值</strong></td><td>3,084 条占全样本 20%。该月市场剧烈波动+大量股票突破平台，bp1 触发爆炸。月度信号量不稳定使日度等权再平衡 NAV 在该月风险暴露异动</td><td>中</td><td>中</td></tr>
<tr><td><strong>D23</strong></td><td><strong>extra_data 缺失</strong></td><td>d6 筹码结构在 extra_data=None 时恒定 6 分（满分 10），N 类评分实际满分只有 84 而非 90。L0 门槛 40 对应实际约 47.6% 得分率（非 44.4%）</td><td>低</td><td>低</td></tr>
<tr><td><strong>D24</strong></td><td><strong>移动止盈几乎不触发</strong></td><td>不止损档 20 日持有内仅 29/15,642 条触发移动止盈（峰值 50% 回撤）——20 日太短，移动止盈形同虚设。出场全部走"到期"</td><td>低</td><td>低</td></tr>
</table>
</div>

<!-- ==================== 八、验收结论 ==================== -->
<h2>八、验收结论（role-07 研报复现验收工程师）</h2>
<div class="card">
<h3>8.1 前视偏差检查</h3>
<table>
<tr><th>#</th><th>检查项</th><th>结果</th></tr>
<tr><td>1</td><td>信号 T 日收盘后可知</td><td><span class="tag tag-green">通过</span> — 技术五条件与 N 类均用 ≤T 日数据</td></tr>
<tr><td>2</td><td>T+1 开盘入场</td><td><span class="tag tag-green">通过</span> — 收益计算 T+1 open → T+20 close</td></tr>
<tr><td>3</td><td>d2 C 部分 anchor</td><td><span class="tag tag-green">通过（修复后）</span> — 曾有前视（offset 越界读未来），已改 jend 修复</td></tr>
<tr><td>4</td><td>RPS 截面</td><td><span class="tag tag-green">通过（修复后）</span> — 截面 ≥w 与原引擎一致</td></tr>
<tr><td>5</td><td>百分位窗口</td><td><span class="tag tag-green">通过（修复后）</span> — 扩展窗口 + 门禁与原引擎一致</td></tr>
<tr><td>6</td><td>n_raw 计算精度</td><td><span class="tag tag-green">通过（修复后）</span> — float64 消除边界翻转</td></tr>
<tr><td>7</td><td>MA 计算一致性</td><td><span class="tag tag-green">通过（修复后）</span> — 滑动窗口 mean 与 np.mean 逐位一致</td></tr>
<tr><td>8</td><td>金标准比对</td><td><span class="tag tag-green">1003/1003 = 100.00%</span></td></tr>
<tr><td>9</td><td>右截断处理</td><td><span class="tag tag-green">通过</span> — 183/15651 剔除</td></tr>
</table>

<h3>8.2 切分方式审查</h3>
<p>训练/验证按信号日 2025-09-08 硬切分（训练 11,176 / 验证 4,292）。验证期样本量充足。OFAT 的参数变体在验证期未翻正，无过拟合美化迹象。</p>

<h3>8.3 最终结论</h3>
<div class="warn">
<strong>复现过程：通过。</strong>校准 1003/1003 = 100.00%，五项修复后与原引擎完全一致。<br><br>
<strong>因子有效性：不通过——三章中最差。</strong><br>
· 信号层：均值 −0.29%、中位 −3.33%、胜率 39.4%、RankIC −0.017（t=−14.1）——负期望且统计极显著<br>
· 组合层：全A NAV <span class="down">−53.2%</span> vs 基准 <span class="up">+31.6%</span>，跑输 84.8 pct；池内 +3.5% vs +82.2%<br>
· 稳健性：14 组变体无一同时在训练/验证期跑赢基准；高波 regime 均值 −3.94%<br>
· 微观机制：突破越猛越差、量放越猛越差——<strong>"平台突破买入"的核心叙事被数据直接否定</strong><br><br>
<strong>唯一残值：</strong>d4 波动率压缩维度（秩相关 +0.141，≥12 分组 +5.56%/胜率 58.3%）——它奖励的是"突破之前"的静默状态。池内信号（+2.15%）靠选股池兜底，但池内基准 +2.54% 随机买入更好。
</div>
</div>

<!-- ==================== 九、修复路径 ==================== -->
<h2>九、基于数据的修复路径（按优先级）</h2>
<div class="card">
<table>
<tr><th>优先级</th><th>方向</th><th>依据</th></tr>
<tr><td><strong>S（立即执行）</strong></td><td><strong>停用 bp1 作为买入信号。</strong>NAV −53.2% vs 基准 +31.6%，无参数组合可救。继续使用的每一笔都是负期望</td><td>4.1 / 8.3</td></tr>
<tr><td><strong>A（有数据支撑的唯一残值）</strong></td><td><strong>拆出 d4 波动率压缩作为独立因子</strong>：秩相关 +0.141、&gt;12 分组 +5.56%/58.3%——测"突破前的静默"而非"突破本身"。可在突破前埋伏（d4 ≥ 10 且未触发技术五条件），而非突破后追入</td><td>6.2</td></tr>
<tr><td><strong>B（若一定要救 bp1 形态）</strong></td><td>三重过滤：① L0 门槛提到 n_raw ≥ 50（+0.06%）② 突破幅度限定 3-5%（+1.40%，禁追 &gt;8%）③ 量比限定 vr60 ≤ 3（+0.78%）。叠加后可能勉强正但样本与稳健性均存疑，<strong>需独立样本外验证后才可考虑</strong></td><td>5.1 / 6.1</td></tr>
<tr><td><strong>C（机制改造）</strong></td><td>反转方向测试：bp1 突破日的<b>卖方</b>（做空/回避突破股）。pre20 &gt;20% 组 −1.47%、突破 &gt;12% 组 −4.21%——反向信号更稳定但 A 股做空受限，可作为回避名单</td><td>6.1 / 6.3</td></tr>
<tr><td><strong>D（出场机制）</strong></td><td>止损失配无解（MAE 中位 −10.57% vs 止损 8% 扫掉 61.4%）。若持有 bp1 形态，止损至少 15%（32.9% 触发）+ 持有期延至 30 日（+0.72%），但整体仍不及基准</td><td>4.4 / 5.1</td></tr>
<tr><td>E（待办）</td><td>池内 RPS 参考系复现（D21）：用 stock_pool.json 建 all_closes_map 重跑，对比全A 参考系的信号集差异</td><td>D21</td></tr>
</table>
<div class="key">
<strong>与前两章的横向对比：</strong>B1（−3.8%）&lt; B2（+16.3%）&lt; 基准（+31.6%）&lt;&lt; <strong>bp1（−53.2%）</strong>。三个信号没有一个跑赢"什么都不做"，但 bp1 是唯一"主动大幅亏钱"的。B1/B2 的教训是"确认层没增量"，bp1 的教训更根本：<strong>"突破追涨"这个动作本身就是负 alpha</strong>。选股池的 alpha（池内 +2.15% vs 池外 −0.92%）依然是唯一可靠的东西。
</div>
</div>

<div class="muted" style="margin-top:30px;text-align:center">
bp1（Kimi买点一·平台突破）因子深度评估报告 · 重构版 ｜ 2026-10-19 ｜ A股量化研报复现专家团 Workflow B ｜ 配套：B1 / B2 章
</div>

</div>
</body>
</html>"""

# 替换模板占位符
html = html.replace("__ROWS_GROUPS__", rows_groups)
html = html.replace("__ROWS_DECILE__", rows_decile)
html = html.replace("__L0_ROWS__", l0_rows)
html = html.replace("__NAIVE_ROWS__", naive_rows)
html = html.replace("__STOP_ROWS__", stop_rows)
html = html.replace("__POOL_ROWS__", pool_rows)
html = html.replace("__ROWS_NAV__", nav_rows)
html = html.replace("__EQUITY__", equity_svg)
html = html.replace("__OFAT_ROWS__", ofat_rows)
html = html.replace("__REGIME_ROWS__", regime_rows)
html = html.replace("__REGIME_CHART__", bar_chart(regime_groups))
html = html.replace("__Q_ROWS__", q_rows)
html = html.replace("__NS_ROWS__", ns_rows)
html = html.replace("__RC_STR__", rc_str)
html = html.replace("__CP_ROWS__", cp_rows)
html = html.replace("__PATH_ROWS__", path_rows)
html = html.replace("__V5_ROWS__", v5_rows)
html = html.replace("__VR_ROWS__", vr_rows)
html = html.replace("__HIST__", hist_chart(edges, counts))

# MAE 表
mae_rows = ""
for th, pct in [(5, mae["frac_below"]["5"]), (8, mae["frac_below"]["8"]), (10, mae["frac_below"]["10"]),
                (12, mae["frac_below"]["12"]), (15, mae["frac_below"]["15"]), (20, mae["frac_below"]["20"]),
                (25, mae["frac_below"]["25"])]:
    note = "8%止损被打掉的比例" if th == 8 else ""
    mae_rows += "<tr><td>MAE ≤ %d%%</td><td class='%s'>%s%%</td><td>%s</td></tr>" % (
        th, "down" if pct > 50 else "", f1(pct), note)
html = html.replace("__MAE_ROWS__", mae_rows)

from healthcheck_chapter import inject
html = inject(html, "bp1(突破追入)")

out_path = os.path.join(HERE, "bp1因子深度评估报告_重构版_20261019.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print("saved ->", out_path)
print("len:", len(html))
