# -*- coding: utf-8 -*-
"""B2因子重构评估报告生成器 — 输出 standalone HTML (内联SVG图表, 无外部依赖)"""
import json, os, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "b2_signal_results.json"), encoding="utf-8"))
P = json.load(open(os.path.join(HERE, "b2_port_results.json"), encoding="utf-8"))
N = json.load(open(os.path.join(HERE, "b2_nav_results.json"), encoding="utf-8"))
EQ = json.load(open(os.path.join(HERE, "b2_nav_equity.json"), encoding="utf-8"))
R = json.load(open(os.path.join(HERE, "b2_robust_results.json"), encoding="utf-8"))
M = json.load(open(os.path.join(HERE, "b2_micro_results.json"), encoding="utf-8"))

def f2(x): return ("%+.2f" % x) if x is not None else "—"
def f1(x): return ("%+.1f" % x) if x is not None else "—"
def cls(x): return "up" if (x or 0) > 0 else "down"

# ---------------- SVG helpers (同B1) ----------------
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
        out.append('<text x="%d" y="%.1f" font-size="11" fill="#6b7280" text-anchor="end">%.2f</text>' % (pl-6, y+4, v))
    for i in range(0, len(xs_all), 4):
        out.append('<text x="%.1f" y="%d" font-size="10.5" fill="#6b7280" text-anchor="middle">%s</text>' % (X(i), height-10, xs_all[i]))
    if vline:
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
    lx = pl
    seen = []
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
        col = "#e74c3c" if edges[i] >= 0 else "#27ae60"
        op = 0.92 if edges[i] >= 0 else 0.75
        out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s" opacity="%.2f"/>' % (x+0.5, pt+ih-h, bw-1, h, col, op))
        if i % 3 == 0:
            out.append('<text x="%.1f" y="%d" font-size="10" fill="#6b7280" text-anchor="middle">%d</text>' % (x+bw/2, height-22, edges[i]))
    out.append('<text x="%d" y="%d" font-size="11" fill="#6b7280">收益(4%%桶) →</text>' % (pl, height-6))
    out.append("</svg>")
    return "".join(out)

# ---------------- 直方图数据(从flat CSV) ----------------
t1v = []
with open(os.path.join(HERE, "b2_signals_flat.csv"), encoding="utf-8") as f:
    for r_ in csv.DictReader(f):
        t1v.append(float(r_["t1_20"]))
t1v = np.array(t1v)
edges = list(range(-68, 344, 4))
counts, _ = np.histogram(t1v, bins=edges)

# ---------------- 数据装配 ----------------
g = S["groups"]; ic = S["ic"]; nv = S["naive"]; dc = S["deciles"]
te = "202509"
months = sorted(EQ["B2_全A"].keys())

def row(label, d, bold=False):
    if not d: return ""
    st = ' style="font-weight:700"' if bold else ""
    t5 = f1(d["top5_share"]) + "%" if d.get("top5_share") is not None else "—"
    return ("<tr%s><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td><td>%s%%</td>"
            "<td>%s%%</td><td>%s%%</td><td>%s</td></tr>") % (
        st, label, "{:,}".format(d["n"]), cls(d["mean"]), f2(d["mean"]), cls(d["median"]), f2(d["median"]),
        f1(d["win"]), f1(d["p10"]), f1(d["p90"]), t5)

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

# 池内 cohort 表
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

# 真实NAV表
nav_rows = ""
for k, lbl in [("B2_全A", "B2信号（全A，日度等权再平衡）"), ("B2_强信号", "B2强信号(score≥7)"),
               ("B1_全A", "B1信号（对照）"), ("B2_池内", "B2信号（池内）"), ("B1_池内", "B1信号（池内，对照）"),
               ("基准_等权全A", "等权全A（主基准）"), ("基准_等权池内", "等权池内（含后视偏差）")]:
    v = N[k]["metrics"]
    nav_rows += ("<tr><td>%s</td><td class='%s'>%s%%</td><td>%s%%</td><td>%s</td><td>%s%%</td>"
                 "<td class='%s'>%s%%</td><td class='%s'>%s%%</td></tr>") % (
        lbl, cls(v["全部"]["total"]), f1(v["全部"]["total"]), f1(v["全部"]["ann"]), f2(v["全部"]["sharpe"]),
        f1(v["全部"]["maxdd"]), cls(v["训练期"]["total"]), f1(v["训练期"]["total"]),
        cls(v["验证期"]["total"]), f1(v["验证期"]["total"]))

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
    regime_groups.append((k, [("B2信号", "#e74c3c", v["sig_mean"]), ("同regime基准", "#2563eb", v["base_mean"])]))

# 配对归因表
pa = S["paired"]
pair_rows = ("<tr><td>A · B1候选直接进（T+1 开盘）</td><td>%s</td><td class='up'>%s%%</td><td>%s%%</td><td>%s%%</td></tr>"
             "<tr><td>B · B2确认后进（T+2 开盘）</td><td>%s</td><td class='up'>%s%%</td><td>%s%%</td><td>%s%%</td></tr>"
             "<tr style='background:#f0f3f7'><td><strong>B − A（等一天再进的代价/收益）</strong></td><td>—</td>"
             "<td><strong>%s pct</strong></td><td colspan='2' class='muted'>训练期 %s pct ｜ 验证期 %s pct</td></tr>") % (
    "{:,}".format(pa["A_B1候选直接进"]["n"]), f2(pa["A_B1候选直接进"]["mean"]), f2(pa["A_B1候选直接进"]["median"]), f1(pa["A_B1候选直接进"]["win"]),
    "{:,}".format(pa["B_B2确认后进"]["n"]), f2(pa["B_B2确认后进"]["mean"]), f2(pa["B_B2确认后进"]["median"]), f1(pa["B_B2确认后进"]["win"]),
    f2(pa["B_B2确认后进"]["mean"]-pa["A_B1候选直接进"]["mean"]),
    f2(M["decompose"]["分期_训练期"]["被确认_T2入场"]-M["decompose"]["分期_训练期"]["未确认_T2入场"]),
    f2(M["decompose"]["分期_验证期"]["被确认_T2入场"]-M["decompose"]["分期_验证期"]["未确认_T2入场"]))

# 分解表
dec = M["decompose"]
dec_rows = ""
for k, lbl in [("被确认_候选日入场", "被确认候选 · 候选日 T+1 开盘入场"), ("被确认_确认日K线内(d0_intra)", "├ 确认日 K 线内（T+1 开→收）"),
               ("被确认_剔除确认日(T+1收->T+21收)", "├ 剔除确认日（T+1 收→T+21 收）"), ("被确认_B2口径(T+2入场)", "└ B2 口径（T+2 开盘入场）"),
               ("未被确认_候选日入场", "未被确认候选 · 候选日 T+1 开盘入场"), ("未被确认_非确认日K线内(d0_intra)", "├ 非确认日 K 线内（T+1 开→收）"),
               ("未被确认_剔除确认日(T+1收->T+21收)", "├ 剔除确认日（T+1 收→T+21 收）"), ("未被确认_同T+2入场", "└ 同 T+2 开盘入场（干净对照）")]:
    v = dec[k]
    strong = " style='font-weight:700'" if k in ("被确认_B2口径(T+2入场)", "未被确认_同T+2入场") else ""
    dec_rows += "<tr%s><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td><td>%s%%</td></tr>" % (
        strong, lbl, "{:,}".format(v["n"]), cls(v["mean"]), f2(v["mean"]), cls(v["median"]), f2(v["median"]), f1(v["win"]))
w = dec["增量检验_被确认T2入场_vs_未确认T2入场"]

# 质量分层表
q_rows = ""
for k, kn in [("vr5", "确认日量比 vr5"), ("chg", "确认日涨幅"), ("upper_shadow", "确认日上影线"),
              ("j_confirm", "确认日 J 值"), ("b2_score", "b2_score 得分")]:
    q_rows += "<tr style='background:#f8fafc'><td colspan='6'><strong>%s</strong></td></tr>" % kn
    for lb, e in M["quality"][k].items():
        q_rows += "<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
            lb, "{:,}".format(e["n"]), cls(e["mean"]), f2(e["mean"]),
            f2(e["训练期"]) if "训练期" in e else "—", f2(e["验证期"]) if "验证期" in e else "—",
            "%.0f%%" % e["win"] if e.get("win") is not None else "—")

# pre60 分层
cp_rows = ""
for k, kn in [("pre20", "确认日前 20 日涨幅"), ("pre60", "确认日前 60 日涨幅"),
              ("dd20", "确认日 20 日回撤"), ("cand_score", "候选日 b1_score")]:
    cp_rows += "<tr style='background:#f8fafc'><td colspan='6'><strong>%s</strong></td></tr>" % kn
    for lb, e in M["counterparty"][k].items():
        cp_rows += "<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
            lb, "{:,}".format(e["n"]), cls(e["mean"]), f2(e["mean"]),
            f2(e["训练期"]) if "训练期" in e else "—", f2(e["验证期"]) if "验证期" in e else "—",
            "%.0f%%" % e["win"] if e.get("win") is not None else "—")

# 路径表
pt = M["path"]
path_rows = ("<tr><td>20日最大不利偏移 MAE（中位 / 均值）</td><td class='down'>%s%% / %s%%</td><td class='down'>%s%% / %s%%</td><td class='down'>%s%% / %s%%</td></tr>"
             "<tr><td>20日最大有利偏移 MFE（中位 / 均值）</td><td>%s%% / %s%%</td><td>%s%% / %s%%</td><td class='up'>%s%% / %s%%</td></tr>"
             "<tr><td>前 5 日最低（中位）</td><td>%s%%</td><td>%s%%</td><td>—</td></tr>"
             "<tr><td>后 15 日最低（中位）</td><td>%s%%</td><td>—</td><td>—</td></tr>"
             "<tr><td>持仓低于成本天数（中位，/20日）</td><td>%s</td><td>%s</td><td>—</td></tr>"
             "<tr><td>MAE 发生时点（前5日 / 5-10日 / 10-20日）</td><td colspan='3'>%s%% / %s%% / %s%%（平均第 %.1f 日）</td></tr>") % (
    f2(pt["B2信号"]["mae20"]["median"]), f2(pt["B2信号"]["mae20"]["mean"]),
    f2(pt["B1候选"]["mae20"]["median"]), f2(pt["B1候选"]["mae20"]["mean"]),
    f2(pt["全A基准"]["mae20"]["median"]), f2(pt["全A基准"]["mae20"]["mean"]),
    f2(pt["B2信号"]["mfe20"]["median"]), f2(pt["B2信号"]["mfe20"]["mean"]),
    f2(pt["B1候选"]["mfe20"]["median"]), f2(pt["B1候选"]["mfe20"]["mean"]),
    f2(pt["全A基准"]["mfe20"]["median"]), f2(pt["全A基准"]["mfe20"]["mean"]),
    f2(pt["B2信号"]["first5_min"]["median"]), f2(pt["B1候选"]["first5_min"]["median"]),
    f2(pt["B2信号"]["mae后15日"]["median"]),
    "%.0f" % pt["B2信号"]["below_days"]["median"], "%.0f" % pt["B1候选"]["below_days"]["median"],
    "%.1f" % pt["B2信号"]["mae_day_hist"]["前5日"], "%.1f" % pt["B2信号"]["mae_day_hist"]["5-10日"],
    "%.1f" % pt["B2信号"]["mae_day_hist"]["10-20日"], pt["B2信号"]["mae_day_hist"]["mean_day"])

# vr5 表
v5_rows = ""
for k, v in M["vr5_trade"].items():
    if k == "秩相关": continue
    v5_rows += "<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td>%s%%</td><td class='%s'>%s%%</td><td>%s%%</td><td class='%s'>%s%%</td></tr>" % (
        k, "{:,}".format(v["n"]), cls(v["前5日"]), f2(v["前5日"]), f1(v["前5日_胜率"]),
        cls(v["后15日"]), f2(v["后15日"]), f1(v["后15日_胜率"]), cls(v["合计"]), f2(v["合计"]))
vr5 = M["vr5_trade"]["秩相关"]
vr_rows = ""
for k, v in M["volratio"].items():
    vr_rows += "<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td></tr>" % (k, "{:,}".format(v["n"]), cls(v["mean"]), f2(v["mean"]))

# 图表: 增量分解
dec_groups = [
    ("候选日入场视角", [("被确认", "#e74c3c", dec["被确认_候选日入场"]["mean"]), ("未确认", "#2563eb", dec["未被确认_候选日入场"]["mean"])]),
    ("确认日K线内", [("被确认", "#e74c3c", dec["被确认_确认日K线内(d0_intra)"]["mean"]), ("未确认", "#2563eb", dec["未被确认_非确认日K线内(d0_intra)"]["mean"])]),
    ("剔除确认日", [("被确认", "#e74c3c", dec["被确认_剔除确认日(T+1收->T+21收)"]["mean"]), ("未确认", "#2563eb", dec["未被确认_剔除确认日(T+1收->T+21收)"]["mean"])]),
    ("同T+2入场(真实)", [("被确认", "#e74c3c", dec["被确认_B2口径(T+2入场)"]["mean"]), ("未确认", "#2563eb", dec["未被确认_同T+2入场"]["mean"])]),
]
# 图表: 质量分层
qual_groups = []
for k, kn in [("vr5", "量比vr5"), ("chg", "确认日涨幅"), ("b2_score", "b2_score")]:
    qual_groups.append((kn, [(lb, "#e74c3c", e["mean"]) for lb, e in M["quality"][k].items()]))
# 图表: pre60
pre60_groups = [(lb, [("B2信号", "#e74c3c", e["mean"])]) for lb, e in M["counterparty"]["pre60"].items()]

html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>B2因子深度评估报告 · 重构版（2026-10-19）</title>
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

<h1>B2 因子深度评估报告 <span class="tag tag-amber">重构版 · 第二章</span></h1>
<div class="sub">生成日期：2026-10-19 ｜ 方法：A股量化研报复现专家团 Workflow B（因子/信号有效性检验）｜ 数据：kline_cache_all.db（前复权日线，2023-06-01 ~ 2026-08-21，5014 只股票）｜ 验收：role-07 研报复现验收工程师｜ 配套：第一章《B1因子深度评估报告_重构版_20261019》</div>

<div class="warn">
<strong>验收结论先行（冷酷版）：</strong>B2（B1候选 + 次日放量阳线确认）作为独立买入信号，<strong>不通过</strong>——但比 B1 好一个档次。三重证据：① 信号层相对同宇宙基准训练期 −1.53 pct/20日、验证期 −0.67 pct，14 个参数变体相对 alpha <strong>全部为负</strong>；② 日度截面 RankIC = −0.0066（t = −4.54），信号组跑输非信号组 1.20 pct/日；③ 真实可投资口径（日度等权再平衡、含成本）全A 累计 <strong>+16.3%</strong> vs 等权全A 基准 <strong>+34.4%</strong>，池内 +97.3% vs 池内基准 +103.8%——B2 显著优于 B1 的全A −3.8%，但<strong>仍然跑输什么都不做</strong>。
</div>

<div class="key">
<strong>一句话给用户的答案：</strong>B2 的"放量阳线确认"表面上有 <span class="up">+4.10 pct</span> 的筛选增量（被确认候选 +2.92% vs 未确认 −1.18%），但增量分解显示其中 <strong>+3.75 pct 是确认日当天 K 线的日内动量</strong>（被确认组 T+1 开盘→收盘 +1.96% vs 未确认组 −1.79%）——这部分你 T+1 收盘前根本买不到；把入场对齐到同一时点（T+2 开盘）后，<strong>确认机制的真实增量只有 +0.19 pct，t = 1.86、p = 0.062 边缘不显著，且验证期反号为 −0.15 pct</strong>。确认环节的全部信息在确认日收盘时点就已经 price in，"新资金确认进场"的叙事不被数据支持。
</div>

<!-- ==================== 一、定义 ==================== -->
<h2>一、因子定义与参数（回源码逐行核对）</h2>
<div class="card">
<p>定义来源：<code>backtest_b2_all_vec.py</code>（B1 候选部分与第一章 B1 同源，已逐行核对一致）。B2 是 B1 的<strong>下游确认层</strong>：先在 T 日筛出 B1 候选（超卖 + 趋势线附近回调不破），再要求 <strong>T+1 日出现"放量阳线"</strong>才算有效信号。确认日 T+1 收盘后可知，<strong>T+2 开盘入场</strong>。</p>
<div class="formula">
<strong>B2（逆势确认）判定规则：</strong><br>
<strong>第一层 · T 日 B1 候选</strong>（与 B1 报告第一章完全一致）：KDJ(9,3,3) J &lt; 20 且最低价触及知行多空线 <code>(MA14+MA28+MA57+MA114)/4</code> 附近回收（low ≤ 线×1.001，close ≥ 线×0.99，low ≥ (线−0.5×0.0214×C)×0.995）且 <code>b1_score ≥ 5</code>（趋势强度加分制，含 MA 多头排列/价格位置/量价配合等 8 项）<br>
<strong>第二层 · T+1 日放量阳线确认</strong>（<code>b2_score ≥ 5</code>，≥7 为强信号）：<br>
　① 硬门禁：<strong>收盘 &gt; 开盘</strong>（非阳线直接 0 分淘汰）<br>
　② 基础 3 分 + 阳线 1 分；量比 vr5 = V/MA5(V) &gt; 1.2 加 2 分（否则加 1 分）<br>
　③ 确认日 J &lt; 55 加 1 分（不许已修复到高位）<br>
　④ 确认日涨幅 chg &gt; 3.9% 加 2 分 / &gt; 1.5% 加 1 分<br>
　⑤ 上影线 &lt; 1.0% 加 1 分（不许冲高回落）
</div>
<table>
<tr><th>参数</th><th>值</th><th>参数</th><th>值</th></tr>
<tr><td>候选层</td><td>B1 全条件 + b1_score≥5</td><td>确认层</td><td>b2_score≥5（强信号 ≥7）</td></tr>
<tr><td>确认日量比</td><td>vr5 = V/MA5(V) &gt; 1.2（+2分）</td><td>确认日涨幅</td><td>&gt;1.5% 起加分，&gt;3.9% 满分</td></tr>
<tr><td>信号日</td><td>确认日 T+1（收盘后可知）</td><td>入场 / 出场</td><td>T+2 开盘 / T+21 收盘（持有 20 个交易日）</td></tr>
<tr><td>信号总量（完整前瞻）</td><td>37,552 条（强 14,644）</td><td>候选总量</td><td>73,562 条（确认率 51.0%）</td></tr>
</table>
<p class="muted">重放校准：与旧 B2 信号 CSV 比对，38,277 条中 0 缺失、0 多余，仅 10 条因分数边界（b2_score 恰为 5 的临界）在数据漂移后翻转，管道校准通过。</p>
</div>

<!-- ==================== 二、九项口径 ==================== -->
<h2>二、九项口径声明（与 B1 报告完全一致）</h2>
<div class="card">
<table>
<tr><th>#</th><th>口径项</th><th>声明值</th></tr>
<tr><td>1</td><td>下期收益定义</td><td>确认日 T+1 收盘后可知 → T+2 开盘入场 → T+21 收盘出场（持有 20 个交易日）。B2 持有期比 B1 晚一个交易日入场，跨 B1/B2 对比时以配对归因（同一事件两路径）消除时点差</td></tr>
<tr><td>2</td><td>分组数</td><td>信号层十分位 + 五分层；另报 b2_score 分层、确认日质量分层、候选集配对分层</td></tr>
<tr><td>3</td><td>多空方向</td><td>仅多头；对照基准为同宇宙等权持有，不设空头腿</td></tr>
<tr><td>4</td><td>股票池过滤</td><td>剔除 3 只指数与 23 只名单外代码；<strong>未做</strong> ST / 涨跌停 / 停牌 / 上市日过滤（D2）</td></tr>
<tr><td>5</td><td>极值处理</td><td>不做 MAD 缩尾（收益率为原始值；分层用日度截面分位）</td></tr>
<tr><td>6</td><td>中性化</td><td><strong>不做</strong>行业 / 市值中性化（数据源不含市值与行业字段，D12）</td></tr>
<tr><td>7</td><td>基准</td><td>等权全A（5014 只，主基准）/ 等权池内（848 只，含后视偏差，上限参考）/ 中证500 / 上证（仅 2025-09 起）</td></tr>
<tr><td>8</td><td>无风险利率</td><td>0（月度夏普 = 均值/标准差×√12，rf=0 口径）</td></tr>
<tr><td>9</td><td>费率</td><td>基准口径不计成本；计费口径：佣金 0.00025 双边 + 印花税 0.0005 卖出 + 滑点 0.001 双边 ≈ 单次往返 0.30%</td></tr>
</table>
<p class="muted">训练/验证切分：确认日 &lt; 2025-09-08 为训练期（24,532 条），≥ 为验证期（13,020 条）。参数为既有生产参数直接沿用，本次不重新寻优。</p>
</div>

<!-- ==================== 三、信号层 ==================== -->
<h2>三、信号层有效性检验</h2>
<div class="card">
<h3>3.1 分组统计（T+2开盘 → T+21收盘，无止损，%）</h3>
<table>
<tr><th>分组</th><th>n</th><th>均值</th><th>中位</th><th>胜率</th><th>P10</th><th>P90</th><th>top5%盈亏贡献</th></tr>
__ROWS_GROUPS__
</table>
<div class="note"><strong>读法：</strong>与 B1 同型的彩票结构——均值 +1.05% 但中位数 −0.73%，盈利全靠右尾（top5% 贡献 +202% 总盈亏，验证期 top5% 贡献 −839%：赢家集体缺席）。验证期胜率从 49.9% 塌到 40.8%。池内信号（+3.10%）明显强于池外（+0.64%），与 B1 一致：alpha 在选股池，不在信号。</div>

<h3>3.2 日度截面 RankIC（对 20 日前向收益，全股票 5014 只）</h3>
<table>
<tr><th>因子形式</th><th>IC 均值</th><th>ICIR</th><th>IC&gt;0 占比</th><th>t 值</th><th>判定</th></tr>
<tr><td>B2 信号（0/1）</td><td class="down">−0.0066</td><td class="down">−0.190</td><td>40.2%</td><td class="down">−4.54</td><td><span class="tag tag-red">负且显著</span></td></tr>
<tr><td>b2_score（连续）</td><td class="down">−0.0141</td><td>−0.093</td><td>48.3%</td><td class="down">−1.66</td><td><span class="tag tag-red">负，边际显著</span></td></tr>
<tr><td>信号组−非信号组价差</td><td class="down">−1.20 pct/日</td><td>—</td><td>42.2%</td><td class="down">−5.40</td><td><span class="tag tag-red">负且显著</span></td></tr>
</table>
<div class="note"><strong>关键对比：</strong>b2_score 连续 IC 为负（−0.0141）意味着<strong>确认日"质量"越高，未来越差</strong>——量比越高、涨幅越大、上影线越短，未来 20 日收益反而越低。这不是噪声：强信号（score≥7，+1.43%）在训练期（+2.72%）好看，验证期塌到 −0.87%。"放量阳线越强越好"的直觉在截面上被直接否定。</div>

<h3>3.3 收益分布（n=37,552，4% 等宽桶）</h3>
__HIST__
<table>
<tr><th>十分位</th><th>收益区间</th><th>数量占比</th><th>盈亏贡献</th><th>组均值</th></tr>
__ROWS_DECILE__
</table>

<h3>3.4 核心配对归因：B1候选直接进 vs B2确认后进（同一事件集，n=73,562）</h3>
<table>
<tr><th>路径</th><th>n</th><th>均值%</th><th>中位%</th><th>胜率%</th></tr>
__ROWS_PAIR__
</table>
<table>
<tr><th>B1 候选集内部（T+1 开盘入场视角）</th><th>n</th><th>均值%</th><th>中位%</th><th>胜率%</th></tr>
__ROWS_CAND__
</table>
<div class="warn"><strong>这是全报告最重要的一张表。</strong>表面看 B2 筛选很神：被确认候选 +2.92% vs 未被确认 −1.18%（差 +4.10 pct）。但注意——<strong>"被确认"的定义就是 T+1 收阳线</strong>，而这 +4.10 pct 里绝大部分是确认日 K 线本身的日内动量：被确认组 T+1 开盘→收盘 <span class="up">+1.96%</span>，未确认组 <span class="down">−1.79%</span>，差 +3.75 pct。B2 的纪律是"T+1 收盘后确认、T+2 开盘才进"，这 +3.75 pct 里你能吃到的部分是零（T+2 入场价已经 price in）。把未确认组也放到同一入场时点（T+2 开盘）后：被确认 +1.05% vs 未确认 +0.86%，<strong>真实增量 +0.19 pct</strong>。 Welch t = 1.86，p = 0.062——连 10% 显著性都不过，且验证期反号。</div>
</div>

<!-- ==================== 四、组合层 ==================== -->
<h2>四、组合层绩效</h2>
<div class="card">
<h3>4.1 真实可投资口径：日度等权再平衡 NAV（含成本，本报告主口径）</h3>
<p class="muted">构造：每个交易日，将账户资金等权分配给当日处于"信号已触发且持有期内"的全部股票，每日再平衡；成本按 0.30%/20日/持仓 计提。这是唯一可以直接对比"策略 vs 基准"的干净口径。</p>
__EQUITY__
<table>
<tr><th>组合（日度等权再平衡，含成本）</th><th>累计%</th><th>年化%</th><th>夏普</th><th>最大回撤%</th><th>训练期%</th><th>验证期%</th></tr>
__ROWS_NAV__
</table>
<div class="warn"><strong>B2 确实比 B1 好，但都没跑赢基准。</strong>B2 全A +16.3%（夏普 0.20）vs B1 全A <span class="down">−3.8%</span>（夏普 −0.05）——确认层把 B1 从"稳定亏损"拉到"小幅盈利"；但同期等权全A <span class="up">+34.4%</span>（夏普 0.38）。验证期 B2 全A −12.4% vs 基准 +1.2%。池内场景同样输：B2 池内 +97.3% vs 池内基准 +103.8%（验证期 +28.1% vs +41.6%）。<strong>B2 的最好用法仍然不是信号，是反指候选池</strong>——但这是下一章的话题。</div>

<h3>4.2 月度 cohort 等权口径（与 B1 报告可比）</h3>
<table>
<tr><th>组合（全A信号，月度 cohort）</th><th>累计%</th><th>年化%</th><th>夏普</th><th>最大回撤%</th><th>训练期%</th><th>验证期%</th><th>计费口径%</th></tr>
__ROWS_STOP__
</table>
<table>
<tr><th>组合（池内信号，月度 cohort）</th><th>累计%</th><th>夏普</th><th>最大回撤%</th><th>训练期%</th><th>验证期%</th><th>计费口径%</th></tr>
__ROWS_POOL__
</table>
<div class="note"><strong>止损档读数：</strong>B2 与 B1 相反——不止损（+37.4%）优于全部止损档（8% +31.7% / 12% +21.4% / 15% +25.4% / 20% +26.7%），因为确认日已过滤掉部分急跌，且 B2 的右尾比 B1 厚。月度 cohort 口径注意 D13 聚合 artifact（同 B1 报告：月度等权放大低信号量灾月）。计费后不止损 +24.9%，仍低于等权全A 基准 +46.7%。</div>

<h3>4.3 止损与波动率失配（MAE 分布，n=37,552）</h3>
<table>
<tr><th>止损档</th><th>5%</th><th>8%</th><th>10%</th><th>12%</th><th>15%</th><th>20%</th><th>25%</th></tr>
<tr><td>信号 20 日内曾触及该回撤的比例</td><td>60.7%</td><td>43.5%</td><td>34.2%</td><td>26.4%</td><td>17.4%</td><td>8.0%</td><td>3.4%</td></tr>
</table>
<p>信号 20 日最大不利偏移（MAE）中位 <strong>−6.81%</strong>、均值 −8.63%——与 B1（中位 −6.0%）和全A 基准（中位 −6.6%）几乎相同。<strong>B2 的确认层没有改善持有体验</strong>：43.5% 的信号在 20 日内曾跌超 8%，8% 止损依旧每 10 笔打掉 4 笔多。B2 选出的不是"更稳的票"，只是"多涨了一天的票"。</p>

<h3>4.4 月度信号量与基准对照</h3>
<table>
<tr><th>口径</th><th>n</th><th>均值%</th><th>中位%</th><th>胜率%</th></tr>
__ROWS_NAIVE__
</table>
<p class="muted">B2 信号 +1.05% vs B1 候选 +0.91% vs B1 信号 +1.24% vs 全股票等权基准 +2.17%（n=241 万）——<strong>三层信号全部跑输"随便买"</strong>。B1 的 +1.24% 略高于 B2 的 +1.05%，因为 B1 样本里包含更多早期（训练期）信号；同窗口配对归因（3.4 节）才是干净对比。</p>
</div>

<!-- ==================== 五、稳健性 ==================== -->
<h2>五、稳健性检验</h2>
<div class="card">
<h3>5.1 参数敏感性（OFAT，相对基准 alpha，pct/20日）</h3>
<table>
<tr><th>变体</th><th>信号数</th><th>均值%</th><th>胜率%</th><th>训练期 rel</th><th>验证期 rel</th></tr>
__ROWS_OFAT__
</table>
<div class="note">14 个变体（b2_score 门槛 4~7、b1_score 门槛 4~6、量比门槛 1.0/1.5、涨幅门槛 (0.5/2.0)/(2.5/5.0)、确认日 J&lt;45/&lt;65、上影线 &lt;0.5%/&lt;2.0%、去掉阳线硬门禁），<strong>训练期与验证期相对 alpha 全部为负</strong>。两个细节：① 强信号（score≥7）训练期 rel −0.57 好于基准变体，但验证期 −1.27 反而更差——严格筛选在样本外有害；② 去掉阳线硬门禁后 n 从 3.76 万膨胀到 7.36 万（=全部候选），mean +0.96%——确认环节的全部贡献就是 +0.09 pct，与配对归因的 +0.19 pct 互为印证（差异来自 T+1 入场视角）。</div>
__REGIME_CHART__
<h3>5.2 市场状态条件（市场 20 日收益 / 波动率分位）</h3>
<table>
<tr><th>状态</th><th>信号 n</th><th>信号均值%</th><th>信号胜率%</th><th>基准 n</th><th>基准均值%</th><th>相对 alpha</th></tr>
__ROWS_REGIME__
</table>
<div class="note">B2 在所有四种市场状态下相对 alpha 均为负（上涨 −0.18 / 下跌 −1.91 / 低波 −0.44 / 高波 −1.46）。下跌市最差（−1.91）而 B2 信号在下跌市占比 24%——与 B1 同款"信号密度与有效性负相关"。值得注意的是 B2 在低波市信号胜率 55.4% 为正——这是唯一像 edge 的角落，但相对 alpha 仍为 −0.44，被基准的低波市强势（+3.55%）盖过。</div>
<h3>5.3 数据质量与截断</h3>
<table>
<tr><th>项目</th><th>数值</th><th>处理</th></tr>
<tr><td>右截断（末尾 20 日内信号无完整前瞻）</td><td>725 / 38,277</td><td>已排除于全部统计</td></tr>
<tr><td>幸存者偏差（退市股缺失）</td><td>73 只</td><td>不可修复；方向=高估历史收益</td></tr>
<tr><td>池内标记后视性（2026-08 池快照标注历史信号）</td><td>全部池内统计</td><td>登记 D3；池内外对比仅作参考</td></tr>
</table>
</div>

<!-- ==================== 六、微观假设 ==================== -->
<h2>六、微观经济假设检验（确认机制到底在确认什么）</h2>
<div class="card">
<p>B2 的微观叙事：<strong>H1</strong> 放量阳线 = 新资金确认进场，趋势未坏，确认后应延续；<strong>H2</strong> 赚"被确认日阳线吓下车的人"与"随后跟风上车者"的钱；<strong>H3</strong> 被"候选日附近埋伏更早的人"（确认日出货）或"认为确认=动能耗尽的人"赚钱。</p>

<h3>6.1 确认日质量分层：质量刻度没有判别力（H1 证伪）</h3>
__QUAL_CHART__
<table>
<tr><th>分层</th><th>n</th><th>均值%</th><th>训练期%</th><th>验证期%</th><th>胜率%</th></tr>
__ROWS_QUAL__
</table>
<div class="warn"><strong>三个反直觉事实：</strong>① 78% 的 B2 信号确认日量比其实 <strong>&lt; 1.2</strong>（靠阳线+低 J+低上影线等其他加分项进门），且该组（+0.95%）与真放量组（1.2~2.0，+1.48%）差距有限、验证期全部归零——<strong>"放量"这个核心特征根本不是必需条件</strong>；② 确认日涨幅 &gt;7% 组未来 20 日 <span class="down">−0.02%</span>（n=1,289），涨幅最大的确认反而最差——大阳线是兑现信号不是启动信号；③ b2_score 7 分组（+1.64%）好于 ≥8 分组（+1.03%），<strong>非单调</strong>——如果 score 在测"确认质量"，不应出现中间最好。全部 5 组质量刻度在验证期的均值都 ≤ +0.54%。</div>

<h3>6.2 确认机制增量分解：+4.10 pct 表观增量的 91% 是确认日日内动量</h3>
__DEC_CHART__
<table>
<tr><th>分解项</th><th>n</th><th>均值%</th><th>中位%</th><th>胜率%</th></tr>
__ROWS_DEC__
</table>
<div class="warn"><strong>H1 证伪的量化版本。</strong>把 +4.10 pct 的表观增量（+2.92 vs −1.18）拆开：确认日 K 线内（T+1 开盘→收盘，被确认 +1.96% vs 未确认 −1.79%，差 <strong>+3.75 pct</strong>）——这是定义使然的日内动量，B2 的 T+2 入场纪律<strong>主动放弃</strong>了它；确认日收盘之后的延续（T+1 收→T+21 收，被确认 +0.98% vs 未确认 +0.79%，差 <strong>+0.19 pct</strong>，t=1.86、p=0.062 边缘不显著；训练期 +0.20、验证期 −0.15 反号）。也就是说：<strong>"放量阳线确认之后趋势延续"在数据里只剩 +0.19 pct 且不稳定</strong>，确认环节的全部有效信息在确认日收盘时点已 price in。若一定要赚这 +3.75 pct，就必须在 T+1 盘中追阳线——那是完全不同的策略（日内动量），不是 B2。</div>

<h3>6.3 对手方分层：60 日涨幅超 20% 后确认信号失效（H3 量化）</h3>
__PRE60_CHART__
<table>
<tr><th>分层</th><th>n</th><th>均值%</th><th>训练期%</th><th>验证期%</th><th>胜率%</th></tr>
__ROWS_CP__
</table>
<div class="warn"><strong>H3 的"被谁赚钱"得到清晰量化：</strong>确认日前 60 日涨幅 &lt;20% 时 B2 有效（+1.64%/+1.84%，且训练验证两期同号），超过 20% 后断崖式转负（20-50% 层 −0.85%、50-100% 层 −3.50%、&gt;100% 层 −3.24%，<strong>且训练期就已经是负的</strong>）——这不是样本外衰减，是结构性的"高位确认=接盘"。你确实在"被更早参与趋势的人赚钱"，而且 B2 系统性地在趋势最疲劳处下注最重（前 60 日涨幅中位数 +14.4%，20% 以上占 24%）。pre20 层同样单调转差（40-80% 层 −4.36%，n=16 样本小）。cand_score（候选日质量）五个分层平坦（+0.95%~+1.29%）——<strong>候选层打分无用，真正的分界在"已经涨了多少"</strong>。</div>

<h3>6.4 持仓路径：B2 没有改善任何持有体验</h3>
<table>
<tr><th>路径指标（T+2 开盘基准，中位数）</th><th>B2 信号</th><th>B1 候选</th><th>全A 基准</th></tr>
__ROWS_PATH__
</table>
<p>B2 的 MAE 中位 −6.81% 与 B1 候选 −6.80%、全A 基准 −6.60% 几乎一致；MFE 中位 +7.38% 反而略低于基准的 +8.35%——<strong>向上空间不如随机，向下暴露一模一样</strong>。MAE 发生时点：49.1% 的信号的最深回撤发生在第 10~20 日（平均第 9.2 日），是<strong>阴跌型</strong>而非急杀型——固定止损在这种路径上反复被磨。持仓 20 日中有 10 日收盘低于成本（中位）。</p>

<h3>6.5 量能事前性复测：确认日放量对未来零预测（H2 证伪）</h3>
<table>
<tr><th>确认日 vr5 五分位</th><th>n</th><th>前5日%</th><th>前5胜率%</th><th>后15日%</th><th>后15胜率%</th><th>合计%</th></tr>
__ROWS_V5__
</table>
<p>确认日量比 vr5 与未来收益的秩相关：对前 5 日 <strong>+0.005</strong>、对后 15 日 <strong>+0.003</strong>——全部归零（对比 B1 报告的 vol5：对前 5 日 +0.377 但第 6 天起归零；B2 连前 5 日的可测性都没有，因为确认日本身就要求放量，组内 vr5 方差被压缩）。事后的跟风量分层（vol_ratio = 后 20 日/前 20 日均量）呈强单调：&lt;0.8 组 <span class="down">−4.44%</span> vs &gt;1.6 组 <span class="up">+15.61%</span>——但这是<strong>与收益同窗口的事后指标</strong>，信号当日不可得，无法转化为规则（与 B1 报告 6.3 节同结论）。</p>
<table>
<tr><th>事后跟风量 vol_ratio 分层</th><th>n</th><th>均值%</th></tr>
__ROWS_VR__
</table>
<div class="note"><strong>H2 判定：证伪。</strong>"赚被洗走者与跟风者的钱"要求放量确认能事前识别"谁会跟风"。实测确认日放量程度对未来收益的预测力为零（秩相关 +0.005），而真正与收益强相关的事后放量又不可事前获得。B2 不是把对手方的钱赚了，而是在一个对手方行为<strong>恰好不可测</strong>的时点入场。</div>

<h3>6.6 入场滑点：不是问题</h3>
<p>T+2 开盘相对 T+1 收盘的跳空：均值 <strong>−0.07%</strong>、中位 0.00%、P10 −0.98%、P90 +0.79%；高开 &gt;1% 占 7.6%、低开 &gt;1% 占 9.5%，与 t1_20 秩相关 +0.027（不显著）。<strong>B2 的 T+2 入场纪律几乎没有滑点代价</strong>——这反过来说明：放弃 +3.75 pct 的确认日日内动量去买一个 +0.19 pct 的不显著增量，性价比极差。</p>
</div>

<!-- ==================== 七、偏离登记 ==================== -->
<h2>七、偏离登记（继承 B1 报告 D1-D15，B2 相关项加注）</h2>
<div class="card">
<table>
<tr><th>#</th><th>偏离项</th><th>影响面</th><th>严重度</th></tr>
<tr><td>D1</td><td>ATR 为常数代理（0.0214×收盘价）——继承 B1</td><td>信号触发集合</td><td>中</td></tr>
<tr><td>D2</td><td>无交易状态过滤（ST/涨跌停/停牌/次新）——继承 B1；B2 确认日阳线若为涨跌停板则不可买入，T+2 开盘假设在涨停延续时不可成交</td><td>收益高估</td><td>高</td></tr>
<tr><td>D3</td><td>池内标记后视性（2026-08 池快照）——继承 B1</td><td>池内外全部对比含选择偏差</td><td>高</td></tr>
<tr><td>D7</td><td>幸存者偏差（73 只退市股缺失）——继承 B1</td><td>历史收益系统性高估</td><td>中</td></tr>
<tr><td>D9</td><td>持有期口径：B2 为 T+2 开盘→T+21 收盘，比 B1 晚一日入场；本报告所有 B1/B2 对比均用配对归因（同一事件两路径）消除时点差</td><td>跨章对比</td><td>中</td></tr>
<tr><td>D11</td><td>止损成交假设（最低价触及即成交，不处理跳空/涨跌停）——继承 B1</td><td>极端日收益虚高</td><td>中</td></tr>
<tr><td>D12</td><td>无中性化（无市值/行业字段）——继承 B1</td><td>IC 解释</td><td>中</td></tr>
<tr><td>D13</td><td>月度 cohort 等权口径为聚合 artifact——继承 B1；本报告 4.1 节另给日度等权再平衡真实口径</td><td>组合层数字</td><td>中</td></tr>
<tr><td>D16</td><td>B2 为 B1 的下游：B2 的全部统计条件于 B1 候选集，B2 无效不构成对 B1 的独立证伪（两者共享 J&lt;20+触线+不破的条件）</td><td>结论外推</td><td>中</td></tr>
<tr><td>D17</td><td>确认日 K 线内收益（+1.96%/−1.79%）为 T+1 开盘→收盘口径，未扣除 T+1 开盘跳空；若 T+1 开盘跳空显著，"日内动量"中有一部分是隔夜动量而非纯日内</td><td>6.2 节分解归因</td><td>低</td></tr>
</table>
</div>

<!-- ==================== 八、验收 ==================== -->
<h2>八、验收结论（role-07 研报复现验收工程师）</h2>
<div class="card">
<h3>8.1 前视偏差检查（9 条，同 B1 口径）</h3>
<table>
<tr><th>#</th><th>检查项</th><th>结论</th></tr>
<tr><td>1</td><td>成交时点晚于信号（T+2 开盘 vs T+1 收盘出信号）</td><td><span class="tag tag-green">通过</span> B2 的确认纪律比 B1 还保守一日</td></tr>
<tr><td>2</td><td>财报/持仓披露滞后</td><td><span class="tag tag-green">不适用</span>（纯量价）</td></tr>
<tr><td>3</td><td>复权使用未来信息</td><td><span class="tag tag-amber">附注通过</span> 同 B1 报告</td></tr>
<tr><td>4</td><td>截面排名使用全样本</td><td><span class="tag tag-green">通过</span> IC 为日度独立截面</td></tr>
<tr><td>5</td><td>滚动窗口未 shift(1)</td><td><span class="tag tag-green">通过</span> vr5/chg/J/上影线均仅用 ≤T+1 日数据</td></tr>
<tr><td>6</td><td>标签跨进测试期</td><td><span class="tag tag-green">通过</span> 右截断信号（725 条）已剔除</td></tr>
<tr><td>7</td><td>标准化使用全样本统计量</td><td><span class="tag tag-green">通过</span> 分层用每日截面分位</td></tr>
<tr><td>8</td><td>停牌/涨跌停事后可知</td><td><span class="tag tag-red">未处理</span> 登记 D2/D11，B2 场景更尖锐（确认日常见涨停）</td></tr>
<tr><td>9</td><td>因子值使用未来数据</td><td><span class="tag tag-green">通过</span></td></tr>
</table>
<h3>8.2 切分方式审查</h3>
<p>固定时间切分（确认日 &lt; 2025-09-08 训练，≥ 验证）——<span class="tag tag-green">可接受</span>；生产参数直接沿用，无全样本寻参风险——<span class="tag tag-green">通过</span>。</p>
<h3>8.3 最终结论</h3>
<div class="warn">
<strong>复现过程：通过。</strong>与旧 B2 CSV 比对 0 缺失 0 多余（仅 10 条分数边界翻转）；B1 候选集与第一章 B1 管道同源一致。<br><br>
<strong>因子有效性：不通过。</strong>B2 相对同宇宙基准的 alpha 在训练期、验证期、14 个参数变体、四种市场状态上一致为负；IC 负且显著；计费组合层跑输等权全A；核心的"确认机制增量"经分解后仅 +0.19 pct 且边缘不显著、验证期反号。<br><br>
<strong>B2 好于 B1 的地方（如实记录）：</strong>① 真实 NAV 口径 +16.3% vs B1 −3.8%——确认层确实把 B1 从稳定亏损拉到小幅盈利；② 计费 cohort +24.9% vs B1 +1.3%；③ 池内信号（+3.10%）比池外（+0.64%）高 2.5 倍，选股池的 alpha 被 B2 比被 B1 更好地保留了。但三者均未达到"跑赢等权基准"的最低门槛。<br><br>
<strong>必须修复项（若继续研究 B2 家族）：</strong>① 补 D2 交易状态过滤（尤其确认日涨停场景）后重跑；② 把 6.3 节发现的"前 60 日涨幅 &lt;20%"作为硬门禁加到 B2 上重测（这是本报告唯一结构正负分明的发现，训练期即成立）；③ 若要赚确认日日内动量（+3.75 pct），应作为独立的日内/隔夜策略单独回测，不要挂在 B2 名下。
</div>
</div>

<!-- ==================== 九、修复路径 ==================== -->
<h2>九、基于数据的修复路径（按优先级）</h2>
<div class="card">
<table>
<tr><th>优先级</th><th>动作</th><th>依据</th></tr>
<tr><td><strong>A（立即）</strong></td><td>停止把 B2 当独立买入信号。若要用，降级为<strong>选股池的过滤器候选</strong>：B2 池内 cohort +168% / 计费 +145.6% 说明池内信号密度和质量都可接受，但必须与"等权持有池子"（+133.2%）做严格配对比较——当前证据是池内 B2 略胜 cohort 口径、略输真实 NAV 口径，未达可用门槛</td><td>4.1 / 4.2 节</td></tr>
<tr><td><strong>B（最有希望的新实验）</strong></td><td><strong>前 60 日涨幅硬门禁</strong>：B2 条件不变，追加 pre60 &lt; 20%（约 87% 的信号保留，剔除最差的 24%）。该分层训练期即成立（+1.84% vs −0.85%），不是样本外过拟合的产物。这是 B2 家族唯一可能翻正的修改，且与第一章 B1 的 H1 证伪结论完全一致（趋势越强越不能追）</td><td>6.3 节</td></tr>
<tr><td><strong>C（独立课题）</strong></td><td>确认日日内动量的单独研究：被确认组 T+1 开盘→收盘 +1.96% 而 T+2 起只剩 +0.19 pct 不显著增量——若"超卖后首根放量阳线"的日内/隔夜动量可独立验证（T+1 尾盘或 T+2 集合竞价买入），那是新策略不是 B2</td><td>6.2 节</td></tr>
<tr><td><strong>D（口径基建）</strong></td><td>① 补 D2/D3 后全量重跑；② 统一 B1/B2 对比口径（配对归因优先，NAV 口径裁决）；③ walk-forward 闸门：任何修复版需连续两段样本外为正才考虑实盘（当前 B2 验证期 −12.4%，远未达到）</td><td>8.3 节</td></tr>
</table>
<div class="note"><strong>方法论备注：</strong>本报告所有基准均为同构造组合（同窗口、同入场出场、同聚合口径、同费率），B2 与基准的唯一差异是选股规则。配对归因（3.4 / 6.2 节）使用同一事件的两种路径，彻底消除了"被确认 vs 未确认"的人群选择差异中除入场时点外的全部因素。</div>
</div>

<div class="sub" style="margin-top:28px">数据与代码：b1_research/（b2_prep.py 特征预处理 → b2_verify.py 校准 → b2_analysis_signal.py / b2_analysis_port.py / b2_analysis_nav.py / b2_analysis_robust.py / b2_analysis_micro.py 五项检验 → 本报告）。全部数字可由上述脚本一键复现。<br>本报告为研究方法论与因子检验，不构成投资建议。</div>

</div>
</body>
</html>
"""

# ---- 填表 ----
def rows_groups():
    order = [("全体信号", "全体", True), ("训练期(<2025-09-08)", "训练期", False), ("验证期(≥2025-09-08)", "验证期", False),
             ("强信号(score≥7)", "强信号(score>=7)", False), ("普通信号(score5-6)", "普通信号(score5-6)", False),
             ("池内", "池内", False), ("池内-训练", "池内-训练", False), ("池内-验证", "池内-验证", False),
             ("池外", "池外", False), ("池外-训练", "池外-训练", False), ("池外-验证", "池外-验证", False)]
    return "\n".join(row(lbl, g[k]) for lbl, k, b in order)

def rows_decile():
    return "\n".join("<tr><td>%s</td><td>%s</td><td>%s%%</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td></tr>" % (
        k, k[3:], f1(v["share"]), cls(v["pnl_share"]), f1(v["pnl_share"]), cls(v["mean"]), f2(v["mean"]))
        for k, v in dc.items())

def rows_pair():
    return pair_rows

def rows_cand():
    cs = S["cand_split"]
    out = ""
    for k, lbl in [("被B2确认", "被 B2 确认（T+1 开盘入场视角）"), ("未被确认", "未被确认（T+1 开盘入场视角）"),
                   ("未被确认_T2入场", "未被确认（改 T+2 开盘入场 = 干净对照）")]:
        v = cs[k]
        strong = " style='font-weight:700'" if k == "未被确认_T2入场" else ""
        out += "<tr%s><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td><td>%s%%</td></tr>" % (
            strong, lbl, "{:,}".format(v["n"]), cls(v["mean"]), f2(v["mean"]), cls(v["median"]), f2(v["median"]), f1(v["win"]))
    out += ("<tr style='background:#fdecea'><td><strong>确认机制真实增量（同 T+2 入场）</strong></td><td>—</td>"
            "<td><strong class='up'>%s pct</strong></td><td colspan='2' class='muted'>t=%.2f, p=%.3f（边缘不显著）｜ 训练期 %s pct ｜ 验证期 %s pct（反号）</td></tr>") % (
        f2(w["diff"]), w["t"], w["p"],
        f2(M["decompose"]["分期_训练期"]["被确认_T2入场"]-M["decompose"]["分期_训练期"]["未确认_T2入场"]),
        f2(M["decompose"]["分期_验证期"]["被确认_T2入场"]-M["decompose"]["分期_验证期"]["未确认_T2入场"]))
    return out

def rows_naive():
    order = ["B2信号(基准)", "B1候选(J<20+触多空线+score>=5)", "B1信号(触任意线)", "全股票等权基准", "池内全股票等权基准"]
    return "\n".join("<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td><td>%s%%</td></tr>" % (
        k, "{:,}".format(nv[k]["n"]), cls(nv[k]["mean"]), f2(nv[k]["mean"]),
        cls(nv[k]["median"]), f2(nv[k]["median"]), f1(nv[k]["win"])) for k in order)

html = (html
    .replace("__ROWS_GROUPS__", rows_groups())
    .replace("__ROWS_DECILE__", rows_decile())
    .replace("__ROWS_PAIR__", rows_pair())
    .replace("__ROWS_CAND__", rows_cand())
    .replace("__ROWS_NAIVE__", rows_naive())
    .replace("__ROWS_STOP__", stop_rows)
    .replace("__ROWS_POOL__", pool_rows)
    .replace("__ROWS_NAV__", nav_rows)
    .replace("__ROWS_OFAT__", ofat_rows)
    .replace("__ROWS_REGIME__", regime_rows)
    .replace("__ROWS_QUAL__", q_rows)
    .replace("__ROWS_DEC__", dec_rows)
    .replace("__ROWS_CP__", cp_rows)
    .replace("__ROWS_PATH__", path_rows)
    .replace("__ROWS_V5__", v5_rows)
    .replace("__ROWS_VR__", vr_rows)
    .replace("__HIST__", hist_chart(edges, list(counts)))
    .replace("__EQUITY__", line_chart([
        ("B2池内", "#d97706", EQ["B2_池内"]),
        ("池内基准", "#b45309", EQ["基准_等权池内"]),
        ("B2全A", "#e74c3c", EQ["B2_全A"]),
        ("B1全A", "#9ca3af", EQ["B1_全A"]),
        ("全A基准", "#2563eb", EQ["基准_等权全A"]),
    ], vline=("20250908" if "20250908" in EQ["B2_全A"] else "20250909"), vlabel="训练|验证", ymin=0.55, ymax=2.1))
    .replace("__REGIME_CHART__", bar_chart(regime_groups))
    .replace("__QUAL_CHART__", bar_chart(qual_groups))
    .replace("__DEC_CHART__", bar_chart(dec_groups))
    .replace("__PRE60_CHART__", bar_chart(pre60_groups))
)

from healthcheck_chapter import inject
html = inject(html, "B2")

out = os.path.join(HERE, "B2因子深度评估报告_重构版_20261019.html")
open(out, "w", encoding="utf-8").write(html)
print("written:", out, "%.1f KB" % (len(html)/1024))
