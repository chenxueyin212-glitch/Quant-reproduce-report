# -*- coding: utf-8 -*-
"""B1因子重构评估报告生成器 — 输出 standalone HTML (内联SVG图表, 无外部依赖)"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "b1_signal_results.json"), encoding="utf-8"))
P = json.load(open(os.path.join(HERE, "b1_port_results.json"), encoding="utf-8"))
PP = json.load(open(os.path.join(HERE, "b1_pool_port.json"), encoding="utf-8"))
R = json.load(open(os.path.join(HERE, "b1_robust_results.json"), encoding="utf-8"))
M = json.load(open(os.path.join(HERE, "b1_micro_results.json"), encoding="utf-8"))
V5 = json.load(open(os.path.join(HERE, "b1_micro_vol5.json"), encoding="utf-8"))
C = json.load(open(os.path.join(HERE, "b1_chart_data.json"), encoding="utf-8"))

def f2(x): return ("%+.2f" % x) if x is not None else "—"
def f1(x): return ("%+.1f" % x) if x is not None else "—"
def pct(x, d=1): return ("%+.*f" % (d, x)) if x is not None else "—"
def cls(x): return "up" if (x or 0) > 0 else "down"

# ---------------- SVG helpers ----------------
def line_chart(series, width=940, height=300, pad=(56, 18, 34, 60), ymin=None, ymax=None, vline=None, vlabel=None):
    """series: list of (name, color, {ym: value})"""
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

def bar_chart(groups, width=940, height=None, pad=(56, 14, 30, 60), hline0=True, fmt="%+.1f", label_rot=False):
    """groups: list of (label, [(series_name, color, value), ...]) — 分组柱状"""
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
    # legend
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

# ---------------- 数据装配 ----------------
g = S["groups"]; ic = S["ic"]; nv = S["naive"]; dc = S["deciles"]; lp = S["legacy_period"]; lw = S["legacy_window_pool"]
eq = C["equity"]; dd = C["drawdown"]; hist = C["hist"]
months = sorted(eq["B1_8pct"].keys())
te = "202509"  # 训练/验证分界月(信号日<20250908为训练)

def row(label, d, bold=False):
    if not d: return ""
    st = ' style="font-weight:700"' if bold else ""
    t5 = f1(d["top5_share"]) + "%" if d.get("top5_share") is not None else "—"
    return ("<tr%s><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td><td>%s%%</td>"
            "<td>%s%%</td><td>%s%%</td><td>%s</td></tr>") % (
        st, label, "{:,}".format(d["n"]), cls(d["mean"]), f2(d["mean"]), cls(d["median"]), f2(d["median"]),
        f1(d["win"]), f1(d["p10"]), f1(d["p90"]), t5)

# 止损表(full-A)
stop_rows = ""
for k in ["不止损", "止损8%", "止损12%", "止损15%", "止损20%"]:
    v = P["results"][k]; a = v["portfolio_all"]; per = v["portfolio_period"]
    co = P["costed"][k]
    stop_rows += ("<tr><td>%s</td><td class='%s'>%s%%</td><td>%s%%</td><td>%s</td><td>%s%%</td>"
                  "<td class='%s'>%s%%</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td></tr>") % (
        k, cls(a["total"]), f1(a["total"]), f1(a["ann"]), f2(a["sharpe"]), f1(a["maxdd"]),
        cls(per["训练期"]["total"]), f1(per["训练期"]["total"]),
        cls(per["验证期"]["total"]), f1(per["验证期"]["total"]),
        cls(co["total"]), f1(co["total"]))

# 池内组合表
pool_rows = ""
for k in ["不止损", "止损8%", "止损12%", "止损15%", "止损20%"]:
    v = PP[k]; a = v["all"]; per = v["period"]; co = v["costed"]
    pool_rows += ("<tr><td>%s</td><td class='%s'>%s%%</td><td>%s</td><td>%s%%</td>"
                  "<td class='%s'>%s%%</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td></tr>") % (
        k, cls(a["total"]), f1(a["total"]), f2(a["sharpe"]), f1(a["maxdd"]),
        cls(per["训练期"]["total"]), f1(per["训练期"]["total"]),
        cls(per["验证期"]["total"]), f1(per["验证期"]["total"]),
        cls(co["total"]), f1(co["total"]))
pb_ = PP["池内基准(等权持有)"]
pool_rows += ("<tr style='background:#fdecea'><td><strong>池内基准(等权持有全部池票)</strong></td>"
              "<td class='up'><strong>%s%%</strong></td><td><strong>%s</strong></td><td><strong>%s%%</strong></td>"
              "<td class='up'><strong>%s%%</strong></td><td class='up'><strong>%s%%</strong></td><td>—</td></tr>") % (
    f1(pb_["all"]["total"]), f2(pb_["all"]["sharpe"]), f1(pb_["all"]["maxdd"]),
    f1(pb_["period"]["训练期"]["total"]), f1(pb_["period"]["验证期"]["total"]))

# OFAT 表
ofat_rows = ""
for k, v in R["ofat"].items():
    bold = ' style="background:#eef4ff;font-weight:700"' if k == "基准" else ""
    ofat_rows += "<tr%s><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td>%s%%</td><td class='%s'>%s</td><td class='%s'>%s</td></tr>" % (
        bold, k, "{:,}".format(v["n_total"]), cls(v["all"]["mean"]), f2(v["all"]["mean"]), f1(v["all"]["win"]),
        cls(v["训练期"]["rel"]), f2(v["训练期"]["rel"]), cls(v["验证期"]["rel"]), f2(v["验证期"]["rel"]))

# Regime 表
regime_rows = ""
regime_groups = []
for k in ["上涨", "下跌", "低波", "高波"]:
    v = R["regime"][k]
    regime_rows += "<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td>%s%%</td><td>%s</td><td class='%s'>%s%%</td><td class='%s'>%s pct</td></tr>" % (
        k, "{:,}".format(v["sig_n"]), cls(v["sig_mean"]), f2(v["sig_mean"]), f1(v["sig_win"]),
        "{:,}".format(v["base_n"]), cls(v["base_mean"]), f2(v["base_mean"]), cls(v["rel"]), f2(v["rel"]))
    regime_groups.append((k, [("B1信号", "#e74c3c", v["sig_mean"]), ("同regime基准", "#2563eb", v["base_mean"])]))

# H1 pre60 表
h1_rows = ""
h1_groups = []
for k, v in M["H1_pre60"].items():
    s, b = v["sig"], v["base"]
    h1_rows += "<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td>%s%%</td><td>%s</td><td class='%s'>%s%%</td><td class='%s'>%s pct</td></tr>" % (
        k, "{:,}".format(s["n"]), cls(s["mean"]), f2(s["mean"]), f1(s["win"]), "{:,}".format(b["n"]),
        cls(b["mean"]), f2(b["mean"]), cls(v["rel"]), f2(v["rel"]))
    h1_groups.append((k.replace("(", "").replace(")", ""), [("B1信号", "#e74c3c", s["mean"]), ("同层基准", "#2563eb", b["mean"])]))

# H2 dd20 表
h2_rows = ""
for k, v in M["H2_dd20"].items():
    s = v["sig"]
    h2_rows += "<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td><td>%s%%</td></tr>" % (
        k, "{:,}".format(s["n"]), cls(s["mean"]), f2(s["mean"]), cls(s["median"]), f2(s["median"]), f1(s["win"]))

# H3 路径表
h3_rows = ""
for k, v in M["H3_paths"].items():
    s = v["stat"]
    if not s: continue
    h3_rows += "<tr><td>%s</td><td>%s%%</td><td>%s</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td><td>%s%%</td></tr>" % (
        k, f1(v["share"]), "{:,}".format(s["n"]), cls(s["mean"]), f2(s["mean"]), cls(s["median"]), f2(s["median"]), f1(s["win"]))

# vol5 表
v5_rows = ""
for k, v in V5["quintiles"].items():
    rf, ra, r5 = v["ret_full"], v["ret_after5"], v["ret_first5"]
    v5_rows += "<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td>%s%%</td><td class='%s'>%s%%</td><td>%s%%</td></tr>" % (
        k, "{:,}".format(rf["n"]), cls(r5["mean"]), f2(r5["mean"]), f1(r5["win"]), cls(ra["mean"]), f2(ra["mean"]), f1(ra["win"]))

# 透支表
ext_rows = ""
for k, v in M["ext_pre20"].items():
    if not v: continue
    ext_rows += "<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td>%s%%</td></tr>" % (
        k, "{:,}".format(v["n"]), cls(v["mean"]), f2(v["mean"]), f1(v["win"]))

html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>B1因子深度评估报告 · 重构版（2026-10-19）</title>
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
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:760px){.grid2{grid-template-columns:1fr}}
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

<h1>B1 因子深度评估报告 <span class="tag tag-red">重构版 · 第一章</span></h1>
<div class="sub">生成日期：2026-10-19 ｜ 方法：A股量化研报复现专家团 Workflow B（因子/信号有效性检验）｜ 数据：kline_cache_all.db（前复权日线，2023-06-01 ~ 2026-08-21，5014 只股票）｜ 验收：role-07 研报复现验收工程师</div>

<div class="warn">
<strong>验收结论先行（冷酷版）：</strong>B1 作为独立买入信号，<strong>不通过</strong>。三重证据一致：① 信号层相对同宇宙基准年化负 alpha（全窗口 −0.54 pct/20日，池内 −0.65，池外 −0.51，训练期与验证期同为负）；② 日度截面 RankIC = −0.005（t = −3.07），信号组跑输非信号组 0.84 pct/20日；③ 组合层（月度 cohort 等权，含最佳止损档）累计 +16.8% / 34个月，同期等权全A +46.7%、池内基准 +133.2%，计费后最好仅 +5.5%。<strong>你的微观假设（趋势持续 + 适度回调）被数据证伪</strong>——详见第六章。选股池本身有 alpha，B1 是在池子 alpha 上<strong>倒扣</strong>的选择器。
</div>

<div class="key">
<strong>一句话给用户的答案：</strong>“真正走趋势的票不会轻易结束”在强趋势端恰恰相反——前60日涨幅最高的 20% 股票，其 B1 回调信号未来 20 日收益 <span class="down">−0.52%</span>（胜率 38.7%），是全样本最差的一层；你确实在“被更早参与趋势的人赚钱”，而 B1 没有让你赚到“被洗走者跟风”的钱——跟风现象真实存在（信号后放量组胜率 81.6%），但只在前 5 天可测（秩相关 +0.377），第 6 天起归零（+0.021），无法转化为入场或持有规则。
</div>

<!-- ==================== 一、定义 ==================== -->
<h2>一、因子定义与参数（回源码逐行核对）</h2>
<div class="card">
<p>定义来源：<code>backtest_b1_all_vec.py</code> 与生产引擎 <code>calc_zge_indicators</code> + <code>detect_b1_support</code>（向量化版已与原版在 8 只样本股票上逐条验证一致）。信号在 <strong>T 日收盘后</strong>产生，当日 K 线数据完整可知。</p>
<div class="formula">
<strong>B1（倒车接人）判定规则，全部条件同时满足：</strong><br>
① KDJ(9,3,3) 的 J 值 &lt; <strong>20</strong>（短期超卖）<br>
② 知行趋势线 = <code>EMA(EMA(Close,10),10)</code> 或知行多空线 = <code>(MA14+MA28+MA57+MA114)/4</code><br>
③ 当日最低价 ≤ 线 × <strong>1.001</strong>（盘中触及线）<br>
④ 当日收盘价 ≥ 线 × <strong>0.99</strong>（收盘仍站在线上方，“不破”）<br>
⑤ 当日最低价 ≥ (线 − <strong>0.5 × ATR</strong>) × 0.995，其中 ATR 用常数代理 <code>0.0214 × 收盘价</code>（非真实 ATR14，见偏离 D1）<br>
⑥ 生效前提：该股票已有 ≥ 60 根日 K 线；趋势线触发生成 <strong>b1_dc</strong> 型，多空线触发生成 <strong>b1_dk</strong> 型（dc 优先）
</div>
<table>
<tr><th>参数</th><th>值</th><th>参数</th><th>值</th></tr>
<tr><td>KDJ 周期</td><td>9,3,3</td><td>J 阈值</td><td>&lt; 20</td></tr>
<tr><td>趋势线</td><td>EMA(EMA(C,10),10)</td><td>多空线</td><td>(MA14+MA28+MA57+MA114)/4</td></tr>
<tr><td>触线容差</td><td>low ≤ line×1.001</td><td>收盘容差</td><td>close ≥ line×0.99</td></tr>
<tr><td>不破深度</td><td>low ≥ (line−0.5×0.0214×C)×0.995</td><td>最少历史</td><td>60 根K线</td></tr>
<tr><td>信号总量（重构重放）</td><td>97,581 条（完整前瞻 94,755）</td><td>池内 / 池外</td><td>15,266 / 79,489</td></tr>
</table>
<p class="muted">重放校准：与 2026-08-22 旧信号 CSV 比对，0 缺失；因前复权数据在两次运行间更新，多出 3,346 条临界触发信号（3.4%，全部为离阈值中位数 0.8% 的边际信号，登记为偏离 D8）。旧框架窗口（2024-07~2026-07，池内）重放结果 +3.93% / 胜率 54.2%，与既有框架 +3.93% / 53.5% 一致——管道校准通过。</p>
</div>

<!-- ==================== 二、九项口径 ==================== -->
<h2>二、九项口径声明（编排角色裁定）</h2>
<div class="card">
<table>
<tr><th>#</th><th>口径项</th><th>声明值</th></tr>
<tr><td>1</td><td>下期收益定义</td><td>T+1 开盘入场 → T+20 收盘出场（持有 20 个交易日），不计股利；信号层同时报告 close[T]/close[T+20] 参考口径</td></tr>
<tr><td>2</td><td>分组数</td><td>信号层十分位 + 五分层；B1 为二元信号，另报 2×2 条件交互与连续得分 IC</td></tr>
<tr><td>3</td><td>多空方向</td><td>仅多头（B1 为买入信号）；对照基准为同宇宙等权持有，不设空头腿</td></tr>
<tr><td>4</td><td>股票池过滤</td><td>剔除 3 只指数（000001/000905/000852）与 23 只名单外代码；<strong>未做</strong> ST / 涨跌停 / 停牌 / 上市日过滤（D2）</td></tr>
<tr><td>5</td><td>极值处理</td><td>不做 MAD 缩尾（收益率为原始值；分层用日度截面分位，对极值稳健）</td></tr>
<tr><td>6</td><td>中性化</td><td><strong>不做</strong>行业 / 市值中性化（数据源不含市值与行业字段，D12）</td></tr>
<tr><td>7</td><td>基准</td><td>等权全A（5014 只，主基准）/ 等权池内（848 只，含后视偏差，上限参考）/ 中证500（000905）/ 上证（000001，仅 2025-09 起）</td></tr>
<tr><td>8</td><td>无风险利率</td><td>0（月度夏普 = 均值/标准差×√12，rf=0 口径）</td></tr>
<tr><td>9</td><td>费率</td><td>基准口径不计成本；另报计费口径：佣金 0.00025 双边 + 印花税 0.0005 卖出 + 滑点 0.001 双边 ≈ 单次往返 0.30%</td></tr>
</table>
<p class="muted">训练/验证切分：信号日 &lt; 2025-09-08 为训练期（66,141 条完整信号），≥ 为验证期（28,614 条）。既有生产参数直接沿用，本次不重新寻优（避免全样本寻参，role-07 切分审查通过）。</p>
</div>

<!-- ==================== 三、信号层 ==================== -->
<h2>三、信号层有效性检验</h2>
<div class="card">
<h3>3.1 分组统计（T+1开盘 → T+20收盘，无止损，%）</h3>
<table>
<tr><th>分组</th><th>n</th><th>均值</th><th>中位</th><th>胜率</th><th>P10</th><th>P90</th><th>top5%盈亏贡献</th></tr>
__ROWS_GROUPS__
</table>
<div class="note"><strong>读法：</strong>均值全为正、中位数几乎全为负——盈利完全由右尾少数极端赢家贡献（top5% 贡献 +163% 的总盈亏，验证期甚至 −659%：赢家都盖不住输家）。彩票结构，不是稳定边际。验证期胜率从 51.8% 塌到 39.9%，中位数从 +0.48% 塌到 −2.22%。</div>

<h3>3.2 日度截面 RankIC（对 20 日前向收益，全股票 5014 只）</h3>
<table>
<tr><th>因子形式</th><th>IC 均值</th><th>ICIR</th><th>IC&gt;0 占比</th><th>t 值</th><th>判定</th></tr>
<tr><td>B1 信号（0/1）</td><td class="down">−0.0052</td><td class="down">−0.124</td><td>44.7%</td><td class="down">−3.07</td><td><span class="tag tag-red">负且显著</span></td></tr>
<tr><td>−J 值（连续）</td><td class="up">+0.0275</td><td>+0.198</td><td>58.0%</td><td>+4.98</td><td><span class="tag tag-amber">弱正</span></td></tr>
<tr><td>−趋势线距离（连续）</td><td class="up">+0.0746</td><td>+0.466</td><td>68.2%</td><td>+11.74</td><td><span class="tag tag-amber">正但方向相反</span></td></tr>
<tr><td>信号组−非信号组价差</td><td class="down">−0.84 pct/日</td><td>—</td><td>44.5%</td><td class="down">−4.03</td><td><span class="tag tag-red">负且显著</span></td></tr>
</table>
<div class="note">“−趋势线距离”IC 强正意味着：<strong>跌得离趋势线越远的股票未来涨得越多</strong>（短线反转），而 B1 偏偏只买“刚碰到线”的最浅回调——连续因子最强的地方，B1 的离散选择刻意避开了。这是本次评估最反直觉的发现。</div>

<h3>3.3 收益分布（n=94,755，4% 等宽桶）</h3>
__HIST__
<table>
<tr><th>十分位</th><th>收益区间</th><th>数量占比</th><th>盈亏贡献</th><th>组均值</th></tr>
__ROWS_DECILE__
</table>

<h3>3.4 条件归因：B1 到底加了什么？</h3>
<table>
<tr><th>规则（同窗口同口径）</th><th>n</th><th>均值%</th><th>中位%</th><th>胜率%</th></tr>
__ROWS_NAIVE__
</table>
<table>
<tr><th>2×2 交互（J&lt;20 × 触线收稳）</th><th>n</th><th>均值%</th><th>胜率%</th></tr>
__ROWS_INTERACT__
</table>
<div class="warn"><strong>三个致命细节：</strong>① B1 交集（+1.54%）<strong>同时差于</strong>它的两个单条件（J&lt;20 单独 +1.68%、触线收稳单独 +1.51%≈基准）和“两个都不满足”（+1.89%）——“超卖+支撑”是负交互，不是正交互；② “不收破”过滤器几乎没有判别力：J&lt;20 触线后收稳 +1.54% vs 破位（收盘跌穿线）+1.48%（n=5,718），<strong>B1 分不清趋势完好和趋势已坏</strong>；③ 朴素动量（前20日收益 top10%）仅 +0.01%——A股此窗口 1 个月尺度反转主导，B1 也不是动量。</div>
</div>

<!-- ==================== 四、组合层 ==================== -->
<h2>四、组合层绩效（月度 cohort 等权）</h2>
<div class="card">
<h3>4.1 净值曲线（全A信号，月度 cohort 复利）</h3>
__EQUITY__
<table>
<tr><th>组合</th><th>累计%</th><th>年化%</th><th>夏普</th><th>最大回撤%</th><th>训练期%</th><th>验证期%</th><th>计费口径%</th></tr>
__ROWS_STOP__
</table>
<table>
<tr><th>基准（同构造）</th><th>累计%</th><th>年化%</th><th>夏普</th><th>最大回撤%</th><th>训练期%</th><th>验证期%</th></tr>
<tr><td>等权全A（5014 只）</td><td class="up">+46.7</td><td>+15.0</td><td>0.68</td><td>−19.6</td><td class="up">+45.6</td><td class="up">+0.8</td></tr>
<tr><td>等权池内（848 只，含后视偏差）</td><td class="up">+133.2</td><td>+37.4</td><td>1.34</td><td>−12.3</td><td class="up">+74.7</td><td class="up">+33.5</td></tr>
<tr><td>等权池外</td><td class="up">+34.0</td><td>+11.2</td><td>0.54</td><td>−19.6</td><td class="up">+41.8</td><td class="down">−5.5</td></tr>
<tr><td>中证500（000905）</td><td class="up">+3.6</td><td>+1.2</td><td>0.20</td><td>−9.4</td><td colspan="2" class="muted">市值加权，此窗口大幅跑输等权小盘</td></tr>
</table>
<div class="note"><strong>止损档的两个必须说清的问题：</strong>（1）8% 止损把累计从 +2.8% 抬到 +16.8%，但<strong>交易级均值反而略降</strong>（不止损 +1.18% vs 8%止损 +1.03%）——组合层的“优势”来自月度等权聚合放大了低信号量灾月（2024-01 仅 439 条信号，月均 −18.3% → −6.0%；2026-07 仅 975 条，−11.2% → −3.7%）。这是聚合口径 artifact，不是止损创造了收益。（2）止损的真实价值在尾部：8% 止损 MDD −22.8% vs 不止损 −33.7%，夏普 0.36 vs 0.16——它切的是左尾，代价是右尾（38.5% 的信号被止损出场；被 8% 止损的交易若不止损平均收 −7.6%，与止损位基本持平，说明止损对这些交易近乎无差别）。</div>

<h3>4.2 池内场景（你的实际用法：池内选 B1）</h3>
<table>
<tr><th>组合（池内信号 15,450 条）</th><th>累计%</th><th>夏普</th><th>最大回撤%</th><th>训练期%</th><th>验证期%</th><th>计费口径%</th></tr>
__ROWS_POOL__
</table>
<div class="warn">池内 B1（+111~117%，计费 +92~98%）<strong>跑输什么都不做、只等权持有整个选股池</strong>（+133.2%）。验证期同样输：+26~31% vs +33.5%。alpha 在池子里，B1 是池子 alpha 上的减项。</div>

<h3>4.3 止损与波动率失配（MAE 分布，n=94,755）</h3>
<table>
<tr><th>止损档</th><th>5%</th><th>8%</th><th>10%</th><th>12%</th><th>15%</th><th>20%</th><th>25%</th></tr>
<tr><td>信号 20 日内曾触及该回撤的比例</td><td>56.5%</td><td>39.2%</td><td>30.4%</td><td>23.0%</td><td>15.1%</td><td>7.1%</td><td>3.2%</td></tr>
</table>
<p>信号 20 日最大不利偏移（MAE）中位 <strong>−6.0%</strong>、均值 −8.0%：B1 信号本身就是高波动品种，固定 8% 止损每 5 笔就打掉 2 笔。这是“止损与波动率失配”的量化版本——不是止损错了，是<strong>信号选的票波动结构和中位 −6% 的路径根本不兼容固定 8% 止损</strong>。</p>

<h3>4.4 水下曲线（B1 8%止损 vs 等权全A）</h3>
__DD__
</div>

<!-- ==================== 五、稳健性 ==================== -->
<h2>五、稳健性检验</h2>
<div class="card">
<h3>5.1 参数敏感性（OFAT，相对基准 alpha，pct/20日）</h3>
<table>
<tr><th>变体</th><th>信号数</th><th>均值%</th><th>胜率%</th><th>训练期 rel</th><th>验证期 rel</th></tr>
__ROWS_OFAT__
</table>
<div class="note">18 个变体（J 阈值 5~30、ATR 深度 0.25×~2×、收盘容差 0.98~0.999、触线容差、趋势线周期 EMA(5,5)/EMA(20,20)、多空线窗口），<strong>验证期相对 alpha 全部为负</strong>（−0.57 ~ −0.90），无一翻正。结论对参数不敏感——不存在“换个参数就能救”的证据。相对最接近零的是多空线 (5,10,20,60)（训练 −0.39）与 J&lt;5（验证 −0.57），但都不构成正向证据。</div>
__REGIME_CHART__
<h3>5.2 市场状态条件（市场 20 日收益 / 波动率分位）</h3>
<table>
<tr><th>状态</th><th>信号 n</th><th>信号均值%</th><th>信号胜率%</th><th>基准 n</th><th>基准均值%</th><th>相对 alpha</th></tr>
__ROWS_REGIME__
</table>
<div class="note">B1 只在<strong>上涨市和低波市</strong>有正相对 alpha（+0.53 / +0.86 pct），在下跌市和高波市显著为负（−1.62 / −1.92）。即：策略需要“市场已经涨起来且波动平稳”才勉强不亏——而信号恰恰在下跌/高波环境中成倍增加（月度信号数 890~6,135 大幅波动，跌势中回调票密集）。<strong>信号密度与信号有效性负相关</strong>，这是时序结构上的第二重问题。</div>
<h3>5.3 数据质量与截断</h3>
<table>
<tr><th>项目</th><th>数值</th><th>处理</th></tr>
<tr><td>右截断（末尾 20 日内信号无完整前瞻）</td><td>2,826 / 97,581</td><td>已排除于全部统计</td></tr>
<tr><td>幸存者偏差（退市股缺失）</td><td>73 只</td><td>不可修复；方向=高估历史收益</td></tr>
<tr><td>池内标记后视性（2026-08 池快照标注历史信号）</td><td>全部池内统计</td><td>登记 D3；池内外对比仅作参考</td></tr>
</table>
</div>

<!-- ==================== 六、微观假设 ==================== -->
<h2>六、微观经济假设检验（你的三个命题）</h2>
<div class="card">
<p>待检验假设：<strong>H1</strong> 真正走趋势的票不会轻易结束（趋势持续）；<strong>H2</strong> 适度回调反而是机会；<strong>H3</strong> 赚“被洗走的人随后跟风上车”的钱；被“更早参与趋势的人”或“认定趋势结束不再参与的人”赚钱。</p>

<h3>6.1 H1 趋势强度分层：证伪</h3>
__H1_CHART__
<table>
<tr><th>前 60 日涨幅分层</th><th>信号 n</th><th>信号均值%</th><th>胜率%</th><th>同层基准 n</th><th>基准均值%</th><th>相对 alpha</th></tr>
__ROWS_H1__
</table>
<p>按 MA114 位置再分：低于 MA114 5% 以上 +1.02%（胜率 46.9%）；贴近 MA114 +1.84%（52.6%）；高于 MA114 20% 以上（强趋势）<span class="down">−0.43%</span>（胜率 40.4%）。</p>
<div class="warn"><strong>H1 证伪。</strong>如果“趋势持续”成立，未来收益应随趋势强度单调上升。实际是平坦后崩塌：最强趋势层（前60日 +21.8%~+429.6%）未来 20 日 <span class="down">−0.52%</span>、胜率 38.7%，连基准（+0.06%）都跑不赢。强趋势股的浅回调不是“机会”，是<strong>为早期趋势参与者提供退出流动性</strong>——你假设里“被谁赚钱”的第一类人（更早参与趋势的人）确实在赚你的钱，而且恰恰赚在 B1 信号最强的区域。</div>

<h3>6.2 H2 适度回调：证伪（没有倒 U，深回调反而强）</h3>
<table>
<tr><th>信号日 20 日回撤深度分层</th><th>信号 n</th><th>均值%</th><th>中位%</th><th>胜率%</th></tr>
__ROWS_H2__
</table>
<p>按 J 值深度：J&lt;0 组 +1.56% > J∈[15,20) 组 +1.30%；按穿线深度（最低价穿破趋势线的幅度）：五个分层 +1.04%~+1.49%，基本平坦。</p>
<div class="warn"><strong>H2 证伪。</strong>“适度回调是机会”预测倒 U 型（浅回调=趋势完好应最好，深回调=趋势坏了应最差）。实测最深回调层（−49.6%~−16.7%）反而最高（+2.11%），且随回调变浅单调走差——这是<strong>反转（跌深了反弹）</strong>的形态，不是<strong>延续（浅调即机会）</strong>的形态。B1 特意挑选的“浅调不破” band 恰好坐在曲线最平淡的位置。它与 3.2 节“−趋势线距离 IC +0.075”互为印证：钱在深调里，不在适度里。</div>

<h3>6.3 H3 洗盘后跟风：现象真实存在，但无事前选择性</h3>
<table>
<tr><th>持仓路径（前 5 日最低 vs 后 15 日最高）</th><th>占比</th><th>n</th><th>均值%</th><th>中位%</th><th>胜率%</th></tr>
__ROWS_H3__
</table>
<p>“洗盘后上行”（先跌超 2% 再涨超 3% 且收正）确实占 12.5%、均值 +11.45%——你描述的“被洗走的人跟风上车”在 1/8 的交易里真实发生。但：直线上攻（34.0%）同样 +10.92% 且更多；先跌后涨组整体 +9.93% 还不如不跌直涨组 +11.92%。<strong>洗盘路径是事后的好结果，不是事前的可选特征。</strong></p>
<p>量能跟风（你说“赚跟风上车者的钱”的直接代理）——全窗口版本：信号后 20 日/前 20 日均量比最高组 +12.44%、胜率 81.6%，最低组 −5.80%、胜率 19.0%。<strong>但这个版本与收益共享同一个 20 天窗口，是同义反复。</strong>事前版本（只用 T+1~T+5 量能）的秩相关：对前 5 日收益 +0.377，对全窗口 +0.208，对第 6~20 日收益 <span class="down">+0.021（归零）</span>；据此设计的动态离场规则（T+5 收盘跌破入场价 X% 则 T+6 离场）相对基准 +1.24% 无增益（+1.19~1.21%）。</p>
<div class="note"><strong>H3 判定：描述为真，利用为假。</strong>跟风行为真实存在且与收益强相关，但其信息 5 个交易日内衰减完毕，本次未能转化为任何事前规则。这仍是 B1 家族里唯一有希望的线索（见 6.4 修复路径 C）。</div>

<h3>6.4 “被谁赚钱”：两类人都赚到了，且都赚在信号最强处</h3>
<table>
<tr><th>B1 信号前 20 日涨幅</th><th>n</th><th>未来 20 日均值%</th><th>胜率%</th></tr>
__ROWS_EXT__
</table>
<p>dc 型（趋势线回调，前期涨幅中位 +5.35%）+1.54% vs dk 型（多空线回调，前期涨幅中位 +8.27%，即更晚的趋势阶段）+0.90%——<strong>越到趋势后期介入，收益越差</strong>，与“早期参与者赚后期接盘者的钱”一致。</p>
<div class="key"><strong>综合判定：</strong>你的博弈叙事在<strong>描述层面基本成立</strong>（趋势有生命周期、回调确实洗人、跟风盘确实后期上车），但在<strong>信号层面全部反向</strong>——B1 系统性地在趋势最强处（H1 Q5）、回调最浅处（H2 Q5）、跟风最热处（信号密集的跌势月）下注最重。信号没有把你的叙事变成 edge，反而把你的叙事变成了别人的 edge。</div>
</div>

<!-- ==================== 七、偏离登记 ==================== -->
<h2>七、偏离登记</h2>
<div class="card">
<table>
<tr><th>#</th><th>偏离项</th><th>参照/原做法</th><th>本实现做法</th><th>影响面</th><th>严重度</th></tr>
<tr><td>D1</td><td>ATR 为常数代理</td><td>真实 ATR(14)</td><td>0.0214×收盘价，“半ATR不破”阈值随价格绝对缩放，低价股阈值偏松、高价股偏紧</td><td>信号触发集合</td><td>中</td></tr>
<tr><td>D2</td><td>无交易状态过滤</td><td>应剔除 ST/涨跌停/停牌/次新</td><td>未做；信号可能落在不可成交状态</td><td>收益高估（不可成交的单据被记为成交）</td><td>高</td></tr>
<tr><td>D3</td><td>池内标记后视性</td><td>信号当日实时池</td><td>2026-08 池快照（848 只）标注 2023-2026 历史信号</td><td>池内外全部对比含选择偏差</td><td>高</td></tr>
<tr><td>D4</td><td>指数混入扫描集</td><td>仅股票</td><td>原 CSV 含 000001/000905/000852 共 71 条信号；重构版已剔除</td><td>全A统计（已修复）</td><td>低</td></tr>
<tr><td>D5</td><td>名单外代码混入</td><td>仅股票名单</td><td>23 只（含 ETF）在原扫描集；重构版已剔除</td><td>全A统计（已修复）</td><td>低</td></tr>
<tr><td>D6</td><td>000001 槽位冲突</td><td>平安银行 K 线</td><td>DB 中 000001 存的是上证指数（2025-09 起），平安银行 K 线不可得</td><td>1 只股票缺失</td><td>低</td></tr>
<tr><td>D7</td><td>幸存者偏差</td><td>含退市股</td><td>73 只退市股 K 线缺失，不可修复</td><td>历史收益系统性高估</td><td>中</td></tr>
<tr><td>D8</td><td>信号集数据漂移</td><td>2026-08-22 旧 CSV</td><td>前复权数据更新后重放多出 3,346 条（3.4%）边际信号，0 缺失</td><td>与旧报告数字不可逐条对齐</td><td>低</td></tr>
<tr><td>D9</td><td>旧“ret20”口径错误</td><td>标称 20 日</td><td>实为 close[T+19]/close[T]（19 日）；组合层为 T+1 开盘→T+20 收盘。重构版统一为后者，两者并存于信号表</td><td>跨报告对比</td><td>中</td></tr>
<tr><td>D10</td><td>移动止盈判定</td><td>含影线回撤</td><td>仅用收盘价（峰值收盘回撤 50%），与原引擎一致</td><td>极少数移动止盈交易（76 笔）</td><td>低</td></tr>
<tr><td>D11</td><td>止损成交假设</td><td>跳空/涨跌停不可成交</td><td>最低价触及即按止损价成交，不处理跳空与涨跌停无法成交</td><td>极端日收益虚高</td><td>中</td></tr>
<tr><td>D12</td><td>无中性化</td><td>市值+行业中性</td><td>数据源无市值/行业字段，IC 与分层含风格暴露（小盘、反转）</td><td>IC 解释</td><td>中</td></tr>
<tr><td>D13</td><td>月度 cohort 等权口径</td><td>真实组合净值</td><td>重叠持仓的月度平均，非 NAV；月度等权放大低信号量灾月（见 4.1）</td><td>组合层数字</td><td>中</td></tr>
<tr><td>D14</td><td>多重检验史</td><td>—</td><td>B1 参数族历史上过多轮扫描；本次 OFAT 仅用于证伪（全变体验证期为负），结论方向不受挑选偏差利好</td><td>结论稳健性</td><td>低</td></tr>
<tr><td>D15</td><td>池内基准后视性</td><td>实时池基准</td><td>+133.2% 的池内基准用 2026-08 池回填历史，为“选择偏差上界”参考，不可作实盘目标</td><td>基准对比</td><td>高</td></tr>
</table>
</div>

<!-- ==================== 八、验收 ==================== -->
<h2>八、验收结论（role-07 研报复现验收工程师）</h2>
<div class="card">
<h3>8.1 前视偏差检查（9 条）</h3>
<table>
<tr><th>#</th><th>检查项</th><th>结论</th></tr>
<tr><td>1</td><td>成交时点晚于信号（T+1 开盘 vs T 日收盘出信号）</td><td><span class="tag tag-green">通过</span></td></tr>
<tr><td>2</td><td>财报/持仓披露滞后</td><td><span class="tag tag-green">不适用</span>（纯量价，无财务数据）</td></tr>
<tr><td>3</td><td>复权使用未来信息</td><td><span class="tag tag-amber">附注通过</span> 前复权价格水平含未来除权信息，但仅影响除权日附近价格刻度，不改变信号时序结构</td></tr>
<tr><td>4</td><td>截面排名使用全样本</td><td><span class="tag tag-green">通过</span> IC 为日度独立截面</td></tr>
<tr><td>5</td><td>滚动窗口未 shift(1)</td><td><span class="tag tag-green">通过</span> KDJ/EMA/MA 均仅用 ≤T 日数据</td></tr>
<tr><td>6</td><td>标签跨进测试期</td><td><span class="tag tag-green">通过</span> 右截断信号（2,826 条）已按完整性剔除</td></tr>
<tr><td>7</td><td>标准化使用全样本统计量</td><td><span class="tag tag-green">通过</span> 分层用每日截面分位</td></tr>
<tr><td>8</td><td>停牌/涨跌停事后可知</td><td><span class="tag tag-red">未处理</span> 登记 D2/D11，影响=收益高估</td></tr>
<tr><td>9</td><td>因子值使用未来数据</td><td><span class="tag tag-green">通过</span></td></tr>
</table>
<h3>8.2 切分方式审查</h3>
<p>固定时间切分（信号日 &lt; 2025-09-08 训练，≥ 验证）——<span class="tag tag-green">可接受</span>；参数为既有生产参数直接沿用，本次未做样本内寻优——<span class="tag tag-green">无全样本寻参风险</span>。</p>
<h3>8.3 最终结论</h3>
<div class="warn">
<strong>复现过程：通过。</strong>管道与旧引擎/旧 CSV/旧框架三重校准一致（0 缺失；旧窗口池内 +3.93%/54.2% vs 旧框架 +3.93%/53.5%）。<br><br>
<strong>因子有效性：不通过。</strong>B1 作为独立买入信号，相对同宇宙基准的 alpha 在训练期、验证期、池内、池外、全参数族五个维度上一致为负；IC 为负且统计显著；组合层计费后为负；无任何止损/参数变体翻正。<br><br>
<strong>必须修复项（若继续研究 B1 家族）：</strong>① 补 D2 交易状态过滤后重跑全部统计；② 池内外对比改用信号当日可得的池快照（消除 D3 后视性）；③ 所有跨期比较固定“T+1 开盘→T+20 收盘”单一口径。<br><br>
<strong>已知限制：</strong>D7 幸存者偏差、D12 无中性化、D13 聚合口径、D15 基准后视性——均使本报告的“B1 无效”结论<strong>偏保守方向相反</strong>（即这些偏差更可能高估而非低估 B1，结论方向安全）。
</div>
</div>

<!-- ==================== 九、修复路径 ==================== -->
<h2>九、基于数据的修复路径（按优先级）</h2>
<div class="card">
<table>
<tr><th>优先级</th><th>动作</th><th>依据</th></tr>
<tr><td><strong>A（立即）</strong></td><td>停止把 B1 当独立买入信号。默认策略改为“等权持有选股池 + 8% 个股止损”（夏普 1.38、回撤 −11.1%、计费 +92%），B1 从“选股器”降级为“池内持仓的<strong>否决项</strong>研究”（验证期池内 B1 −1.47 pct 相对 alpha，至少说明它不该被主动买入）</td><td>4.2 节池内对比</td></tr>
<tr><td><strong>B（下一步实验）</strong></td><td>检验“深回调反弹”：B1 条件不变但把入场从“触线”改为“跌穿线 1.5~2.5 倍 ATR”，或直接测 −趋势线距离连续因子最高分层。这是全报告唯一 IC 强正（+0.075）的方向，但与 B1 哲学相反，属于新因子</td><td>3.2 / 6.2 节</td></tr>
<tr><td><strong>C（有价值的线索）</strong></td><td>量能跟风的事前化：T+1~T+3 量价确认 + 剩余仓位规则（如“前 3 日量比 &gt;1.2 且收阳则持有满 20 日，否则第 4 日离场”），用训练期设计、验证期盲测。本次已证伪简单版，需要更精细的条件结构</td><td>6.3 节 vol5 检验</td></tr>
<tr><td><strong>D（口径基建）</strong></td><td>修复 D2/D3 后重跑；建立 walk-forward 连续两段样本外为正才进实盘的闸门（当前验证期组合层 −3.5%~−11.6%，远未达到）</td><td>8.3 节</td></tr>
</table>
<div class="note"><strong>方法论备注：</strong>本报告所有“基准”均为同构造等权组合（同窗口、同入场出场、同聚合口径），B1 与基准的唯一差异是<strong>选股规则</strong>本身。因此所有相对数字都是对“B1 选股能力”的干净度量，不含时机、容量、费率结构差异。</div>
</div>

<div class="sub" style="margin-top:28px">数据与代码：b1_research/（b1_prep.py 特征预处理 → b1_verify.py 校准 → b1_analysis_signal.py / b1_analysis_port.py / b1_analysis_robust.py / b1_analysis_micro.py / b1_micro_vol5.py / b1_pool_port.py / b1_series.py 七项检验 → 本报告）。全部数字可由上述脚本一键复现。<br>本报告为研究方法论与因子检验，不构成投资建议。</div>

</div>
</body>
</html>
"""

# ---- 填表 ----
def rows_groups():
    order = [("全体信号", "全体", True), ("训练期(<2025-09-08)", "训练期", False), ("验证期(≥2025-09-08)", "验证期", False),
             ("池内", "池内", False), ("池内-训练", "池内-训练", False), ("池内-验证", "池内-验证", False),
             ("池外", "池外", False), ("池外-训练", "池外-训练", False), ("池外-验证", "池外-验证", False),
             ("dc型(趋势线)", "dc型", False), ("dk型(多空线)", "dk型", False)]
    return "\n".join(row(lbl, g[k]) for lbl, k, b in order)

def rows_decile():
    return "\n".join("<tr><td>%s</td><td>%s</td><td>%s%%</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td></tr>" % (
        k, k[3:], f1(v["share"]), cls(v["pnl_share"]), f1(v["pnl_share"]), cls(v["mean"]), f2(v["mean"]))
        for k, v in dc.items())

def rows_naive():
    order = ["B1信号(基准)", "J<20单独(超卖反弹)", "触趋势线不收稳单独", "触多空线不收稳单独", "前20日收益top10%(动量)",
             "全股票等权基准", "池内全股票等权基准", "池外全股票等权基准"]
    return "\n".join("<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td><td>%s%%</td></tr>" % (
        k, "{:,}".format(nv[k]["n"]), cls(nv[k]["mean"]), f2(nv[k]["mean"]),
        cls(nv[k]["median"]), f2(nv[k]["median"]), f1(nv[k]["win"])) for k in order)

def rows_interact():
    keys = [("J<20=0,触线收稳=0", "两者都不满足(基准行为)"), ("J<20=0,触线收稳=1", "仅触线收稳"),
            ("J<20=1,触线收稳=0", "仅 J<20(超卖)"), ("J<20=1,触线收稳=1", "B1 交集(两者都满足)")]
    return "\n".join("<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td>%s%%</td></tr>" % (
        lbl, "{:,}".format(R["interact"][k]["n"]), cls(R["interact"][k]["mean"]),
        f2(R["interact"][k]["mean"]), f1(R["interact"][k]["win"])) for k, lbl in keys)

def rows_h1():
    return h1_rows

def rows_h2():
    return h2_rows

def rows_h3():
    return h3_rows

def rows_ext():
    return ext_rows

def rows_v5():
    return v5_rows

html = (html
    .replace("__ROWS_GROUPS__", rows_groups())
    .replace("__ROWS_DECILE__", rows_decile())
    .replace("__ROWS_NAIVE__", rows_naive())
    .replace("__ROWS_INTERACT__", rows_interact())
    .replace("__ROWS_STOP__", stop_rows)
    .replace("__ROWS_POOL__", pool_rows)
    .replace("__ROWS_OFAT__", ofat_rows)
    .replace("__ROWS_REGIME__", regime_rows)
    .replace("__ROWS_H1__", h1_rows)
    .replace("__ROWS_H2__", h2_rows)
    .replace("__ROWS_H3__", h3_rows)
    .replace("__ROWS_EXT__", ext_rows)
    .replace("__ROWS_V5__", v5_rows)
    .replace("__HIST__", hist_chart(hist["edges"], hist["counts"]))
    .replace("__EQUITY__", line_chart([
        ("等权池内(含后视)", "#d97706", eq["eq_pool"]),
        ("等权全A", "#2563eb", eq["eq_all"]),
        ("B1(8%止损)", "#e74c3c", eq["B1_8pct"]),
        ("B1(不止损)", "#9ca3af", eq["B1_nostop"]),
        ("中证500", "#6b7280", eq["zz500"]),
    ], vline=te, vlabel="训练|验证", ymin=0.85, ymax=2.45))
    .replace("__DD__", line_chart([
        ("B1(8%止损)", "#e74c3c", dd["B1_8pct"]),
        ("等权全A", "#2563eb", dd["eq_all"]),
    ], vline=te, vlabel="训练|验证", ymin=-40, ymax=2))
    .replace("__REGIME_CHART__", bar_chart(regime_groups))
    .replace("__H1_CHART__", bar_chart(h1_groups))
)

from healthcheck_chapter import inject
html = inject(html, "B1(倒车/多空)")

out = os.path.join(HERE, "B1因子深度评估报告_重构版_20261019.html")
open(out, "w", encoding="utf-8").write(html)
print("written:", out, "%.1f KB" % (len(html)/1024))
