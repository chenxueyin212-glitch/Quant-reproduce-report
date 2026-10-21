# -*- coding: utf-8 -*-
"""六信号因子体检补全报告 — 回应"IC/RankIC/Barra暴露/换手/自相关为何缩水"，统一补齐并修正前六章结论"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
H = json.load(open(os.path.join(HERE, "healthcheck_results.json"), encoding="utf-8"))

ORDER = ["B1(倒车/多空)", "B2", "bp1(突破追入)", "补票(单针下20)", "搬砖(多空临界)", "bp2(均线回踩)"]
SHORT = {"B1(倒车/多空)": "B1", "B2": "B2", "bp1(突破追入)": "bp1",
         "补票(单针下20)": "补票", "搬砖(多空临界)": "搬砖", "bp2(均线回踩)": "bp2"}

def f2(x): return ("%+.2f" % x) if x is not None else "—"
def f3(x): return ("%+.3f" % x) if x is not None else "—"
def f4(x): return ("%+.4f" % x) if x is not None else "—"
def f1(x): return ("%+.1f" % x) if x is not None else "—"
def pct(x): return "%.1f%%" % (100*x) if x is not None else "—"
def cls(x): return "up" if (x or 0) > 0 else "down"

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
    bw = 84
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

# ==================== 表格构建 ====================
# --- 一、差距审计表 ---
audit_rows = ""
AUDIT = [
    ("全截面二元 RankIC（信号 vs 截面未来收益）", ["ok:4口径", "ok", "ok", "no:组内IC", "no:组内IC", "no:组内IC"], "✅ 补齐"),
    ("全截面 Pearson IC", ["no"]*6, "✅ 补齐"),
    ("IC 衰减（h=5/10/20）", ["no"]*6, "✅ 补齐"),
    ("风格暴露（Barra 或代理）", ["no:仅D12", "no:仅D12", "no:仅D12", "no:仅D12", "no:仅D12", "no:仅D12"], "✅ 代理5因子"),
    ("风格调整后 alpha（异常收益归因）", ["no"]*6, "✅ 补齐"),
    ("换手率 / 持仓重叠", ["no", "no", "no", "≈密度", "≈密度", "≈密度"], "✅ 补齐"),
    ("收益序列自相关", ["no"]*6, "✅ 补齐"),
]
def mark(m):
    if m == "ok": return "<span class='tag tag-green'>做</span>"
    if m.startswith("ok"): return "<span class='tag tag-green'>做·%s</span>" % m.split(":", 1)[1]
    if m.startswith("≈"): return "<span class='tag tag-amber'>近似%s</span>" % m[1:]
    if m.startswith("no:"): return "<span class='tag tag-red'>缺·%s</span>" % m.split(":", 1)[1]
    return "<span class='tag tag-red'>缺</span>"
for name, marks, fix in AUDIT:
    audit_rows += "<tr><td>%s</td>%s<td>%s</td></tr>" % (
        name, "".join("<td style='text-align:center'>%s</td>" % mark(m) for m in marks), fix)

# --- 三、IC 表 ---
ic_rows = ""
for s in ORDER:
    d = H[s]; sh = SHORT[s]
    for kind, kn in [("pearson", "Pearson"), ("rank", "Rank")]:
        for h in ("5", "10", "20"):
            v = d["ic_full"][kind][h]
            if not v:
                ic_rows += "<tr><td>%s</td><td>%s</td><td>h=%s</td><td colspan='4'>—</td></tr>" % (sh, kn, h)
                continue
            sig = "显著" if abs(v.get("t") or 0) > 2 else "不显著"
            ic_rows += ("<tr><td>%s</td><td>%s</td><td>h=%s</td><td class='%s'>%s</td><td>%s</td>"
                        "<td>%.1f%%</td><td class='%s'>%+.2f</td></tr>") % (
                sh, kn, h, cls(v["mean"]), f4(v["mean"]), f2(v["icir"]),
                v["pos_ratio"], cls(v["t"]), v["t"])
# h=20 rank 一行摘要
ic20_chart = bar_chart([
    (SHORT[s], [("RankIC h=20", "#8b5cf6" if H[s]["ic_full"]["rank"]["20"]["mean"] >= 0 else "#27ae60",
                 H[s]["ic_full"]["rank"]["20"]["mean"])]) for s in ORDER
], fmt="%+.4f")

# --- 四、风格暴露表 ---
exp_rows = ""
STY = [("ln_amt", "ln_amt20"), ("beta", "beta60"), ("mom60", "mom60"), ("vol20", "vol20"), ("rev20", "rev20")]
for s in ORDER:
    d = H[s]["exposure"]
    cells = ""
    for k, _ in STY:
        v = d.get(k)
        if not v:
            cells += "<td>—</td>"
            continue
        strong = abs(v.get("t") or 0) > 2
        cells += "<td class='%s'>%s%s</td>" % (cls(v["mean"]), f2(v["mean"]),
                                               " <span class='muted'>(t=%+.1f)</span>" % v["t"] if strong else "")
    exp_rows += "<tr><td>%s</td>%s</tr>" % (SHORT[s], cells)

# --- 风格 alpha 表（核心）---
alpha_rows = ""
alpha_chart_groups = []
verdict = {
    "B1(倒车/多空)": ("tag-gray", "全是风格"),
    "B2": ("tag-gray", "全是风格·日度残差略负"),
    "bp1(突破追入)": ("tag-red", "负 alpha"),
    "补票(单针下20)": ("tag-green", "唯一残差 alpha·样本外存活"),
    "搬砖(多空临界)": ("tag-red", "全是风格·残差偏负"),
    "bp2(均线回踩)": ("tag-amber", "有残差·基率不足"),
}
for s in ORDER:
    a = H[s]["style_alpha"]
    t_d = a.get("abn_t_daily")
    t_p = a.get("abn_t")
    t_show = t_d if t_d is not None else t_p
    t_note = "池化 %s" % f2(t_p) if t_d is not None else "(池化)"
    vtag, vtxt = verdict[s]
    alpha_rows += ("<tr><td><strong>%s</strong></td><td class='%s'>%s%%</td><td class='%s'>%s%%</td>"
                   "<td class='%s'>%s<span class='muted'> %s</span></td><td class='%s'>%s</td><td class='%s'>%s</td>"
                   "<td><span class='tag %s'>%s</span></td></tr>") % (
        SHORT[s], cls(a["raw_mean"]), f2(a["raw_mean"]), cls(a["abn_mean"]), f2(a["abn_mean"]),
        cls(t_show), f2(t_show), t_note,
        cls(a["abn_train"]), f2(a["abn_train"]), cls(a["abn_valid"]), f2(a["abn_valid"]),
        vtag, vtxt)
    alpha_chart_groups.append((SHORT[s], [
        ("raw", "#94a3b8", a["raw_mean"]),
        ("风格调整后", "#2563eb" if a["abn_mean"] >= 0 else "#27ae60", a["abn_mean"]),
    ]))
alpha_chart = bar_chart(alpha_chart_groups)

# --- 五、换手/自相关表 ---
turn_rows = ""
for s in ORDER:
    t = H[s]["turnover"]
    turn_rows += ("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>") % (
        SHORT[s], pct(t["mean_daily"]), pct(t["median_daily"]), pct(t["overlap"]),
        pct(t["stickiness"]), "{:,}".format(t["n_days"]), f3(H[s]["nav_ac1"]))

# --- 六、修正对照表 ---
fix_rows = ""
FIXES = [
    ("补票", "有条件可用（cohort 止损8% +45.6%，门禁后 +52.5%）",
     "<strong>升级：唯一残差 alpha</strong>（+0.75%，按日 t=2.62，验证 +0.45%）——可用性论据改为『残差 alpha + 门禁后接近基准』，绝对收益仍落后基准 +56.8%"),
    ("B1", "组合层 −3.8%，信号层有边际",
     "<strong>降级：全是风格</strong>（raw +1.25% → abn +0.03%，按日 t=1.59 不显著）——低β强势回调风格体，绝对收益来自风格 β 不是选股"),
    ("B2", "cohort +16.3%，六信号中游",
     "<strong>降级：全是风格</strong>（raw +1.05% → abn +0.03%，按日 t=−2.83 略负）——同 B1，风格体；按日口径残差甚至偏负"),
    ("bp1", "cohort −42.7%，全系列最差",
     "<strong>确认并加严：负 alpha</strong>（raw −0.29% → abn −1.12%，按日 t=−4.18）——同风格票中系统性跑输，rev20 暴露 +1.30 是接盘标志，与彩票型曲线互证"),
    ("搬砖", "信号层 +1.54% 但边际 +0.13 pct 最低，cohort +26.8%",
     "<strong>改写：raw 全是风格，残差偏负</strong>（abn −0.11%，按日 t=−1.92）——『最弱正边际』实为零/微负 alpha"),
    ("bp2", "全系列最弱，唯一信号层负边际，建议下线",
     "<strong>部分翻案：有残差但基率不足</strong>（abn +0.56%，按日 t=3.39，验证 +0.61%）——raw +0.58% 仍低于宇宙 +1.41%，处置维持下线/观察"),
]
for name, old, new in FIXES:
    fix_rows += "<tr><td><strong>%s</strong></td><td>%s</td><td>%s</td></tr>" % (name, old, new)

n_sig_rows = "".join(
    "<tr><td>%s</td><td>%s</td></tr>" % (SHORT[s], "{:,}".format(H[s]["n_sig"])) for s in ORDER)

meta_note = H["meta"]["barra_note"]

html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>六信号因子体检补全报告（IC/RankIC/风格暴露/换手/自相关）· 2026-10-20</title>
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

<h1>六信号因子体检补全报告 <span class="tag tag-amber">IC · RankIC · 风格暴露 · 换手 · 自相关</span></h1>
<div class="sub">生成日期：2026-10-20 ｜ 回应指令：标准因子检查项为何前几章做了、后几章缩水 ｜ 数据：kline_cache_all.db 全A 4,999 只，2023-11-01 ~ 2026-08-21 ｜ 口径：T日收盘后可知 → T+1 开盘入场 → T+h 收盘（h=5/10/20），训练期 &lt;20250908 ｜ 体检脚本：factor_healthcheck.py</div>

<div class="warn">
<strong>先回答"为什么缩水"——这是错误取舍，如实登记：</strong>IC/RankIC、Barra 风格暴露、换手率、自相关是标准因子评估协议的固定检查项（分别回答"有没有排序力/是不是风格马甲/成本可不可行/时序有没有结构"）。前三个报告（B1/B2/bp1）做了<strong>全截面二元 RankIC</strong>（B1 四口径），但 <strong>Pearson IC 与 IC 衰减六章全缺</strong>；后三章（补票/搬砖/bp2）模板化复用时把检查单<strong>裁剪成"组内特征 IC"</strong>；Barra 暴露六章只有一行 <code>D12 不做中性化</code> 的<strong>登记</strong>（登记 ≠ 已检验）；换手/自相关从未进模板（后三章仅有信号密度近似）。根因：模板复用时检查单没有锁死全项，越写越省。本报告一次性补齐六信号全部缺失项，并如实修正前六章结论。
</div>

<div class="key">
<strong>补全后的核心发现（改变系列叙事）：</strong>把六信号的 20 日收益对五个风格代理（规模/β/动量/波动/反转）做日截面中性化后——<strong>补票是唯一残差 alpha 显著为正且验证期存活的信号</strong>（abn <span class="up">+0.75%</span>/20日，按日 t=2.62，训练 +0.84 / 验证 +0.45）；<strong>B1、B2 的 raw 边际被风格完全解释</strong>（abn ≈ +0.03%，按日 t 不显著/略负）；<strong>搬砖 raw +1.54% 全是风格、残差偏负</strong>（−0.11%，按日 t=−1.92）；<strong>bp1 是显著负 alpha</strong>（−1.12%，按日 t=−4.18）；bp2 有小残差（+0.56%，按日 t=3.39）但基率不足。全截面 IC 六信号 |IC|&lt;0.02——它们都不是"排序因子"，是"事件触发器"。
</div>

<!-- ============ 一、差距审计 ============ -->
<h2>一、六章差距审计（哪些章做了什么、漏了什么）</h2>
<div class="card">
<table>
<tr><th>检查项</th><th>B1</th><th>B2</th><th>bp1</th><th>补票</th><th>搬砖</th><th>bp2</th><th>本报告</th></tr>
__AUDIT_ROWS__
</table>
<p class="muted">"组内IC"＝只在信号票内部对特征分层的 IC（如 dist_close、score），不能替代全截面 IC；"密度近似"＝用信号密度当换手的粗代理；"仅D12"＝口径表登记了不做中性化，但从未实际检验风格暴露。</p>
<div class="note">
<strong>为什么要这五项：</strong>① Pearson IC/RankIC——信号在<strong>当日全市场截面</strong>上的排序力（预测力本体）；② IC 衰减 h=5/10/20——预测力的半衰期（对照 20 日持有口径）；③ 风格暴露/风格 alpha——raw 边际是选股能力还是风格马甲（中性化归因）；④ 换手率——成本可行性（0.30% 往返 × 换手）；⑤ 自相关——日收益时序有无可利用结构（连亏降档/动量再入场的依据）。缺任何一项，因子结论都不完整。
</div>
</div>

<!-- ============ 二、方法 ============ -->
<h2>二、补全方法与口径</h2>
<div class="card">
<div class="formula">
<strong>模块 A · 全截面 IC：</strong>逐日把 0/1 信号哑变量与当日宇宙内 T+1开盘→T+h收盘收益做 Pearson 与 Spearman 相关，h=5/10/20，按日序列给 mean/ICIR/t/正IC占比。<br>
<strong>模块 B · 风格暴露与 alpha：</strong>五个 OHLCV 代理风格（ln 20日均成交额、beta60 对等权市场、mom60、vol20、rev20）当日截面 z 分 → 暴露 = 信号行 z 均值（按日聚合 t）；alpha = 日截面 OLS <code>r20 ~ 1 + z_ln_amt + z_beta + z_mom60 + z_vol20 + z_rev20</code> 取信号行残差（异常收益）。<br>
<strong>模块 C · 换手：</strong>持仓按 T+1 入场持有 20 日展开，日度换手 = (新开+平仓)/日均持仓数，另给持仓重叠率与信号粘性 P(sig[t+1]|sig[t])。<br>
<strong>模块 D · 自相关：</strong>组合日收益（扣费拖 1.5bp/日）lag-1 自相关。
</div>
<table>
<tr><th>信号</th><th>信号行数</th></tr>
__N_SIG_ROWS__
</table>
<div class="warn">
<strong>风格因子局限（D41）：</strong>__META_NOTE__ 真 Barra（市值/行业/成长/盈利/杠杆等）需财务与行业数据源，不在 kline_cache_all.db 内——本报告风格归因结论仅对上述五个代理风格成立，行业/市值残差风险无法排除。
</div>
<p class="muted">样本说明：体检窗口 2023-11-01 起（风格预热需要 60 日历史），且剔除风格缺失行，故均值与各章报告（2023-06 起全样本）有小差异（如 bp2 raw +0.58% vs 章内 +0.51%）——结论方向不受影响。</p>
</div>

<!-- ============ 三、模块A ============ -->
<h2>三、模块 A：全截面 Pearson IC 与 RankIC（h=5/10/20）</h2>
<div class="card">
<h3>3.1 全截面 IC 明细</h3>
<table>
<tr><th>信号</th><th>类型</th><th>持有期</th><th>IC 均值</th><th>ICIR</th><th>正IC占比</th><th>t</th></tr>
__IC_ROWS__
</table>
__IC_CHART__
<div class="note">
<strong>读法与语义限制（D42）：</strong>六信号 |IC| 全部 &lt; 0.02 且<strong>多数显著为负</strong>——信号票在当日截面中排名靠后。bp1 的 RankIC h=10 <span class="down">−0.0168</span>（t=−15.4）是全表最强负值：突破追入系统性买在截面高点。补票是唯一 Pearson 为正的（h=20 +0.0017，弱）。但<strong>二元稀疏信号的点二列相关被稀疏度机械压缩</strong>（每天只有少数票取 1），|IC| 小 ≠ 无边际——IC 回答"能不能当排序因子"，答案是六信号都不能；事件触发器的边际要看信号行均值与风格 alpha（下一章）。
</div>
</div>

<!-- ============ 四、模块B ============ -->
<h2>四、模块 B：风格暴露与风格调整后 alpha（核心章）</h2>
<div class="card">
<h3>4.1 风格暴露（信号行当日截面 z 均值，t 按日聚合）</h3>
<table>
<tr><th>信号</th><th>ln_amt20（规模/流动性）</th><th>beta60（市场β）</th><th>mom60（动量）</th><th>vol20（波动）</th><th>rev20（反转/近20日涨幅）</th></tr>
__EXP_ROWS__
</table>
<p><strong>共同风格体：</strong>六信号全部 <span class="down">低 β</span>（−0.28 ~ −0.42）、全部正 mom60/rev20——典型"低β强势股回调"风格。bp1/补票 的 rev20 暴露最高（+1.30 / +1.73），接盘属性最强；搬砖独有负 vol20（−0.10），是唯一偏低波的信号。</p>

<h3>4.2 风格调整后 alpha（raw → 中性化残差）★ 改变系列叙事</h3>
<table>
<tr><th>信号</th><th>raw 均值</th><th>风格调整后 abn</th><th>t</th><th>训练期 abn</th><th>验证期 abn</th><th>判定</th></tr>
__ALPHA_ROWS__
</table>
__ALPHA_CHART__
<div class="key">
<strong>逐信号判词：</strong>① <strong>补票</strong>——raw +1.31% 中有 +0.75% 是同风格票拿不到的（按日 t=2.62），验证期 +0.45% 仍存活，<strong>六信号中唯一真 alpha</strong>；② <strong>B1/B2</strong>——abn ≈ +0.03%（按日 t=1.59 / −2.83），raw 全是"低β强势回调"风格 β，B2 日度口径残差甚至略负；③ <strong>搬砖</strong>——raw +1.54% 最高但 abn <span class="down">−0.11%</span>（按日 t=−1.92），同风格票做得更好；④ <strong>bp1</strong>——abn <span class="down">−1.12%</span>（按日 t=−4.18，两期都约 −1.1%），突破追入在同风格票中系统性亏钱；⑤ <strong>bp2</strong>——abn +0.56%（按日 t=3.39，验证 +0.61%）有真残差，但 raw 只有 +0.58%，扣 0.30% 费率后所剩无几，基率不足以致无操作价值。
</div>
<div class="note">t 主口径为<strong>按日聚合</strong>（每日信号行残差均值为观测，消除同日截面相关对 t 的膨胀，D43），括号内为信号行池化 t（偏高）。按日口径下：补票 t=2.62、bp2 t=3.39、bp1 t=−4.18 绝对值 &gt; 2 结论不变；B2 池化 t=0.50 但按日 t=−2.83——池化被少数大残差行抬正，按日更真实。</div>
</div>

<!-- ============ 五、模块C/D ============ -->
<h2>五、模块 C/D：换手率、持仓结构与自相关</h2>
<div class="card">
<table>
<tr><th>信号</th><th>日均换手</th><th>换手中位</th><th>持仓重叠率</th><th>信号粘性</th><th>持仓日数</th><th>NAV ac(1)</th></tr>
__TURN_ROWS__
</table>
<div class="note">
<strong>读法：</strong>① 94-96% 的持仓重叠是 <strong>20 日持有结构的机械结果</strong>（每天只换掉 1/20 仓位），不代表信号粘性；真正的信号连发概率是粘性列——bp2 <span class="up">43.1%</span> 最高（均线回踩常连日触发，与其 pullback 持续多日一致），补票 7.1% 最低（单针事件一次性）。② 日均换手 8.2-12.2%，按 0.30% 往返、20 日持有摊销 ≈ 1.5bp/日费拖，已计入 NAV。③ NAV lag-1 自相关 +0.04~+0.08——轻微正但远不足以做 P&amp;L 动量再入场；日收益近似独立，<strong>"连续 10 次止损是常态"的既有铁律与此一致</strong>（没有均值回复结构保护连亏）。
</div>
</div>

<!-- ============ 六、修正 ============ -->
<h2>六、对前六章结论的修正（本报告的增量）</h2>
<div class="card">
<table>
<tr><th>信号</th><th>前章结论</th><th>体检修正后</th></tr>
__FIX_ROWS__
</table>
<div class="key">
<strong>系列一句话（修正版）：</strong>六信号里<strong>真残差 alpha 只有补票一个</strong>（验证期存活）；B1/B2/搬砖是"低β强势回调"风格体（raw 边际来自风格 β）；bp1 是负 alpha（追高接盘）；bp2 有小残差但基率不足。原系列"只剩补票有条件可用"的处置结论<strong>方向不变，但论据换轨</strong>：从"绝对收益相对最好"换成"唯一中性化后仍有正 alpha"——这对产品化更重要，因为风格体的收益会随风格轮动消失，而补票的残差不依赖风格延续。
</div>
<div class="warn">
<strong>但也不许过度翻案：</strong>补票 cohort 绝对收益（止损8% +45.6%，门禁后 +52.5%）仍落后基准 +56.8%——"唯一 alpha"不等于"值得上实盘"。在修复验证完成前（用户全局纪律），全系列仍不实盘。风格体信号若要用，只能以风格择时工具的身份、并接受风格回撤，不能以"选股 alpha"记账。
</div>
</div>

<!-- ============ 七、偏离登记 ============ -->
<h2>七、偏离登记（本报告新增）</h2>
<div class="card">
<table>
<tr><th>编号</th><th>偏离</th><th>处置</th></tr>
<tr><td>D41</td><td>真 Barra 风格不可得（市值/行业/基本面缺失），用 OHLCV 代理五因子；行业暴露无法检验</td><td>如实登记；风格归因结论仅对五代理成立，行业/市值残差风险不排除</td></tr>
<tr><td>D42</td><td>二元稀疏信号的全截面 IC 被稀疏度机械压缩，|IC| 小 ≠ 无边际</td><td>IC 仅用于"能否当排序因子"判定（答案：否）；边际以信号行均值 + 风格 alpha 为准</td></tr>
<tr><td>D43</td><td>abn 若按信号行池化计算 t，同日截面相关会膨胀显著性</td><td>主口径改按日聚合 t；两口径并列披露</td></tr>
<tr><td>D12（继承）</td><td>不做行业/市值中性化（数据源不含）</td><td>由本报告代理风格中性化部分覆盖，登记保留</td></tr>
</table>
</div>

<!-- ============ 八、结论 ============ -->
<h2>八、验收结论与后续</h2>
<div class="card">
<div class="key">
<strong>检查单闭环：</strong>IC ✅ / RankIC ✅ / IC 衰减 ✅ / 风格暴露 ✅（代理）/ 风格 alpha ✅ / 换手 ✅ / 自相关 ✅——五项标准要求全部补齐六信号，前六章的对应缺口以本报告为准合并解读。
</div>
<p><strong>后续（按优先级）：</strong>① 若接入市值/行业/基本面数据源，用真 Barra 重做模块 B，验证补票残差 alpha 在行业/市值中性化后是否存活（最关键复核）；② 补票门禁组合的换手敏感性（费率 0.30%→0.5% 压力）；③ 风格体信号（B1/B2/搬砖）若保留，加风格拥挤度开关（风格滚动收益转负即停）；④ 本报告口径并入六信号系列报告模板，后续任何新信号的体检单必须含本五项。</p>
<p class="muted">配套文件：healthcheck_results.json（原始数值）｜ factor_healthcheck.py（体检脚本）｜ 前六章：B1/B2/bp1/补票/搬砖/bp2 重构版报告。</p>
</div>

</div>
</body>
</html>"""

html = (html.replace("__AUDIT_ROWS__", audit_rows)
            .replace("__IC_ROWS__", ic_rows)
            .replace("__IC_CHART__", ic20_chart)
            .replace("__EXP_ROWS__", exp_rows)
            .replace("__ALPHA_ROWS__", alpha_rows)
            .replace("__ALPHA_CHART__", alpha_chart)
            .replace("__TURN_ROWS__", turn_rows)
            .replace("__FIX_ROWS__", fix_rows)
            .replace("__N_SIG_ROWS__", n_sig_rows)
            .replace("__META_NOTE__", meta_note))

out = os.path.join(HERE, "六信号因子体检补全报告_20261020.html")
open(out, "w", encoding="utf-8").write(html)
print("saved ->", out, len(html), "chars")

# 表外 tr/td 校验（浏览器会丢弃 <table> 外的 tr/td）
import re
tbl_spans = [(m.start(), m.end()) for m in re.finditer(r"<table[\s\S]*?</table>", html)]
bad = 0
for m in re.finditer(r"</?t[dhr][\s>]", html):
    if not any(a <= m.start() < b for a, b in tbl_spans):
        bad += 1
        print("  [WARN] 表外标签 @", m.start(), html[max(0, m.start()-40):m.start()+20].replace("\n", " "))
print("表外 tr/td 校验:", "通过" if bad == 0 else "%d 处违规" % bad)
