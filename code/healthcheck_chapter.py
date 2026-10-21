# -*- coding: utf-8 -*-
"""因子体检补全章（IC/RankIC/风格暴露/风格alpha/换手/自相关）——注入各因子专门报告。
用法：from healthcheck_chapter import inject; html = inject(html, "<signal_key>")
数据源：healthcheck_results.json（factor_healthcheck.py 产出，按日聚合 t 为主口径）。
注入位置：报告开头第一张 key 卡后放"体检修正"警示框；<h2>偏离登记</h2> 前插"附·因子体检补全"章。
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
H = json.load(open(os.path.join(HERE, "healthcheck_results.json"), encoding="utf-8"))

SHORT = {"B1(倒车/多空)": "B1", "B2": "B2", "bp1(突破追入)": "bp1",
         "补票(单针下20)": "补票", "搬砖(多空临界)": "搬砖", "bp2(均线回踩)": "bp2"}

# 每信号叙事（数值以 healthcheck_results.json 为准，此处为判词）
NARR = {
    "B1(倒车/多空)": {
        "top": "风格中性化后残差仅 +0.03%（按日 t=1.59 不显著）——raw +1.25% 的边际几乎全是『低β强势回调』风格 β，不是选股 alpha。组合层绝对结论不变，归因必须换轨。",
        "verdict": ("tag-gray", "全是风格"),
        "ic": "六项 IC 全负且弱（rank h=20 −0.0038，t=−2.51）——无截面排序力；『倒车/多空』语义是事件触发器，不是排序因子。",
        "exp": "低β（−0.42）+ 高 mom60/rev20（+0.39/+0.34）+ 高 vol20（+0.18）——典型『低β强势回调』风格体。",
        "fix": "<strong>降级：全是风格。</strong>raw +1.25% → abn +0.03%（按日 t=1.59 不显著；训练 +0.10 / 验证 −0.15）。B1 的绝对收益来自『低β强势回调』风格 β，不是选股能力——风格轮动消失时这部分收益会消失。原 cohort −3.8% 判定不变。",
    },
    "B2": {
        "top": "风格中性化后残差 +0.03%（按日 t=−2.83，日度口径略负）——cohort +16.3% 的收益来自风格 β 不是选股 alpha。",
        "verdict": ("tag-gray", "全是风格·日度残差略负"),
        "ic": "rank h=10 −0.0066（t=−5.02）为全系列第二负——触发日后 10 日截面排名偏后。",
        "exp": "与 B1 同画像：β −0.38、mom60 +0.34、vol20 +0.19（rev20 +0.08 相对温和）——低β强势回调风格体。",
        "fix": "<strong>降级：全是风格。</strong>raw +1.05% → abn +0.03%（池化 t=0.50，但按日 t=−2.83 略负——池化被少数大残差行抬正）。同 B1，风格体；『六信号中游』的 cohort +16.3% 应记为风格 β 收益。",
    },
    "bp1(突破追入)": {
        "top": "风格中性化后残差 −1.12%（按日 t=−4.18）——确认负 alpha：同风格票中系统性亏钱，rev20 暴露 +1.30 是接盘标志。原『全系列最差』判定成立且加严。",
        "verdict": ("tag-red", "负 alpha"),
        "ic": "rank h=10 −0.0168（t=−15.4）全系列最强负值——突破追入系统性买在截面高点，与负 alpha、彩票型曲线三方互证。",
        "exp": "rev20 +1.30 全系列最高 + mom60 +0.65——接盘属性最强的突破追入，与其负 alpha 直接对应。",
        "fix": "<strong>确认并加严：负 alpha。</strong>raw −0.29% → abn −1.12%（按日 t=−4.18；训练 −1.16 / 验证 −1.02 两期一致）。bp1 不只是绝对收益差（cohort −42.7%），而是<strong>同风格票中也系统性跑输</strong>——买强势突破票的钱被别的强势票赚走。处置：下线。",
    },
    "补票(单针下20)": {
        "top": "升级：风格中性化后残差 +0.75%（按日 t=2.62，验证 +0.45% 存活）——六信号中唯一真残差 alpha。可用性论据换轨为『中性化后仍有正 alpha』（绝对收益仍落后基准 +56.8%，不许过度翻案）。",
        "verdict": ("tag-green", "唯一残差 alpha·样本外存活"),
        "ic": "唯一 Pearson 为正的信号（h=20 +0.0017，t=1.50 弱）——弱正排序力，与残差 alpha 方向一致。",
        "exp": "rev20 +1.73 / mom60 +1.11 全系列最高 + ln_amt +0.30——高动量高波动强势票上的事件；其残差 alpha 是在这桶票里仍跑赢同风格票。",
        "fix": "<strong>升级：唯一残差 alpha。</strong>raw +1.31% 中有 +0.75% 是同风格票拿不到的（按日 t=2.62；训练 +0.84 / 验证 +0.45 存活）。<strong>但不许过度翻案：</strong>cohort 绝对收益（止损8% +45.6%，门禁后 +52.5%）仍落后基准 +56.8%——『唯一 alpha』≠『可上实盘』，修复验证完成前不实盘（全局纪律）。",
    },
    "搬砖(多空临界)": {
        "top": "改写：raw +1.54% 全系列最高，但残差 −0.11%（按日 t=−1.92）——『最弱正边际』实为零/微负 alpha，raw edge 全是风格暴露。",
        "verdict": ("tag-red", "全是风格·残差偏负"),
        "ic": "rank h=20 −0.0065（t=−4.21）——信号票在当日截面中排名靠后，raw 均值高是风格桶抬的。",
        "exp": "唯一负 vol20（−0.10）+ 低β（−0.28）——六信号中唯一偏低波的画像；残差转负说明其选股不如同风格低波票。",
        "fix": "<strong>改写：raw 全是风格、残差偏负。</strong>raw +1.54%（全系列最高）→ abn −0.11%（按日 t=−1.92；训练 −0.06 / 验证 −0.18）。前章『信号层边际 +0.13 pct 全系列最低』的表述不够狠——中性化后根本没有正边际，是低波风格桶的顺风。处置维持不通过。",
    },
    "bp2(均线回踩)": {
        "top": "部分翻案：残差 +0.56%（按日 t=3.39，验证 +0.61% 存活）——有真残差，但 raw 才 +0.58%（低于宇宙 +1.41%），基率不足；处置维持下线/池内观察。",
        "verdict": ("tag-amber", "有残差·基率不足"),
        "ic": "pearson h=20 ≈ 0（+0.0000）、rank 弱负——完全无排序力，『均线回踩』不构成选股依据。",
        "exp": "mom60 +0.96 / rev20 +0.55 强势画像 + β −0.35；raw +0.58% 里 +0.56% 是残差——不靠风格，但绝对水平不够。",
        "fix": "<strong>部分翻案：有残差但基率不足。</strong>abn +0.56%（按日 t=3.39；训练 +0.55 / 验证 +0.61 两期一致）——『均线回踩』有真实残差选股力，<strong>这是本章修正的增量</strong>；但 raw +0.58% 低于宇宙 +1.41%、扣 0.30% 费率所剩无几，cohort +6.2% vs 基准 +56.8% 的巨大差距无法翻转。处置维持：下线或极窄甜区（pullback&lt;0.2）池内观察。",
    },
}

def f2(x): return ("%+.2f" % x) if x is not None else "—"
def f3(x): return ("%+.3f" % x) if x is not None else "—"
def f4(x): return ("%+.4f" % x) if x is not None else "—"
def cls(x): return "up" if (x or 0) > 0 else "down"

def top_box(key):
    n = NARR[key]
    return ('<div class="warn"><strong>【2026-10-20 因子体检修正】</strong>%s'
            '（详见文末『附·因子体检补全』章）</div>\n' % n["top"])

def chapter(key):
    d = H[key]; n = NARR[key]; sh = SHORT[key]
    # --- IC 表 ---
    ic_rows = ""
    for kind, kn in [("pearson", "Pearson"), ("rank", "Rank")]:
        for h in ("5", "10", "20"):
            v = d["ic_full"][kind][h]
            if not v:
                ic_rows += "<tr><td>%s</td><td>h=%s</td><td colspan='4'>—</td></tr>" % (kn, h)
                continue
            ic_rows += ("<tr><td>%s</td><td>h=%s</td><td class='%s'>%s</td><td>%s</td>"
                        "<td>%.1f%%</td><td class='%s'>%+.2f</td></tr>") % (
                kn, h, cls(v["mean"]), f4(v["mean"]), f2(v["icir"]),
                v["pos_ratio"], cls(v["t"]), v["t"])
    # --- 暴露表 ---
    exp_rows = ""
    for k, lbl in [("ln_amt", "ln_amt20（规模/流动性）"), ("beta", "beta60（市场β）"),
                   ("mom60", "mom60（动量）"), ("vol20", "vol20（波动）"), ("rev20", "rev20（近20日涨幅）")]:
        v = d["exposure"].get(k)
        if not v:
            exp_rows += "<tr><td>%s</td><td colspan='2'>—</td></tr>" % lbl
            continue
        exp_rows += "<tr><td>%s</td><td class='%s'>%s</td><td class='%s'>%+.2f</td></tr>" % (
            lbl, cls(v["mean"]), f2(v["mean"]), cls(v["t"]), v["t"])
    # --- alpha 表 ---
    a = d["style_alpha"]
    t_d = a.get("abn_t_daily"); t_p = a.get("abn_t")
    t_show = t_d if t_d is not None else t_p
    t_note = "（按日 t；池化 %s）" % f2(t_p) if t_d is not None else "（池化）"
    alpha_rows = ("<tr><td>raw 均值</td><td class='%s'>%s%%</td></tr>"
                  "<tr><td>风格调整后 abn</td><td class='%s'>%s%%</td></tr>"
                  "<tr><td>t %s</td><td class='%s'>%s</td></tr>"
                  "<tr><td>训练期 abn / 验证期 abn</td><td><span class='%s'>%s%%</span> / <span class='%s'>%s%%</span></td></tr>"
                  "<tr><td>信号行数 / 观测日数</td><td>%s / %s</td></tr>") % (
        cls(a["raw_mean"]), f2(a["raw_mean"]), cls(a["abn_mean"]), f2(a["abn_mean"]),
        t_note, cls(t_show), f2(t_show),
        cls(a["abn_train"]), f2(a["abn_train"]), cls(a["abn_valid"]), f2(a["abn_valid"]),
        "{:,}".format(a["n"]), "{:,}".format(a.get("abn_days") or 0))
    # --- 换手表 ---
    t = d["turnover"]
    turn_rows = ("<tr><td>日均换手 / 换手中位</td><td>%.1f%% / %.1f%%</td></tr>"
                 "<tr><td>持仓重叠率</td><td>%.1f%%</td></tr>"
                 "<tr><td>信号粘性 P(sig[t+1]|sig[t])</td><td>%.1f%%</td></tr>"
                 "<tr><td>持仓日数</td><td>%s</td></tr>"
                 "<tr><td>NAV lag-1 自相关</td><td class='%s'>%s</td></tr>") % (
        100*t["mean_daily"], 100*t["median_daily"], 100*t["overlap"],
        100*t["stickiness"], "{:,}".format(t["n_days"]), cls(d["nav_ac1"]), f3(d["nav_ac1"]))
    vtag, vtxt = n["verdict"]
    return """
<!-- ============ 附·因子体检补全（2026-10-20 补入） ============ -->
<h2>附·因子体检补全：IC / RankIC / 风格暴露 / 换手 / 自相关（2026-10-20）</h2>
<div class="card">
<p><strong>为什么要补：</strong>这五项是标准因子评估协议的固定检查项——IC/RankIC 答"有没有排序力"、IC 衰减答"预测力半衰期"、风格暴露与调整 alpha 答"是不是风格马甲"、换手答"成本可不可行"、自相关答"时序有无可利用结构"。此前系列模板存在缺口：B1/B2/bp1 章做过全截面 RankIC，但 Pearson IC 与 IC 衰减全缺；风格暴露只有一行 D12"不做中性化"的登记（<strong>登记 ≠ 已检验</strong>）；换手/自相关从未进模板。本章对 <strong>%s</strong> 一次性补齐（脚本 factor_healthcheck.py，数值 healthcheck_results.json），与前章冲突处以本章为准。</p>
<div class="warn"><strong>风格因子局限（D41）：</strong>真 Barra 风格需市值/行业/基本面，数据源不含（D12 属实）——本章用 OHLCV 代理五因子（ln成交额/β60/mom60/vol20/rev20），行业与市值残差风险不排除。t 主口径为<strong>按日聚合</strong>（每日信号行残差均值为观测，消除同日截面相关对 t 的膨胀，D43）。</div>
</div>

<div class="card">
<h3>附.1 全截面 IC（0/1 信号哑变量 × 当日截面未来收益）</h3>
<table>
<tr><th>类型</th><th>持有期</th><th>IC 均值</th><th>ICIR</th><th>正IC占比</th><th>t</th></tr>
%s
</table>
<p>%s</p>
<p class="muted">语义限制（D42）：二元稀疏信号的点二列相关被稀疏度机械压缩，|IC| 小 ≠ 无边际——IC 只回答"能否当排序因子"；事件触发器的边际以信号行均值与下方风格 alpha 为准。</p>
</div>

<div class="card">
<h3>附.2 风格暴露与风格调整后 alpha</h3>
<table>
<tr><th>风格代理</th><th>信号行截面 z 均值</th><th>t（按日）</th></tr>
%s
</table>
<p>%s</p>
<table>
<tr><th colspan="2">20 日收益的日截面中性化归因（r20 ~ 1 + 5 风格 z）</th></tr>
%s
</table>
<div class="key"><strong>对本章结论的修正（判定：%s）：</strong>%s</div>
</div>

<div class="card">
<h3>附.3 换手率、持仓结构与自相关</h3>
<table>
<tr><th>指标</th><th>值</th></tr>
%s
</table>
<p>读法：持仓重叠 94-96%% 是 <strong>20 日持有结构的机械结果</strong>（每天只换 1/20 仓位），不反映信号粘性；粘性列才是信号连发概率。日均换手 8-12%% 按 0.30%% 往返 ≈ 1.5bp/日费拖（已计入 NAV）。ac1 +0.04~+0.08 无时序结构可挖——"连续止损是常态"与此一致。</p>
</div>

<div class="card">
<h3>附.4 偏离登记补录（与偏离登记章清单并读）</h3>
<table>
<tr><th>编号</th><th>偏离</th><th>处置</th></tr>
<tr><td>D41</td><td>真 Barra 风格不可得（市值/行业/基本面缺失），用 OHLCV 代理五因子</td><td>风格归因结论仅对五代理成立；行业/市值残差风险不排除</td></tr>
<tr><td>D42</td><td>二元稀疏信号的全截面 IC 被稀疏度机械压缩</td><td>IC 仅用于排序力判定；边际以信号行均值 + 风格 alpha 为准</td></tr>
<tr><td>D43</td><td>abn 按信号行池化计算 t 会被同日截面相关膨胀</td><td>主口径按日聚合 t，双口径披露</td></tr>
</table>
</div>
""" % (sh, ic_rows, n["ic"], exp_rows, n["exp"], alpha_rows, vtag, vtxt, turn_rows)

def inject(html, key):
    """开头 key 卡后插修正框；偏离登记章前插补全章。"""
    tb = top_box(key)
    ch = chapter(key)
    m = re.search(r'<div class="key">[\s\S]*?</div>', html)
    if m:
        html = html[:m.end()] + "\n" + tb + html[m.end():]
    else:
        html = html.replace("<body>", "<body>\n" + tb, 1)
    anchor = None
    for pat in [r'<!--[^>]*偏离登记[^>]*-->\s*<h2>[^<]*</h2>',
                r'<h2>[^<]*偏离登记[^<]*</h2>']:
        mm = re.search(pat, html)
        if mm:
            anchor = mm.start()
            break
    if anchor is None:
        mm = re.search(r'</div>\s*</body>', html)
        anchor = mm.start() if mm else len(html)
    html = html[:anchor] + ch + "\n" + html[anchor:]
    return html
