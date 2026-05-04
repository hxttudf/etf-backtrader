#!/usr/bin/env python3
"""ETF双动量轮动 — 可视化运行界面 (Streamlit)

用法: streamlit run etf_app.py
"""

import sys
from datetime import datetime
from pathlib import Path

import math

import numpy as np
import pandas as pd
import streamlit as st

# Ensure the script dir is on sys.path so imports work both in dev and PyInstaller
sys.path.insert(0, str(Path(__file__).parent))

from etf_data import DEFAULT_CONFIG, calc_indicators, load_config, load_prices, load_prices_extended
from etf_backtrader import run_backtest_bt, position_dist_bt

st.set_page_config(page_title="ETF双动量轮动", layout="wide")
st.title("ETF 双动量轮动")

st.markdown("""
<style>
/* metric 数值不截断，缩小字号完整显示 */
[data-testid="stMetricValue"] {
    font-size: 1.1rem !important;
    white-space: nowrap;
    overflow: visible !important;
    text-overflow: clip !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    white-space: nowrap;
    overflow: visible !important;
    text-overflow: clip !important;
}
/* 列之间减少间距 */
[data-testid="column"] {
    padding-left: 0.3rem !important;
    padding-right: 0.3rem !important;
}
    /* dataframe 按内容自适应宽度，超出屏幕自动滚动 */
    [data-testid="stDataFrame"] > div:first-child {
        width: fit-content !important;
        max-width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def cached_prices(etfs: dict, group_name: str, source: str = "tencent") -> pd.DataFrame:
    return load_prices_extended(etfs, group_name, source=source)


def run_backtest(prices, mode, start_date, end_date, ma_days, roc_days, min_hold=0):
    """Inline backtest so the app stays self-contained."""
    etf_names = list(prices.columns)
    returns = prices.pct_change(fill_method=None)
    ma60, roc20, _ = calc_indicators(prices, ma_days, roc_days)
    COMMISSION = 0.0001  # 万1 per side, 免五
    STAMP_DUTY = 0.0005  # 印花税 0.05%, 卖出时收取

    strat_ret = pd.Series(0.0, index=prices.index)
    holding = None
    trades = 0
    trade_dates = []
    trade_details = []
    is_friday = prices.index.dayofweek == 4
    last_trade_idx = -999

    for i in range(ma_days, len(prices)):
        dt = prices.index[i]

        if holding is not None:
            r = returns[holding].iloc[i]
            strat_ret.iloc[i] = r if not pd.isna(r) else 0.0

        should_check = True if mode == "daily" else is_friday[i]
        if should_check and (min_hold <= 0 or i - last_trade_idx >= min_hold):
            above = {}
            for name in etf_names:
                px = prices[name].iloc[i]
                ma = ma60[name].iloc[i]
                roc = roc20[name].iloc[i]
                if not pd.isna(ma) and px > ma and not pd.isna(roc):
                    above[name] = roc

            new_holding = max(above, key=above.get) if above else None
            if new_holding != holding:
                if holding is not None:
                    strat_ret.iloc[i] -= COMMISSION + STAMP_DUTY
                if new_holding is not None:
                    strat_ret.iloc[i] -= COMMISSION
                trades += 1
                trade_dates.append(dt)
                trade_details.append((dt, holding, new_holding))
                last_trade_idx = i
                holding = new_holding

    trim = (prices.index >= start_date) & (prices.index <= end_date)
    ret = strat_ret[trim]
    bench_ret = returns[trim].mean(axis=1)
    nav = (1 + ret).cumprod()
    bench_nav = (1 + bench_ret).cumprod()
    return nav, bench_nav, ret, bench_ret, trades, trade_dates, trade_details


def calc_metrics(nav, ret):
    r = ret.dropna()
    if len(r) < 5:
        return {}
    total = nav.iloc[-1] - 1
    ann = (1 + total) ** (252 / len(r)) - 1
    vol = r.std() * (252 ** 0.5)
    sharpe = (ann - 0.03) / vol if vol > 0 else 0
    dd_series = nav / nav.cummax() - 1
    dd = dd_series.min()
    calmar = ann / abs(dd) if dd != 0 else 0
    max_loss = (nav - 1).min()
    max_loss_dt = nav.idxmin()
    underwater_days = int((nav < 1).sum())
    holding_days = len(r)
    # longest consecutive period below initial capital (NAV < 1.0)
    longest_loss_days = 0
    longest_loss_start = None
    longest_loss_end = None
    cur_start = None
    cur_len = 0
    for dt, val in nav.items():
        if val < 1.0:
            if cur_start is None:
                cur_start = dt
            cur_len += 1
        else:
            if cur_len > longest_loss_days:
                longest_loss_days = cur_len
                longest_loss_start = cur_start
                longest_loss_end = nav.index[nav.index.get_loc(dt) - 1]
            cur_start = None
            cur_len = 0
    if cur_len > longest_loss_days:
        longest_loss_days = cur_len
        longest_loss_start = cur_start
        longest_loss_end = nav.index[-1]
    loss_range = f"{longest_loss_start.strftime('%Y-%m-%d')} ~ {longest_loss_end.strftime('%Y-%m-%d')}" if longest_loss_start else "N/A"
    return {"累计收益": total, "年化收益": ann, "年化波动": vol, "夏普比率": sharpe,
            "最大回撤": dd, "卡尔玛比率": calmar, "最大亏损": max_loss,
            "最大亏损日期": max_loss_dt, "水下天数": underwater_days, "持有天数": holding_days,
            "最长亏损持续": longest_loss_days, "最长亏损区间": loss_range}


def _nav_one_backtest(prices, daily_ret, ma60, roc20, etf_names, start_date, end_date, mode, min_hold, ma_days):
    """Run one backtest pass, return final NAV. etf_names = active ETF pool.
    Uses additive commission (matching run_backtest) for exact comparability."""
    is_friday = prices.index.dayofweek == 4
    COMMISSION = 0.0001
    STAMP_DUTY = 0.0005
    strat_ret = pd.Series(0.0, index=prices.index)
    holding = None
    last_trade_idx = -999
    for i in range(ma_days, len(prices)):
        dt = prices.index[i]
        if holding is not None and dt >= pd.Timestamp(start_date) and dt <= pd.Timestamp(end_date):
            r = daily_ret[holding].iloc[i]
            if not pd.isna(r):
                strat_ret.iloc[i] = r
        should_check = True if mode == "daily" else is_friday[i]
        if should_check and (min_hold <= 0 or i - last_trade_idx >= min_hold):
            above = {}
            for name in etf_names:
                px = prices[name].iloc[i]
                ma = ma60[name].iloc[i]
                roc = roc20[name].iloc[i]
                if not pd.isna(ma) and px > ma and not pd.isna(roc):
                    above[name] = roc
            new_holding = max(above, key=above.get) if above else None
            if new_holding is not None and new_holding != holding:
                last_trade_idx = i
            if new_holding != holding:
                if holding is not None:
                    strat_ret.iloc[i] -= COMMISSION + STAMP_DUTY
                if new_holding is not None:
                    strat_ret.iloc[i] -= COMMISSION
            holding = new_holding
    trim = (prices.index >= pd.Timestamp(start_date)) & (prices.index <= pd.Timestamp(end_date))
    r = strat_ret[trim].dropna()
    return (1 + r).prod() if len(r) > 0 else 1.0


def position_dist(prices, start_date, end_date, mode, ma_days, roc_days, min_hold=0):
    """返回 (持有天数dict, 买入次数dict, 收益占比dict, 持有期累计收益dict, 胜率dict)
    收益占比 = 各ETF持有期间的对数收益 / 总对数收益，加总=100%，正=赚钱负=亏钱
    持有期累计收益 = 持有该ETF期间的累计收益率"""
    etf_names = list(prices.columns)
    daily_ret = prices.pct_change(fill_method=None)
    ma60, roc20, _ = calc_indicators(prices, ma_days, roc_days)
    is_friday = prices.index.dayofweek == 4
    COMMISSION_RATE = 0.0001 + 0.0005  # 万1佣金 + 万5印花税（卖出），简化为双向万3
    days = {n: 0 for n in etf_names}
    days["CASH"] = 0
    buys = {n: 0 for n in etf_names}
    nav = {n: 1.0 for n in etf_names}
    pos_days = {n: 0 for n in etf_names}
    hold_days_for_wr = {n: 0 for n in etf_names}
    log_ret = {n: 0.0 for n in etf_names}
    log_ret["CASH"] = 0.0
    holding = None
    last_trade_idx = -999
    for i in range(ma_days, len(prices)):
        dt = prices.index[i]
        in_range = dt >= pd.Timestamp(start_date) and dt <= pd.Timestamp(end_date)
        if in_range:
            h = holding or "CASH"
            days[h] += 1
            if h != "CASH":
                r = daily_ret[h].iloc[i]
                if not pd.isna(r):
                    nav[h] *= (1 + r)
                    hold_days_for_wr[h] += 1
                    if r > 0:
                        pos_days[h] += 1
                    log_ret[h] += math.log(1 + r)
        should_check = True if mode == "daily" else is_friday[i]
        if should_check and (min_hold <= 0 or i - last_trade_idx >= min_hold):
            above = {}
            for name in etf_names:
                px = prices[name].iloc[i]
                ma = ma60[name].iloc[i]
                roc = roc20[name].iloc[i]
                if not pd.isna(ma) and px > ma and not pd.isna(roc):
                    above[name] = roc
            new_holding = max(above, key=above.get) if above else None
            if new_holding is not None and new_holding != holding:
                buys[new_holding] += 1
                last_trade_idx = i
                # 佣金从当天持仓的 log return 扣除
                if in_range and holding is not None:
                    log_ret[holding] += math.log(1 - COMMISSION_RATE)
            holding = new_holding

    total_log = sum(log_ret.values())
    contrib = {}
    for name in etf_names:
        contrib[name] = log_ret[name] / total_log if abs(total_log) > 1e-12 else 0.0
    contrib["CASH"] = 0.0

    cum_ret = {}
    for k in etf_names:
        cum_ret[k] = nav[k] - 1.0 if days[k] > 0 else 0.0
    cum_ret["CASH"] = 0.0

    win_rate = {}
    for k in etf_names:
        win_rate[k] = pos_days[k] / hold_days_for_wr[k] if hold_days_for_wr[k] > 0 else 0.0
    win_rate["CASH"] = 0.0

    return days, buys, contrib, cum_ret, win_rate


def yearly_returns(ret):
    result = {}
    for yr in sorted(set(ret.index.year)):
        mask = ret.index.year == yr
        if mask.sum() > 10:
            result[yr] = (1 + ret[mask]).prod() - 1
    return result


def trade_win_rate(ret, trade_dates):
    """计算交易胜率：相邻调仓日之间的持有期收益 > 0 记为一次盈利"""
    if not trade_dates:
        return 0.0
    nav = (1 + ret).cumprod()
    valid = sorted({d for d in trade_dates if d in nav.index})
    if not valid:
        return 0.0
    breaks = [ret.index[0]] + valid
    wins = total = 0
    for i in range(len(breaks) - 1):
        seg = nav.loc[breaks[i]:breaks[i + 1]]
        if len(seg) >= 2:
            total += 1
            if seg.iloc[-1] / seg.iloc[0] - 1 > 0:
                wins += 1
    seg = nav.loc[valid[-1]:]
    if len(seg) >= 2:
        total += 1
        if seg.iloc[-1] / seg.iloc[0] - 1 > 0:
            wins += 1
    return wins / total if total > 0 else 0.0


def grid_search(prices, modes, start, end, ma_values, roc_values, progress_bar):
    """网格搜索最优MA/ROC，返回所有结果DataFrame"""
    import itertools

    rows = []
    total = len(ma_values) * len(roc_values) * len(modes)
    done = 0
    for ma, roc in itertools.product(ma_values, roc_values):
        for mode in modes:
            nav, bnav, ret, bret, trades, trade_dates, trade_details = run_backtest(
                prices, mode, start, end, ma, roc)
            m = calc_metrics(nav, ret)
            wr = trade_win_rate(ret, trade_dates)
            rows.append({
                "MA": ma, "ROC": roc, "模式": mode,
                "累计收益": m.get("累计收益", 0),
                "年化收益": m.get("年化收益", 0),
                "夏普比率": m.get("夏普比率", 0),
                "最大回撤": m.get("最大回撤", 0),
                "最大亏损": m.get("最大亏损", 0),
                "水下天数": m.get("水下天数", 0),
                "持有天数": m.get("持有天数", 0),
                "卡尔玛比率": m.get("卡尔玛比率", 0),
                "胜率": wr,
                "交易次数": trades,
            })
            done += 1
            progress_bar.progress(done / total,
                                  text=f"MA={ma} ROC={roc} {mode}  ({done}/{total})")
    return pd.DataFrame(rows)


def build_optimizer_heatmap(df, metric):
    """Plotly热力图"""
    import plotly.graph_objects as go

    modes = df["模式"].unique()
    figs = {}
    for mode in modes:
        sub = df[df["模式"] == mode].pivot_table(index="MA", columns="ROC", values=metric)
        if sub.empty:
            continue
        fig = go.Figure(data=go.Heatmap(
            z=sub.values,
            x=list(sub.columns),
            y=list(sub.index),
            colorscale="RdYlGn",
            text=np.vectorize(lambda v: f"{v:.2f}" if abs(v) < 10 else f"{v:.1%}")(sub.values),
            texttemplate="%{text}",
            textfont={"size": 8},
            hovertemplate="MA=%{y}<br>ROC=%{x}<br>%{z:.4f}<extra></extra>",
        ))
        fig.update_layout(
            title=f"{METRIC_LABELS.get(metric, metric)} ({mode})",
            xaxis_title="ROC 天数", yaxis_title="MA 天数",
            height=500,
        )
        figs[mode] = fig
    return figs


METRIC_LABELS = {
    "累计收益": "累计收益", "年化收益": "年化收益", "夏普比率": "夏普比率",
    "最大回撤": "最大回撤", "最大亏损": "最大亏损", "水下天数": "水下天数",
    "卡尔玛比率": "卡尔玛比率", "胜率": "胜率",
}

# Percentage columns in optimizer results
_OPT_PCT_COLS = {"累计收益", "年化收益", "最大回撤", "最大亏损", "胜率"}
_OPT_FLOAT_COLS = {"夏普比率", "卡尔玛比率"}
_OPT_INT_COLS = {"水下天数", "持有天数", "交易次数"}


def _fmt_optimize_table(df: pd.DataFrame) -> None:
    """In-place format optimizer result columns for display."""
    for c in _OPT_PCT_COLS:
        if c in df.columns:
            df[c] = df[c].map(lambda v: f"{v:.1%}")
    for c in _OPT_FLOAT_COLS:
        if c in df.columns:
            df[c] = df[c].map(lambda v: f"{v:.2f}")
    for c in _OPT_INT_COLS:
        if c in df.columns:
            df[c] = df[c].astype(int)


def build_plotly_fig(prices, etf_codes, modes_data, start, end, cmp_data=None):
    """Reuse the same Plotly logic as etf_backtest.plot_interactive but embedded.
    cmp_data: optional dict of {mode: (nav, bnav, ret, bret, trades, trade_dates, trade_details)}
    for the min_hold=10 comparison variant."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    strategy_colors = {"daily": "#2196F3", "friday": "#FF9800"}
    cmp_colors = {"daily": "#64B5F6", "friday": "#FFB74D"}
    etf_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.05,
        subplot_titles=("净值曲线", "回撤"),
    )

    nav0 = prices[prices.index >= start].copy()
    etf_color_map = {}
    for i, name in enumerate(prices.columns):
        etf_nav = (1 + nav0[name].pct_change().fillna(0)).cumprod()
        code = etf_codes.get(name, "")
        label = f"持有{name} ({code})" if code else f"持有{name}"
        color = etf_colors[i % len(etf_colors)]
        etf_color_map[name] = color
        fig.add_trace(go.Scatter(
            x=etf_nav.index, y=etf_nav.values, mode="lines",
            name=label, line=dict(color=color, width=1.5), opacity=0.7,
        ), row=1, col=1)

    bench_added = False
    for mode, (nav, bnav, trade_dates, trade_details) in modes_data.items():
        color = strategy_colors.get(mode, "black")

        cum_ret = nav - 1
        ret_customdata = [(d.strftime("%Y-%m-%d"), r) for d, r in zip(nav.index, cum_ret)]
        hovertemplate = f"策略({mode}) " + "%{customdata[0]} 净值%{y:.3f} 累计%{customdata[1]:+.2%}<extra></extra>"

        fig.add_trace(go.Scatter(
            x=nav.index, y=nav.values, mode="lines",
            name=f"策略({mode})",
            line=dict(color=color, width=2.5),
            customdata=ret_customdata,
            hovertemplate=hovertemplate,
        ), row=1, col=1)

        # Underwater
        uw = nav.copy()
        uw[uw >= 1] = pd.NA
        if uw.notna().any():
            fig.add_trace(go.Scatter(
                x=nav.index, y=uw.values, mode="lines",
                name=f"水下({mode})",
                line=dict(color="#DC2626", width=3),
                hovertemplate=f"水下({mode}) " + "%{customdata[0]} 净值%{y:.3f}<extra></extra>",
                customdata=ret_customdata,
            ), row=1, col=1)

        # Comparison variant (min_hold=10) — dashed line + trade markers
        if cmp_data and mode in cmp_data:
            c_nav, _, _, _, _, _, c_tdets = cmp_data[mode]
            c_color = cmp_colors.get(mode, "#888")
            c_cum_ret = c_nav - 1
            c_customdata = [(d.strftime("%Y-%m-%d"), r) for d, r in zip(c_nav.index, c_cum_ret)]
            fig.add_trace(go.Scatter(
                x=c_nav.index, y=c_nav.values, mode="lines",
                name=f"策略-最小持有10天({mode})",
                line=dict(color=c_color, width=2.5, dash="dash"),
                customdata=c_customdata,
                hovertemplate=f"策略-最小持有10天({mode}) " + "%{customdata[0]} 净值%{y:.3f} 累计%{customdata[1]:+.2%}<extra></extra>",
            ), row=1, col=1)
            # Trade markers for comparison
            if c_tdets:
                c_valid = [(dt, frm, to) for dt, frm, to in c_tdets if dt in c_nav.index]
                if c_valid:
                    c_dates = [dt for dt, _, _ in c_valid]
                    c_navs = [c_nav.loc[dt] for dt in c_dates]
                    c_mcolors = [etf_color_map.get(to, "#999") for _, _, to in c_valid]
                    c_htexts = []
                    for dt, frm, to in c_valid:
                        dt_str = dt.strftime("%Y-%m-%d")
                        if frm is None: c_htexts.append(f"{dt_str} 买入 {to}")
                        elif to is None: c_htexts.append(f"{dt_str} 卖出 {frm} → 空仓")
                        else: c_htexts.append(f"{dt_str} 卖出 {frm} → 买入 {to}")
                    fig.add_trace(go.Scatter(
                        x=c_dates, y=c_navs, mode="markers",
                        name=f"调仓-最小持有10天({mode})",
                        marker=dict(color=c_mcolors, size=8, symbol="triangle-up",
                                   line=dict(color="white", width=1)),
                        hovertemplate="%{customdata} 净值%{y:.3f}<extra></extra>",
                        customdata=c_htexts,
                    ), row=1, col=1)

        # Trade markers
        if trade_details:
            valid = [(dt, frm, to) for dt, frm, to in trade_details if dt in nav.index]
            if valid:
                dates = [dt for dt, _, _ in valid]
                navs_v = [nav.loc[dt] for dt in dates]
                marker_colors = [etf_color_map.get(to, "#999") for _, _, to in valid]
                hover_texts = []
                for dt, frm, to in valid:
                    dt_str = dt.strftime("%Y-%m-%d")
                    if frm is None:
                        hover_texts.append(f"{dt_str}<br>买入 <b>{to}</b>")
                    elif to is None:
                        hover_texts.append(f"{dt_str}<br>卖出 <b>{frm}</b> → <b>空仓</b>")
                    else:
                        hover_texts.append(f"{dt_str}<br>卖出 <b>{frm}</b> → 买入 <b>{to}</b>")
                fig.add_trace(go.Scatter(
                    x=dates, y=navs_v, mode="markers",
                    name=f"调仓({mode})",
                    marker=dict(color=marker_colors, size=10, symbol="diamond",
                               line=dict(color="white", width=1)),
                    hovertemplate="%{customdata} 净值%{y:.3f}<extra></extra>",
                    customdata=hover_texts,
                ), row=1, col=1)

        # Drawdown
        dd = nav / nav.cummax() - 1
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, mode="lines",
            name=f"回撤({mode})",
            line=dict(color=color, width=1),
            fill="tozeroy", fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.2)",
            showlegend=False,
        ), row=2, col=1)

        if not bench_added:
            fig.add_trace(go.Scatter(
                x=bnav.index, y=bnav.values, mode="lines",
                name="等权基准", line=dict(color="#666", width=1.2, dash="dot"),
            ), row=1, col=1)
            bench_added = True

    fig.update_xaxes(range=[start, end], row=1, col=1)
    fig.update_xaxes(range=[start, end], row=2, col=1)
    fig.update_yaxes(title_text="累计净值", row=1, col=1)
    fig.update_yaxes(title_text="回撤", tickformat=".0%", row=2, col=1)
    fig.update_layout(
        hovermode="x unified",
        legend=dict(font=dict(size=10), orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template="plotly_white",
        height=650,
    )
    return fig


def signal_for_date(prices, target_date, ma_days, roc_days):
    ma60, roc20, day_chg = calc_indicators(prices, ma_days, roc_days)
    dt = pd.Timestamp(target_date)
    if dt not in prices.index:
        available = prices.index[prices.index <= dt]
        if len(available) == 0:
            return None, None, None
        dt = available[-1]

    rows = []
    qualified = {}
    for name in prices.columns:
        px = float(prices[name].loc[dt])
        ma = float(ma60[name].loc[dt]) if not pd.isna(ma60[name].loc[dt]) else float("nan")
        roc = float(roc20[name].loc[dt]) if not pd.isna(roc20[name].loc[dt]) else float("nan")
        chg = float(day_chg[name].loc[dt]) if not pd.isna(day_chg[name].loc[dt]) else float("nan")
        rows.append({"ETF": name, "收盘价": px, "当日涨幅": chg, f"{ma_days}日均线": ma, f"{roc_days}日涨幅": roc})
        if not pd.isna(ma) and px > ma and not pd.isna(roc):
            qualified[name] = roc

    df = pd.DataFrame(rows)
    best = max(qualified, key=qualified.get) if qualified else None
    return best, df, dt


# ── Session state ──────────────────────────────────────────
if "cfg" not in st.session_state:
    st.session_state.cfg = load_config()

cfg = st.session_state.cfg

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.header("📊 回测参数")

# ── Restore from URL query params (survives browser refresh) ──
qp = st.query_params
_qp = lambda k, d: qp[k] if k in qp else d

# Group selector + config button
col1, col2 = st.sidebar.columns([3, 1])
with col1:
    group_names = list(cfg["groups"].keys())
    sel_group = st.selectbox("组合", group_names,
                             index=group_names.index(_qp("g", group_names[0])) if _qp("g", group_names[0]) in group_names else 0,
                             key="sel_group")
with col2:
    st.write(" ")
    if st.button("⚙️", help="管理组合", use_container_width=True):
        st.session_state.show_config = not st.session_state.get("show_config", False)

# Group config expander
if st.session_state.get("show_config", False):
    with st.sidebar.expander("组合管理", expanded=True):
        import json
        raw = json.dumps(cfg["groups"], ensure_ascii=False, indent=2)
        edited = st.text_area(
            "直接编辑JSON", raw, height=300, key="cfg_json",
            help="格式: {\"组合名\": {\"ETF名\": \"代码\", ...}}",
        )
        c1, c2 = st.columns(2)
        if c1.button("💾 保存", type="primary", use_container_width=True):
            try:
                parsed = json.loads(edited)
                cfg["groups"] = parsed
                with open(DEFAULT_CONFIG, "w") as f:
                    json.dump(cfg, f, ensure_ascii=False, indent=2)
                st.success("已保存")
            except json.JSONDecodeError as e:
                st.error(f"JSON格式错误: {e}")
        if c2.button("↩ 撤销", use_container_width=True):
            st.session_state.pop("cfg_json", None)
            st.rerun()

# Re-sync group list after edits
group_names = list(cfg["groups"].keys())
if sel_group not in group_names:
    sel_group = group_names[0]

start_date = st.sidebar.date_input("开始日期",
    pd.Timestamp(_qp("start", "2025-04-30")),
    min_value=pd.Timestamp("2010-01-01"),
    max_value=pd.Timestamp.today(),
    key="sb_start")
end_date = st.sidebar.date_input("结束日期",
    pd.Timestamp(_qp("end", datetime.today().strftime("%Y-%m-%d"))),
    min_value=pd.Timestamp("2010-01-01"),
    max_value=pd.Timestamp.today(),
    key="sb_end")
mode = st.sidebar.radio("调仓模式", ["daily", "friday", "both"], horizontal=True,
                        index=["daily","friday","both"].index(_qp("mode", "daily")),
                        format_func=lambda x: {"daily": "每日", "friday": "周五", "both": "两者"}[x], key="sb_mode")
source = st.sidebar.selectbox("数据源", ["tencent", "akshare"],
                              index=["tencent","akshare"].index(_qp("src", "akshare")),
                              format_func=lambda x: {"tencent": "腾讯财经", "akshare": "AKShare"}[x], key="sb_source")
source_hint = {"tencent": "⚠️ 仅约800交易日（~3年）", "akshare": "✅ 全历史（自ETF上市起, 2011+）"}
st.sidebar.caption(source_hint[source])
ma_days = st.sidebar.slider("MA 均线天数", 10, 200, int(_qp("ma", "60")), step=5, key="sb_ma")
roc_days = st.sidebar.slider("ROC 动量天数", 5, 120, int(_qp("roc", "25")), step=5, key="sb_roc")
compare_min_hold = st.sidebar.checkbox("最小持有10天对比", value=_qp("cmp", "0") == "1",
    help="开启后同时显示 原始策略 vs 最小持有10天 两条曲线")
run_btn = st.sidebar.button("🚀 开始回测", type="primary", use_container_width=True)
use_backtrader = st.sidebar.checkbox("使用 Backtrader 引擎", value=True, help="勾选使用 backtrader 专业回测引擎，取消使用手写回测")

# ── Persist to URL query params (survives F5 refresh) ──
st.query_params.update({
    "g": sel_group, "start": str(start_date), "end": str(end_date),
    "mode": mode, "src": source, "ma": str(ma_days), "roc": str(roc_days),
    "cmp": "1" if compare_min_hold else "0",
})

st.sidebar.divider()
st.sidebar.header("🔍 参数遍历")
optimize = st.sidebar.checkbox("开启 (搜索最优MA×ROC配置)", value=False)
if optimize:
    opt_ma_step = st.sidebar.slider("MA 步长", 5, 40, 10, help="MA 遍历步长（范围 10~200）")
    opt_roc_step = st.sidebar.slider("ROC 步长", 5, 40, 10, help="ROC 遍历步长（范围 5~120）")
    opt_metric = st.sidebar.selectbox("排序指标", ["夏普比率", "卡尔玛比率", "年化收益", "胜率", "最大亏损", "水下天数"])
    opt_top = st.sidebar.number_input("显示前N组", 5, 100, 20, step=5)

st.sidebar.divider()
st.sidebar.header("📡 每日信号")
sig_date = st.sidebar.date_input("查询日期", pd.Timestamp.today(), key="sig_date")
sig_btn = st.sidebar.button("🔍 查询信号", use_container_width=True)

# ── Main area ────────────────────────────────────────────
if run_btn:
    etfs = cfg["groups"][sel_group]
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    lookback = (pd.Timestamp(start_str) - pd.Timedelta(days=200)).strftime("%Y-%m-%d")

    with st.spinner("加载数据 & 运行回测..."):
        prices_full = cached_prices(etfs, sel_group, source=source)
        prices_full = prices_full[prices_full.index >= lookback]
        etf_codes = {name: code for name, code in etfs.items()}

        modes_to_run = ["daily", "friday"] if mode == "both" else [mode]
        modes_data = {}
        all_metrics = {}

        for m in modes_to_run:
            bt_fn = run_backtest_bt if use_backtrader else run_backtest
            nav, bnav, ret, bret, trades, trade_dates, trade_details = bt_fn(
                prices_full, m, start_str, end_str, ma_days, roc_days)
            metrics_dict = calc_metrics(nav, ret)
            bench_metrics = calc_metrics(bnav, bret)
            all_metrics[m] = (metrics_dict, bench_metrics, trades, ret, nav, bnav)
            modes_data[m] = (nav, bnav, trade_dates, trade_details)

        # Comparison: min_hold=10 variant
        cmp_data = {}
        if compare_min_hold:
            for m in modes_to_run:
                bt_fn = run_backtest_bt if use_backtrader else run_backtest
                nav2, bnav2, ret2, bret2, trades2, td2, tdets2 = bt_fn(
                    prices_full, m, start_str, end_str, ma_days, roc_days, min_hold=10)
                cmp_data[m] = (nav2, bnav2, ret2, bret2, trades2, td2, tdets2)

    st.subheader(f"回测结果: {sel_group}  |  {start_str} ~ {end_str}")

    # Metrics cards
    for m in modes_to_run:
        mm, bm, trades, ret, nav, bnav = all_metrics[m]
        trade_dates = modes_data[m][2]
        wr = trade_win_rate(ret, trade_dates)
        st.markdown(f"**{m.upper()} 调仓**")

        def render_metrics(mm, trades, wr, metric_keys, cols_per_row=6):
            """Render metrics in rows, cols_per_row per row. No hard column count."""
            for row_start in range(0, len(metric_keys), cols_per_row):
                row_keys = metric_keys[row_start:row_start + cols_per_row]
                cols = st.columns(cols_per_row)
                for ci, key in enumerate(row_keys):
                    if key == "交易次数":
                        cols[ci].metric(key, f"{trades}", help=metric_help.get(key))
                    elif key in ("持有天数", "水下天数", "最长亏损持续"):
                        cols[ci].metric(key, f"{int(mm.get(key, 0))}", help=metric_help.get(key))
                    elif key == "最长亏损区间":
                        cols[ci].metric(key, mm.get(key, "N/A"), help=metric_help.get(key))
                    elif key == "最大亏损日期":
                        dt_val = mm.get("最大亏损日期")
                        cols[ci].metric(key, dt_val.strftime("%Y-%m-%d") if dt_val else "N/A", help=metric_help.get(key))
                    elif key == "胜率":
                        cols[ci].metric(key, f"{wr:.0%}", help=metric_help.get(key))
                    elif key in ("夏普比率", "卡尔玛比率"):
                        cols[ci].metric(key, f"{mm.get(key, 0):.2f}", help=metric_help.get(key))
                    else:
                        cols[ci].metric(key, f"{mm.get(key, 0):.2%}", help=metric_help.get(key))
            # pad empty columns so layout doesn't shift
            for ci in range(len(row_keys), cols_per_row):
                cols[ci].markdown("")

        metric_help = {
            "累计收益": "策略总收益率 = (最终净值 - 1) × 100%",
            "年化收益": "年化复合收益率，按 252 个交易日折算",
            "夏普比率": "(年化收益 - 无风险利率 3%) / 年化波动率，衡量风险调整后收益",
            "最大回撤": "策略净值从峰值到谷底的最大跌幅（峰值不一定是1.0）",
            "最大亏损": "策略净值相对本金(1.0)的最大亏损，衡量实际亏本金额度。下方显示最大亏损发生日期",
            "最大亏损日期": "最大亏损发生的具体交易日",
            "持有天数": "回测区间内的有效交易日总数",
            "水下天数": "净值低于1.0（处于亏损状态）的交易日数",
            "最长亏损持续": "净值连续低于本金(1.0)的最大交易日数（最长亏损持续期）",
            "最长亏损区间": "最长连续亏损的起止日期",
            "卡尔玛比率": "年化收益 / |最大回撤|，衡量回撤调整后收益",
            "交易次数": "调仓交易总次数（含买卖双边）",
            "胜率": "盈利持仓期数 / 总持仓期数，相邻调仓日之间记为一个持仓期",
        }
        metric_keys = ["累计收益", "年化收益", "夏普比率", "最大回撤", "最大亏损",
                       "最大亏损日期", "水下天数", "最长亏损持续", "最长亏损区间",
                       "持有天数", "卡尔玛比率", "交易次数", "胜率"]
        render_metrics(mm, trades, wr, metric_keys)
        pos_fn = position_dist_bt if use_backtrader else position_dist
        pos_days, pos_buys, pos_contrib, pos_cum, pos_wr = pos_fn(prices_full, start_str, end_str, m, ma_days, roc_days)
        total = sum(pos_days.values())
        pos_rows = []
        for k in sorted(pos_days.keys(), key=lambda x: -pos_days[x]):
            d = pos_days[k]
            b = pos_buys.get(k, 0)
            ct = pos_contrib.get(k, 0)
            cr = pos_cum.get(k, 0)
            wr = pos_wr.get(k, 0)
            pos_rows.append({"ETF": k, "持有天数": d, "占比": f"{d/total:.0%}", "买入次数": b,
                             "收益占比": f"{ct:+.1%}", "持有期累计收益": f"{cr:+.1%}", "胜率": f"{wr:.0%}"})
        st.dataframe(pd.DataFrame(pos_rows), hide_index=True, use_container_width=False)
        st.caption("收益占比=各ETF对数收益÷总对数收益(加总=100%) | 持有期累计收益=∏(1+r)-1 | 胜率=上涨天数÷持有天数")

        # Yearly returns
        yr = yearly_returns(ret)
        if len(yr) > 1:
            yr_badges = [f"` {y}: {r:+.1%} `" for y, r in yr.items()]
            st.markdown("**逐年收益**  " + "  ".join(yr_badges))

    # Comparison: full metrics for min_hold=10 variant
    if compare_min_hold and cmp_data:
        st.divider()
        st.subheader("🔹 最小持有10天", divider="orange")
        for m in modes_to_run:
            c_nav, _, c_ret, _, c_trades, c_td, _ = cmp_data[m]
            c_mm = calc_metrics(c_nav, c_ret)
            c_wr = trade_win_rate(c_ret, c_td)
            st.markdown(f"**{m.upper()} 调仓**")
            render_metrics(c_mm, c_trades, c_wr, metric_keys)
            # Position distribution for comparison
            pos_fn = position_dist_bt if use_backtrader else position_dist
            c_days, c_buys, c_contrib, c_cum, c_wr = pos_fn(prices_full, start_str, end_str, m, ma_days, roc_days, min_hold=10)
            c_total = sum(c_days.values())
            c_rows = []
            for k in sorted(c_days.keys(), key=lambda x: -c_days[x]):
                d = c_days[k]; b = c_buys.get(k, 0)
                ct = c_contrib.get(k, 0); cr = c_cum.get(k, 0); wr = c_wr.get(k, 0)
                c_rows.append({"ETF": k, "持有天数": d, "占比": f"{d/c_total:.0%}", "买入次数": b,
                               "收益占比": f"{ct:+.1%}", "持有期累计收益": f"{cr:+.1%}", "胜率": f"{wr:.0%}"})
            st.dataframe(pd.DataFrame(c_rows), hide_index=True, use_container_width=False)
            st.caption("收益占比=各ETF对数收益÷总对数收益(加总=100%) | 持有期累计收益=∏(1+r)-1 | 胜率=上涨天数÷持有天数")
            # Yearly returns for comparison
            c_yr = yearly_returns(c_ret)
            if len(c_yr) > 1:
                c_yr_items = [f"` {y}: {r:+.1%} `" for y, r in c_yr.items()]
                st.markdown("**逐年收益**  " + "  ".join(c_yr_items))

    # Plotly chart
    st.plotly_chart(
        build_plotly_fig(prices_full, etf_codes, modes_data, start_str, end_str, cmp_data if compare_min_hold else None),
        use_container_width=True,
    )

    # ── Parameter optimization ────────────────────────────
    if optimize:
        st.divider()
        st.subheader("🔍 参数遍历结果", divider="orange")

        ma_range = list(range(10, 201, opt_ma_step))
        roc_range = list(range(5, 121, opt_roc_step))
        total_combo = len(ma_range) * len(roc_range) * len(modes_to_run)

        with st.status(f"搜索 {len(ma_range)}×{len(roc_range)} = {total_combo} 种组合...", expanded=False) as status:
            prog = st.progress(0, text="初始化...")
            df_opt = grid_search(prices_full, modes_to_run, start_str, end_str,
                                 ma_range, roc_range, prog)
            status.update(label=f"完成 {total_combo} 种组合", state="complete")

        # ── Result table per mode ──
        opt_tab = st.tabs([f"{m.upper()}调仓 TOP{opt_top}" for m in modes_to_run] + ["全部 TOP"])
        for ti, mode in enumerate(modes_to_run):
            with opt_tab[ti]:
                sub = df_opt[df_opt["模式"] == mode].sort_values(opt_metric, ascending=False).head(opt_top).copy()
                _fmt_optimize_table(sub)
                st.dataframe(sub, hide_index=True)

                # Heatmap for this mode
                figs = build_optimizer_heatmap(df_opt, opt_metric)
                if mode in figs:
                    st.plotly_chart(figs[mode], use_container_width=True)

        # ── All-mode summary ──
        with opt_tab[-1]:
            top_all = df_opt.sort_values(opt_metric, ascending=False).head(opt_top).copy()
            _fmt_optimize_table(top_all)
            st.dataframe(top_all, hide_index=True)

        # CSV download
        csv = df_opt.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("📥 下载全量CSV", csv, "etf_optimize.csv", "text/csv", use_container_width=True)

# ── Signal query ─────────────────────────────────────────
if sig_btn:
    etfs = cfg["groups"][sel_group]
    with st.spinner("查询信号..."):
        prices = cached_prices(etfs, sel_group, source=source)
        best, df, actual_dt = signal_for_date(prices, sig_date.strftime("%Y-%m-%d"), ma_days, roc_days)

    if df is None:
        st.warning("数据不足，无法查询")
    else:
        st.subheader(f"信号: {actual_dt.strftime('%Y-%m-%d')} [{sel_group}]")
        if best:
            st.success(f"持有 **{best}**")
        else:
            st.warning(f"空仓 (全部低于{ma_days}日均线)")

        # Highlight qualified rows
        def highlight_above(row):
            ma_col = f"{ma_days}日均线"
            if row.get(ma_col, None) and not (isinstance(row[ma_col], float) and pd.isna(row[ma_col])):
                if row["收盘价"] > row[ma_col]:
                    return ["background-color: #e8f5e9"] * len(row)
            return [""] * len(row)

        st.dataframe(df.style.apply(highlight_above, axis=1), hide_index=True)
