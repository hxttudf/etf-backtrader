#!/usr/bin/env python3
"""ETF双动量轮动 - 回测脚本

用法:
  python etf_backtest.py --start 2025-04-30                     # 至今，daily+Friday对比
  python etf_backtest.py --start 2023-01-01 --end 2025-12-31    # 指定区间
  python etf_backtest.py --start 2024-01-01 --mode daily         # 仅每日调仓
  python etf_backtest.py --start 2024-01-01 --group 默认组合
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Microsoft YaHei", "SimHei", "PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

from etf_data import DEFAULT_CONFIG, calc_indicators, load_config, load_prices, load_prices_extended
from etf_backtrader import run_backtest_bt, position_dist_bt, STRATEGIES

COMMISSION = 0.0001  # 万1 per side, 免五
STAMP_DUTY = 0.0005  # 印花税 0.05%, 卖出时收取


def run_backtest(prices: pd.DataFrame, mode: str, start_date: str, end_date: str,
                 ma_days: int = 60, roc_days: int = 20, min_hold: int = 0,
                 open_prices: pd.DataFrame | None = None,
                 midday_prices: pd.DataFrame | None = None,
                 afternoon_open_prices: pd.DataFrame | None = None):
    """mode: 'daily' | 'friday'  → (nav, bench_nav, ret, bench_ret, trades, trade_dates, trade_details)

    信号在 T 日收盘判定，T+1 执行。
    - 无开盘价/中午价: 信号在 T+1 日收盘执行（close-to-close）
    - 有开盘价: 信号在 T+1 日开盘执行（close→open 过夜 + open→close 日内）
    - 有中午价+下午开盘价: 中午信号执行（close[i-1]→midday[i] 上午 + afternoon_open[i]→close[i] 下午）
    """
    etf_names = list(prices.columns)
    returns = prices.pct_change(fill_method=None)
    ma60, roc20, _ = calc_indicators(prices, ma_days, roc_days)

    strat_ret = pd.Series(0.0, index=prices.index)
    holding = None
    trades = 0
    trade_dates: list[pd.Timestamp] = []
    trade_details: list[tuple[pd.Timestamp, str | None, str | None]] = []
    is_friday = prices.index.dayofweek == 4
    signal = None
    last_trade_i = -999
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    _use_open = open_prices is not None
    _use_midday = midday_prices is not None and afternoon_open_prices is not None

    for i in range(ma_days, len(prices)):
        dt = prices.index[i]

        # Execute pending signal
        if signal != holding and i - last_trade_i >= min_hold:
            if _use_midday and i > 0:
                # Midday execution: morning(old) + afternoon(new)
                # Morning: close[i-1] → midday[i] of old holding
                mid_dt = midday_prices.index[midday_prices.index <= dt]
                aft_dt = afternoon_open_prices.index[afternoon_open_prices.index <= dt]
                if len(mid_dt) > 0 and len(aft_dt) > 0:
                    mid_key = mid_dt[-1]
                    aft_key = aft_dt[-1]
                    mid_ok = holding is None or (holding in midday_prices.columns and mid_key in midday_prices.index and not np.isnan(midday_prices[holding].loc[mid_key]))
                    aft_ok = signal is None or (signal in afternoon_open_prices.columns and aft_key in afternoon_open_prices.index and not np.isnan(afternoon_open_prices[signal].loc[aft_key]))
                    if (holding is None or mid_ok) and (signal is None or aft_ok):
                        # Morning leg
                        if holding is not None:
                            prev_close = prices[holding].iloc[i - 1]
                            mid_px = midday_prices[holding].loc[mid_key]
                            if not np.isnan(prev_close) and prev_close > 0:
                                strat_ret.iloc[i] = mid_px / prev_close - 1
                            strat_ret.iloc[i] -= COMMISSION + STAMP_DUTY
                        if signal is not None:
                            strat_ret.iloc[i] -= COMMISSION
                        # Afternoon leg
                        new_h = signal
                        if new_h is not None:
                            aft_o = afternoon_open_prices[new_h].loc[aft_key]
                            day_c = prices[new_h].iloc[i]
                            if not np.isnan(aft_o) and not np.isnan(day_c) and aft_o > 0:
                                afternoon_ret = day_c / aft_o - 1
                                strat_ret.iloc[i] = (1 + strat_ret.iloc[i]) * (1 + afternoon_ret) - 1
                    else:
                        # Fallback to close-to-close when midday data missing
                        if holding is not None:
                            r = returns[holding].iloc[i]
                            strat_ret.iloc[i] = r if not np.isnan(r) else 0.0
                            strat_ret.iloc[i] -= COMMISSION + STAMP_DUTY
                        if signal is not None:
                            strat_ret.iloc[i] -= COMMISSION
                else:
                    # No midday data for this date, fallback close-to-close
                    if holding is not None:
                        r = returns[holding].iloc[i]
                        strat_ret.iloc[i] = r if not np.isnan(r) else 0.0
                        strat_ret.iloc[i] -= COMMISSION + STAMP_DUTY
                    if signal is not None:
                        strat_ret.iloc[i] -= COMMISSION
            elif _use_open and i > 0:
                # T+1 open execution: overnight(old) + swap + intraday(new)
                # Overnight: close[i-1] → open[i] of old holding
                if holding is not None:
                    prev_close = prices[holding].iloc[i - 1]
                    today_open_old = open_prices[holding].iloc[i]
                    if not np.isnan(prev_close) and not np.isnan(today_open_old) and prev_close > 0:
                        strat_ret.iloc[i] = today_open_old / prev_close - 1
                    strat_ret.iloc[i] -= COMMISSION + STAMP_DUTY
                if signal is not None:
                    strat_ret.iloc[i] -= COMMISSION
                # Intraday: open[i] → close[i] of NEW holding
                new_h = signal
                if new_h is not None:
                    o = open_prices[new_h].iloc[i]
                    c = prices[new_h].iloc[i]
                    if not np.isnan(o) and not np.isnan(c) and o > 0:
                        intraday = c / o - 1
                        strat_ret.iloc[i] = (1 + strat_ret.iloc[i]) * (1 + intraday) - 1
            else:
                # Close-to-close execution (fallback)
                if holding is not None:
                    r = returns[holding].iloc[i]
                    strat_ret.iloc[i] = r if not np.isnan(r) else 0.0
                    strat_ret.iloc[i] -= COMMISSION + STAMP_DUTY
                if signal is not None:
                    strat_ret.iloc[i] -= COMMISSION

            if start_ts <= dt <= end_ts:
                trades += 1
                trade_dates.append(dt)
                trade_details.append((dt, holding, signal))
            last_trade_i = i
            holding = signal

        elif holding is not None:
            # Normal hold: close-to-close return
            r = returns[holding].iloc[i]
            strat_ret.iloc[i] = r if not np.isnan(r) else 0.0

        # Compute signal at close → for next trading day
        should_check = True if mode == "daily" else is_friday[i]
        if should_check:
            above = {}
            for name in etf_names:
                px = prices[name].iloc[i]
                ma = ma60[name].iloc[i]
                roc = roc20[name].iloc[i]
                if not np.isnan(ma) and px > ma and not np.isnan(roc):
                    above[name] = roc
            signal = max(above, key=above.get) if above else None

    # Trim returns to target range
    trim = (prices.index >= start_date) & (prices.index <= end_date)
    ret = strat_ret[trim]
    bench_ret = returns[trim].mean(axis=1)
    nav = (1 + ret).cumprod()
    bench_nav = (1 + bench_ret).cumprod()

    return nav, bench_nav, ret, bench_ret, trades, trade_dates, trade_details


def metrics(nav: pd.Series, ret: pd.Series) -> dict:
    r = ret.dropna()
    if len(r) < 5:
        return {}
    total = nav.iloc[-1] - 1
    ann = (1 + total) ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = (ann - 0.03) / vol if vol > 0 else 0
    dd = (nav / nav.cummax() - 1).min()
    calmar = ann / abs(dd) if dd != 0 else 0
    max_loss = (nav - 1).min()
    max_loss_dt = nav.idxmin()
    underwater_days = int((nav < 1).sum())
    holding_days = len(r)
    return {"累计收益": total, "年化收益": ann, "年化波动": vol, "夏普比率": sharpe,
            "最大回撤": dd, "卡尔玛比率": calmar, "最大亏损": max_loss,
            "最大亏损日期": max_loss_dt, "水下天数": underwater_days, "持有天数": holding_days}


def position_dist(prices: pd.DataFrame, start_date: str, end_date: str, mode: str,
                  ma_days: int = 60, roc_days: int = 20, min_hold: int = 0) -> tuple[dict, dict, dict, dict, dict]:
    """返回 (持有天数dict, 买入次数dict, 收益占比dict, 持有期累计收益dict, 上涨天数占比dict)"""
    import math

    etf_names = list(prices.columns)
    ma60, roc20, _ = calc_indicators(prices, ma_days, roc_days)
    daily_ret = prices.pct_change(fill_method=None)
    is_friday = prices.index.dayofweek == 4

    days = {n: 0 for n in etf_names}
    days["CASH"] = 0
    buys = {n: 0 for n in etf_names}
    nav = {n: 1.0 for n in etf_names}
    pos_days = {n: 0 for n in etf_names}
    hold_days_for_wr = {n: 0 for n in etf_names}
    log_ret = {n: 0.0 for n in etf_names}
    log_ret["CASH"] = 0.0

    holding = None
    signal = None
    last_trade_i = -999
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    for i in range(ma_days, len(prices)):
        dt = prices.index[i]

        # Execute pending signal
        if signal != holding:
            if i - last_trade_i >= min_hold:
                if start_ts <= dt <= end_ts:
                    buys[signal] += 1
                    if holding is not None and holding in log_ret:
                        log_ret[holding] += math.log(1 - (COMMISSION + STAMP_DUTY))
                    if signal is not None:
                        log_ret[signal] += math.log(1 - COMMISSION)
                last_trade_i = i
                holding = signal

        should_check = True if mode == "daily" else is_friday[i]
        if should_check:
            above = {}
            for name in etf_names:
                px = prices[name].iloc[i]
                ma = ma60[name].iloc[i]
                roc = roc20[name].iloc[i]
                if not np.isnan(ma) and px > ma and not np.isnan(roc):
                    above[name] = roc
            signal = max(above, key=above.get) if above else None

        if start_ts <= dt <= end_ts:
            h = holding or "CASH"
            days[h] += 1
            if holding is not None:
                r = daily_ret[holding].iloc[i]
                if not pd.isna(r):
                    nav[holding] *= (1 + r)
                    hold_days_for_wr[holding] += 1
                    if r > 0:
                        pos_days[holding] += 1
                    log_ret[holding] += math.log(1 + r)

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


def yearly_returns(ret: pd.Series) -> dict:
    result = {}
    for yr in sorted(set(ret.index.year)):
        mask = ret.index.year == yr
        if mask.sum() > 10:
            result[yr] = (1 + ret[mask]).prod() - 1
    return result


def trade_win_rate(ret: pd.Series, trade_details: list, prices: pd.DataFrame) -> float:
    """Per-trade win rate: each closed position = one observation.

    Tracks entry price when an ETF is bought, compares to exit price when sold.
    Standard industry metric — every round-trip trade counts once.
    """
    if not trade_details:
        return 0.0
    entry_prices: dict[str, float] = {}
    wins = 0
    total = 0
    for dt, sold, bought in trade_details:
        # Close sold position
        if sold is not None and sold in entry_prices:
            exit_px = prices[sold].get(dt, np.nan) if dt in prices.index else np.nan
            if not np.isnan(exit_px):
                total += 1
                if exit_px > entry_prices[sold]:
                    wins += 1
            del entry_prices[sold]
        # Open bought position — record entry price
        if bought is not None:
            entry_px = prices[bought].get(dt, np.nan) if dt in prices.index else np.nan
            if not np.isnan(entry_px):
                entry_prices[bought] = entry_px
    # Close remaining open positions at last available date
    last_dt = ret.index[-1]
    for etf, entry_px in entry_prices.items():
        exit_px = prices[etf].get(last_dt, np.nan) if last_dt in prices.index else np.nan
        if not np.isnan(exit_px):
            total += 1
            if exit_px > entry_px:
                wins += 1
    return wins / total if total > 0 else 0.0


def plot_result(prices: pd.DataFrame, modes_data: dict, group_name: str, start: str, end: str,
                etf_codes: dict | None = None) -> Path:
    """modes_data: {mode: (nav, bnav, trade_dates)}"""
    n_modes = len(modes_data)
    fig, axes = plt.subplots(2, 1, figsize=(14, 5 + 2.5 * n_modes), gridspec_kw={"height_ratios": [2, 1]})
    ax1, ax2 = axes

    strategy_colors = {"daily": "#2196F3", "friday": "#FF9800"}
    etf_colors = plt.cm.tab10.colors

    # -- Top: Cumulative NAV --
    nav0 = prices[prices.index >= start].copy()
    etf_navs = {}
    for i, name in enumerate(prices.columns):
        etf_nav = (1 + nav0[name].pct_change(fill_method=None).fillna(0)).cumprod()
        etf_navs[name] = etf_nav
        color = etf_colors[i % len(etf_colors)]
        code = etf_codes.get(name, "") if etf_codes else ""
        label = f"持有{name} ({code})" if code else f"持有{name}"
        ax1.plot(etf_nav.index, etf_nav.values, linewidth=1.0, color=color, alpha=0.7, label=label)

    # Strategy NAVs + benchmark
    bench_added = False
    all_trade_dates: set[pd.Timestamp] = set()
    for mode, (nav, bnav, trade_dates) in modes_data.items():
        color = strategy_colors.get(mode, "black")
        ax1.plot(nav.index, nav.values, linewidth=2.0, color=color, label=f"策略({mode})")
        # Trade markers — large diamonds on the NAV line
        if trade_dates:
            valid_dates = [d for d in trade_dates if d in nav.index]
            trade_navs = [nav.loc[d] for d in valid_dates]
            all_trade_dates.update(valid_dates)
            ax1.scatter(valid_dates, trade_navs, s=60, color=color, marker="D",
                        edgecolors="white", linewidths=1.2, zorder=6, alpha=0.95)
        if not bench_added:
            ax1.plot(bnav.index, bnav.values, linewidth=1.2, color="#666", linestyle=":", label="等权基准")
            bench_added = True

    ax1.set_title(f"ETF双动量轮动 [{group_name}]  {start} ~ {end}", fontsize=13, fontweight="bold")
    ax1.set_ylabel("累计净值")
    ax1.legend(loc="upper left", fontsize=8, ncol=2)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax1.grid(True, alpha=0.3)

    # -- Bottom: Drawdown --
    for mode, (nav, _bnav, _) in modes_data.items():
        dd = nav / nav.cummax() - 1
        color = strategy_colors.get(mode, "black")
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.25, color=color, label=f"{mode}")
        ax2.plot(dd.index, dd.values, linewidth=0.8, color=color)

    ax2.set_ylabel("回撤")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Vertical dashed lines at all trade dates across both subplots
    if all_trade_dates:
        for dt in sorted(all_trade_dates):
            ax1.axvline(x=dt, color="#333", linestyle="--", linewidth=0.6, alpha=0.3, zorder=1)
            ax2.axvline(x=dt, color="#333", linestyle="--", linewidth=0.6, alpha=0.3, zorder=1)

    plt.tight_layout()
    out = Path(__file__).parent / f"etf_backtest_{group_name.replace(' ','_')}_{start}_{end}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def plot_interactive(groups_data: list[dict], start: str, end: str) -> Path:
    """groups_data: [{"name": str, "prices": DataFrame, "etf_codes": dict, "modes_data": dict}, ...]

    modes_data: {mode: (nav, bnav, trade_dates, trade_details)}
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    strategy_colors = {"daily": "#2196F3", "friday": "#FF9800"}
    etf_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    n_groups = len(groups_data)

    first_name = groups_data[0]["name"]
    title = f"ETF双动量轮动 [{first_name}]  {start} ~ {end}" if n_groups == 1 else f"ETF双动量轮动 {start} ~ {end}"

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.05,
                        subplot_titles=(title, "回撤"))

    trace_idx = 0

    for gi, g in enumerate(groups_data):
        group_name = g["name"]
        prices = g["prices"]
        etf_codes = g["etf_codes"]
        modes_data = g["modes_data"]

        visible = gi == 0  # only first group visible initially

        # ETF individual curves
        nav0 = prices[prices.index >= start].copy()
        etf_color_map: dict[str, str] = {}
        for i, name in enumerate(prices.columns):
            etf_nav = (1 + nav0[name].pct_change(fill_method=None).fillna(0)).cumprod()
            code = etf_codes.get(name, "")
            label = f"持有{name} ({code})" if code else f"持有{name}"
            color = etf_colors[i % len(etf_colors)]
            etf_color_map[name] = color
            fig.add_trace(go.Scatter(
                x=etf_nav.index, y=etf_nav.values, mode="lines",
                name=label, line=dict(color=color, width=1.5), opacity=0.7,
                meta=group_name, visible=visible,
            ), row=1, col=1)
            trace_idx += 1

        # Benchmark (only once — shared across all groups)
        if gi == 0:
            for mode, (nav, bnav, trade_dates, trade_details) in modes_data.items():
                fig.add_trace(go.Scatter(
                    x=bnav.index, y=bnav.values, mode="lines",
                    name="等权基准", line=dict(color="#666", width=1.2, dash="dot"),
                    meta="__benchmark__", visible=True,
                ), row=1, col=1)
                trace_idx += 1
                break  # only one benchmark trace

        # Strategy NAVs + trade markers + drawdown
        for mode, (nav, bnav, trade_dates, trade_details) in modes_data.items():
            color = strategy_colors.get(mode, "black")

            # Cumulative return for hover
            cum_ret = nav - 1
            ret_hovertemplate = "%{customdata[0]}<br>净值: %{y:.3f}<br>累计收益: %{customdata[1]:+.2%}<extra></extra>"
            ret_customdata = [(d.strftime("%Y-%m-%d"), r) for d, r in zip(nav.index, cum_ret)]

            # Strategy line
            strat_name = f"策略({mode}) [{group_name}]" if n_groups > 1 else f"策略({mode})"
            fig.add_trace(go.Scatter(
                x=nav.index, y=nav.values, mode="lines",
                name=strat_name,
                line=dict(color=color, width=2.5),
                customdata=ret_customdata,
                hovertemplate=ret_hovertemplate,
                meta=group_name, visible=visible,
            ), row=1, col=1)
            trace_idx += 1

            # Underwater overlay — red line segments where NAV < 1
            uw = nav.copy()
            uw[uw >= 1] = np.nan
            if uw.notna().any():
                fig.add_trace(go.Scatter(
                    x=nav.index, y=uw.values, mode="lines",
                    name=f"水下({mode}) [{group_name}]" if n_groups > 1 else f"水下({mode})",
                    line=dict(color="#DC2626", width=3),
                    hovertemplate="%{customdata[0]}<br>净值: %{y:.3f}<br>累计收益: %{customdata[1]:+.2%}<extra></extra>",
                    customdata=ret_customdata,
                    meta=group_name, visible=visible,
                ), row=1, col=1)
                trace_idx += 1

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
                    trade_name = f"调仓({mode}) [{group_name}]" if n_groups > 1 else f"调仓({mode})"
                    fig.add_trace(go.Scatter(
                        x=dates, y=navs_v, mode="markers",
                        name=trade_name,
                        marker=dict(color=marker_colors, size=10, symbol="diamond",
                                   line=dict(color="white", width=1)),
                        hovertemplate="%{customdata}<br>净值: %{y:.3f}<extra></extra>",
                        customdata=hover_texts,
                        meta=group_name, visible=visible,
                    ), row=1, col=1)
                    trace_idx += 1

            # Drawdown
            dd = nav / nav.cummax() - 1
            fig.add_trace(go.Scatter(
                x=dd.index, y=dd.values, mode="lines",
                name=f"回撤({mode}) [{group_name}]" if n_groups > 1 else f"回撤({mode})",
                line=dict(color=color, width=1),
                fill="tozeroy", fillcolor=_hex_to_rgba(color, 0.2),
                meta=group_name, visible=visible, showlegend=False,
            ), row=2, col=1)
            trace_idx += 1

    fig.update_xaxes(range=[start, end], row=1, col=1)
    fig.update_xaxes(range=[start, end], row=2, col=1)
    fig.update_yaxes(title_text="累计净值", row=1, col=1)
    fig.update_yaxes(title_text="回撤", tickformat=".0%", row=2, col=1)

    # Build dropdown for group switching (uses meta, not legendgroup)
    if n_groups > 1:
        buttons = []
        for gi, g in enumerate(groups_data):
            vis = []
            for ti in range(trace_idx):
                t = fig.data[ti]
                vis.append(t.meta in (g["name"], "__benchmark__"))
            buttons.append(dict(label=g["name"], method="update", args=[{"visible": vis},
                               {"title": f"ETF双动量轮动 [{g['name']}]  {start} ~ {end}"}]))
        fig.update_layout(
            updatemenus=[dict(active=0, buttons=buttons, x=1.0, y=1.15, xanchor="right",
                              bgcolor="#eee", bordercolor="#ccc", borderwidth=1,
                              font=dict(size=12))],
        )

    fig.update_layout(
        hovermode="x unified",
        legend=dict(font=dict(size=10), orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template="plotly_white",
        height=700,
    )

    safe_name = "all" if n_groups > 1 else groups_data[0]["name"].replace(" ", "_")
    out = Path(__file__).parent / f"etf_backtest_{safe_name}_{start}_{end}.html"
    fig.write_html(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="ETF双动量轮动 回测")
    parser.add_argument("--group", default=None, help="组合名称（不指定则HTML回测全部，PNG回测第一个）")
    parser.add_argument("--start", required=True, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"), help="结束日期")
    parser.add_argument("--mode", default="both", choices=["daily", "friday", "both"], help="调仓模式")
    parser.add_argument("--ma", type=int, default=60, help="均线天数 (默认60)")
    parser.add_argument("--roc", type=int, default=20, help="动量天数 (默认20)")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--source", default="tencent", choices=["tencent", "akshare"], help="数据源 (默认tencent)")
    parser.add_argument("--html", action="store_true", help="输出交互式HTML图表（可hover查看调仓ETF）")
    parser.add_argument("--backtrader", action="store_true", default=False, help="使用 backtrader 引擎回测")
    parser.add_argument("--strategy", "-s", default="momentum", choices=list(STRATEGIES.keys()),
                        help="策略类型 (默认momentum)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    all_groups = cfg["groups"]

    if args.group and args.group not in all_groups:
        print(f"组合 '{args.group}' 不存在。可用: {', '.join(all_groups.keys())}")
        return

    modes = ["daily", "friday"] if args.mode == "both" else [args.mode]
    start_dt = pd.Timestamp(args.start)
    lookback_start = (start_dt - pd.Timedelta(days=200)).strftime("%Y-%m-%d")

    # Determine which groups to run
    if args.group:
        group_names = [args.group]
    elif args.html:
        group_names = list(all_groups.keys())  # HTML: all groups with dropdown
    else:
        group_names = [next(iter(all_groups.keys()))]  # PNG: just the first

    all_groups_data: list[dict] = []
    first_prices = None
    first_etf_codes = None
    first_metrics = None

    for gi, group_name in enumerate(group_names):
        etfs = all_groups[group_name]
        prices_full = load_prices_extended(etfs, group_name, source=args.source) if args.source == "akshare" else load_prices(etfs, group_name, source=args.source)
        prices_full = prices_full[prices_full.index >= lookback_start]
        etf_codes = {name: code for name, code in etfs.items()}

        print(f"\n=== [{group_name}] 回测: {args.start} ~ {args.end} ===\n")
        print(f"交易日: {(prices_full.index >= args.start).sum()}  |  ETF: {', '.join(etfs.keys())}")

        all_metrics = {}
        for mode in modes:
            if args.backtrader:
                nav, bnav, ret, bret, trades, trade_dates, trade_details, daily_signals = \
                    run_backtest_bt(prices_full, mode, args.start, args.end, args.ma, args.roc,
                                    strategy=args.strategy)
            else:
                nav, bnav, ret, bret, trades, trade_dates, trade_details = \
                    run_backtest(prices_full, mode, args.start, args.end, args.ma, args.roc)
                daily_signals = []
            m = metrics(nav, ret)
            bm = metrics(bnav, bret)
            all_metrics[mode] = (m, bm, trades, ret, nav, bnav, trade_dates, trade_details, daily_signals)

            print(f"\n-- {mode.upper()} 调仓 --")
            print(f"{'指标':<14} {'策略':>10} {'基准':>10}")
            print("-" * 38)
            for key in ["累计收益", "年化收益", "年化波动", "夏普比率", "最大回撤", "最大亏损", "水下天数", "持有天数", "卡尔玛比率"]:
                sv = m.get(key, 0)
                bv = bm.get(key, 0)
                if key in ("夏普比率", "卡尔玛比率"):
                    print(f"{key:<14} {sv:>10.2f} {bv:>10.2f}")
                elif key == "水下天数":
                    print(f"{key:<14} {sv:>10.0f} {bv:>10.0f}")
                else:
                    print(f"{key:<14} {sv:>9.1%} {bv:>9.1%}")
            wr = trade_win_rate(ret, trade_details, prices_full)
            print(f"{'交易次数':<14} {trades:>10}")
            print(f"{'胜率':<14} {wr:>9.1%}")

            pd_result = (position_dist_bt(prices_full, args.start, args.end, mode, args.ma, args.roc, strategy=args.strategy)
                         if args.backtrader else
                         position_dist(prices_full, args.start, args.end, mode, args.ma, args.roc))
            pos_days, pos_buys = pd_result[:2]
            total = sum(pos_days.values())
            print("\n  持仓分布 (天数/买入次数):")
            for k in sorted(pos_days.keys(), key=lambda x: -pos_days[x]):
                d = pos_days[k]
                b = pos_buys.get(k, 0)
                print(f"    {k}: {d}天 ({d/total:.1%})  买入{b}次")

            yr = yearly_returns(ret)
            if len(yr) > 1:
                print("  逐年收益:")
                for y, r in yr.items():
                    print(f"    {y}: {r:>+8.1%}")

            # Daily signal summary
            if daily_signals:
                print(f"\n  每日信号摘要 (最近5个调仓日):")
                trade_signal_dates = sorted({d for d in trade_dates if d >= pd.Timestamp(args.start) and d <= pd.Timestamp(args.end)})
                recent = trade_signal_dates[-5:]
                for td in recent:
                    match = next((s for s in daily_signals if s.get('_dt') == td), None)
                    if match is None:
                        # Try to find by date string
                        for s in daily_signals:
                            s_dt = s.get('_dt')
                            if s_dt is not None and str(s_dt)[:10] == str(td)[:10]:
                                match = s
                                break
                    if match:
                        scores = {k: v for k, v in match.items() if k not in ('holding', '_dt')}
                        score_str = "  ".join(f"{k}: {v:+.4f}" if isinstance(v, float) and v is not None else f"{k}: N/A" for k, v in sorted(scores.items()))
                        print(f"    {str(td)[:10]}  [{match.get('holding', 'CASH')}]  {score_str}")
                    else:
                        print(f"    {str(td)[:10]}  (无信号数据)")

        # Store group data for combined HTML
        if args.html or len(group_names) > 1:
            all_groups_data.append({
                "name": group_name,
                "prices": prices_full,
                "etf_codes": etf_codes,
                "modes_data": {
                    m: (all_metrics[m][4], all_metrics[m][5], all_metrics[m][6], all_metrics[m][7])
                    for m in modes
                },
            })

        if gi == 0:
            first_prices = prices_full
            first_etf_codes = etf_codes
            first_metrics = all_metrics

    # Generate chart
    if args.html:
        chart_path = plot_interactive(all_groups_data, args.start, args.end)
    else:
        assert first_prices is not None and first_etf_codes is not None and first_metrics is not None
        chart_path = plot_result(first_prices, {
            m: (first_metrics[m][4], first_metrics[m][5], first_metrics[m][6]) for m in modes
        }, group_names[0], args.start, args.end, first_etf_codes)
    print(f"\n图表: {chart_path}")


if __name__ == "__main__":
    main()
