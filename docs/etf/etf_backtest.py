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
from etf_backtrader import run_backtest_bt, position_dist_bt

COMMISSION = 0.0001  # 万1 per side, 免五
STAMP_DUTY = 0.0005  # 印花税 0.05%, 卖出时收取


def run_backtest(prices: pd.DataFrame, mode: str, start_date: str, end_date: str,
                 ma_days: int = 60, roc_days: int = 25):
    """mode: 'daily' | 'friday'  → (nav, bench_nav, ret, bench_ret, trades, trade_dates, trade_details)"""
    etf_names = list(prices.columns)
    returns = prices.pct_change(fill_method=None)
    ma60, roc20, _ = calc_indicators(prices, ma_days, roc_days)

    strat_ret = pd.Series(0.0, index=prices.index)
    holding = None
    trades = 0
    trade_dates: list[pd.Timestamp] = []
    trade_details: list[tuple[pd.Timestamp, str | None, str | None]] = []
    is_friday = prices.index.dayofweek == 4

    for i in range(ma_days, len(prices)):
        dt = prices.index[i]

        # Apply return of current holding
        if holding is not None:
            r = returns[holding].iloc[i]
            strat_ret.iloc[i] = r if not np.isnan(r) else 0.0

        # Check if we should rebalance
        should_check = True if mode == "daily" else is_friday[i]
        if should_check:
            above = {}
            for name in etf_names:
                px = prices[name].iloc[i]
                ma = ma60[name].iloc[i]
                roc = roc20[name].iloc[i]
                if not np.isnan(ma) and px > ma and not np.isnan(roc):
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
                holding = new_holding

    # Trim to target range
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
                  ma_days: int = 60, roc_days: int = 25) -> tuple[dict, dict]:
    """返回 (持有天数dict, 买入次数dict)"""
    etf_names = list(prices.columns)
    ma60, roc20, _ = calc_indicators(prices, ma_days, roc_days)
    is_friday = prices.index.dayofweek == 4

    days = {n: 0 for n in etf_names}
    days["CASH"] = 0
    buys = {n: 0 for n in etf_names}
    holding = None

    for i in range(ma_days, len(prices)):
        dt = prices.index[i]
        if dt < pd.Timestamp(start_date) or dt > pd.Timestamp(end_date):
            continue

        should_check = True if mode == "daily" else is_friday[i]
        if should_check:
            above = {}
            for name in etf_names:
                px = prices[name].iloc[i]
                ma = ma60[name].iloc[i]
                roc = roc20[name].iloc[i]
                if not np.isnan(ma) and px > ma and not np.isnan(roc):
                    above[name] = roc
            new_holding = max(above, key=above.get) if above else None
            if new_holding is not None and new_holding != holding:
                buys[new_holding] += 1
            holding = new_holding

        days[holding or "CASH"] += 1
    return days, buys


def yearly_returns(ret: pd.Series) -> dict:
    result = {}
    for yr in sorted(set(ret.index.year)):
        mask = ret.index.year == yr
        if mask.sum() > 10:
            result[yr] = (1 + ret[mask]).prod() - 1
    return result


def trade_win_rate(ret: pd.Series, trade_dates: list) -> float:
    """计算交易胜率 = 盈利持仓期数 / 总持仓期数

    相邻调仓日之间的累计收益率 > 0 记为一次盈利。
    """
    if not trade_dates:
        return 0.0
    nav = (1 + ret).cumprod()
    valid = sorted({d for d in trade_dates if d in nav.index})
    if len(valid) < 1:
        return 0.0
    breaks = [ret.index[0]] + valid  # 首个分段起点
    wins = total = 0
    for i in range(len(breaks) - 1):
        seg = nav.loc[breaks[i]:breaks[i + 1]]
        if len(seg) >= 2:
            total += 1
            if seg.iloc[-1] / seg.iloc[0] - 1 > 0:
                wins += 1
    # 最后一段: 末次调仓 → 末尾
    seg = nav.loc[valid[-1]:]
    if len(seg) >= 2:
        total += 1
        if seg.iloc[-1] / seg.iloc[0] - 1 > 0:
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
        etf_nav = (1 + nav0[name].pct_change().fillna(0)).cumprod()
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
            etf_nav = (1 + nav0[name].pct_change().fillna(0)).cumprod()
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
    parser.add_argument("--roc", type=int, default=25, help="动量天数 (默认25)")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--source", default="tencent", choices=["tencent", "akshare"], help="数据源 (默认tencent)")
    parser.add_argument("--html", action="store_true", help="输出交互式HTML图表（可hover查看调仓ETF）")
    parser.add_argument("--backtrader", action="store_true", default=False, help="使用 backtrader 引擎回测")
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
            nav, bnav, ret, bret, trades, trade_dates, trade_details = (
                run_backtest_bt(prices_full, mode, args.start, args.end, args.ma, args.roc)
                if args.backtrader else
                run_backtest(prices_full, mode, args.start, args.end, args.ma, args.roc))
            m = metrics(nav, ret)
            bm = metrics(bnav, bret)
            all_metrics[mode] = (m, bm, trades, ret, nav, bnav, trade_dates, trade_details)

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
            wr = trade_win_rate(ret, trade_dates)
            print(f"{'交易次数':<14} {trades:>10}")
            print(f"{'胜率':<14} {wr:>9.1%}")

            pd_result = (position_dist_bt(prices_full, args.start, args.end, mode, args.ma, args.roc)
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
