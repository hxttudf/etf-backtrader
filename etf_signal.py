#!/usr/bin/env python3
"""ETF双动量轮动 - 每日信号脚本
规则：每组合4个ETF，选价格在60日均线上方 + 近20日涨幅最大的持有；全部低于60日线则空仓。

用法:
  python etf_signal.py                        # 默认组合，今日信号
  python etf_signal.py --all                   # 所有组合
  python etf_signal.py --group 默认组合         # 指定组合
  python etf_signal.py --date 2026-03-15       # 历史信号
  python etf_signal.py --config my_config.json  # 自定义配置
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from etf_data import DEFAULT_CONFIG, calc_indicators, load_config, load_prices


def signal_for_date(prices: pd.DataFrame, target_date: str,
                    ma_days: int = 60, roc_days: int = 25) -> tuple[str | None, pd.DataFrame, pd.Timestamp]:
    """计算指定日期的信号，返回 (持仓名称, 明细DataFrame)"""
    ma60, roc20, day_chg = calc_indicators(prices, ma_days, roc_days)

    dt = pd.Timestamp(target_date)
    if dt not in prices.index:
        # 取最近交易日
        available = prices.index[prices.index <= dt]
        if len(available) == 0:
            raise ValueError(f"日期 {target_date} 无数据")
        dt = available[-1]

    rows = []
    qualified: dict[str, float] = {}
    for name in prices.columns:
        px = float(prices[name].loc[dt])
        ma = float(ma60[name].loc[dt]) if not pd.isna(ma60[name].loc[dt]) else np.nan
        roc = float(roc20[name].loc[dt]) if not pd.isna(roc20[name].loc[dt]) else np.nan
        chg = float(day_chg[name].loc[dt]) if not pd.isna(day_chg[name].loc[dt]) else np.nan
        rows.append({"ETF": name, "收盘价": px, "当日涨幅": chg, "60日均线": ma, "20日涨幅": roc})

        if not np.isnan(ma) and px > ma and not np.isnan(roc):
            qualified[name] = roc

    df = pd.DataFrame(rows)
    best = max(qualified, key=qualified.get) if qualified else None
    return best, df, dt


def print_signal(prices: pd.DataFrame, group_name: str, target_date: str,
                 ma_days: int = 60, roc_days: int = 25) -> None:
    best, df, actual_dt = signal_for_date(prices, target_date, ma_days, roc_days)

    print(f"\n=== {actual_dt.strftime('%Y-%m-%d')} [{group_name}] ETF双动量信号 ===")
    if actual_dt.strftime("%Y-%m-%d") != target_date:
        print(f"(最近交易日，请求日期 {target_date})")

    print(f"{'ETF':<12} {'收盘价':>8} {'当日涨幅':>8} {f'{ma_days}日均线':>8} {'线上?':>6} {f'{roc_days}日涨幅':>8}")
    print("-" * 58)

    qualified: dict[str, float] = {}
    for _, row in df.iterrows():
        name = row["ETF"]
        px = row["收盘价"]
        chg = row["当日涨幅"]
        ma = row["60日均线"]
        roc = row["20日涨幅"]

        above = "是" if not np.isnan(ma) and px > ma else "否"
        chg_str = f"{chg:>+7.2%}" if not np.isnan(chg) else "    N/A"
        roc_str = f"{roc:>7.1%}" if not np.isnan(roc) else "    N/A"
        ma_str = f"{ma:>8.3f}" if not np.isnan(ma) else "    N/A"
        print(f"{name:<12} {px:>8.3f} {chg_str} {ma_str} {above:>6} {roc_str}")

        if above == "是" and not np.isnan(roc):
            qualified[name] = roc

    print("-" * 58)

    if qualified:
        best = max(qualified, key=qualified.get)
        details = ", ".join(f"{k} {v:.1%}" for k, v in sorted(qualified.items(), key=lambda x: -x[1]))
        print(f"\n>> 信号: 持有 [{best}]  (排名: {details})")
    else:
        print(f"\n>> 信号: 空仓 (全部低于{ma_days}日均线)")


def main() -> None:
    parser = argparse.ArgumentParser(description="ETF双动量轮动 每日信号")
    parser.add_argument("--group", default=None, help="组合名称 (默认第一个)")
    parser.add_argument("--all", action="store_true", help="运行所有组合")
    parser.add_argument("--date", default=datetime.today().strftime("%Y-%m-%d"), help="日期 YYYY-MM-DD")
    parser.add_argument("--ma", type=int, default=60, help="均线天数 (默认60)")
    parser.add_argument("--roc", type=int, default=25, help="动量天数 (默认25)")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--source", default="tencent", choices=["tencent", "akshare"], help="数据源 (默认tencent)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    all_groups = cfg["groups"]

    if args.all:
        groups_to_run = list(all_groups.items())
    elif args.group:
        if args.group not in all_groups:
            print(f"组合 '{args.group}' 不存在。可用: {', '.join(all_groups.keys())}")
            return
        groups_to_run = [(args.group, all_groups[args.group])]
    else:
        first = next(iter(all_groups.items()))
        groups_to_run = [first]

    for group_name, etfs in groups_to_run:
        prices = load_prices(etfs, group_name, source=args.source)
        if len(prices) < args.ma:
            print(f"[{group_name}] 数据不足: {len(prices)}天, 需>{args.ma}天")
            continue
        print_signal(prices, group_name, args.date, args.ma, args.roc)


if __name__ == "__main__":
    main()
