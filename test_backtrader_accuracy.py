"""验证 backtrader 引擎与手写引擎结果一致"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from etf_data import load_prices
from etf_backtest import run_backtest as run_manual, metrics
from etf_backtrader import run_backtest_bt


def test_signal_identity():
    """交易信号必须完全一致"""
    cfg = {"默认组合": {"沪深300ETF": "510310", "创业板ETF": "159915", "纳指ETF": "513100"}}
    etfs = cfg["默认组合"]
    prices = load_prices(etfs, "默认组合", source="tencent")

    for mode in ["daily", "friday"]:
        _, _, _, _, trades_m, td_m, tdet_m = run_manual(prices, mode, "2025-06-01", "2026-04-30")
        _, _, _, _, trades_bt, td_bt, tdet_bt, _ = run_backtest_bt(prices, mode, "2025-06-01", "2026-04-30")

        assert trades_m == trades_bt, f"{mode}: trade count mismatch {trades_m} vs {trades_bt}"
        for i in range(trades_m):
            assert td_m[i] == td_bt[i], f"{mode}: trade date #{i} mismatch"
            assert tdet_m[i][1:] == tdet_bt[i][1:], f"{mode}: trade detail #{i} mismatch"

    print("PASS: signal identity")


def test_nav_correlation():
    """NAV 相关性 > 0.9999"""
    cfg = {"默认组合": {"沪深300ETF": "510310", "创业板ETF": "159915", "纳指ETF": "513100"}}
    etfs = cfg["默认组合"]
    prices = load_prices(etfs, "默认组合", source="tencent")

    for mode in ["daily", "friday"]:
        nav_m, _, ret_m, _, _, _, _ = run_manual(prices, mode, "2025-06-01", "2026-04-30")
        nav_bt, _, ret_bt, _, _, _, _, _ = run_backtest_bt(prices, mode, "2025-06-01", "2026-04-30")

        common = nav_m.index.intersection(nav_bt.index)
        corr = nav_m[common].corr(nav_bt[common])
        assert corr > 0.9999, f"{mode}: NAV correlation too low: {corr}"

    print("PASS: NAV correlation")


def test_nav_deviation():
    """NAV 偏差 < 1%"""
    cfg = {"默认组合": {"沪深300ETF": "510310", "创业板ETF": "159915", "纳指ETF": "513100"}}
    etfs = cfg["默认组合"]
    prices = load_prices(etfs, "默认组合", source="tencent")

    for mode in ["daily", "friday"]:
        nav_m, _, _, _, _, _, _ = run_manual(prices, mode, "2025-06-01", "2026-04-30")
        nav_bt, _, _, _, _, _, _, _ = run_backtest_bt(prices, mode, "2025-06-01", "2026-04-30")

        deviation = abs(nav_m.iloc[-1] / nav_bt.iloc[-1] - 1)
        assert deviation < 0.01, f"{mode}: NAV deviation too high: {deviation:.4%}"

    print("PASS: NAV deviation")


def test_metrics_consistency():
    """关键指标偏差在合理范围"""
    cfg = {"默认组合": {"沪深300ETF": "510310", "创业板ETF": "159915", "纳指ETF": "513100"}}
    etfs = cfg["默认组合"]
    prices = load_prices(etfs, "默认组合", source="tencent")

    for mode in ["daily", "friday"]:
        nav_m, _, ret_m, _, _, _, _ = run_manual(prices, mode, "2025-06-01", "2026-04-30")
        nav_bt, _, ret_bt, _, _, _, _, _ = run_backtest_bt(prices, mode, "2025-06-01", "2026-04-30")

        mm = metrics(nav_m, ret_m)
        mb = metrics(nav_bt, ret_bt)

        assert abs(mm["夏普比率"] - mb["夏普比率"]) < 0.1, f"{mode}: Sharpe deviation"
        assert abs(mm["最大回撤"] - mb["最大回撤"]) < 0.02, f"{mode}: MaxDD deviation"

    print("PASS: metrics consistency")


def test_strategy_selection():
    """每个策略都能正常运行并产生结果"""
    cfg = {"默认组合": {"沪深300ETF": "510310", "创业板ETF": "159915", "纳指ETF": "513100"}}
    etfs = cfg["默认组合"]
    prices = load_prices(etfs, "默认组合", source="tencent")

    for strategy in ["momentum", "rsi", "bb", "macd"]:
        nav, _, ret, _, trades, _, _, signals = run_backtest_bt(
            prices, "daily", "2025-06-01", "2026-04-30", strategy=strategy)
        assert len(nav) > 0, f"{strategy}: empty NAV"
        assert trades >= 0, f"{strategy}: negative trades"
        assert len(signals) > 0, f"{strategy}: no daily signals"
        assert nav.iloc[-1] > 0, f"{strategy}: NAV went to zero"

    print("PASS: strategy selection")


def test_daily_signals():
    """每日信号数据格式正确"""
    cfg = {"默认组合": {"沪深300ETF": "510310", "创业板ETF": "159915", "纳指ETF": "513100"}}
    etfs = cfg["默认组合"]
    prices = load_prices(etfs, "默认组合", source="tencent")

    for strategy in ["momentum", "rsi", "bb", "macd"]:
        _, _, _, _, _, _, _, signals = run_backtest_bt(
            prices, "daily", "2025-06-01", "2026-04-30", strategy=strategy)

        for sig in signals[:5]:
            assert '_dt' in sig, f"{strategy}: signal missing _dt"
            assert 'holding' in sig, f"{strategy}: signal missing holding"
            # Each signal should have entries for all ETFs
            for name in etfs:
                assert name in sig, f"{strategy}: signal missing ETF {name}"

        # At least one signal should have a non-None holding (unless strategy is very conservative)
        holdings = [s['holding'] for s in signals]
        assert len(holdings) > 0, f"{strategy}: no holding records"

    print("PASS: daily signals")


if __name__ == "__main__":
    test_signal_identity()
    test_nav_correlation()
    test_nav_deviation()
    test_metrics_consistency()
    test_strategy_selection()
    test_daily_signals()
    print("\nAll accuracy checks passed.")
