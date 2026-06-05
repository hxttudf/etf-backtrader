#!/usr/bin/env python3
"""Analyze ALL drawdowns > 5% for 价纳创黄C3, using app's full backtest parameters.
Run with: python3 analyze_drawdowns.py"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from etf_data import calc_indicators, load_prices

GROUP_NAME = "价纳创黄C3"
START = "2018-01-01"
END = "2026-06-05"
SOURCE = "akshare"
MA_DAYS = 55
ROC_DAYS = 20
COMMISSION = 0.0001
STAMP_DUTY = 0.0
CRASH_SIGMA = 2.6
CRASH_WIN = 51

etfs = {"价值": "512040", "纳指": "513100", "创业板": "159952", "黄金": "159937"}
print(f"Loading prices for {GROUP_NAME}...")
prices = load_prices(etfs, GROUP_NAME, source=SOURCE)
prices = prices[prices.index >= "2017-06-01"]
print(f"Loaded {len(prices)} rows")

print("Running backtest (app full params)...")
returns = prices.pct_change(fill_method=None)
ma60, roc20, _ = calc_indicators(prices, MA_DAYS, ROC_DAYS)
strat_ret = pd.Series(0.0, index=prices.index)
holding = None
for i in range(MA_DAYS, len(prices)):
    dt = prices.index[i]
    above = {}
    for name in etfs:
        px = prices[name].iloc[i]
        ma = ma60[name].iloc[i]
        roc = roc20[name].iloc[i]
        if not pd.isna(ma) and px > ma and not pd.isna(roc):
            if CRASH_SIGMA is not None and i >= CRASH_WIN:
                ret_t = returns[name].iloc[i]
                if not pd.isna(ret_t):
                    std = returns[name].iloc[i-CRASH_WIN+1:i+1].std()
                    if std > 0 and ret_t < -CRASH_SIGMA * std:
                        continue
            above[name] = roc
    signal = max(above, key=above.get) if above else None
    if signal != holding:
        if holding is not None:
            r = returns[holding].iloc[i]
            strat_ret.iloc[i] = r if not np.isnan(r) else 0.0
            strat_ret.iloc[i] -= COMMISSION + STAMP_DUTY
        if signal is not None:
            strat_ret.iloc[i] -= COMMISSION
        holding = signal
    elif holding is not None:
        r = returns[holding].iloc[i]
        strat_ret.iloc[i] = r if not np.isnan(r) else 0.0
trim = (prices.index >= pd.Timestamp(START)) & (prices.index <= pd.Timestamp(END))
nav = (1 + strat_ret[trim]).cumprod()
print(f"Total return: {nav.iloc[-1]-1:.2%}")

# ─── Drawdown extraction ───
peak = nav.cummax()
dd = nav / peak - 1
drawdowns = []
in_dd = False
for i in range(len(nav)):
    dt = nav.index[i]
    v = nav.iloc[i]
    current_peak = peak.iloc[i]
    if not in_dd:
        if dd.iloc[i] < -0.05:
            in_dd = True
            dd_start, dd_trough, dd_trough_val, max_dd = dt, dt, v, dd.iloc[i]
    else:
        if dd.iloc[i] < max_dd:
            max_dd, dd_trough, dd_trough_val = dd.iloc[i], dt, v
        if v >= current_peak:
            in_dd = False
            drawdowns.append({
                'start': dd_start, 'trough': dd_trough, 'end': dt,
                'max_dd': max_dd, 'dd_days': (dd_trough - dd_start).days,
                'recv_days': (dt - dd_trough).days, 'total_days': (dt - dd_start).days,
            })
if in_dd:
    drawdowns.append({
        'start': dd_start, 'trough': dd_trough, 'end': nav.index[-1],
        'max_dd': max_dd, 'dd_days': (dd_trough - dd_start).days,
        'recv_days': (nav.index[-1] - dd_trough).days, 'total_days': (nav.index[-1] - dd_start).days,
        'ongoing': True,
    })

# ─── 5% bucket stats ───
buckets = {'5~10%': [], '10~15%': [], '15~20%': [], '20~25%': []}
for d in drawdowns:
    pct = abs(d['max_dd'])
    if pct < 0.10: buckets['5~10%'].append(d)
    elif pct < 0.15: buckets['10~15%'].append(d)
    elif pct < 0.20: buckets['15~20%'].append(d)
    else: buckets['20~25%'].append(d)

print(f"\n{'='*100}")
print("各区间回撤统计")
print(f"{'='*100}")
print(f"{'区间':<10} {'次数':>4} {'下跌到底均值':>12} {'修复均值':>10} {'修复中位数':>10} {'修复最短':>8} {'修复最长':>8}")
print('-' * 65)
for bk in ['5~10%','10~15%','15~20%','20~25%']:
    v = buckets.get(bk, [])
    if not v:
        print(f"{bk:<10} {0:>4} {'—':>12} {'—':>10} {'—':>10} {'—':>8} {'—':>8}")
        continue
    dd_d = np.array([d['dd_days'] for d in v if not d.get('ongoing')])
    recv = np.array([d['recv_days'] for d in v if not d.get('ongoing')])
    n_recv = len(recv)
    print(f"{bk:<10} {len(v):>4} {dd_d.mean():>7.0f}d    {recv.mean():>5.0f}d    {np.median(recv):>5.0f}d    {recv.min() if n_recv else '-':>3}    {recv.max() if n_recv else '-':>3}")

print(f"\n{'='*100}")
print("阈值跨过次数（含更深回撤）")
for th in [5, 10, 15, 20]:
    cnt = sum(1 for d in drawdowns if abs(d['max_dd']) * 100 >= th)
    print(f"  ≥ {th:>2}%: {cnt}次 ({cnt/8.5:.1f}次/年)")

# ─── Yearly distribution ───
print(f"\n{'='*100}")
print("按年份分布")
print(f"{'='*100}")
print(f"{'年份':<6} {'次数':>4} {'平均回撤':<10} {'最深回撤':<8} {'平均下跌':<8} {'平均修复':<8} {'最长修复':<8}")
print('-' * 65)
by_year = {}
for d in drawdowns:
    by_year.setdefault(str(d['start'].year), []).append(d)
for y in sorted(by_year.keys()):
    v = [d for d in by_year[y] if not d.get('ongoing')]
    if not v:
        continue
    avg_dd = np.mean([abs(d['max_dd']) for d in v])
    max_dd_v = max([abs(d['max_dd']) for d in v])
    avg_days = np.mean([d['dd_days'] for d in v])
    avg_recv = np.mean([d['recv_days'] for d in v])
    max_recv = max([d['recv_days'] for d in v])
    print(f"{y:<6} {len(v):>4} {avg_dd:<10.2%} {max_dd_v:<8.2%} {avg_days:<8.0f} {avg_recv:<8.0f} {max_recv:<8.0f}")

# ─── Detail list ───
print(f"\n{'='*100}")
print("完整回撤列表")
print(f"{'='*100}")
for d in drawdowns:
    ong = " *" if d.get('ongoing') else ""
    print(f"  {d['start'].strftime('%Y-%m-%d')} → {d['trough'].strftime('%Y-%m-%d')} → {d['end'].strftime('%Y-%m-%d')}{ong}")
    print(f"    {d['max_dd']:.2%} | 下跌{d['dd_days']}天 | 修复{d['recv_days']}天 | 共{d['total_days']}天")
