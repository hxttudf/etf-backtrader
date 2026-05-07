#!/usr/bin/env python3
"""ETF双动量轮动 — 入场时机分析

遍历过去三年每一天作为回测起点，分析：
- 亏损时间最长的入场点（最大回撤持续天数）
- 盈利最多的入场点（最终累计收益）
- 2026年5月是否适合入场

用法:
  python etf_entry_analysis.py
  python etf_entry_analysis.py --years 3 --group 默认组合
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from etf_backtest import run_backtest, metrics
from etf_data import load_config, load_prices, load_prices_extended


def max_drawdown_duration(nav: pd.Series) -> int:
    """最长水下持续天数（连续低于前高的交易日数）"""
    peak = nav.cummax()
    underwater = nav < peak
    if not underwater.any():
        return 0

    durations: list[int] = []
    count = 0
    for u in underwater:
        if u:
            count += 1
        else:
            if count > 0:
                durations.append(count)
            count = 0
    if count > 0:
        durations.append(count)
    return max(durations) if durations else 0


def entry_scan(prices: pd.DataFrame, mode: str, ma: int, roc: int,
               start_scan: str, end_scan: str, end_fixed: str) -> pd.DataFrame:
    """遍历每个交易日作为入场点，返回每条入场的结果"""
    dates = prices.loc[start_scan:end_scan].index
    rows: list[dict] = []
    total = len(dates)

    for i, entry_dt in enumerate(dates):
        entry_str = entry_dt.strftime("%Y-%m-%d")
        nav, bnav, ret, bret, trades, trade_dates, trade_details = run_backtest(
            prices, mode, entry_str, end_fixed, ma, roc)

        m = metrics(nav, ret)
        dd_dur = max_drawdown_duration(nav)

        rows.append({
            "入场日期": entry_dt,
            "累计收益": m.get("累计收益", 0),
            "年化收益": m.get("年化收益", 0),
            "最大回撤": m.get("最大回撤", 0),
            "夏普比率": m.get("夏普比率", 0),
            "水下最长天数": dd_dur,
            "持有天数": len(ret),
        })

        if (i + 1) % 50 == 0 or i == total - 1:
            print(f"  进度: {i + 1}/{total}")

    return pd.DataFrame(rows)


def analyze(df: pd.DataFrame) -> None:
    """输出分析结果"""
    if df.empty:
        print("无数据")
        return

    # ── 亏损时间最长（水下天数最多） ──
    print("\n" + "=" * 80)
    print("  亏损时间最长 — 水下持续天数 TOP 10")
    print("=" * 80)
    worst_dur = df.nlargest(10, "水下最长天数")
    for _, r in worst_dur.iterrows():
        print(f"  {r['入场日期'].strftime('%Y-%m-%d')}  "
              f"水下{r['水下最长天数']:>4d}天  "
              f"回撤{r['最大回撤']:>8.1%}  "
              f"最终{r['累计收益']:>+8.1%}  "
              f"年化{r['年化收益']:>+8.1%}")

    # ── 亏损最多（最终收益最差） ──
    print("\n" + "=" * 80)
    print("  最终亏损最多 — 累计收益最低 TOP 10")
    print("=" * 80)
    worst_ret = df.nsmallest(10, "累计收益")
    for _, r in worst_ret.iterrows():
        print(f"  {r['入场日期'].strftime('%Y-%m-%d')}  "
              f"累计{r['累计收益']:>+8.1%}  "
              f"水下{r['水下最长天数']:>4d}天  "
              f"回撤{r['最大回撤']:>8.1%}  "
              f"持有{r['持有天数']:>4d}天")

    # ── 盈利最多（最终收益最高） ──
    print("\n" + "=" * 80)
    print("  最终盈利最多 — 累计收益最高 TOP 10")
    print("=" * 80)
    best_ret = df.nlargest(10, "累计收益")
    for _, r in best_ret.iterrows():
        print(f"  {r['入场日期'].strftime('%Y-%m-%d')}  "
              f"累计{r['累计收益']:>+8.1%}  "
              f"年化{r['年化收益']:>+8.1%}  "
              f"水下{r['水下最长天数']:>4d}天  "
              f"回撤{r['最大回撤']:>8.1%}")

    # ── 统计摘要 ──
    print("\n" + "=" * 80)
    print("  统计摘要（所有入场点）")
    print("=" * 80)
    for col in ["累计收益", "最大回撤", "水下最长天数", "夏普比率"]:
        vals = df[col]
        print(f"  {col:>10}:  "
              f"均值 {vals.mean():>8.3f}  "
              f"中位 {vals.median():>8.3f}  "
              f"最小 {vals.min():>8.3f}  "
              f"最大 {vals.max():>8.3f}")

    # ── 正收益占比 ──
    win_pct = (df["累计收益"] > 0).mean()
    print(f"\n  正收益比例: {win_pct:.1%}  ({int(win_pct * len(df))}/{len(df)} 个入场点盈利)")

    # ── 季度分析 ──
    df_q = df.copy()
    df_q["季度"] = df_q["入场日期"].dt.to_period("Q")
    q_stats = df_q.groupby("季度")["累计收益"].agg(["mean", "count", lambda x: (x > 0).mean()])
    q_stats.columns = ["均值", "样本数", "胜率"]
    print("\n  各季度入场表现:")
    for q, r in q_stats.iterrows():
        bar = "█" * int(r["胜率"] * 20)
        print(f"    {q}:  均值{r['均值']:>+7.1%}  胜率{r['胜率']:>6.0%}  {bar}  (n={int(r['样本数'])})")


def main() -> None:
    parser = argparse.ArgumentParser(description="ETF双动量轮动 — 入场时机分析")
    parser.add_argument("--group", default=None)
    parser.add_argument("--years", type=int, default=3, help="回溯年数 (默认3)")
    parser.add_argument("--ma", type=int, default=60)
    parser.add_argument("--roc", type=int, default=25)
    parser.add_argument("--mode", default="friday", choices=["daily", "friday"])
    parser.add_argument("--source", default="akshare", choices=["tencent", "akshare"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    group_name = args.group or next(iter(cfg["groups"].keys()))
    etfs = cfg["groups"][group_name]

    end_date = datetime.today().strftime("%Y-%m-%d")
    start_scan = (datetime.today() - pd.DateOffset(years=args.years)).strftime("%Y-%m-%d")
    lookback = (pd.Timestamp(start_scan) - pd.Timedelta(days=200)).strftime("%Y-%m-%d")

    print(f"组合: {group_name}  ({', '.join(etfs.keys())})")
    print(f"参数: MA={args.ma} ROC={args.roc} mode={args.mode}")
    print(f"扫描范围: {start_scan} ~ {end_date}")

    prices = load_prices_extended(etfs, group_name, source=args.source) if args.source == "akshare" else load_prices(etfs, group_name, source=args.source)
    prices = prices[prices.index >= lookback]
    print(f"数据: {len(prices)} 个交易日\n")

    print("开始逐日扫描...")
    df = entry_scan(prices, args.mode, args.ma, args.roc, start_scan, end_date, end_date)
    print(f"扫描完成: {len(df)} 个入场点\n")

    analyze(df)

    # ── 2026年5月分析 ──
    print("\n" + "=" * 80)
    print("  2026年5月入场分析")
    print("=" * 80)
    may_entries = df[df["入场日期"].between("2026-05-01", "2026-05-31")]
    if len(may_entries) > 0:
        avg_ret = may_entries["累计收益"].mean()
        avg_dd = may_entries["最大回撤"].mean()
        win_pct_may = (may_entries["累计收益"] > 0).mean()
        print(f"  5月可入场日: {len(may_entries)}")
        print(f"  平均累计收益: {avg_ret:+.2%}")
        print(f"  平均最大回撤: {avg_dd:.1%}")
        print(f"  正收益比例: {win_pct_may:.0%}")
    else:
        print(f"  5月尚无可用交易日（数据可能不足）")

    # 最近的趋势
    recent = df[df["入场日期"] >= "2026-04-01"]
    if len(recent) > 0:
        print(f"\n  2026年4月以来入场: {len(recent)} 天")
        print(f"    平均收益: {recent['累计收益'].mean():+.2%}")
        print(f"    胜率: {(recent['累计收益'] > 0).mean():.0%}")

    # ── 结论 ──
    print("\n" + "=" * 80)
    print("  结论")
    print("=" * 80)

    # 整体胜率
    overall_win = (df["累计收益"] > 0).mean()
    recent_avg = df[df["入场日期"] >= "2026-03-01"]["累计收益"].mean()
    dd_now = df.iloc[-1]["最大回撤"] if len(df) > 0 else 0

    if overall_win > 0.6:
        print(f"  ✅ 过去{args.years}年任意入场正收益概率 {overall_win:.0%}，策略整体有效")
    elif overall_win > 0.4:
        print(f"  ⚠️  过去{args.years}年任意入场正收益概率 {overall_win:.0%}，择时很重要")
    else:
        print(f"  ❌ 过去{args.years}年任意入场正收益概率仅 {overall_win:.0%}，策略近期表现不佳")

    print(f"  近2月入场平均收益: {recent_avg:+.2%}")
    print(f"  最近入场最大回撤: {dd_now:.1%}")

    if recent_avg > 0.05:
        print(f"  🟢 近期信号偏多，2026年5月较适合入场")
    elif recent_avg > -0.05:
        print(f"  🟡 近期信号中性，2026年5月可谨慎入场")
    else:
        print(f"  🔴 近期信号偏空，建议观望等待更好的入场时机")

    # 保存结果
    csv_path = Path(__file__).parent / f"etf_entry_analysis_{group_name.replace(' ','_')}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n全量结果: {csv_path}")


if __name__ == "__main__":
    main()
