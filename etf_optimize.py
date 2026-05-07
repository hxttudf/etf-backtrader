#!/usr/bin/env python3
"""ETF双动量轮动 — 参数优化：网格搜索最优 MA / ROC 配置

用法:
  python etf_optimize.py --start 2020-01-01 --end 2025-12-31
  python etf_optimize.py --start 2020-01-01 --ma-min 20 --ma-max 120 --ma-step 10 \\
                         --roc-min 10 --roc-max 60 --roc-step 5
  python etf_optimize.py --start 2020-01-01 --metric calmar --top 30 --heatmap
  python etf_optimize.py --start 2020-01-01 --mode daily    # 仅每日调仓
"""

import argparse
import itertools
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from etf_backtest import run_backtest, metrics, trade_win_rate
from etf_data import load_config, load_prices, load_prices_extended

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    matplotlib.use("Agg")
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Microsoft YaHei", "SimHei", "PingFang SC", "Heiti SC"]
    plt.rcParams["axes.unicode_minus"] = False
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

METRIC_LABELS = {
    "年化收益": "年化收益",
    "夏普比率": "夏普比率",
    "卡尔玛比率": "卡尔玛比率",
    "胜率": "胜率",
    "最大回撤": "最大回撤",
    "累计收益": "累计收益",
    "年化波动": "年化波动",
}


def grid_search(prices: pd.DataFrame, modes: list[str], start: str, end: str,
                ma_values: list[int], roc_values: list[int]) -> pd.DataFrame:
    """网格搜索所有 (MA, ROC) 组合，返回结果 DataFrame。"""
    rows: list[dict] = []
    total = len(ma_values) * len(roc_values) * len(modes)
    done = 0

    for ma, roc in itertools.product(ma_values, roc_values):
        for mode in modes:
            nav, bnav, ret, bret, trades, trade_dates, trade_details = run_backtest(
                prices, mode, start, end, ma, roc)
            m = metrics(nav, ret)
            wr = trade_win_rate(ret, trade_dates)
            rows.append({
                "MA": ma,
                "ROC": roc,
                "模式": mode,
                "累计收益": m.get("累计收益", 0),
                "年化收益": m.get("年化收益", 0),
                "年化波动": m.get("年化波动", 0),
                "夏普比率": m.get("夏普比率", 0),
                "最大回撤": m.get("最大回撤", 0),
                "最大亏损": m.get("最大亏损", 0),
                "水下天数": m.get("水下天数", 0),
                "卡尔玛比率": m.get("卡尔玛比率", 0),
                "胜率": wr,
                "交易次数": trades,
            })
            done += 1
        print(f"  MA={ma:3d} 完成 ({done}/{total})")

    return pd.DataFrame(rows)


def plot_heatmaps(df: pd.DataFrame, metric: str, out_dir: Path) -> list[Path]:
    """为每个模式生成一张热力图，保存到 out_dir。"""
    if not HAS_MPL:
        print("[WARN] matplotlib 未安装，跳过热力图")
        return []

    modes = df["模式"].unique()
    files: list[Path] = []
    label = METRIC_LABELS.get(metric, metric)

    for mode in modes:
        sub = df[df["模式"] == mode].pivot_table(index="MA", columns="ROC", values=metric)
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(sub.values, cmap="RdYlGn", aspect="auto", interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, shrink=0.75)
        cbar.set_label(label, fontsize=11)

        ax.set_xticks(range(len(sub.columns)))
        ax.set_xticklabels([str(c) for c in sub.columns], fontsize=9)
        ax.set_yticks(range(len(sub.index)))
        ax.set_yticklabels([str(r) for r in sub.index], fontsize=9)
        ax.set_xlabel("ROC 天数", fontsize=11)
        ax.set_ylabel("MA 天数", fontsize=11)
        ax.set_title(f"参数网格 — {label} ({mode}调仓)", fontsize=13, fontweight="bold")

        # 在每个格子中标注数值
        for i in range(len(sub.index)):
            for j in range(len(sub.columns)):
                v = sub.values[i, j]
                if metric in ("夏普比率", "卡尔玛比率"):
                    txt = f"{v:.2f}"
                elif metric in ("胜率",):
                    txt = f"{v:.0%}"
                elif metric in ("最大回撤",):
                    txt = f"{v:.1%}"
                else:
                    txt = f"{v:.1%}"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7, color="black" if 0.3 < im.norm(v) < 0.7 else "white")

        fig.tight_layout()
        out = out_dir / f"etf_heatmap_{metric}_{mode}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        files.append(out)

    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="ETF双动量轮动 — 参数优化")
    parser.add_argument("--group", default=None, help="组合名称")
    parser.add_argument("--start", required=True, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"), help="结束日期")
    parser.add_argument("--mode", default="both", choices=["daily", "friday", "both"], help="调仓模式")

    parser.add_argument("--ma-min", type=int, default=20, help="MA 最小值 (默认20)")
    parser.add_argument("--ma-max", type=int, default=120, help="MA 最大值 (默认120)")
    parser.add_argument("--ma-step", type=int, default=10, help="MA 步长 (默认10)")

    parser.add_argument("--roc-min", type=int, default=10, help="ROC 最小值 (默认10)")
    parser.add_argument("--roc-max", type=int, default=60, help="ROC 最大值 (默认60)")
    parser.add_argument("--roc-step", type=int, default=5, help="ROC 步长 (默认5)")

    parser.add_argument("--metric", default="夏普比率",
                        choices=["夏普比率", "卡尔玛比率", "年化收益", "胜率", "累计收益", "最大回撤", "年化波动"],
                        help="排序依据 (默认 夏普比率)")
    parser.add_argument("--top", type=int, default=20, help="显示前N组 (默认20)")
    parser.add_argument("--heatmap", action="store_true", help="生成热力图")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--source", default="tencent", choices=["tencent", "akshare"], help="数据源")
    args = parser.parse_args()

    modes = ["daily", "friday"] if args.mode == "both" else [args.mode]

    ma_values = list(range(args.ma_min, args.ma_max + 1, args.ma_step))
    roc_values = list(range(args.roc_min, args.roc_max + 1, args.roc_step))
    print(f"MA 范围: {ma_values}")
    print(f"ROC 范围: {roc_values}")
    print(f"组合数: {len(ma_values) * len(roc_values) * len(modes)}")
    print(f"调仓模式: {modes}")

    cfg = load_config(args.config)
    all_groups = cfg["groups"]
    group_name = args.group or next(iter(all_groups.keys()))
    etfs = all_groups[group_name]

    print(f"\n组合: {group_name}  ({', '.join(etfs.keys())})")

    start_dt = pd.Timestamp(args.start)
    lookback_start = (start_dt - pd.Timedelta(days=200)).strftime("%Y-%m-%d")
    prices_full = load_prices_extended(etfs, group_name, source=args.source) if args.source == "akshare" else load_prices(etfs, group_name, source=args.source)
    prices_full = prices_full[prices_full.index >= lookback_start]
    print(f"数据: {len(prices_full)} 个交易日\n")

    print("开始网格搜索...")
    df = grid_search(prices_full, modes, args.start, args.end, ma_values, roc_values)
    print("搜索完成!\n")

    # --- 按指标排序输出 ---
    for mode in modes:
        sub = df[df["模式"] == mode].sort_values(args.metric, ascending=False)
        print(f"\n{'='*60}")
        print(f"  [{mode.upper()}调仓] 按 {args.metric} 排序 TOP {args.top}")
        print(f"{'='*60}")
        cols = ["MA", "ROC", args.metric, "胜率", "年化收益", "最大回撤", "最大亏损", "水下天数", "卡尔玛比率", "交易次数"]
        header = "  ".join(f"{c:>10}" for c in cols)
        print(f"  {header}")
        print(f"  {'-' * (11 * len(cols))}")
        for _, r in sub.head(args.top).iterrows():
            vals = []
            for c in cols:
                v = r[c]
                if c in ("夏普比率", "卡尔玛比率"):
                    vals.append(f"{v:>10.2f}")
                elif c in ("胜率",):
                    vals.append(f"{v:>9.0%}")
                elif c in ("水下天数", "交易次数"):
                    vals.append(f"{v:>10.0f}")
                elif c in ("最大回撤", "最大亏损", "年化收益", "累计收益"):
                    vals.append(f"{v:>9.1%}")
                else:
                    vals.append(f"{v:>10}")
            print("  " + "  ".join(vals))

    # --- 全模式汇总 TOP ---
    if len(modes) > 1:
        top_all = df.sort_values(args.metric, ascending=False).head(args.top)
        print(f"\n{'='*60}")
        print(f"  全部模式汇总 TOP {args.top} (按 {args.metric})")
        print(f"{'='*60}")
        cols = ["MA", "ROC", "模式", args.metric, "胜率", "年化收益", "最大回撤", "最大亏损", "水下天数", "卡尔玛比率", "交易次数"]
        header = "  ".join(f"{c:>10}" for c in cols)
        print(f"  {header}")
        print(f"  {'-' * (11 * len(cols))}")
        for _, r in top_all.iterrows():
            vals = []
            for c in cols:
                v = r[c]
                if c in ("夏普比率", "卡尔玛比率"):
                    vals.append(f"{v:>10.2f}")
                elif c in ("胜率",):
                    vals.append(f"{v:>9.0%}")
                elif c in ("水下天数", "交易次数"):
                    vals.append(f"{v:>10.0f}")
                elif c in ("最大回撤", "最大亏损", "年化收益", "累计收益"):
                    vals.append(f"{v:>9.1%}")
                else:
                    vals.append(f"{v:>10}")
            print("  " + "  ".join(vals))

    # --- 热力图 ---
    if args.heatmap:
        out_dir = Path(__file__).parent
        files = plot_heatmaps(df, args.metric, out_dir)
        for f in files:
            print(f"\n热力图: {f}")

    # 保存全量 CSV
    csv_path = Path(__file__).parent / f"etf_optimize_{group_name.replace(' ','_')}_{args.start}_{args.end}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n全量结果: {csv_path}")


if __name__ == "__main__":
    main()
