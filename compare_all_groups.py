"""Compare all ETF groups and find top 3 by Sharpe ratio."""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from etf_data import load_prices, calc_indicators
from etf_backtest import run_backtest, trade_win_rate

CFG = json.load(open(Path(__file__).parent / "etf_config.json"))
GROUPS = CFG["groups"]

# Build unique code set (name__code → code)
all_unique: dict[str, str] = {}
for gname, getfs in GROUPS.items():
    for name, code in getfs.items():
        all_unique[f"{name}__{code}"] = code

print(f"Loading prices for {len(all_unique)} unique codes...")
all_prices = load_prices(all_unique, "_batch", source="akshare")
print(f"Price data: {all_prices.shape[0]} days, {all_prices.shape[1]} codes")

# Default backtest params
MA, ROC = 60, 20
LOOKBACK = all_prices.index[MA]

results = []
total = len(GROUPS)
t0 = time.time()

for idx, (gname, getfs) in enumerate(GROUPS.items()):
    # Map unique column names back to group category names
    col_map = {f"{name}__{code}": name for name, code in getfs.items()}
    gprices = all_prices[list(col_map.keys())].rename(columns=col_map).dropna(how="all")

    # Trim to lookback
    gprices = gprices[gprices.index >= LOOKBACK]

    # Find actual start (all ETFs have data)
    firsts = [gprices[name].first_valid_index() for name in getfs]
    valid_firsts = [d for d in firsts if d is not None]
    if not valid_firsts:
        continue
    actual_start = max(max(valid_firsts), LOOKBACK)
    end_date = gprices.index[-1].strftime("%Y-%m-%d")
    start_str = actual_start.strftime("%Y-%m-%d")

    try:
        nav, _, ret, _, trades, _, tdets = run_backtest(
            gprices, "daily", start_str, end_date, MA, ROC)
    except Exception as e:
        continue

    if trades < 3:
        continue

    # Calc metrics
    cum_ret = nav.iloc[-1] - 1.0
    daily_rf = 0.02 / 252
    excess = ret - daily_rf
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0
    peak = nav.expanding().max()
    dd = (nav - peak) / peak
    max_dd = float(dd.min())
    calmar = float(cum_ret / abs(max_dd)) if max_dd != 0 else 0
    wr = trade_win_rate(ret, tdets, gprices)
    ann_ret = float((1 + cum_ret) ** (252 / len(ret)) - 1) if len(ret) > 0 else 0

    results.append({
        "组合": gname,
        "ETF": ", ".join(f"{n}({c})" for n, c in getfs.items()),
        "累计收益": cum_ret,
        "年化收益": ann_ret,
        "夏普比率": sharpe,
        "最大回撤": max_dd,
        "卡尔玛比率": calmar,
        "胜率": wr,
        "交易次数": trades,
        "数据起始": start_str,
        "数据截止": end_date,
        "数据天数": len(gprices),
    })

    if (idx + 1) % 50 == 0:
        elapsed = time.time() - t0
        print(f"  {idx+1}/{total} ({elapsed:.1f}s)")

elapsed = time.time() - t0
print(f"\nDone: {len(results)} groups in {elapsed:.1f}s\n")

if not results:
    print("No valid results!")
    sys.exit(1)

df = pd.DataFrame(results)

# Top 3 by Sharpe
print("=" * 80)
print("TOP 3 by 夏普比率")
print("=" * 80)
top_sharpe = df.nlargest(3, "夏普比率")
for _, r in top_sharpe.iterrows():
    print(f"  {r['组合']}")
    print(f"    ETF: {r['ETF']}")
    print(f"    夏普={r['夏普比率']:.2f}  年化={r['年化收益']:.2%}  回撤={r['最大回撤']:.2%}  "
          f"卡尔玛={r['卡尔玛比率']:.2f}  胜率={r['胜率']:.0%}  交易={r['交易次数']}")
    print(f"    区间={r['数据起始']}~{r['数据截止']} ({r['数据天数']}天)")
    print()

print("=" * 80)
print("TOP 3 by 卡尔玛比率")
print("=" * 80)
top_calmar = df.nlargest(3, "卡尔玛比率")
for _, r in top_calmar.iterrows():
    print(f"  {r['组合']}")
    print(f"    ETF: {r['ETF']}")
    print(f"    夏普={r['夏普比率']:.2f}  年化={r['年化收益']:.2%}  回撤={r['最大回撤']:.2%}  "
          f"卡尔玛={r['卡尔玛比率']:.2f}  胜率={r['胜率']:.0%}  交易={r['交易次数']}")
    print(f"    区间={r['数据起始']}~{r['数据截止']} ({r['数据天数']}天)")
    print()

print("=" * 80)
print("TOP 3 by 年化收益")
print("=" * 80)
top_ret = df.nlargest(3, "年化收益")
for _, r in top_ret.iterrows():
    print(f"  {r['组合']}")
    print(f"    ETF: {r['ETF']}")
    print(f"    夏普={r['夏普比率']:.2f}  年化={r['年化收益']:.2%}  回撤={r['最大回撤']:.2%}  "
          f"卡尔玛={r['卡尔玛比率']:.2f}  胜率={r['胜率']:.0%}  交易={r['交易次数']}")
    print(f"    区间={r['数据起始']}~{r['数据截止']} ({r['数据天数']}天)")
    print()

# Save full results
csv_path = Path(__file__).parent / "etf_all_groups_compare.csv"
df.sort_values("夏普比率", ascending=False).to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"Full results saved to: {csv_path}")
