"""ETF 数据模块 — 多数据源 + 本地缓存
共享于 etf_signal.py 和 etf_backtest.py
"""

from datetime import datetime, timedelta
from pathlib import Path
import json
import urllib.request

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).parent
DEFAULT_CONFIG = Path(__file__).parent / "etf_config.json"

# ── 数据源注册 ──────────────────────────────────────────

def _market(code: str) -> str:
    """5xxxxx=sh, 1xxxxx=sz"""
    return "sh" if code.startswith("5") else "sz"


def fetch_one_tencent(code: str, days: int = 800) -> pd.Series:
    """腾讯财经 — 最多约800个交易日"""
    m = _market(code)
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={m}{code},day,,,{days},qfq"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    inner = data["data"][f"{m}{code}"]
    klines = inner.get("qfqday") or inner.get("day") or []
    rows = [{"日期": k[0], "开盘": float(k[1]), "收盘": float(k[2])} for k in klines]
    df = pd.DataFrame(rows)
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.set_index("日期")
    close = df["收盘"]
    close._open = df["开盘"]
    return close


def _fix_splits(s: pd.Series, threshold: float = 0.5) -> pd.Series:
    """修复基金份额折算（分拆）导致的单位净值跳变。
    从最新段往前逐个分拆点修正，保持最新段价格不变。
    threshold=0.5：单日跌幅超50%视为分拆事件，A股真实行情跌幅上限约20%。
    """
    s = s.copy()
    pct = s.pct_change(fill_method=None)
    splits = pct[pct < -threshold].index.tolist()
    for split_dt in reversed(splits):
        iloc = s.index.get_loc(split_dt)
        split_ratio = s.iloc[iloc] / s.iloc[iloc - 1]
        s.iloc[:iloc] = s.iloc[:iloc] * split_ratio
    return s


def fetch_one_akshare(code: str, days: int = 0) -> pd.Series:
    """Sina 全量历史（2011年起）+ 腾讯前复权锚点拼接。
    腾讯覆盖段（近800天）直接使用腾讯数据，更早历史用 Sina 并修复内部分拆跳变，
    在衔接点按中位数比值对齐到腾讯尺度。days 参数忽略。
    """
    import time
    import akshare as ak

    m = _market(code)
    for attempt in range(3):
        try:
            df = ak.fund_etf_hist_sina(symbol=f"{m}{code}")
            df["日期"] = pd.to_datetime(df["date"])
            sina_close = df.set_index("日期")["close"].sort_index()
            sina_open = df.set_index("日期")["open"].sort_index()
            break
        except Exception as e:
            if attempt < 2:
                wait = (attempt + 1) * 5
                print(f"  [{code}] Sina 拉取失败: {e}, {wait}s 后重试...")
                time.sleep(wait)
            else:
                raise

    tencent = fetch_one_tencent(code)
    tencent_start = tencent.index[0]
    sina_early_close = sina_close[sina_close.index < tencent_start]
    sina_early_open = sina_open[sina_open.index < tencent_start]
    if len(sina_early_close) == 0:
        # Cache open alongside close (stored by caller)
        return sina_close  # type: ignore

    overlap = sina_close.index.intersection(tencent.index)
    if len(overlap) == 0:
        return sina_close  # type: ignore

    sina_early_close_fixed = _fix_splits(sina_early_close)
    # Apply the SAME split ratios to open prices (splits are corporate actions,
    # not price movements — close and open must use identical adjustment factors)
    close_adj_factor = sina_early_close_fixed / sina_early_close
    sina_early_open_fixed = sina_early_open * close_adj_factor
    ratio = (tencent.loc[overlap[:10]] / sina_close.loc[overlap[:10]]).median()
    # Store open prices as an attribute so caller can retrieve them
    result_close = pd.concat([sina_early_close_fixed * ratio, tencent]).sort_index()
    # Build open: Sina early + tencent open (from tencent OHLC data)
    tencent_open = get_open_from_result(tencent)
    if tencent_open is None:
        tencent_open = pd.Series(np.roll(tencent.values, 1), index=tencent.index)
        tencent_open.iloc[0] = tencent.iloc[0]
    result_open = pd.concat([sina_early_open_fixed * ratio, tencent_open]).sort_index()
    result_open = result_open[~result_open.index.duplicated()]
    # Attach open to result as attribute (hack to avoid changing return type)
    result_close._open = result_open
    return result_close


def get_open_from_result(result: pd.Series) -> pd.Series | None:
    """Extract cached open prices from a fetch_one_akshare result, if available."""
    return getattr(result, '_open', None)


def fetch_one_em(code: str, days: int = 0) -> pd.Series:
    """AKShare EM source — 东方财富前复权，复权质量更高。需要可访问 EastMoney API。days 参数忽略。"""
    import time
    import akshare as ak

    last_error = None
    for attempt in range(3):
        try:
            df = ak.fund_etf_hist_em(symbol=code, adjust="qfq")
            df["日期"] = pd.to_datetime(df["日期"])
            df = df.set_index("日期")
            close = df["收盘"]
            if "开盘" in df.columns:
                close._open = df["开盘"]
            return close
        except Exception as e:
            last_error = e
            if attempt < 2:
                wait = (attempt + 1) * 8
                print(f"  [{code}] EM 拉取失败 (尝试 {attempt+1}/3): {e}, {wait}s 后重试...")
                time.sleep(wait)

    raise ConnectionError(
        f"东方财富数据源不可用（{code}）：{last_error}\n"
        f"建议：切换到 AKShare(Sina) 或腾讯财经数据源。"
    ) from last_error


# 数据源名称 → 拉取函数
SOURCES = {
    "tencent": fetch_one_tencent,
    "akshare": fetch_one_akshare,
    "em": fetch_one_em,
}


# ── 缓存 ────────────────────────────────────────────────

def _cache_path(source: str) -> Path:
    return CACHE_DIR / f"etf_prices_{source}.csv"


def _cache_path_open(source: str) -> Path:
    return CACHE_DIR / f"etf_prices_{source}_open.csv"


def _migrate_old_cache() -> bool:
    """旧缓存文件 etf_prices_all.csv → etf_prices_tencent.csv"""
    old = CACHE_DIR / "etf_prices_all.csv"
    new = _cache_path("tencent")
    if old.exists() and not new.exists():
        old.rename(new)
        return True
    return False


# ── 数据加载 ────────────────────────────────────────────

def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p) as f:
        return json.load(f)


def load_prices(etfs: dict, group_name: str = "default", source: str = "tencent") -> pd.DataFrame:
    """加载缓存或拉取，返回价格DataFrame。

    etfs: {"名称": "代码"}
    source: "tencent" | "akshare"
    每个数据源独立缓存文件，不交叉污染。
    """
    _migrate_old_cache()
    fetch_fn = SOURCES[source]
    cache_file = _cache_path(source)

    if cache_file.exists():
        cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        cached = pd.DataFrame()

    # 历史数据不可变，仅缓存为空/数据不完整/过期时才重拉。用户可通过UI按钮手动刷新。
    is_stale = len(cached) == 0
    if not is_stale and source in ("akshare", "em") and len(cached) > 0:
        if cached.index[0] > pd.Timestamp("2018-01-01"):
            is_stale = True
    if not is_stale and source in ("akshare", "em") and not _cache_path_open(source).exists():
        is_stale = True
    # Auto-refresh if cached data is more than 5 calendar days behind (no trading data for a week+)
    if not is_stale and len(cached) > 0:
        cache_latest = cached.index[-1]
        today = pd.Timestamp.now().normalize()
        if cache_latest < today - pd.Timedelta(days=5):
            print(f"[{source}] 缓存过期 ({cache_latest.strftime('%Y-%m-%d')})，重新拉取...")
            is_stale = True
    # Intraday / post-close refresh for today's data
    if not is_stale and len(cached) > 0:
        today = pd.Timestamp.now().normalize()
        now = pd.Timestamp.now()
        is_trading_day = now.dayofweek < 5
        tracked = [c for c in etfs.values() if c in cached.columns]

        if today in cached.index:
            today_has_nan = bool(tracked) and cached.loc[today, tracked].isna().any() if tracked else False
            is_trading = is_trading_day and 9 <= now.hour < 15

            if is_trading:
                # During trading hours: re-fetch if cache file hasn't been updated in 5+ min
                cache_mtime = pd.Timestamp(cache_file.stat().st_mtime, unit='s')
                if (now - cache_mtime).total_seconds() > 300:
                    print(f"[{source}] 交易时段缓存已过{int((now - cache_mtime).total_seconds() / 60)}分钟，重新拉取...")
                    is_stale = True
            elif now.hour >= 15:
                # st_mtime is POSIX UTC, convert to local datetime for hour check
                from datetime import datetime as dt_mod
                cache_mtime_local = pd.Timestamp(
                    dt_mod.fromtimestamp(cache_file.stat().st_mtime))
                already_refreshed_today = (cache_mtime_local.normalize() == today
                                           and cache_mtime_local.hour >= 15)
                if not already_refreshed_today:
                    was_fetched_intraday = cache_mtime_local.hour < 15
                    if was_fetched_intraday:
                        print(f"[{source}] 盘中缓存需刷新为收盘价，重新拉取...")
                        is_stale = True
                    elif today_has_nan:
                        print(f"[{source}] 今日盘中缓存不完整，收盘后重新拉取...")
                    is_stale = True
        elif is_trading_day and today > cached.index[-1] and now.hour >= 9:
            # Today not in cache at all → only fetch after market opens
            print(f"[{source}] 缺少今日数据，拉取...")
            is_stale = True
    new_codes = [c for c in set(etfs.values()) if c not in cached.columns]

    if is_stale or new_codes:
        codes_to_fetch: list[str] = list(cached.columns) if is_stale else []
        for c in new_codes:
            if c not in codes_to_fetch:
                codes_to_fetch.append(c)

        label = f"[{source}] 拉取数据..." if is_stale or len(new_codes) == len(codes_to_fetch) else f"[{source}] 拉取新ETF..."
        print(label)
        results = {}
        open_results = {}
        failed = []
        for code in codes_to_fetch:
            try:
                s = fetch_fn(code)
                results[code] = s
                o = get_open_from_result(s)
                if o is not None:
                    open_results[code] = o
            except Exception as e:
                failed.append((code, str(e)))
                print(f"  [{code}] 拉取失败，跳过: {e}")
        if failed:
            print(f"  ⚠ {len(failed)}/{len(codes_to_fetch)} 个ETF拉取失败: {[c for c, _ in failed]}")
        if not results:
            if is_stale and len(cached) > 0:
                print(f"  ⚠ 数据源 [{source}] 所有ETF拉取失败，继续使用过期缓存 ({cached.index[-1].strftime('%Y-%m-%d')})")
                print(f"  建议：刷新数据缓存或切换到其他数据源（推荐 AKShare(Sina)）。")
            else:
                raise RuntimeError(
                    f"数据源 [{source}] 所有ETF拉取均失败。"
                    f"请检查网络连接或切换到其他数据源（推荐 AKShare(Sina)）。"
                )
        new_data = pd.DataFrame(results).dropna(how='all')

        if is_stale:
            # merge to preserve today's data if re-fetch didn't return it
            if len(cached) > 0 and today not in new_data.index and today in cached.index:
                cached = cached.combine_first(new_data)
            else:
                cached = new_data
        else:
            cached = cached.combine_first(new_data)

        cached.to_csv(cache_file, encoding="utf-8-sig")

        # Cache open prices separately when available
        if open_results:
            open_cache_file = _cache_path_open(source)
            new_open = pd.DataFrame(open_results).dropna()
            if open_cache_file.exists():
                old_open = pd.read_csv(open_cache_file, index_col=0, parse_dates=True)
                if is_stale:
                    cached_open = new_open
                else:
                    cached_open = old_open.combine_first(new_open)
            else:
                cached_open = new_open
            cached_open.to_csv(open_cache_file, encoding="utf-8-sig")

        print(f"[{source}] {len(cached)}天 ({cached.index[0].strftime('%Y-%m-%d')} ~ {cached.index[-1].strftime('%Y-%m-%d')})")

    col_map = {code: name for name, code in etfs.items()}
    result = cached[[c for c in col_map if c in cached.columns]].rename(columns=col_map)
    return result.loc[result.index >= result.first_valid_index()]


def load_open_prices(etfs: dict, group_name: str = "default", source: str = "akshare") -> pd.DataFrame | None:
    """加载开盘价缓存。akshare / em 源支持。无缓存时返回 None。"""
    open_cache_file = _cache_path_open(source)
    if not open_cache_file.exists():
        return None
    cached_open = pd.read_csv(open_cache_file, index_col=0, parse_dates=True)
    col_map = {code: name for name, code in etfs.items()}
    available = [c for c in col_map if c in cached_open.columns]
    if not available:
        return None
    result = cached_open[available].rename(columns=col_map)
    return result.loc[result.index >= result.first_valid_index()]


# ── 指数映射 (ETF代码 → 底层指数) ──────────────────────────
# 用于在ETF上市前用指数数据补全历史

ETF_INDEX_MAP: dict[str, tuple[str, str]] = {
    # ETF代码 → (指数代码, 指数名称)
    "518880": ("XAU", "黄金期货"),           # 黄金ETF, ETF 2013上市, XAU 2006起
    "159934": ("XAU", "黄金期货"),           # 黄金ETF(深)
    "513100": (".IXIC", "纳斯达克综合"),     # 纳指ETF, ETF 2013上市, IXIC 2004起
    "159941": (".IXIC", "纳斯达克综合"),     # 纳指ETF(深)
    "159915": ("sz399006", "创业板指"),       # 创业板ETF, ETF 2011上市, 399006 2010起
    "159952": ("sz399006", "创业板指"),       # 创业板ETF(深)
    "512890": ("sh000015", "上证红利"),       # 红利低波ETF, ETF 2013上市, 000015 2005起
    "510310": ("sh000300", "沪深300"),        # 沪深300ETF
}


def _fetch_index_data(index_code: str) -> pd.Series | None:
    """拉取指数日线收盘价。国内指数用AKShare，美股指数用Sina。"""
    import time

    if index_code.startswith("sh") or index_code.startswith("sz"):
        # 国内指数
        for attempt in range(2):
            try:
                df = __import__("akshare").stock_zh_index_daily(symbol=index_code)
                df["date"] = pd.to_datetime(df["date"])
                return df.set_index("date")["close"].sort_index()
            except Exception:
                if attempt == 0:
                    time.sleep(3)
        return None

    if index_code.startswith("."):
        # 美股指数 (Sina)
        for attempt in range(2):
            try:
                df = __import__("akshare").index_us_stock_sina(symbol=index_code)
                df["date"] = pd.to_datetime(df["date"])
                return df.set_index("date")["close"].sort_index()
            except Exception:
                if attempt == 0:
                    time.sleep(3)
        return None

    if index_code == "XAU":
        for attempt in range(2):
            try:
                df = __import__("akshare").futures_foreign_hist(symbol="XAU")
                df["date"] = pd.to_datetime(df["date"])
                return df.set_index("date")["close"].sort_index()
            except Exception:
                if attempt == 0:
                    time.sleep(3)
        return None

    return None


def load_prices_extended(etfs: dict, group_name: str = "default",
                         source: str = "tencent") -> pd.DataFrame:
    """加载ETF价格，自动用底层指数补全ETF上市前的历史。

    算法：在ETF上市首日计算 ETF价格/指数值 的比值，
    上市前的指数值 × 比值 = 合成ETF价格。上市后用真实ETF数据。
    各ETF独立补全，最终对齐到所有列都有数据的第一个交易日。
    """
    df_etf = load_prices(etfs, group_name, source=source)

    extended: dict[str, pd.Series] = {}
    for name, code in etfs.items():
        if code not in ETF_INDEX_MAP:
            extended[name] = df_etf[name]
            continue

        index_code, index_name = ETF_INDEX_MAP[code]
        etf_series = df_etf[name].dropna()
        etf_start = etf_series.index[0]

        index_series = _fetch_index_data(index_code)
        if index_series is None:
            extended[name] = df_etf[name]
            continue
        if index_series.empty:
            extended[name] = df_etf[name]
            continue

        # 找到ETF上市首日前后各30天的重合窗口，取中位数比值
        overlap = index_series.index[
            (index_series.index >= etf_start - pd.Timedelta(days=30)) &
            (index_series.index <= etf_start + pd.Timedelta(days=30))
        ]
        if len(overlap) < 3:
            extended[name] = df_etf[name]
            continue

        ratio = (etf_series.loc[etf_start] / index_series.loc[etf_start])
        if not (0.000001 < ratio < 100000):
            extended[name] = df_etf[name]
            continue

        # 上市前的指数数据 × 比值 = 合成ETF价格
        early_idx = index_series[index_series.index < etf_start]
        if len(early_idx) == 0:
            extended[name] = df_etf[name]
            continue

        synthetic = early_idx * ratio
        combined = pd.concat([synthetic, etf_series])
        extended[name] = combined[~combined.index.duplicated(keep="last")].sort_index()

        # 报告
        orig_start = df_etf[name].dropna().index[0].strftime("%Y-%m-%d")
        new_start = combined.index[0].strftime("%Y-%m-%d")
        extra = len(early_idx)
        print(f"  {name}: {orig_start} → {new_start} (+{extra}天, 来自{index_name})")

    result = pd.DataFrame(extended)
    # 对齐：保留所有数据，用最早的有数据日期（各列独立，允许前导NaN）
    first_valid = max(s.dropna().index[0] for _, s in result.items())
    result = result.loc[result.index >= first_valid]
    print(f"  最终范围: {result.index[0].strftime('%Y-%m-%d')} ~ {result.index[-1].strftime('%Y-%m-%d')} ({len(result)}天)")
    return result


def load_midday_prices(etfs: dict) -> pd.DataFrame | None:
    """加载中午收盘价（11:30 60分钟K线收盘价）"""
    path = CACHE_DIR / "etf_midday_sina.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    available = [name for name in etfs if name in df.columns]
    if not available:
        return None
    return df[available]


def load_afternoon_open_prices(etfs: dict) -> pd.DataFrame | None:
    """加载下午开盘价（14:00 60分钟K线开盘价）"""
    path = CACHE_DIR / "etf_afternoon_open_sina.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    available = [name for name in etfs if name in df.columns]
    if not available:
        return None
    return df[available]


def midday_data_available(etfs: dict) -> bool:
    """检查中午收盘价数据是否可用于给定ETF组合"""
    midday = load_midday_prices(etfs)
    aft = load_afternoon_open_prices(etfs)
    if midday is None or aft is None:
        return False
    return len(midday.columns) >= len(etfs) and len(aft.columns) >= len(etfs)


def calc_indicators(prices: pd.DataFrame, ma: int = 60, roc: int = 25):
    """计算均线和动量指标。先 ffill 处理节假日缺失值再计算。"""
    p = prices.ffill()
    return p.rolling(ma).mean(), p.pct_change(roc, fill_method=None), p.pct_change(fill_method=None)
