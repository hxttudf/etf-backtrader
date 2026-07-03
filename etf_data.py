"""ETF 数据模块 — 多数据源 + 本地缓存 (SQLite + CSV)
共享于 etf_signal.py 和 etf_backtest.py
"""

from datetime import datetime, timedelta
from pathlib import Path
import json
import os
import urllib.request

import numpy as np
import pandas as pd

CACHE_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).parent))
DEFAULT_CONFIG = CACHE_DIR / "etf_config.json"
_CSV_FALLBACK = os.environ.get("ETF_CSV_FALLBACK", "0") == "1"

from etf_db import PriceDB, get_last_fetch_time, _set_last_fetch


# ── 数据源注册 ──────────────────────────────────────────

def _market(code: str) -> str:
    """5xxxxx=sh, 1xxxxx=sz"""
    return "sh" if code.startswith("5") else "sz"


def fetch_one_tencent(code: str, days: int = 800) -> pd.Series:
    """腾讯财经 — 最多约800个交易日（2位小数）。days=0 仅今日（3位小数）。"""
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
    """修复基金份额折算（分拆）导致的单位净值跳变。"""
    s = s.copy()
    pct = s.pct_change(fill_method=None)
    splits = pct[pct < -threshold].index.tolist()
    for split_dt in reversed(splits):
        iloc = s.index.get_loc(split_dt)
        split_ratio = s.iloc[iloc] / s.iloc[iloc - 1]
        s.iloc[:iloc] = s.iloc[:iloc] * split_ratio
    return s


def fetch_one_akshare(code: str, days: int = 0) -> pd.Series:
    """Sina 全量历史（2011年起）+ 腾讯前复权锚点拼接。"""
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
        return sina_close

    overlap = sina_close.index.intersection(tencent.index)
    if len(overlap) == 0:
        return sina_close

    sina_early_close_fixed = _fix_splits(sina_early_close)
    close_adj_factor = sina_early_close_fixed / sina_early_close
    sina_early_open_fixed = sina_early_open * close_adj_factor
    ratio = (tencent.loc[overlap[:10]] / sina_close.loc[overlap[:10]]).median()
    result_close = pd.concat([sina_early_close_fixed * ratio, tencent]).sort_index()
    tencent_open = get_open_from_result(tencent)
    if tencent_open is None:
        tencent_open = pd.Series(np.roll(tencent.values, 1), index=tencent.index)
        tencent_open.iloc[0] = tencent.iloc[0]
    result_open = pd.concat([sina_early_open_fixed * ratio, tencent_open]).sort_index()
    result_open = result_open[~result_open.index.duplicated()]
    result_close._open = result_open
    return result_close


def fetch_tencent_qt(code: str) -> dict | None:
    """腾讯实时行情 — 3位小数精度，用于信号查询"""
    m = _market(code)
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={m}{code},day,,,1,qfq"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        qt = data.get("data", {}).get(f"{m}{code}", {}).get("qt", {}).get(f"{m}{code}")
        if qt and len(qt) > 5:
            return {"open": float(qt[5]), "price": float(qt[3]), "prev_close": float(qt[4])}
    except Exception:
        pass
    return None

def get_open_from_result(result: pd.Series) -> pd.Series | None:
    """Extract cached open prices from a fetch_one_akshare result, if available."""
    return getattr(result, '_open', None)


def fetch_one_em(code: str, days: int = 0) -> pd.Series:
    """AKShare EM source — 东方财富前复权。"""
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


# ── 缓存 (SQLite 为主, CSV 为向下兼容) ──────────────────

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


def _read_cache_from_db(codes: list, source: str) -> pd.DataFrame:
    """从 SQLite 读取收盘价，返回 index=日期, columns=codes 的 DataFrame"""
    if _CSV_FALLBACK:
        return pd.DataFrame()
    return PriceDB.load_close(codes, source)


def _read_open_cache_from_db(codes: list, source: str) -> pd.DataFrame:
    """从 SQLite 读取开盘价"""
    if _CSV_FALLBACK:
        return pd.DataFrame()
    return PriceDB.load_open(codes, source)


def _write_cache_to_db(df_close: pd.DataFrame, df_open: pd.DataFrame, source: str) -> None:
    """写入价格数据到 SQLite"""
    try:
        if df_close is not None and not df_close.empty:
            PriceDB.save_close(df_close, source)
        if df_open is not None and not df_open.empty:
            PriceDB.save_open(df_open, source)
    except Exception as e:
        print(f"  [DB] SQLite 写入失败: {e}")


def _cache_has_data(codes: list, source: str) -> bool:
    """SQLite 中是否有缓存数据"""
    if _CSV_FALLBACK:
        return False
    return PriceDB.has_source(codes, source)


def _cache_latest_date(codes: list, source: str) -> pd.Timestamp | None:
    """SQLite 中缓存的最新日期"""
    if _CSV_FALLBACK:
        return None
    d = PriceDB.latest_date(codes, source)
    if d:
        return pd.Timestamp(d)
    return None


def _cache_earliest_date(codes: list, source: str) -> pd.Timestamp | None:
    """SQLite 中缓存的最旧日期"""
    if _CSV_FALLBACK:
        return None
    d = PriceDB.earliest_date(codes, source)
    if d:
        return pd.Timestamp(d)
    return None


# ── 数据加载 ────────────────────────────────────────────

def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p) as f:
        return json.load(f)


def load_prices(etfs: dict, group_name: str = "default", source: str = "tencent") -> pd.DataFrame:
    """加载缓存或拉取，返回价格DataFrame。

    etfs: {"名称": "代码"}
    source: "tencent" | "akshare"
    缓存优先从 SQLite 读取，同时保持 CSV 文件作为向下兼容。
    """
    _migrate_old_cache()
    fetch_fn = SOURCES[source]
    codes = list(etfs.values())
    cache_file = _cache_path(source)
    open_cache_file = _cache_path_open(source)

    # ── 尝试从 SQLite 读取缓存 ──
    cached = _read_cache_from_db(codes, source)
    cached_open = _read_open_cache_from_db(codes, source)

    # ── SQLite 无数据时回退到 CSV ──
    if cached.empty and cache_file.exists():
        cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        _write_cache_to_db(cached, None, source)
    if cached_open.empty and open_cache_file.exists():
        cached_open = pd.read_csv(open_cache_file, index_col=0, parse_dates=True)
        _write_cache_to_db(None, cached_open, source)

    # ── 过期检测 ──
    is_stale = len(cached) == 0
    if not is_stale and source in ("akshare", "em") and len(cached) > 0:
        if cached.index[0] > pd.Timestamp("2018-01-01"):
            is_stale = True
    if not is_stale and source in ("akshare", "em") and cached_open.empty:
        is_stale = True
    if not is_stale and len(cached) > 0:
        cache_latest = cached.index[-1]
        today = pd.Timestamp.now().normalize()
        if cache_latest < today - pd.Timedelta(days=5):
            print(f"[{source}] 缓存过期 ({cache_latest.strftime('%Y-%m-%d')})，重新拉取...")
            is_stale = True

    # Intraday / post-close refresh
    if not is_stale and len(cached) > 0:
        today = pd.Timestamp.now().normalize()
        now = pd.Timestamp.now()
        is_trading_day = now.dayofweek < 5
        tracked = [c for c in codes if c in cached.columns]

        if today in cached.index:
            today_has_nan = bool(tracked) and cached.loc[today, tracked].isna().any() if tracked else False
            is_trading = is_trading_day and 9 <= now.hour < 15

            if is_trading:
                last_fetch = get_last_fetch_time(source) or pd.Timestamp(0)
                if (now - last_fetch).total_seconds() > 300:
                    print(f"[{source}] 交易时段缓存已过{int((now - last_fetch).total_seconds() / 60)}分钟，重新拉取...")
                    is_stale = True
            elif now.hour >= 15:
                last_fetch = get_last_fetch_time(source) or pd.Timestamp(0)
                already_refreshed_today = (last_fetch.normalize() == today and last_fetch.hour >= 15)
                if not already_refreshed_today:
                    was_fetched_intraday = last_fetch.hour < 15
                    if was_fetched_intraday:
                        print(f"[{source}] 盘中缓存需刷新为收盘价，重新拉取...")
                        is_stale = True
                    elif today_has_nan:
                        print(f"[{source}] 今日盘中缓存不完整，收盘后重新拉取...")
                    is_stale = True
        elif is_trading_day and today > cached.index[-1] and now.hour >= 9:
            print(f"[{source}] 缺少今日数据，拉取...")
            is_stale = True

    new_codes = [c for c in codes if c not in cached.columns]

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

        cached_latest = cached.index.max() if len(cached) > 0 else None
        missing_days = (pd.Timestamp.now().normalize() - cached_latest).days if cached_latest else None

        for code in codes_to_fetch:
            try:
                if source == "tencent" and cached_latest is not None and code in cached.columns:
                    fetch_days = max(missing_days + 10, 10) if missing_days else 10
                    s = fetch_one_tencent(code, days=fetch_days)
                else:
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

        cached = cached.combine_first(new_data)

        # ── 腾讯 days=0 补今天高精度收盘/开盘 ──
        today_ts = pd.Timestamp.now().normalize()
        if source == "akshare" and today_ts in cached.index:
            for code in codes:
                try:
                    tod = fetch_one_tencent(code, days=0)
                    if code not in cached.columns or today_ts not in tod.index:
                        continue
                    cached.loc[today_ts, code] = tod.loc[today_ts]
                    tod_open = get_open_from_result(tod)
                    if tod_open is not None and today_ts in tod_open.index:
                        if code not in open_results:
                            open_results[code] = tod_open
                        else:
                            open_results[code].loc[today_ts] = tod_open.loc[today_ts]
                except Exception:
                    pass

        # ── 写入 SQLite + CSV ──
        _write_cache_to_db(cached, None, source)
        if open_results:
            new_open = pd.DataFrame(open_results).dropna()
            if cached_open.empty:
                cached_open = new_open
            else:
                cached_open = cached_open.combine_first(new_open)
            _write_cache_to_db(None, cached_open, source)
            cached_open.to_csv(open_cache_file, encoding="utf-8-sig")

        cached.to_csv(cache_file, encoding="utf-8-sig")

        _set_last_fetch(source)

        print(f"[{source}] {len(cached)}天 ({cached.index[0].strftime('%Y-%m-%d')} ~ {cached.index[-1].strftime('%Y-%m-%d')})")

    col_map = {code: name for name, code in etfs.items()}
    result = cached[[c for c in col_map if c in cached.columns]].rename(columns=col_map)
    return result.loc[result.index >= result.first_valid_index()]


def load_open_prices(etfs: dict, group_name: str = "default", source: str = "akshare") -> pd.DataFrame | None:
    """加载开盘价缓存。akshare / em 源支持。无缓存时返回 None。"""
    codes = list(etfs.values())
    df = _read_open_cache_from_db(codes, source)
    if df.empty:
        open_cache_file = _cache_path_open(source)
        if not open_cache_file.exists():
            return None
        df = pd.read_csv(open_cache_file, index_col=0, parse_dates=True)
        _write_cache_to_db(None, df, source)
    col_map = {code: name for name, code in etfs.items()}
    available = [c for c in col_map if c in df.columns]
    if not available:
        return None
    result = df[available].rename(columns=col_map)
    return result.loc[result.index >= result.first_valid_index()]


# ── 指数映射 (ETF代码 → 底层指数) ──────────────────────────

ETF_INDEX_MAP: dict[str, tuple[str, str]] = {
    "518880": ("XAU", "黄金期货"),
    "159934": ("XAU", "黄金期货"),
    "513100": (".IXIC", "纳斯达克综合"),
    "159941": (".IXIC", "纳斯达克综合"),
    "159915": ("sz399006", "创业板指"),
    "159952": ("sz399006", "创业板指"),
    "512890": ("sh000015", "上证红利"),
    "510310": ("sh000300", "沪深300"),
}


def _fetch_index_data(index_code: str) -> pd.Series | None:
    """拉取指数日线收盘价。"""
    import time

    if index_code.startswith("sh") or index_code.startswith("sz"):
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
    """加载ETF价格，自动用底层指数补全ETF上市前的历史。"""
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
        if index_series is None or index_series.empty:
            extended[name] = df_etf[name]
            continue

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

        early_idx = index_series[index_series.index < etf_start]
        if len(early_idx) == 0:
            extended[name] = df_etf[name]
            continue

        synthetic = early_idx * ratio
        combined = pd.concat([synthetic, etf_series])
        extended[name] = combined[~combined.index.duplicated(keep="last")].sort_index()

        orig_start = df_etf[name].dropna().index[0].strftime("%Y-%m-%d")
        new_start = combined.index[0].strftime("%Y-%m-%d")
        extra = len(early_idx)
        print(f"  {name}: {orig_start} → {new_start} (+{extra}天, 来自{index_name})")

    result = pd.DataFrame(extended)
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
