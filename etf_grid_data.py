"""网格交易 — 独立数据加载模块（分钟级 OHLC）

独立于 etf_data.py（动量策略用 daily close），
为网格回测提供分钟级 high/low 数据检测网格穿越。

数据源：
  - fund_etf_hist_min_em: ETF 分钟线（1/5/15/30/60）
  - stock_zh_a_hist_min_em: 个股分钟线（1/5/15/30/60）
"""

from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import akshare as ak

CACHE_DIR = Path(__file__).parent


def _detect_type(symbol: str) -> str:
    """判断是 ETF 还是个股"""
    return "etf" if symbol.startswith(("5", "1")) else "stock"


def _market(symbol: str) -> str:
    return "sh" if symbol.startswith(("5", "6")) else "sz"


def _cache_path(symbol: str, period: str) -> Path:
    fname = f"grid_{symbol}_{period}min.csv"
    return CACHE_DIR / fname


def _fetch_minute(symbol: str, period: str, start_date: str, end_date: str) -> pd.DataFrame:
    """拉取分钟 K 线，返回 OHLC DataFrame"""
    typ = _detect_type(symbol)
    try:
        if typ == "etf":
            df = ak.fund_etf_hist_min_em(
                symbol=symbol, period=period,
                start_date=start_date, end_date=end_date,
                adjust="qfq",
            )
        else:
            df = ak.stock_zh_a_hist_min_em(
                symbol=symbol, period=period,
                start_date=start_date, end_date=end_date,
                adjust="qfq",
            )
    except Exception as e:
        raise RuntimeError(f"拉取 {symbol} ({period}min) 失败: {e}")

    if df is None or len(df) == 0:
        return pd.DataFrame()

    # 统一列名
    col_map = {"时间": "datetime", "开盘": "open", "收盘": "close",
               "最高": "high", "最低": "low", "成交量": "volume"}
    df = df.rename(columns=col_map)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    df = df[["open", "high", "low", "close", "volume"]].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.astype(float)


def load_grid_data(symbol: str, period: str = "5",
                   start_date: str | None = None,
                   end_date: str | None = None,
                   force_refresh: bool = False) -> pd.DataFrame:
    """加载分钟 K 线数据（缓存优先）

    Args:
        symbol: ETF/个股代码
        period: 分钟级别 "1"/"5"/"15"/"30"/"60"
        start_date: 起始日 "2026-01-01"，默认 1 年前
        end_date: 结束日，默认今天
        force_refresh: 强制重新拉取

    Returns:
        DataFrame index=datetime, columns=['open','high','low','close','volume']
    """
    now = pd.Timestamp.now()
    if end_date is None:
        end_date = now.strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (now - pd.Timedelta(days=365)).strftime("%Y-%m-%d")

    cache_file = _cache_path(symbol, period)

    # 从缓存加载
    if not force_refresh and cache_file.exists():
        cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        # 检查缓存是否足够新
        cache_end = cached.index[-1]
        cache_start = cached.index[0]
        # 缓存覆盖范围足够则直接返回
        if cache_start <= pd.Timestamp(start_date) and cache_end >= pd.Timestamp(end_date):
            trim = cached[(cached.index >= start_date) & (cached.index <= end_date)]
            if len(trim) > 0:
                return trim

    # 拉取数据（拉取范围比请求范围更大，避免后续频繁重拉）
    fetch_start = (pd.Timestamp(start_date) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    fetch_end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    df = _fetch_minute(symbol, period, fetch_start, fetch_end)
    if len(df) == 0:
        # 拉取失败，尝试用缓存兜底
        if cache_file.exists():
            cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            trim = cached[(cached.index >= start_date) & (cached.index <= end_date)]
            if len(trim) > 0:
                return trim
        return df

    # 合并旧缓存（如果有新数据拉不到的部分）
    if cache_file.exists():
        old = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        combined = pd.concat([old, df[~df.index.isin(old.index)]]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        combined = df

    combined.to_csv(cache_file)

    trim = combined[(combined.index >= start_date) & (combined.index <= end_date)]
    return trim


def daily_from_minute(minute_df: pd.DataFrame) -> pd.DataFrame:
    """分钟 K 线聚合为日线 OHLC（用于对比/展示）"""
    if len(minute_df) == 0:
        return minute_df
    daily = minute_df.resample("D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open"])
    return daily
