"""网格交易 — 独立数据加载模块（分钟级/daily OHLC）

独立于 etf_data.py（动量策略用 daily close），
为网格回测提供 high/low 数据检测网格穿越。

数据源：
  - em: 东方财富分钟线（1/5/15/30/60）和日线
  - akshare: Sina 日线（最稳定，ETF 和个股均有 OHLC）
"""

from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import akshare as ak

CACHE_DIR = Path(__file__).parent


def _detect_type(symbol: str) -> str:
    return "etf" if symbol.startswith(("5", "1")) else "stock"


def _market(symbol: str) -> str:
    return "sh" if symbol.startswith(("5", "6")) else "sz"


def _cache_path(symbol: str, period: str, source: str) -> Path:
    if period == "daily":
        fname = f"grid_{symbol}_daily.csv"
    else:
        fname = f"grid_{symbol}_{period}min_{source}.csv"
    return CACHE_DIR / fname


def _fetch_minute_em(symbol: str, period: str, start_date: str, end_date: str) -> pd.DataFrame:
    """拉取东方财富分钟 K 线"""
    typ = _detect_type(symbol)
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
    if df is None or len(df) == 0:
        return pd.DataFrame()
    col_map = {"时间": "datetime", "开盘": "open", "收盘": "close",
               "最高": "high", "最低": "low", "成交量": "volume"}
    df = df.rename(columns=col_map)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    df = df[["open", "high", "low", "close", "volume"]].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.astype(float)


def _fetch_daily_sina(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """拉取 Sina 日线 OHLC（ETF 和个股都支持）"""
    typ = _detect_type(symbol)
    if typ == "etf":
        df = ak.fund_etf_hist_sina(symbol=f"{_market(symbol)}{symbol}")
        df["日期"] = pd.to_datetime(df["date"])
        df = df.set_index("日期")
        # Sina 列: date,open,close,high,low,volume
        df = df[["open", "high", "low", "close", "volume"]].sort_index()
    else:
        df = ak.stock_zh_a_hist(symbol=symbol, start_date=start_date, end_date=end_date, adjust="qfq")
        col_map = {"日期": "date", "开盘": "open", "收盘": "close",
                   "最高": "high", "最低": "low", "成交量": "volume"}
        df = df.rename(columns=col_map)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df = df[["open", "high", "low", "close", "volume"]]
    return df.astype(float)


def _fetch_daily_em(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """拉取东方财富日线 OHLC"""
    typ = _detect_type(symbol)
    if typ == "etf":
        df = ak.fund_etf_hist_em(symbol=symbol, period="daily",
                                  start_date=start_date, end_date=end_date, adjust="qfq")
    else:
        df = ak.stock_zh_a_hist_em(symbol=symbol, start_date=start_date,
                                    end_date=end_date, adjust="qfq")
    col_map = {"日期": "date", "开盘": "open", "收盘": "close",
               "最高": "high", "最低": "low", "成交量": "volume"}
    df = df.rename(columns=col_map)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df = df[["open", "high", "low", "close", "volume"]].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.astype(float)


def _fetch_data(symbol: str, period: str, start_date: str, end_date: str,
                source: str = "akshare") -> pd.DataFrame:
    """拉取数据，带重试"""
    last_err = None
    for attempt in range(3):
        try:
            if period == "daily":
                if source == "akshare":
                    return _fetch_daily_sina(symbol, start_date, end_date)
                elif source == "em":
                    return _fetch_daily_em(symbol, start_date, end_date)
                else:
                    raise ValueError(f"日线不支持数据源: {source}")
            else:
                if source == "em":
                    return _fetch_minute_em(symbol, period, start_date, end_date)
                else:
                    raise ValueError(f"分钟线仅支持 em 数据源, 不支持: {source}")
        except Exception as e:
            last_err = e
            import time
            time.sleep(2 * (attempt + 1))
            continue
    raise RuntimeError(f"拉取 {symbol} ({period}) [{source}] 失败: {last_err}")


def load_grid_data(symbol: str, period: str = "5",
                   start_date: str | None = None,
                   end_date: str | None = None,
                   force_refresh: bool = False,
                   source: str = "akshare") -> pd.DataFrame:
    """加载网格 OHLC 数据（缓存优先）

    Args:
        symbol: ETF/个股代码
        period: "daily" 或分钟级别 "1"/"5"/"15"/"30"/"60"
        start_date: 起始日 "2026-01-01"
        end_date: 结束日
        force_refresh: 强制重新拉取
        source: "akshare"(Sina, 日线) / "em"(东方财富, 日线+分钟线)

    Returns:
        DataFrame index=datetime, columns=['open','high','low','close','volume']
    """
    now = pd.Timestamp.now()
    if end_date is None:
        end_date = now.strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (now - pd.Timedelta(days=365)).strftime("%Y-%m-%d")

    cache_file = _cache_path(symbol, period, source)

    if not force_refresh and cache_file.exists():
        cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        if len(cached) > 0:
            cache_s, cache_e = cached.index[0], cached.index[-1]
            if cache_s <= pd.Timestamp(start_date) and cache_e >= pd.Timestamp(end_date):
                trim = cached[(cached.index >= start_date) & (cached.index <= end_date)]
                if len(trim) > 0:
                    return trim

    df = _fetch_data(symbol, period, start_date, end_date, source=source)
    if len(df) == 0 and cache_file.exists():
        cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        trim = cached[(cached.index >= start_date) & (cached.index <= end_date)]
        return trim if len(trim) > 0 else df
    if len(df) == 0:
        return df

    # 合并旧缓存
    if cache_file.exists():
        old = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        combined = pd.concat([old, df[~df.index.isin(old.index)]]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        combined = df

    combined.to_csv(cache_file)
    trim = combined[(combined.index >= start_date) & (combined.index <= end_date)]
    return trim
