import locale
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# Fix macOS locale issue: Python 3.12+ on macOS without LANG set defaults to ASCII,
# which breaks urllib when reading HTTP responses with UTF-8 content.
os.environ.setdefault("LANG", "en_US.UTF-8")
os.environ.setdefault("LC_ALL", "en_US.UTF-8")
try:
    locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, "C.UTF-8")
    except locale.Error:
        pass
import _locale
_locale._getdefaultlocale = lambda *args: ("en_US", "UTF-8")

_ETF_BT_DIR = Path(__file__).parents[1]
if str(_ETF_BT_DIR) not in sys.path:
    sys.path.insert(0, str(_ETF_BT_DIR))


class DataLoader:
    def __init__(self, source: str = "tencent", cache_dir: str | None = None):
        self.source = source
        self.cache_dir = Path(cache_dir) if cache_dir else _ETF_BT_DIR / "cache"
        self.cache_dir.mkdir(exist_ok=True)

    def _read_cache_direct(self, universe: dict[str, str]) -> pd.DataFrame | None:
        """Read cached CSV directly, no network calls."""
        from etf_data import _cache_path
        cache = _cache_path(self.source)
        open_cache = _cache_path(self.source + "_open")
        if not cache.exists():
            return None
        try:
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
            code_map = {v: k for k, v in universe.items()}
            avail = [c for c in code_map if c in df.columns]
            if not avail:
                return None
            result = df[avail].rename(columns=code_map)
            return result.loc[result.index >= result.first_valid_index()]
        except Exception:
            return None

    def _read_cache_open_direct(self, universe: dict[str, str]) -> pd.DataFrame | None:
        """Read cached open prices directly, no network calls."""
        from etf_data import _cache_path
        open_cache = _cache_path(self.source + "_open")
        if not open_cache.exists():
            return None
        try:
            df = pd.read_csv(open_cache, index_col=0, parse_dates=True)
            code_map = {v: k for k, v in universe.items()}
            avail = [c for c in code_map if c in df.columns]
            if not avail:
                return None
            result = df[avail].rename(columns=code_map)
            return result.loc[result.index >= result.first_valid_index()]
        except Exception:
            return None

    def load_extended_prices(self, universe: dict[str, str]) -> pd.DataFrame:
        import time

        # Try cache first (no network)
        cached = self._read_cache_direct(universe)
        if cached is not None and len(cached) > 100:
            print(f"[loader] 从缓存读取 {len(cached)} 天数据 ({self.source})")
            return cached

        sources_to_try = [self.source]
        if self.source == "tencent":
            sources_to_try.append("akshare")
        elif self.source == "akshare":
            sources_to_try.append("tencent")
        else:
            sources_to_try.extend(["akshare", "tencent"])

        last_error = None
        for src in sources_to_try:
            try:
                from etf_data import load_prices_extended as _extended
                # Set timeout for network
                import socket
                old_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(20)
                try:
                    df = _extended(
                        {v: k for k, v in universe.items()},
                        source=src,
                    )
                finally:
                    socket.setdefaulttimeout(old_timeout)
                df.columns = list(universe.keys())
                self.source = src
                return df
            except Exception as e:
                last_error = e
                print(f"[loader] 数据源 {src} 失败: {e}，尝试下一个...")

        # Final fallback: try reading ANY cached CSV directly
        for alt_src in ["akshare", "tencent", "em"]:
            if alt_src == self.source:
                continue
            self.source = alt_src
            cached = self._read_cache_direct(universe)
            if cached is not None and len(cached) > 100:
                print(f"[loader] 从缓存读取 {len(cached)} 天数据 ({alt_src})")
                return cached
            self.source = sources_to_try[0]

        raise RuntimeError(
            f"所有数据源均失败。最后错误: {last_error}\n"
            f"请检查网络连接。\n"
            f"常见修复: LANG=en_US.UTF-8 streamlit run etf_app.py"
        )

    def _fetch_one_ohlc(self, name: str, code: str) -> pd.DataFrame | None:
        try:
            import akshare as ak
            m = "sh" if code.startswith("5") else "sz"
            df = ak.fund_etf_hist_sina(symbol=f"{m}{code}")
            df["日期"] = pd.to_datetime(df["date"])
            df = df.set_index("日期")[["open", "high", "low", "close", "volume"]].sort_index()
            return df
        except Exception:
            return None

    def _ohlc_cache_path(self, code: str) -> Path:
        return self.cache_dir / f"ohlc_{code}.parquet"

    def _load_cached_ohlc(self, universe: dict[str, str]) -> dict[str, pd.DataFrame]:
        result = {}
        for name, code in universe.items():
            p = self._ohlc_cache_path(code)
            if p.exists():
                try:
                    result[name] = pd.read_parquet(p)
                except Exception:
                    pass
        return result

    def _fetch_missing_ohlc(
        self, universe: dict[str, str], cached: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        missing = {n: c for n, c in universe.items() if n not in cached}
        if not missing:
            return {}

        result = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            fut = {pool.submit(self._fetch_one_ohlc, n, c): n for n, c in missing.items()}
            for f in as_completed(fut):
                name = fut[f]
                try:
                    df = f.result()
                    if df is not None:
                        result[name] = df
                        df.to_parquet(self._ohlc_cache_path(universe[name]))
                except Exception:
                    pass
        return result

    def load_ohlc(self, universe: dict[str, str]) -> dict[str, pd.DataFrame]:
        cached = self._load_cached_ohlc(universe)
        fetched = self._fetch_missing_ohlc(universe, cached)
        result = {**cached, **fetched}

        vol_cache = self.cache_dir / "volume.parquet"
        if not vol_cache.exists() and result:
            vol_df = pd.DataFrame(
                {name: df["volume"] for name, df in result.items()}
            )
            vol_df.to_parquet(vol_cache)

        return result

    def load_volume(self, universe: dict[str, str]) -> pd.DataFrame | None:
        vol_cache = self.cache_dir / "volume.parquet"
        if vol_cache.exists():
            return pd.read_parquet(vol_cache)

        ohlc = self.load_ohlc(universe)
        if not ohlc:
            return None
        vol_df = pd.DataFrame(
            {name: df["volume"] for name, df in ohlc.items()}
        )
        vol_df.to_parquet(vol_cache)
        return vol_df
