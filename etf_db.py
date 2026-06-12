"""轻量数据库层 — SQLite 存储用户配置、K线数据、分析缓存
取代 JSON/CSV 文件，提供 ACID 保证和并发安全。
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent / "data")))
DB_PATH = DATA_DIR / "etf.db"

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        _init_schema(conn)
        _local.conn = conn
    return _local.conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS configs (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
        );
        CREATE TABLE IF NOT EXISTS analysis_cache (
            id TEXT PRIMARY KEY,
            config_hash TEXT NOT NULL,
            config_snapshot TEXT NOT NULL,
            result TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
        );
        CREATE TABLE IF NOT EXISTS cache_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
        );
        CREATE TABLE IF NOT EXISTS daily_close (
            code TEXT NOT NULL,
            date TEXT NOT NULL,
            close REAL NOT NULL,
            source TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (code, date, source)
        );
        CREATE TABLE IF NOT EXISTS daily_open (
            code TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL NOT NULL,
            source TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (code, date, source)
        );
        CREATE INDEX IF NOT EXISTS idx_close_code ON daily_close(code);
        CREATE INDEX IF NOT EXISTS idx_open_code ON daily_open(code);
    """)


class ConfigDB:
    KEY_MOMENTUM = "momentum"
    KEY_GRID_CONFIG = "grid_config"
    KEY_GRID_SYMBOLS = "grid_symbols"
    KEY_MA_SYMBOLS = "ma_symbols"
    KEY_CHANLUN = "chanlun"

    @classmethod
    def get(cls, key: str) -> Optional[dict]:
        conn = _get_conn()
        row = conn.execute("SELECT value FROM configs WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            return None

    @classmethod
    def set(cls, key: str, value: dict) -> None:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO configs (key, value, updated_at) VALUES (?,?,datetime('now','localtime')) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, json.dumps(value, ensure_ascii=False)),
        )
        conn.commit()

    @classmethod
    def delete(cls, key: str) -> None:
        conn = _get_conn()
        conn.execute("DELETE FROM configs WHERE key=?", (key,))
        conn.commit()

    @classmethod
    def migrate_from_json(cls) -> int:
        mapping = {
            "etf_momentum_config.json": cls.KEY_MOMENTUM,
            "etf_grid_config.json": cls.KEY_GRID_CONFIG,
            "etf_grid_symbols.json": cls.KEY_GRID_SYMBOLS,
            "etf_ma_symbols.json": cls.KEY_MA_SYMBOLS,
            "etf_chanlun_config.json": cls.KEY_CHANLUN,
        }
        migrated = 0
        proj_root = Path(__file__).parent
        for json_name, db_key in mapping.items():
            json_path = proj_root / json_name
            if json_path.exists() and cls.get(db_key) is None:
                try:
                    data = json.loads(json_path.read_text(encoding="utf-8"))
                    cls.set(db_key, data)
                    migrated += 1
                except (json.JSONDecodeError, OSError):
                    pass
        return migrated


class CacheDB:
    @classmethod
    def get(cls, cache_id: str, config_hash: str) -> Optional[Any]:
        conn = _get_conn()
        row = conn.execute(
            "SELECT result FROM analysis_cache WHERE id=? AND config_hash=?",
            (cache_id, config_hash),
        ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["result"])
        except (json.JSONDecodeError, TypeError):
            return None

    @classmethod
    def set(cls, cache_id: str, config_hash: str, config_snapshot: dict, result: Any) -> None:
        conn = _get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO analysis_cache (id, config_hash, config_snapshot, result, created_at) "
            "VALUES (?,?,?,?,datetime('now','localtime'))",
            (cache_id, config_hash, json.dumps(config_snapshot, ensure_ascii=False),
             json.dumps(result, ensure_ascii=False, default=str)),
        )
        conn.commit()


class PriceDB:
    """K 线数据存储 — 每日收盘/开盘价"""

    @classmethod
    def save_close(cls, df: pd.DataFrame, source: str) -> None:
        """df: index=日期, columns=ETF代码, values=收盘价"""
        conn = _get_conn()
        rows = []
        for date_str, row in df.iterrows():
            date = pd.Timestamp(date_str).strftime("%Y-%m-%d")
            for code in row.index:
                v = row[code]
                if pd.notna(v):
                    rows.append((str(code), date, float(v), source))
        if not rows:
            return
        conn.executemany(
            "INSERT OR REPLACE INTO daily_close (code, date, close, source) VALUES (?,?,?,?)",
            rows,
        )
        conn.commit()

    @classmethod
    def save_open(cls, df: pd.DataFrame, source: str) -> None:
        """df: index=日期, columns=ETF代码, values=开盘价"""
        conn = _get_conn()
        rows = []
        for date_str, row in df.iterrows():
            date = pd.Timestamp(date_str).strftime("%Y-%m-%d")
            for code in row.index:
                v = row[code]
                if pd.notna(v):
                    rows.append((str(code), date, float(v), source))
        if not rows:
            return
        conn.executemany(
            "INSERT OR REPLACE INTO daily_open (code, date, open, source) VALUES (?,?,?,?)",
            rows,
        )
        conn.commit()

    @classmethod
    def load_close(cls, codes: list[str], source: str) -> pd.DataFrame:
        """返回 index=日期, columns=codes 的收盘价 DataFrame"""
        conn = _get_conn()
        placeholders = ",".join("?" for _ in codes)
        rows = conn.execute(
            f"SELECT date, code, close FROM daily_close WHERE code IN ({placeholders}) AND source=? ORDER BY date",
            [*codes, source],
        ).fetchall()
        data: dict[str, dict[str, float]] = {}
        for date, code, close in rows:
            if date not in data:
                data[date] = {}
            data[date][code] = close
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(data, orient="index")
        df.index = pd.to_datetime(df.index)
        df.index.name = "日期"
        # fill missing codes with NaN
        for c in codes:
            if c not in df.columns:
                df[c] = None
        return df[sorted(codes, key=lambda x: codes.index(x))]

    @classmethod
    def load_open(cls, codes: list[str], source: str) -> pd.DataFrame:
        """返回 index=日期, columns=codes 的开盘价 DataFrame"""
        conn = _get_conn()
        placeholders = ",".join("?" for _ in codes)
        rows = conn.execute(
            f"SELECT date, code, open FROM daily_open WHERE code IN ({placeholders}) AND source=? ORDER BY date",
            [*codes, source],
        ).fetchall()
        data: dict[str, dict[str, float]] = {}
        for date, code, open_val in rows:
            if date not in data:
                data[date] = {}
            data[date][code] = open_val
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(data, orient="index")
        df.index = pd.to_datetime(df.index)
        df.index.name = "日期"
        for c in codes:
            if c not in df.columns:
                df[c] = None
        return df[sorted(codes, key=lambda x: codes.index(x))]

    @classmethod
    def has_source(cls, codes: list[str], source: str) -> bool:
        """检查指定数据源是否有缓存数据"""
        conn = _get_conn()
        placeholders = ",".join("?" for _ in codes)
        row = conn.execute(
            f"SELECT COUNT(*) FROM daily_close WHERE code IN ({placeholders}) AND source=?",
            [*codes, source],
        ).fetchone()
        return row is not None and row[0] > 0

    @classmethod
    def latest_date(cls, codes: list[str], source: str) -> Optional[str]:
        """缓存中最新的日期"""
        conn = _get_conn()
        placeholders = ",".join("?" for _ in codes)
        row = conn.execute(
            f"SELECT MAX(date) FROM daily_close WHERE code IN ({placeholders}) AND source=?",
            [*codes, source],
        ).fetchone()
        return row[0] if row and row[0] else None

    @classmethod
    def earliest_date(cls, codes: list[str], source: str) -> Optional[str]:
        """缓存中最旧的日期"""
        conn = _get_conn()
        placeholders = ",".join("?" for _ in codes)
        row = conn.execute(
            f"SELECT MIN(date) FROM daily_close WHERE code IN ({placeholders}) AND source=?",
            [*codes, source],
        ).fetchone()
        return row[0] if row and row[0] else None


def _set_last_fetch(source: str) -> None:
    """记录数据拉取时间戳"""
    try:
        conn = _get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO cache_meta (key, value, updated_at) VALUES (?,?,datetime('now','localtime'))",
            (f"last_fetch_{source}", datetime.now().isoformat()),
        )
        conn.commit()
    except Exception:
        pass


def get_last_fetch_time(source: str) -> Optional[pd.Timestamp]:
    """获取指定数据源最近拉取时间"""
    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT updated_at FROM cache_meta WHERE key=?", (f"last_fetch_{source}"),
        ).fetchone()
        if row and row["updated_at"]:
            return pd.Timestamp(row["updated_at"])
    except Exception:
        pass
    return None


def get_config_path() -> Path:
    """返回 data 目录路径，供其他模块计算缓存文件路径"""
    return DATA_DIR
