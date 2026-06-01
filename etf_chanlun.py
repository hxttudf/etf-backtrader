#!/usr/bin/env python3
"""缠论分析模块 — 基于 czsc 的分型、笔识别与可视化

独立模块，不依赖 etf_data/etf_backtrader 等动量策略组件。
"""

from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

_CACHE_DIR = Path(__file__).parent


def _is_etf(symbol: str) -> bool:
    return symbol.startswith(("5", "1"))


def _market(symbol: str) -> str:
    return "sh" if symbol.startswith(("5", "6")) else "sz"


def _col_map_standard(df: pd.DataFrame) -> pd.DataFrame:
    m = {"日期": "dt", "开盘": "open", "收盘": "close",
         "最高": "high", "最低": "low", "成交量": "vol",
         "成交额": "amount", "date": "dt", "volume": "vol"}
    df = df.rename(columns=m)
    df["dt"] = pd.to_datetime(df["dt"])
    return df


def fetch_ohlc(symbol: str, start_date: str = "2020-01-01",
               end_date: str | None = None, source: str = "sina") -> pd.DataFrame:
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    import akshare as ak
    if source == "sina":
        if _is_etf(symbol):
            df = ak.fund_etf_hist_sina(symbol=f"{_market(symbol)}{symbol}")
        else:
            df = ak.stock_zh_a_daily(symbol=f"{_market(symbol)}{symbol}", adjust="")
        df = _col_map_standard(df)
    else:
        if _is_etf(symbol):
            df = ak.fund_etf_hist_em(symbol=symbol, period="daily",
                                     start_date=start_date, end_date=end_date, adjust="qfq")
        else:
            df = ak.stock_zh_a_hist_em(symbol=symbol, start_date=start_date,
                                       end_date=end_date, adjust="qfq")
        df = _col_map_standard(df)
    df = df[~df["dt"].duplicated(keep="last")]
    df = df[(df["dt"] >= start_date) & (df["dt"] <= end_date)]
    df["symbol"] = symbol
    cols = [c for c in ["dt", "symbol", "open", "close", "high", "low", "vol", "amount"] if c in df.columns]
    df = df[cols].sort_values("dt").reset_index(drop=True)
    df = df.astype({c: float for c in ["open", "close", "high", "low", "vol", "amount"]})
    return df


def check_quality(df: pd.DataFrame) -> dict:
    """数据质量检查"""
    from czsc.utils.kline_quality import check_kline_quality
    return check_kline_quality(df)


def run_chanlun(df: pd.DataFrame, freq: str = "日线", max_bi: int = 50) -> object:
    from czsc import CZSC, Freq, format_standard_kline
    bars = format_standard_kline(df, freq=Freq(freq))
    cz = CZSC(bars, max_bi_num=max_bi)
    return cz


def generate_signals(cz: object) -> pd.DataFrame:
    """生成缠论买卖点信号（一买/二买/三买 + MACD背驰）"""
    from czsc import generate_czsc_signals
    signals_config = [
        {'name': 'czsc.signals.cxt_first_buy_V221126', 'freq': str(cz.freq), 'di': 1},
        {'name': 'czsc.signals.cxt_second_bs_V230320', 'freq': str(cz.freq), 'di': 1},
        {'name': 'czsc.signals.cxt_third_bs_V230318', 'freq': str(cz.freq), 'di': 1},
        {'name': 'czsc.signals.tas_macd_bc_V221201', 'freq': str(cz.freq), 'di': 1},
    ]
    result = generate_czsc_signals(cz.bars_raw, signals_config, df=True)
    return result


def _parse_signal_value(val: str) -> tuple:
    """解析信号值，如 '一买_任意_任意_1' -> ('一买', 1)"""
    if not isinstance(val, str) or val == "":
        return ("", 0)
    parts = val.split("_")
    if len(parts) < 4:
        return ("", 0)
    sig_type = parts[0]
    strength = int(parts[-1]) if parts[-1].isdigit() else 0
    return (sig_type, strength)


def extract_buy_sell_points(signal_df: pd.DataFrame) -> pd.DataFrame:
    """从信号DataFrame中提取买卖点"""
    rows = []
    for col in signal_df.columns:
        label = None
        if "BUY1" in col or "一买" in col:
            label = "一买"
        elif "BS2" in col or "二买" in col or "二卖" in col:
            label = "二买/卖"
        elif "BS3" in col or "三买" in col or "三卖" in col:
            label = "三买/卖"
        elif "MACD_BC" in col or "背驰" in col:
            label = "背驰"
        else:
            continue
        sig_type_key = col
        for idx, row in signal_df.iterrows():
            val = row[sig_type_key]
            sig_type, strength = _parse_signal_value(val)
            if strength > 0:
                rows.append({
                    "时间": row["dt"],
                    "信号": label,
                    "类型": sig_type,
                    "强度": strength,
                    "价格": row["close"],
                })
    return pd.DataFrame(rows)


def plot_signals(fig: "plotly.graph_objs.Figure", signals: pd.DataFrame) -> None:
    import numpy as np
    import plotly.graph_objects as go
    colors = {"一买": "red", "二买/卖": "orange", "三买/卖": "green", "背驰": "purple"}
    symbols_map = {"一买": "triangle-up", "二买/卖": "diamond", "三买/卖": "triangle-down", "背驰": "star"}
    for _, row in signals.iterrows():
        typ = row["信号"]
        dt_ns = np.datetime64(row["时间"], "ns")
        fig.add_annotation(
            x=dt_ns, y=row["价格"],
            text=typ, showarrow=False,
            font=dict(color=colors.get(typ, "gray"), size=11),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=colors.get(typ, "gray"),
            borderwidth=1.5,
            xref="x", yref="y",
            row=1, col=1,
        )


def get_bi_stats(cz: object) -> pd.DataFrame:
    """笔统计信息（力度/斜率/角度等）"""
    from czsc.utils.bi_info import calculate_bi_info
    return calculate_bi_info(cz.bars_raw)


def get_chanlun_stats(cz: object) -> dict:
    return {
        "笔数量": len(cz.bi_list),
        "分型数量": len(cz.fx_list),
        "已完成笔": len(cz.finished_bis),
        "K线数量": len(cz.bars_raw),
        "频率": str(cz.freq),
    }


def get_bi_data(cz: object) -> pd.DataFrame:
    rows = []
    for bi in cz.bi_list:
        rows.append({
            "起始": bi.fx_a.dt,
            "结束": bi.fx_b.dt,
            "方向": "向上" if bi.direction == 1 else "向下",
            "涨跌幅": f"{(bi.high / bi.low - 1) * 100:.2f}%",
            "力度": f"{bi.power:.2f}" if hasattr(bi, "power") else "",
            "斜率": f"{bi.slope:.2f}" if hasattr(bi, "slope") else "",
            "角度": f"{bi.angle:.1f}°" if hasattr(bi, "angle") else "",
        })
    return pd.DataFrame(rows)


def get_zs_data(cz: object) -> list[dict]:
    from czsc import ZS
    bis = cz.finished_bis
    zs_list = []
    for i in range(len(bis) - 2):
        triple = bis[i:i+3]
        try:
            zs = ZS(list(triple))
        except Exception:
            continue
        valid = zs.is_valid() if callable(zs.is_valid) else zs.is_valid
        if not valid:
            continue
        zs_list.append({
            "起始": zs.sdt,
            "结束": zs.edt,
            "中枢顶(ZG)": zs.zg,
            "中枢底(ZD)": zs.zd,
            "中枢中(ZZ)": zs.zz,
            "最高(GG)": zs.gg,
            "最低(DD)": zs.dd,
            "起始方向": zs.sdir,
            "结束方向": zs.edir,
        })
    return zs_list


def plot_zs(fig: "plotly.graph_objs.Figure", zs_list: list[dict]) -> None:
    import numpy as np
    for zs in zs_list:
        fig.add_shape(
            type="rect",
            y0=zs["中枢底(ZD)"], y1=zs["中枢顶(ZG)"],
            x0=np.datetime64(zs["起始"], "ns"), x1=np.datetime64(zs["结束"], "ns"),
            fillcolor="rgba(100, 149, 237, 0.15)",
            layer="below", line_width=0,
            xref="x", yref="y",
        )


def plot_chanlun(cz: object) -> "plotly.graph_objs.Figure":
    from czsc.utils.plotting.kline import plot_czsc_chart
    kc = plot_czsc_chart(cz)
    fig = kc.fig
    for tr in fig.data:
        if tr.name == "分型":
            tr.line.color = "rgba(255,165,0,0.5)"
            tr.marker.color = "rgba(255,165,0,1)"
        elif tr.name == "笔":
            tr.line.color = "rgba(100,149,237,0.5)"
    return fig


def plot_chanlun_echarts(cz: object, signals: pd.DataFrame | None = None,
                         title: str = "缠中说禅K线分析") -> str:
    """生成 TradingView 风格 HTML 图表

    Returns:
        HTML 字符串，用于 st.components.v1.html()
    """
    from czsc.utils.echarts_plot import kline_pro

    kline_data = []
    for bar in cz.bars_raw:
        kline_data.append({
            "dt": bar.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "open": bar.open,
            "close": bar.close,
            "high": bar.high,
            "low": bar.low,
            "vol": bar.vol,
        })

    fx_data = []
    for fx in cz.fx_list:
        fx_data.append({
            "dt": fx.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "fx": fx.fx,
            "mark": "G" if fx.mark == 1 else "D",
            "value": fx.fx,
        })

    bi_data = []
    for fx in cz.fx_list:
        bi_data.append({
            "dt": fx.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "bi": fx.fx,
        })

    bs_data = []
    if signals is not None and len(signals) > 0:
        for _, row in signals.iterrows():
            bs_data.append({
                "dt": row["时间"].strftime("%Y-%m-%d %H:%M:%S") if hasattr(row["时间"], "strftime") else str(row["时间"]),
                "bs": row["信号"],
                "price": row["价格"],
            })

    chart = kline_pro(
        kline=kline_data,
        fx=fx_data,
        bi=bi_data,
        bs=bs_data,
        title=title,
        t_seq=[5, 13, 21, 34, 55],
    )
    return chart.render_embed()


def load_config() -> dict:
    import json
    path = Path(__file__).parent / "etf_chanlun_config.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def save_config(cfg: dict) -> None:
    import json
    path = Path(__file__).parent / "etf_chanlun_config.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
