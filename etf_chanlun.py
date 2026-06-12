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


def run_chanlun(df: pd.DataFrame, freq: str = "日线", max_bi: int | None = None) -> object:
    from czsc import CZSC, Freq, format_standard_kline
    if max_bi is None:
        max_bi = max(50, len(df) // 8)
    bars = format_standard_kline(df, freq=Freq(freq))
    cz = CZSC(bars, max_bi_num=max_bi)
    return cz


def generate_signals(cz: object, init_n: int | None = None) -> pd.DataFrame:
    """生成缠论买卖点信号（一买/二买/三买 + MACD背驰）

    Args:
        cz: CZSC 对象
        init_n: 信号生成初始K线数，None则自适应取 max(30, len(bars_raw)//4)
    """
    from czsc import generate_czsc_signals
    signals_config = [
        {'name': 'czsc.signals.cxt_first_buy_V221126', 'freq': str(cz.freq), 'di': 1},
        {'name': 'czsc.signals.cxt_second_bs_V230320', 'freq': str(cz.freq), 'di': 1},
        {'name': 'czsc.signals.cxt_third_bs_V230318', 'freq': str(cz.freq), 'di': 1},
        {'name': 'czsc.signals.tas_macd_bc_V221201', 'freq': str(cz.freq), 'di': 1},
    ]
    if init_n is None:
        n_raw = len(cz.bars_raw)
        init_n = max(30, min(200, n_raw // 4))
    init_n = min(init_n, max(0, len(cz.bars_raw) - 1))
    result = generate_czsc_signals(cz.bars_raw, signals_config, df=True, init_n=init_n)
    return result


def _parse_signal_value(val: str) -> tuple:
    """解析信号值，如 '一买_17笔_任意_1' -> ('一买', '17笔')"""
    if not isinstance(val, str) or val == "":
        return ("", "")
    parts = val.split("_")
    if len(parts) < 4:
        return ("", "")
    sig_type = parts[0]
    bi_count = parts[1]
    return (sig_type, bi_count)


def extract_buy_sell_points(signal_df: pd.DataFrame) -> pd.DataFrame:
    """从信号DataFrame中提取买卖点

    买点用 low，卖点用 high，背驰用 close
    结果列：时间、信号（一买/二买/二卖/三买/三卖/背驰）、价格、笔数
    """
    buy_types = {"一买", "二买", "三买"}
    sell_types = {"二卖", "三卖"}

    def _price(row: pd.Series, sig_type: str) -> float:
        if sig_type in buy_types:
            return row["low"]
        if sig_type in sell_types:
            return row["high"]
        return row["close"]

    rows = []
    for col in signal_df.columns:
        sig_type_key = col
        prev_type = "其他"
        for idx, row in signal_df.iterrows():
            val = row[sig_type_key]
            sig_type, bi_count = _parse_signal_value(val)
            if sig_type != prev_type and sig_type != "其他":
                rows.append({
                    "时间": row["dt"],
                    "信号": sig_type,
                    "价格": _price(row, sig_type),
                    "笔数": bi_count,
                    "close": row["close"],
                })
            prev_type = sig_type if sig_type else "其他"
    return pd.DataFrame(rows)


def plot_signals(fig: "plotly.graph_objs.Figure", signals: pd.DataFrame) -> None:
    import plotly.graph_objects as go
    config = {
        "一买":  {"color": "red",    "symbol": "triangle-up",   "size": 12},
        "二买":  {"color": "#FF6B6B","symbol": "diamond",       "size": 10},
        "二卖":  {"color": "#4ECDC4","symbol": "diamond-tall",  "size": 10},
        "三买":  {"color": "#FF6B6B","symbol": "triangle-down", "size": 10},
        "三卖":  {"color": "#4ECDC4","symbol": "triangle-down", "size": 10},
        "背驰":  {"color": "purple", "symbol": "star",          "size": 12},
    }
    for typ, grp in signals.groupby("信号"):
        cfg = config.get(typ, {})
        if not cfg:
            continue
        fig.add_trace(go.Scatter(
            x=grp["时间"].apply(_fmt_x), y=grp["价格"],
            mode="markers+text",
            xaxis="x",
            marker=dict(symbol=cfg["symbol"], size=cfg["size"],
                        color=cfg["color"], line=dict(width=1, color="white")),
            text=typ, textfont=dict(color=cfg["color"], size=10),
            textposition="top center", name=typ, showlegend=False,
        ))


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


def resample_ohlc(df: pd.DataFrame, target_freq: str) -> pd.DataFrame:
    freq_map = {"周线": "W", "月线": "ME"}
    rule = freq_map[target_freq]
    symbol = df["symbol"].iloc[0] if "symbol" in df.columns else None
    cols = {"open": "first", "high": "max", "low": "min", "close": "last", "vol": "sum"}
    if "amount" in df.columns:
        cols["amount"] = "sum"
    resampled = df.resample(rule, on="dt").agg(cols).dropna().reset_index()
    if symbol is not None:
        resampled["symbol"] = symbol
    return resampled


def run_multi_freq_analysis(df: pd.DataFrame, max_bi: int | None = None) -> dict:
    df_daily = df.copy()
    df_weekly = resample_ohlc(df_daily, "周线")
    df_monthly = resample_ohlc(df_daily, "月线")

    freq_data = {"日线": df_daily, "周线": df_weekly, "月线": df_monthly}
    result = {}

    for freq_label, freq_df in freq_data.items():
        cz = run_chanlun(freq_df, freq="日线", max_bi=max_bi)
        signal_df = generate_signals(cz)
        bs = extract_buy_sell_points(signal_df)
        zs = get_zs_data(cz)
        stats = get_chanlun_stats(cz)
        fig = plot_chanlun(cz)
        if len(bs) > 0:
            plot_signals(fig, bs)
        if zs:
            plot_zs(fig, zs)
        result[freq_label] = {"cz": cz, "bs": bs, "zs": zs, "fig": fig, "stats": stats}

    rows = []
    for freq_label, data in result.items():
        for _, row in data["bs"].iterrows():
            rows.append({"时间": row["时间"], "信号": row["信号"], "频率": freq_label})

    if rows:
        resonance_df = pd.DataFrame(rows)
        sig_counts = resonance_df.groupby("信号").agg(
            共振次数=("频率", "nunique"),
            参与频率=("频率", lambda x: "/".join(sorted(set(x)))),
        ).reset_index()
        sig_counts = sig_counts[sig_counts["共振次数"] >= 2]
        resonant_types = set(sig_counts["信号"])
        res_detail = resonance_df[resonance_df["信号"].isin(resonant_types)].copy()
        res_grouped = res_detail.groupby("信号").agg(
            共振次数=("频率", "nunique"),
            参与频率=("频率", lambda x: "/".join(sorted(set(x)))),
            时间=("时间", "max"),
        ).reset_index()
        result["resonance"] = res_grouped.sort_values("共振次数", ascending=False).reset_index(drop=True)
    else:
        result["resonance"] = pd.DataFrame()

    return result


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
            "起始": triple[0].fx_a.dt,
            "结束": triple[2].fx_b.dt,
            "中枢顶(ZG)": zs.zg,
            "中枢底(ZD)": zs.zd,
            "中枢中(ZZ)": zs.zz,
            "最高(GG)": zs.gg,
            "最低(DD)": zs.dd,
        })
    return zs_list


def plot_zs(fig: "plotly.graph_objs.Figure", zs_list: list[dict]) -> None:
    import numpy as np
    for zs in zs_list:
        x0 = _fmt_x(zs["起始"])
        x1 = _fmt_x(zs["结束"])
        y_zd, y_zg = zs["中枢底(ZD)"], zs["中枢顶(ZG)"]
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y_zd, y1=y_zg,
                      fillcolor="rgba(100, 149, 237, 0.15)", layer="below",
                      line=dict(width=1, color="rgba(100, 149, 237, 0.4)", dash="dot"),
                      row=1, col=1)
        fig.add_shape(type="line", x0=x0, x1=x1, y0=y_zg, y1=y_zg,
                      line=dict(width=1.5, color="rgba(100, 149, 237, 0.7)", dash="dash"),
                      row=1, col=1)
        fig.add_shape(type="line", x0=x0, x1=x1, y0=y_zd, y1=y_zd,
                      line=dict(width=1.5, color="rgba(100, 149, 237, 0.7)", dash="dash"),
                      row=1, col=1)


def _fmt_x(x):
    import numpy as np
    import pandas as pd
    if isinstance(x, np.datetime64):
        return pd.Timestamp(x).strftime("%Y-%m-%d")
    if hasattr(x, "strftime"):
        return x.strftime("%Y-%m-%d")
    return str(x)[:10]


def plot_chanlun(cz: object) -> "plotly.graph_objs.Figure":
    import pandas as pd
    from czsc.utils.plotting.kline import plot_czsc_chart
    kc = plot_czsc_chart(cz)
    fig = kc.fig

    # 修复 hover 时间格式（去掉 .000000000）
    for tr in fig.data:
        if hasattr(tr, "x") and tr.x is not None and len(tr.x) > 0:
            tr.x = [_fmt_x(v) for v in tr.x]
    for shp in fig.layout.shapes:
        if hasattr(shp, "x0") and shp.x0 is not None:
            shp.x0 = _fmt_x(shp.x0)
        if hasattr(shp, "x1") and shp.x1 is not None:
            shp.x1 = _fmt_x(shp.x1)

    for tr in fig.data:
        if tr.name == "分型":
            tr.line.color = "rgba(255,165,0,0.5)"
            tr.marker.color = "rgba(255,165,0,1)"
        elif tr.name == "笔":
            tr.line.color = "rgba(100,149,237,0.5)"
            tr.text = None
        elif tr.name == "DIFF":
            tr.line.color = "#00E5FF"
            tr.line.width = 1.5
        elif tr.name == "DEA":
            tr.line.color = "#FF9100"
            tr.line.width = 1.5

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
                "dt": row["时间"].strftime("%Y-%m-%d") if hasattr(row["时间"], "strftime") else str(row["时间"]),
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


def backtest_signals(bs_points: pd.DataFrame, df: pd.DataFrame) -> dict:
    """根据买卖点信号做简单回测

    规则：
    - 买点（一买/二买/三买）→ 开多
    - 卖点（二卖/三卖）→ 平多
    - 以信号日 close 成交，仅做多
    """
    signals = bs_points.sort_values("时间").reset_index(drop=True)
    trades = []
    capital = 1.0
    in_pos = False
    entry_price = 0.0
    entry_date = None

    for _, sig in signals.iterrows():
        t = sig["信号"]
        dt = sig["时间"]
        px = sig["close"]

        if t in ("一买", "二买", "三买") and not in_pos:
            entry_price = px
            entry_date = dt
            in_pos = True
        elif t in ("二卖", "三卖") and in_pos:
            ret = px / entry_price - 1
            capital *= (1 + ret)
            trades.append({
                "买入时间": entry_date,
                "卖出时间": dt,
                "持有天数": (dt - entry_date).days,
                "买入价": round(entry_price, 3),
                "卖出价": round(px, 3),
                "收益率": round(ret * 100, 2),
            })
            in_pos = False

    if in_pos:
        last = df.iloc[-1]
        ret = last["close"] / entry_price - 1
        capital *= (1 + ret)
        trades.append({
            "买入时间": entry_date,
            "卖出时间": last["dt"],
            "持有天数": (last["dt"] - entry_date).days,
            "买入价": round(entry_price, 3),
            "卖出价": round(last["close"], 3),
            "收益率": round(ret * 100, 2),
        })

    trades_df = pd.DataFrame(trades)
    n = len(trades_df)

    if n > 0:
        wins = (trades_df["收益率"] > 0).sum()
        win_rate = wins / n * 100
        avg_ret = trades_df["收益率"].mean()
        total_ret = (capital - 1) * 100

        cum = capital
        eq = [1.0]
        for _, r in trades_df.iterrows():
            eq.append(eq[-1] * (1 + r["收益率"] / 100))
        peak = pd.Series(eq).cummax()
        dd = (pd.Series(eq) / peak - 1).min() * 100
    else:
        win_rate = avg_ret = total_ret = dd = 0.0

    return {
        "总收益率": round(total_ret, 2),
        "年化收益率": round(total_ret / max(1, ((df.iloc[-1]["dt"] - df.iloc[0]["dt"]).days / 365)), 2),
        "最大回撤": round(dd, 2),
        "交易次数": n,
        "胜率": round(win_rate, 1),
        "平均收益": round(avg_ret, 2),
        "交易明细": trades_df,
    }


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
    try:
        from etf_db import ConfigDB
        ConfigDB.set(ConfigDB.KEY_CHANLUN, cfg)
    except ImportError:
        pass
    path = Path(__file__).parent / "etf_chanlun_config.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))


def _get_czsc_divergence(df: pd.DataFrame) -> pd.DataFrame:
    """使用czsc的tas_macd_bc_V221201信号检测背驰

    逐bar处理，信号位置=背驰确认的K线位置（无未来数据）。
    Returns: 信号DataFrame（时间/信号/价格/close）
    """
    from czsc import CZSC, Freq, format_standard_kline, generate_czsc_signals
    if len(df) < 30:
        return pd.DataFrame()
    bars = format_standard_kline(df, freq=Freq("日线"))
    cz = CZSC(bars, max_bi_num=max(50, len(df) // 8))
    config = [{'name': 'czsc.signals.tas_macd_bc_V221201', 'freq': str(cz.freq), 'di': 1}]
    n_raw = len(cz.bars_raw)
    init_n = max(30, min(200, n_raw // 4))
    init_n = min(init_n, max(0, len(cz.bars_raw) - 1))
    result = generate_czsc_signals(cz.bars_raw, config, df=True, init_n=init_n)
    div_rows = []
    for _, row in result.iterrows():
        for col in result.columns:
            if col == "dt":
                continue
            val = row[col]
            sig_type, _ = _parse_signal_value(val)
            if sig_type == "顶部":
                div_rows.append({"时间": row["dt"], "信号": "顶背驰",
                                 "价格": row["high"], "close": row["close"]})
            elif sig_type == "底部":
                div_rows.append({"时间": row["dt"], "信号": "底背驰",
                                 "价格": row["low"], "close": row["close"]})
    return pd.DataFrame(div_rows)


def ta_signal_analysis(df: pd.DataFrame, resonance_days: int = 0, weekly_filter: bool = False) -> dict:
    """基于 MACD 金叉/死叉、MA5/10 金叉/死叉、背驰的技术信号分析

    Args:
        resonance_days: 信号共振确认天数，0=禁用。金叉与背驰在此天数内同时出现才视为有效
        weekly_filter: 周线金叉/死叉加入共振池参与配对

    Returns:
        signals: 信号DataFrame (时间/信号/价格/close)
        fig: Plotly图表
        backtest: 回测结果
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    df = df.sort_values("dt").reset_index(drop=True)
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()

    close_s = pd.Series(close)
    ema12 = close_s.ewm(span=12, adjust=False).mean().to_numpy()
    ema26 = close_s.ewm(span=26, adjust=False).mean().to_numpy()
    diff = ema12 - ema26
    dea = pd.Series(diff).ewm(span=9, adjust=False).mean().to_numpy()
    hist = diff - dea

    ma5 = close_s.rolling(5).mean().to_numpy()
    ma10 = close_s.rolling(10).mean().to_numpy()

    signals = []

    def _ok(*vals):
        return all(not np.isnan(v) for v in vals)

    for i in range(1, len(df)):
        if not _ok(diff[i], dea[i], diff[i-1], dea[i-1], ma5[i], ma10[i], ma5[i-1], ma10[i-1]):
            continue

        if diff[i-1] <= dea[i-1] and diff[i] > dea[i]:
            signals.append({"时间": df["dt"].iloc[i], "信号": "MACD金叉", "价格": close[i], "close": close[i]})
        elif diff[i-1] >= dea[i-1] and diff[i] < dea[i]:
            signals.append({"时间": df["dt"].iloc[i], "信号": "MACD死叉", "价格": close[i], "close": close[i]})

        if ma5[i-1] <= ma10[i-1] and ma5[i] > ma10[i]:
            signals.append({"时间": df["dt"].iloc[i], "信号": "MA金叉", "价格": close[i], "close": close[i]})
        elif ma5[i-1] >= ma10[i-1] and ma5[i] < ma10[i]:
            signals.append({"时间": df["dt"].iloc[i], "信号": "MA死叉", "价格": close[i], "close": close[i]})

    div_df = _get_czsc_divergence(df)
    if len(div_df) > 0:
        for _, row in div_df.iterrows():
            signals.append(row.to_dict())

    sig_df = pd.DataFrame(signals)
    if len(sig_df) > 0:
        sig_df = sig_df.sort_values("时间").reset_index(drop=True)

        # 周线金叉/死叉信号：检测周线MA5/MA10交叉
        if weekly_filter:
            weekly = df.set_index("dt").resample("W-FRI").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "vol": "sum",
            }).dropna().reset_index()
            weekly["ma5"] = weekly["close"].rolling(5).mean()
            weekly["ma10"] = weekly["close"].rolling(10).mean()
            weekly = weekly.dropna(subset=["ma5", "ma10"]).reset_index(drop=True)
            wk_sigs = []
            for i in range(1, len(weekly)):
                prev_ok = not np.isnan(weekly["ma5"].iloc[i-1]) and not np.isnan(weekly["ma10"].iloc[i-1])
                cur_ok = not np.isnan(weekly["ma5"].iloc[i]) and not np.isnan(weekly["ma10"].iloc[i])
                if prev_ok and cur_ok:
                    prev_bull = weekly["ma5"].iloc[i-1] > weekly["ma10"].iloc[i-1]
                    cur_bull = weekly["ma5"].iloc[i] > weekly["ma10"].iloc[i]
                    if not prev_bull and cur_bull:
                        wk_sigs.append({"时间": weekly["dt"].iloc[i], "信号": "周线金叉",
                            "价格": weekly["close"].iloc[i], "close": weekly["close"].iloc[i]})
                    elif prev_bull and not cur_bull:
                        wk_sigs.append({"时间": weekly["dt"].iloc[i], "信号": "周线死叉",
                            "价格": weekly["close"].iloc[i], "close": weekly["close"].iloc[i]})
            if wk_sigs:
                sig_df = pd.concat([sig_df, pd.DataFrame(wk_sigs)], ignore_index=True).sort_values("时间").reset_index(drop=True)

        # 信号共振：买入需金叉+底背驰配对；卖出任何死叉/顶背驰独立触发
        if resonance_days > 0 and len(sig_df) > 0:
            jc_dates = set(sig_df[sig_df["信号"].isin(["MACD金叉", "MA金叉", "周线金叉"])]["时间"])
            dc_dates = set(sig_df[sig_df["信号"] == "底背驰"]["时间"])

            def _has_nearby(target, candidates, days):
                if not candidates:
                    return False
                return any(abs((d - target).days) <= days for d in candidates)

            bt_signals = []
            for _, row in sig_df.iterrows():
                t, d = row["信号"], row["时间"]
                if t in ("MACD金叉", "MA金叉", "周线金叉") and _has_nearby(d, dc_dates, resonance_days):
                    bt_signals.append({"时间": d, "信号": "买入", "价格": row["价格"], "close": row["close"]})
                elif t == "底背驰" and _has_nearby(d, jc_dates, resonance_days):
                    bt_signals.append({"时间": d, "信号": "买入", "价格": row["价格"], "close": row["close"]})
                elif t in ("MACD死叉", "MA死叉", "周线死叉", "顶背驰"):
                    bt_signals.append({"时间": d, "信号": "卖出", "价格": row["价格"], "close": row["close"]})
            bt_sig_df = pd.DataFrame(bt_signals).drop_duplicates().sort_values("时间").reset_index(drop=True) if bt_signals else pd.DataFrame(columns=["时间", "信号", "价格", "close"])
        else:
            bt_sig_df = sig_df

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(
        x=df["dt"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="K线", showlegend=False,
        increasing_line_color='#E53935', decreasing_line_color='#43A047',
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["dt"], y=ma5, mode="lines",
        name="MA5", line=dict(color="#FF8A80", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["dt"], y=ma10, mode="lines",
        name="MA10", line=dict(color="#82B1FF", width=1.5)), row=1, col=1)

    fig.add_trace(go.Scatter(x=df["dt"], y=diff, mode="lines",
        name="DIFF", line=dict(color="#00E5FF", width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["dt"], y=dea, mode="lines",
        name="DEA", line=dict(color="#FF9100", width=1.5)), row=2, col=1)
    hist_c = ["#E53935" if v < 0 else "#43A047" for v in hist]
    fig.add_trace(go.Bar(x=df["dt"], y=hist, name="MACD",
        marker_color=hist_c, showlegend=False), row=2, col=1)

    if len(sig_df) > 0:
        mc = {
            "MACD金叉": ("red", "triangle-up", 12),
            "MACD死叉": ("green", "triangle-down", 12),
            "MA金叉": ("#FF6B6B", "diamond", 10),
            "MA死叉": ("#4ECDC4", "diamond", 10),
            "顶背驰": ("purple", "star", 14),
            "底背驰": ("orange", "star", 14),
            "周线金叉": ("#00E676", "diamond-wide", 14),
            "周线死叉": ("#FF1744", "diamond-wide", 14),
        }
        for typ, grp in sig_df.groupby("信号"):
            c = mc.get(typ)
            if c is None:
                continue
            fig.add_trace(go.Scatter(
                x=grp["时间"], y=grp["价格"],
                mode="markers+text",
                marker=dict(symbol=c[1], size=c[2], color=c[0], line=dict(width=1, color="white")),
                text=typ, textfont=dict(color=c[0], size=9), textposition="top center",
                name=typ, showlegend=True,
            ), row=1, col=1)

    fig.update_layout(height=650, template="plotly_white", hovermode="x unified",
                      xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    fig.update_xaxes(title_text="日期", row=2, col=1)
    fig.update_yaxes(title_text="价格", row=1, col=1)

    bt = _ta_backtest(bt_sig_df, df)

    return {"signals": sig_df, "fig": fig, "backtest": bt}


def _ta_backtest(signals: pd.DataFrame, df: pd.DataFrame) -> dict:
    """信号回测（次交易日开盘执行，无未来数据）"""
    buy_sigs = {"MACD金叉", "MA金叉", "底背驰", "买入"}
    sell_sigs = {"MACD死叉", "MA死叉", "顶背驰", "卖出"}
    ds = df.sort_values("dt").reset_index(drop=True)
    dt_to_idx = {dt: i for i, dt in enumerate(ds["dt"])}
    trades = []
    capital = 1.0
    in_pos = False
    entry_price = 0.0
    entry_date = None

    for _, sig in signals.iterrows():
        t, dt = sig["信号"], sig["时间"]
        idx = dt_to_idx.get(dt)
        if idx is None or idx + 1 >= len(ds):
            continue
        nxt = idx + 1
        exec_price = ds.iloc[nxt]["open"]
        exec_date = ds.iloc[nxt]["dt"]

        if t in buy_sigs and not in_pos:
            entry_price, entry_date, in_pos = exec_price, exec_date, True
        elif t in sell_sigs and in_pos:
            ret = exec_price / entry_price - 1
            capital *= (1 + ret)
            trades.append({"买入时间": entry_date, "卖出时间": exec_date,
                           "持有天数": (exec_date - entry_date).days,
                           "买入价": round(entry_price, 3), "卖出价": round(exec_price, 3),
                           "收益率": round(ret * 100, 2), "信号类型": t})
            in_pos = False

    if in_pos:
        last = ds.iloc[-1]
        ret = last["close"] / entry_price - 1
        capital *= (1 + ret)
        trades.append({"买入时间": entry_date, "卖出时间": last["dt"],
                       "持有天数": (last["dt"] - entry_date).days,
                       "买入价": round(entry_price, 3), "卖出价": round(last["close"], 3),
                       "收益率": round(ret * 100, 2), "信号类型": "持仓中"})

    trades_df = pd.DataFrame(trades)
    n = len(trades_df)
    if n > 0:
        wins = (trades_df["收益率"] > 0).sum()
        win_rate = wins / n * 100
        total_ret = (capital - 1) * 100
        avg_ret = trades_df["收益率"].mean()
        eq = [1.0]
        for _, r in trades_df.iterrows():
            eq.append(eq[-1] * (1 + r["收益率"] / 100))
        peak = pd.Series(eq).cummax()
        dd = (pd.Series(eq) / peak - 1).min() * 100
    else:
        win_rate = total_ret = avg_ret = dd = 0.0

    return {
        "总收益率": round(total_ret, 2),
        "年化收益率": round(total_ret / max(1, ((ds.iloc[-1]["dt"] - ds.iloc[0]["dt"]).days / 365)), 2),
        "最大回撤": round(dd, 2),
        "交易次数": n,
        "胜率": round(win_rate, 1),
        "平均收益": round(avg_ret, 2),
        "交易明细": trades_df,
    }
