"""价纳创黄C3 双动量轮动 - JoinQuant 回测脚本

策略：
  1. 四只ETF：价值(512040) / 纳指(513100) / 创业板(159952) / 黄金(159937)
  2. 价格 > MA55 才候选
  3. 候选标的中选 ROC20 最高 → 全仓
  4. 暴跌过滤：日跌幅 > 2.6σ(51日滚动) 剔除候选
  5. 无候选 → 空仓
  6. T日收盘信号，T日收盘执行

参数：MA=55, ROC=20, CRASH_SIGMA=2.6, CRASH_WIN=51
"""

import pandas as pd


def initialize(context):
    g.etfs = {
        '价值': '512040.XSHG',
        '纳指': '513100.XSHG',
        '创业板': '159952.XSHE',
        '黄金': '159937.XSHE',
    }
    g.MA = 55
    g.ROC = 20
    g.CRASH_SIGMA = 2.6
    g.CRASH_WIN = 51

    set_option('use_real_price', True)
    set_order_cost(OrderCost(open_tax=0, close_tax=0,
                             open_commission=0.0001, close_commission=0.0001,
                             close_today_commission=0, min_commission=0),
                   type='stock')

    run_daily(trade, time='14:55')


def trade(context):
    cd = context.current_dt
    codes = list(g.etfs.values())
    names = list(g.etfs.keys())
    need = max(g.MA, g.CRASH_WIN) + g.ROC + 10

    closes = {}
    for code in codes:
        df = get_price(code, count=need, end_date=cd,
                       frequency='daily', fields=['close'])
        if df is None or len(df) < need:
            log.warn(f'{code} 数据不足')
            return
        closes[code] = df['close']

    close_df = pd.DataFrame(closes)

    ma = close_df.rolling(g.MA).mean()
    roc = close_df.pct_change(g.ROC, fill_method=None)
    daily_ret = close_df.pct_change(fill_method=None)

    latest = close_df.iloc[-1]
    latest_ma = ma.iloc[-1]
    latest_roc = roc.iloc[-1]
    latest_ret = daily_ret.iloc[-1]

    trailing = daily_ret.iloc[-g.CRASH_WIN:]
    std = trailing.std(ddof=1)

    candidates = {}
    for code in codes:
        if latest_ma[code] != latest_ma[code] or latest[code] <= latest_ma[code]:
            continue
        if latest_roc[code] != latest_roc[code]:
            continue
        if latest_ret[code] == latest_ret[code] and std[code] > 0:
            if latest_ret[code] < -g.CRASH_SIGMA * std[code]:
                name = names[codes.index(code)]
                log.info(f'{name}({code}) 暴跌排除: {latest_ret[code]:.2%}')
                continue
        candidates[code] = latest_roc[code]

    total_value = context.portfolio.total_value

    if not candidates:
        for code in codes:
            order_target(code, 0)
        log.info(f'{cd.strftime("%Y-%m-%d")} 无候选 → 空仓')
        return

    best = max(candidates, key=candidates.get)
    name = names[codes.index(best)]
    log.info(f'{cd.strftime("%Y-%m-%d")} → {name}({best})  ROC={candidates[best]:.2%}')

    # 先卖非选中的，再全仓买入选中的
    for code in codes:
        if code != best:
            order_target(code, 0)
    order_target_value(best, total_value)
