# 网格交易功能设计

## 概述

在现有 ETF 双动量策略之外，独立新增网格交易模块。支持 A 股 ETF/个股网格策略回测，Web 可视化。

## 文件结构

```
etf_grid.py          ← 新增：网格引擎 + 回测逻辑
etf_grid_data.py     ← 新增：独立数据加载（分钟级 OHLC）
etf_app.py           ← 修改：加 tab 切换「网格交易」
```

不修改 `etf_data.py`、`etf_backtrader.py` 等现有动量策略模块。

## 数据层（etf_grid_data.py）

### 数据源

| 来源 | 函数 | 类型 | 粒度 |
|------|------|------|------|
| 东方财富 | `fund_etf_hist_min_em` | ETF | 1/5/15/30/60 分钟 |
| 东方财富 | `stock_zh_a_hist_min_em` | 个股 | 1/5/15/30/60 分钟 |

### 缓存策略

- 按标的+分钟级别独立缓存 CSV（如 `grid_510050_5min.csv`）
- 每次回测前检查最新数据日期，过期自动拉取
- 不清除动量策略的缓存

### 返回格式

```python
# DataFrame, index=datetime, columns=['open','high','low','close','volume']
# 分钟级：每根 K 线包含完整 OHLC
```

## 网格引擎（etf_grid.py）

### 架构

参考 `etf-quant` 的 `Signal` 模型 + `grider` 的网格计算器。

```
GridEngine
├── config: GridConfig
│   ├── symbol, grid_type, n_levels, price_low, price_high
│   ├── amount_per_grid, max_positions, commission, slippage
│   └── min_interval (分钟级粒度)
├── state: GridState
│   ├── levels: [(price, buy_filled, sell_filled), ...]
│   ├── cash, position, total_value, trades
│   └── base_price
├── calculators
│   ├── ArithmeticGrid (等差，参考 grider)
│   ├── GeometricGrid (等比，参考 grider)
│   └── VolatilityGrid (ATR 动态，参考 grider)
├── run(ohlc_df) → GridResult
│   └── 遍历每根 K 线，检测穿越，触发交易
└── calc_metrics(result) → dict
    └── 收益、夏普、回撤、交易次数、胜率
```

### 网格类型

| 类型 | 算法 | 说明 |
|------|------|------|
| `arithmetic` | 等差 | 每格固定差价，参考 grider `ArithmeticGridCalculator` |
| `geometric` | 等比 | 每格固定百分比，参考 grider `GeometricGridCalculator` |
| `volatility` | ATR 动态 | 格距=ATR×倍数，参考 grider `GridOptimizer` |

### 穿越检测

逐根 K 线，按 `open → high → low → close` 路径检查：

```
对于每根 K 线:
  1. 从当前 base_price 开始
  2. 价格从 open 移动到 high：
     - 每穿过一个卖网格线 → 卖出（如果持仓）
     - 更新 base_price
  3. 价格从 high 移动到 low：
     - 每穿过一个买网格线 → 买入（如果有现金）
     - 更新 base_price
  4. 不处理 close（假设 close 在 low 之后已处理）
```

执行时参考 grider 的 `_execute_buy`/`_execute_sell`——成交后立即更新 `base_price`，下一格自动重算。

**成交价**：网格价格（模拟限价单成交）。

### T+1 规则

A 股 T+1：当日买入的份额不可当日卖出。实现：

- `position` 拆为 `today_bought` 和 `available`
- 当天买入的记入 `today_bought`，次日开盘转入 `available`
- 卖出时只允许卖出 `available` 部分

### 多标的

引擎单次只跑一个标的，但 `etf_app.py` 中可配置并排运行多个标的。

## UI 设计（etf_app.py）

### 页面结构

顶部 tab：`[双动量轮动] │ [网格交易]`

### 参数面板（左侧）

模仿 OctoBot 风格——卡片式参数分组：

```
┌─ 标的 ──────────────┐
│ 代码     ████████   │
│ 分钟粒度 █ 5/15/30  │
│ 日期范围 ██ ~ ██    │
├─ 网格参数 ──────────┤
│ 类型     █ 等差/等比 │
│ 下界 ███  上界 ███   │
│ 格数 ███  每格金额 █ │
│ 最大持仓 ██ 格       │
├─ 费用 ──────────────┤
│ 佣金 ███  滑点 ███   │
└─────────────────────┘
```

### 结果展示

```
┌─ 收益指标 ─────────────────────────────┐
│ 总收益  年化  夏普  最大回撤  交易次数  │
├─ 净值曲线 ─────────────────────────────┤
│  📈 plotly 图（策略净值 vs 持有不动）   │
├─ 网格 + K 线 ──────────────────────────┤
│  水平网格线叠加在 K 线上（plotly）       │
│  买标记 🟢 / 卖标记 🔴 标在触达位置     │
├─ 交易明细 ─────────────────────────────┤
│  时间  方向  价格  数量  盈亏           │
├─ 持仓分布 ─────────────────────────────┤
│  饼图/柱状图各价格区间持仓比例          │
└────────────────────────────────────────┘
```

### 网格可视化

重点参考 OctoBot 的网格线展示方式——在价格走势图上叠加水平网格线，每个网格线旁标注买卖状态（已成交/待成交）。

## 参考项目

| 项目 | 参考内容 |
|------|---------|
| yansongwel/etf-quant | `strategies/grid.py` 的 Signal 模型、`engine/backtest.py` 的 T+1 执行逻辑 |
| jorben/grider | `arithmetic_grid.py` / `geometric_grid.py` 的网格计算器、`trading_logic.py` 的穿越检测 |
| OctoBot | UI 风格参考，`staggered_orders_trading_mode` 的网格移动设计思路 |

## 不做的

- ❌ 实盘下单（仅回测）
- ❌ 多标的混合网格（单标的）
- ❌ 回测框架外接 backtrader
- ❌ 修改现有动量策略任何代码
