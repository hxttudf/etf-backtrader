# ETF双动量轮动策略工具

## 策略规则

1. 4个ETF为一组，价格在**N日均线上方**才具备资格（默认60日）
2. 在具备资格的ETF中，选**近M日涨幅最大**的持有（默认20日）
3. 全部低于均线则**空仓**
4. 交易成本：万一佣金（免五）+ 印花税0.05%（卖出收取）

## 文件说明

```
docs/
├── etf_config.json    # ETF组合配置
├── etf_data.py        # 数据模块（多数据源 + 本地缓存）
├── etf_signal.py      # 每日信号脚本（CLI）
├── etf_backtest.py    # 回测脚本（CLI）
├── etf_app.py         # 可视化运行界面（Streamlit）
├── build_exe.py       # 跨平台打包脚本
├── requirements.txt   # 依赖清单
├── etf_README.md      # 本文件
└── etf_prices_*.csv   # 价格缓存（按数据源自动生成）
```

## 可视化界面 `etf_app.py`（推荐）

浏览器操作，无需手动敲命令：

```bash
cd docs/etf
pip install -r requirements.txt
streamlit run etf_app.py
```

浏览器自动打开 `http://localhost:8501`，侧边栏设置参数，点击按钮即可回测/查信号。

## 每日信号 `etf_signal.py`

```bash
cd docs/etf

# 默认组合，今日信号
python etf_signal.py

# 自定义均线和动量参数
python etf_signal.py --ma 30 --roc 10

# 指定组合
python etf_signal.py --group 组合1

# 切换数据源 (tencent / akshare)
python etf_signal.py --source akshare

# 所有组合
python etf_signal.py --all

# 历史日期
python etf_signal.py --date 2026-03-15
```

## 回测 `etf_backtest.py`

```bash
cd docs/etf

# 从指定日期至今，每日+周五对比
python etf_backtest.py --start 2025-04-30

# 指定区间
python etf_backtest.py --start 2023-01-01 --end 2025-12-31

# 仅每日调仓 或 仅周五调仓
python etf_backtest.py --start 2024-01-01 --mode daily
python etf_backtest.py --start 2024-01-01 --mode friday

# 指定组合
python etf_backtest.py --start 2025-01-01 --group 组合1

# 自定义参数
python etf_backtest.py --start 2025-01-01 --ma 30 --roc 10

# 切换数据源 (tencent / akshare)
python etf_backtest.py --start 2025-01-01 --source akshare

# 输出交互式HTML图表（可hover查看调仓详情，支持多组合切换）
python etf_backtest.py --start 2025-04-30 --html
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ma` | 60 | 均线天数，价格需站在线上方 |
| `--roc` | 20 | 动量天数，选涨幅最大的 |
| `--mode` | both | daily（每日）/ friday（周五）/ both |
| `--start` | — | 回测开始日期 YYYY-MM-DD |
| `--end` | 今天 | 回测结束日期 |
| `--group` | 第一个组合 | 组合名称 |
| `--source` | tencent | 数据源: tencent / akshare |
| `--html` | — | 输出交互式HTML图表 |

## 跨平台打包

```bash
cd docs/etf
python build_exe.py
```

输出 `dist/etf_signal[.exe]` 和 `dist/etf_backtest[.exe]`，可在对应平台直接运行。

可视化界面无需打包：`streamlit run etf_app.py`

## 配置组合

编辑 `etf_config.json`，格式：

```json
{
  "groups": {
    "组合名称": {
      "ETF名称": "代码"
    }
  }
}
```

代码规则：5开头=沪市(sh)，1开头=深市(sz)，自动识别。

## 数据源

| 数据源 | `--source` | 最早数据 | 说明 |
|--------|-----------|---------|------|
| 腾讯财经 | `tencent`（默认） | 2023年1月 | 约800个交易日 |
| AKShare (新浪) | `akshare` | 2019年1月 | 约1800个交易日，需 `pip install akshare` |

每个数据源使用独立缓存文件，互不污染。切换数据源后首次运行会自动拉取全量数据。

---

## 多因子轮动（新增）

在 `etf_app.py` 顶部选择"多因子轮动"模式，即可使用基于 IC 加权评分的多因子 ETF 选股系统。

### 工作流程

```
数据加载 → 计算各因子信号 → 市场中性化 + 标准化
  → IC加权合成综合评分 → Top-N选股 → 定期再平衡 → 输出净值
```

### 因子库（10个因子，4个类别）

| 类别 | 因子 | 说明 |
|------|------|------|
| 动量 | ROC多周期动量 | 20/60/120/252日收益率指数加权 |
| 动量 | 52周新高 | 当前价格距52周高点的距离 |
| 动量 | 风险调整动量 | 收益率 ÷ 波动率 = Sharpe风格动量 |
| 趋势 | 价格/均线 | 价格相对20/60/200日均线的偏离度 |
| 趋势 | 均线斜率 | 60日均线的变化率 |
| 趋势 | ADX | 平均趋向指数，衡量趋势强度 |
| 波动率 | 历史波动率 | 20/60日年化波动率（负值，低波动获高分） |
| 波动率 | Parkinson波动率 | 利用OHLC估算的波动率（负值） |
| 波动率 | 下行风险 | 半方差（仅下跌日，负值） |
| 成交量 | 成交量趋势 | 当日成交量 / 20日均量 |
| 成交量 | 成交量动量 | 成交量的20日变化率 |
| 成交量 | 流动性筛选 | 成交额低于阈值则扣分 |

### 评分方法

- **IC加权**（推荐）：计算每个因子每日的 RankIC，用指数衰减（半衰期60天）求均值作为因子权重。预测能力越强的因子权重越高。
- **等权**：所有因子权重相同。

### 回测结果解读

| Tab | 内容 |
|-----|------|
| 回测结果 | 净值曲线、持仓权重、关键指标（CAGR/Sharpe/Calmar/最大回撤） |
| 因子信号 | 各ETF每日的综合评分（正值=看好，负值=不看好） |
| IC分析 | 各因子的 RankIC 均值、ICIR、命中率，以及IC走势 |
| 分层回测 | 按评分分层（L1-L5），检验评分是否能区分好坏 |

### 使用步骤

1. 运行 `streamlit run etf_app.py`
2. 顶部选择 **多因子轮动**
3. 左侧配置：ETF组合、时间范围、持仓数量、再平衡频率
4. 可在"因子开关"中启用/关闭不需要的因子类别
5. 点击 **🚀 运行多因子回测**
6. 查看4个Tab的分析结果

## 注意事项

- 缓存文件：`etf_prices_tencent.csv` / `etf_prices_akshare.csv`（按数据源独立）
- 不同数据源的收盘价可能略有差异，属正常现象
- 非交易日运行信号脚本会自动取最近交易日
