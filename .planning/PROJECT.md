# 有责取消开票改造方案

## What This Is

外卖「有责取消」订单支持用户申请发票的功能改造。当用户因个人原因（如"计划有变不想要"）取消外卖订单，需支付取消费用时，允许用户对该取消费用申请开具发票。

## Core Value

用户能对「已取消」且「有取消费用」的外卖订单，通过现有开票流程申请并获得发票。

## Context

- **业务背景**：外卖有责取消场景产生了实际消费（取消费），用户有合理的开票诉求，且满足税务合规要求
- **涉及端**：美团App、美团微信小程序、外卖App、外卖微信小程序（iOS/安卓/鸿蒙）
- **本次范围**：仅覆盖「订单详情页」入口，不含IM入口和发票中心可开列表改造
- **PRD**：https://km.sankuai.com/collabpage/2746697856
- **接口文档**：https://km.sankuai.com/collabpage/2750960983

## 涉及仓库

| 仓库 | 角色 | 改造内容 |
|------|------|---------|
| `idp-user-web` | HTTP网关，对接前端 | 放开已取消订单开票入口限制；调用外卖TSP查询取消费用作为开票金额 |
| `invoice-customer-extension` | 有状态RPC服务，处理开票业务逻辑 | 引入外卖TSP售后接口；实现有责取消判断与金额提取逻辑 |

## Requirements

### Validated（现有能力）

- ✓ 正常已完成订单开票全流程（申请→任务分发→美团配送费开票/商家开票）
- ✓ `@ApiConfig` AOP鉴权体系（forceLogin等）
- ✓ 通过外卖TSP Pigeon RPC查询订单信息（`WmOrderResult`）
- ✓ 开票金额查询接口 `checkInvoiceAmount`（当前仅支持已完成订单）
- ✓ 发票任务创建与分发（美团配送费/商家开票两条路径）
- ✓ `invoice-customer-extension` 已集成外卖TSP（`WaimaiOrderClient`）

### Active（本次新增）

- [ ] **ENTRY-01**：`idp-user-web` 放开「已取消」状态订单的开票申请入口限制
- [ ] **AMOUNT-01**：通过外卖TSP「查询订单退款信息」接口获取有责取消费用作为开票金额
- [ ] **AMOUNT-02**：`isDutyCancelAfterSaleRecord` 判断售后单是否为有责取消类型（`reason_ext.bizType=31`）
- [ ] **AMOUNT-03**：`supportDutyCancel` 灰度开关控制（Lion配置）
- [ ] **TASK-01**：取消费用发票任务按现有逻辑分发（美团配送→平台开票；商家商品/自配→商家开票）
- [ ] **COMPAT-01**：确保有责取消场景不与原订单开票逻辑并存（外卖侧保证）
- [ ] **SAFE-01**：时序问题保护——若用户先申请发票后TSP调红冲，需评估影响（待确认）

### Out of Scope

- IM入口改造 — 本期不覆盖，需单独讨论
- 发票中心可开订单列表改造 — 本期不覆盖，待评估
- 0元售后单（`z_a=1`）的开票 — 金额为0无需开票

## Key Decisions

| 决策 | 理由 | 结论 |
|------|------|------|
| 开票金额来源 | 通过外卖「查询订单退款信息」接口查退款单，取`ccf`(cCancelFee)字段 | 已确定 |
| 入口控制方 | 订单详情页入口由外卖交易侧控制（已取消+取消费用不为0+属于新流程） | 外卖侧自行控制，发票侧放开限制 |
| 灰度方案 | 使用Lion开关+`WaiMaiAfterSaleUtil#supportDutyCancel`方法做灰度 | 已确定 |
| 发票任务处理 | 沿用现有分发逻辑，无需新增任务类型 | 已确定 |
| 红冲风险 | 外卖侧调`invoiceOrderRefundNotify`时平台判断是否存在发票，无发票不处理 | 待确认时序问题 |

---
*Last updated: 2026-04-13 after initialization*
