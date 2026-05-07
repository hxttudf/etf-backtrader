# Requirements — 有责取消开票改造

## v1 Requirements

### 入口控制 (ENTRY)

- [ ] **ENTRY-01**: 放开 `idp-user-web` 中「已取消」状态订单的开票申请入口限制（当前代码屏蔽了非已完成订单）
  - 条件：订单状态=已取消 AND 取消费用>0 AND 属于有责取消新流程（由外卖侧控制入口，发票侧只需放开校验）

### 开票金额查询 (AMOUNT)

- [ ] **AMOUNT-01**: `invoice-customer-extension` 引入外卖TSP售后接口依赖（`com.sankuai.tsp.order.aftersale.starlight`）
- [ ] **AMOUNT-02**: 实现「查询订单退款信息」调用，从退款单中提取有责取消费用（`ccf` / `cCancelFee`字段）
- [ ] **AMOUNT-03**: 实现 `isDutyCancelAfterSaleRecord` 判断：`reason_ext.bizType == 31`（有责取消类型）
- [ ] **AMOUNT-04**: 实现 `supportDutyCancel` 灰度开关（Lion配置：`waimai_duty_cancel_support_all` 等）
- [ ] **AMOUNT-05**: `idp-user-web` 的 `checkInvoiceAmount` 接口支持已取消订单，返回取消费用金额

### 发票任务处理 (TASK)

- [ ] **TASK-01**: 取消费用发票任务沿用现有分发逻辑
  - 美团配送费部分 → 走「美团配送费」流程（平台开票）
  - 商家商品费/自配费部分 → 走「商家开票」流程（门店能力分发）
- [ ] **TASK-02**: 取消费用发票不支持退款红冲，但支持用户自行发起换开操作

### 安全与兼容 (SAFE)

- [ ] **SAFE-01**: 有责取消场景与原订单开票不并存（外卖侧保证，发票侧需校验）
- [ ] **SAFE-02**: 0元取消费用（`z_a=1` 或 `ccf=0`）时返回「无法开票」错误提示
- [ ] **SAFE-03**: 评估 `invoiceOrderRefundNotify` 红冲通知对取消费用发票的影响（待确认）

## v2 Requirements（本期不做）

- IM入口支持有责取消订单开票跳转
- 发票中心可开订单列表新增有责取消订单
- 智能客服入口边界处理优化

## Out of Scope

- IM入口改造 — 需单独讨论方案
- 发票中心列表 — TSP接口改造量大，待评估
- 非外卖业务线的有责取消开票 — 本期仅覆盖外卖

## Traceability

| REQ-ID | Phase | Status |
|--------|-------|--------|
| ENTRY-01 | Phase 1 | pending |
| AMOUNT-01~05 | Phase 2 | pending |
| TASK-01~02 | Phase 3 | pending |
| SAFE-01~03 | Phase 3 | pending |
