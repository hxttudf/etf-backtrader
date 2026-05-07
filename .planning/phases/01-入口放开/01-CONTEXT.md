# Phase 1: 入口放开 - Context

**Gathered:** 2026-04-13
**Status:** Ready for planning

<domain>
## Phase Boundary

放开 `idp-user-web` 中外卖订单状态校验，使有责取消订单（status=9）能通过入口校验进入开票流程。
本 Phase 仅改 `idp-user-web`，不涉及 `invoice-customer-extension`。

</domain>

<decisions>
## Implementation Decisions

### 校验放开方式
- **D-01:** 扩展现有 Lion 配置 `wm_order_can_invoice_status`，灰度时将取消状态码加入该配置（不改代码逻辑）
  - 当前默认值：`[8]`（已完成）
  - 灰度放开后配置值：`[8, 9]`（已完成 + 有责取消）
  - 修改位置：`InvoiceService.java` 的 `checkWmOrderStatus` 方法（Line 1262）无需改动，配置驱动
  - 两处调用点（Line 536、Line 1213）均通过同一方法校验，扩展配置即可覆盖两处

### 外卖订单状态码
- **D-02:** 有责取消订单 status = **9**（用户取消，含有责取消场景）
  - 已完成 = 8，有责取消 = 9

### 灰度控制
- **D-03:** 通过 Lion 配置 `wm_order_can_invoice_status` 控制，默认 `[8]`，灰度时改为 `[8, 9]`
  - 无需新建 Lion 开关，利用现有配置扩展
  - Phase 2 的 `WaiMaiAfterSaleUtil#supportDutyCancel` 是独立的金额查询灰度，不是本 Phase 的范围

### 依赖版本
- **D-04:** Phase 1 **不升级** `invoice-customer-extension-api`（当前 0.6.28）
  - Phase 1 不调用新接口，依赖版本不影响本次改动
  - 升级留给 Phase 2（引入售后接口时一并处理）

### Claude's Discretion
- 错误信息文案可沿用现有 `"订单状态非已完成，暂时无法进行开票"`，Phase 1 不改文案（入口由外卖交易侧控制，正常用户不会触发此错误）
- 单元测试覆盖 `checkWmOrderStatus` 的 status=9 场景

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### 核心改动文件
- `~/IdeaProjects/idp-user-web/src/main/java/com/sankuai/invoice/user/service/InvoiceService.java` — 含 `checkWmOrderStatus`（Line 1262），两处调用点（Line 536, 1213）

### 项目规划文件
- `.planning/ROADMAP.md` — Phase 1 改造点说明
- `.planning/REQUIREMENTS.md` — ENTRY-01 验收标准
</canonical_refs>
