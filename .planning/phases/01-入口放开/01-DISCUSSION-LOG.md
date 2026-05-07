# Phase 1: 入口放开 - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-13
**Phase:** 01-入口放开
**Areas discussed:** 校验放开方式, 依赖升级, 取消状态码

---

## 校验放开方式

| Option | Description | Selected |
|--------|-------------|----------|
| 扩展现有 Lion 配置 | 灰度时把取消状态码(9)加入 `wm_order_can_invoice_status`，无需改代码 | ✓ |
| 新建专用 Lion 开关 | 新增 `waimai_duty_cancel_entry_enable`，在 checkWmOrderStatus 内加 if 判断 | |
| 在两个调用点各自加判断 | Line 536 和 Line 1213 分别绕过状态校验 | |

**User's choice:** 扩展现有 Lion 配置（推荐）
**Notes:** 最小侵入，配置驱动，无需改代码逻辑

---

## 依赖升级

| Option | Description | Selected |
|--------|-------------|----------|
| Phase 1 不升级 | 保持 0.6.28，Phase 1 不调新接口 | ✓ |
| Phase 1 一起升级 | 升到 0.6.41，Phase 2 省一步 | |

**User's choice:** Phase 1 不升级（推荐）

---

## 取消状态码

**User's choice:** 9（有责取消订单 status=9）
