# Roadmap — 有责取消开票改造

**3 phases** | **10 requirements mapped** | All v1 requirements covered ✓

| # | Phase | Goal | Requirements | 预估工作量 |
|---|-------|------|--------------|-----------|
| 1 | 入口放开 | idp-user-web 支持已取消订单进入开票流程 | ENTRY-01 | 0.5天 |
| 2 | 金额查询 | 实现有责取消费用查询与灰度控制 | AMOUNT-01~05 | 2天 |
| 3 | 任务分发与兜底 | 发票任务正确分发，边界安全处理 | TASK-01~02, SAFE-01~03 | 1天 |

---

## Phase 1: 入口放开

**Goal**: `idp-user-web` 放开已取消状态订单的开票入口校验，使有责取消订单能进入申请页面。

**Requirements**: ENTRY-01

**改造点**:

### idp-user-web

1. **`InvoiceService.java` / `CollectionService.java`**
   - 找到订单状态校验逻辑（当前限制只有「已完成」状态才能开票）
   - 新增条件：订单状态=已取消 AND 取消费用>0，允许进入开票流程
   - 建议用 Lion 开关控制，默认关闭，灰度放量

2. **`checkInvoiceAmount` 相关接口**
   - 当订单状态为已取消时，不直接返回错误，而是继续走取消费用查询逻辑（Phase 2 实现）

**Success Criteria**:
1. 已取消+有取消费用的订单，调用开票相关接口不再返回「订单状态不支持开票」错误
2. 已完成订单的现有开票流程不受影响
3. 已取消+无取消费用的订单仍返回错误

---

## Phase 2: 金额查询

**Goal**: 通过外卖TSP售后接口查询有责取消费用，作为开票金额返回给前端。

**Requirements**: AMOUNT-01~05

**改造点**:

### invoice-customer-extension（主要改造仓库）

1. **引入依赖** (`invoice-customer-extension-service/pom.xml`)
   ```xml
   <dependency>
       <groupId>com.sankuai.tsp.order.aftersale.starlight</groupId>
       <artifactId>waimai-aftersale-starlight-api</artifactId>
       <version><!-- 最新版本 --></version>
   </dependency>
   ```

2. **新增 Integration 类** `integration/waimai/WaimaiAfterSaleClient.java`
   - 封装调用外卖TSP「查询订单退款信息」接口（Pigeon RPC）
   - 返回 `List<WmApplyRefundRecordResult>`
   - 加 Rhino 熔断保护

3. **新增工具方法**（参考接口文档中的 `WaiMaiAfterSaleUtil`）
   ```java
   // 判断是否有责取消售后单（bizType == 31）
   public static boolean isDutyCancelAfterSaleRecord(WmApplyRefundRecordResult record)
   
   // 从退款单列表中找到有责取消单
   public static WmApplyRefundRecordResult getDutyCancelRecord(List<WmApplyRefundRecordResult> records)
   
   // 提取开票金额：取 common_ext.ccf (cCancelFee)
   public static long extractInvoiceAmount(WmApplyRefundRecordResult record)
   ```

4. **灰度控制** `util/DutyCancelGrayUtil.java`
   - 封装 `supportDutyCancel(DutyCancelGrayParam)` 方法
   - Lion 开关：
     - `waimai_duty_cancel_support_all`（全量开关）
     - `waimai_duty_cancel_poi_id_whitelist`（门店白名单）
     - `waimai_duty_cancel_degrade`（降级开关）

5. **改造开票金额查询服务**（`InvoiceApplyServiceImpl` 或对应的金额查询方法）
   - 当订单状态=已取消时，调用 `WaimaiAfterSaleClient` 查退款单
   - 命中有责取消且 `ccf > 0` → 返回 `ccf` 作为开票金额
   - 未命中或 `ccf == 0` → 返回「无法开票」

### idp-user-web

6. **`InvoiceService.java`** 中调用 extension 的金额查询接口时，传入订单状态=已取消的场景标识

**关键字段说明**（来自接口文档）:

| 字段 | 类型 | 含义 |
|------|------|------|
| `common_ext.dc` | int | dutyCancel，1=用户责任取消 |
| `common_ext.ccf` | long | cCancelFee，C端展示给用户的取消费用（**开票金额来源**） |
| `common_ext.ccst` | long | 补偿商家金额 |
| `common_ext.ccrt` | long | 补偿骑手金额 |
| `common_ext.ciut` | long | 保险承担费用 |
| `common_ext.scst` | long | 结算侧给商家补偿金额 |
| `reason_ext.bizType` | int | **31 = 有责取消**（新增枚举值） |

**Success Criteria**:
1. 有责取消订单调用开票金额查询接口，返回 `ccf` 字段值（单位：分）
2. 灰度开关关闭时，有责取消订单走原有逻辑（返回不支持）
3. 灰度开关开启时，正确返回取消费用金额
4. TSP接口超时/异常时，Rhino熔断生效，返回友好错误

---

## Phase 3: 任务分发与兜底

**Goal**: 确保发票任务按现有逻辑正确分发，处理边界场景。

**Requirements**: TASK-01~02, SAFE-01~03

**改造点**:

### invoice-customer-extension

1. **发票任务创建**（`InvoiceApplyServiceImpl`）
   - 有责取消发票任务沿用现有 `serviceType` 分发逻辑：
     - 美团配送费部分 → `serviceType=配送费` → 走开普勒配送费开票
     - 商家商品费/自配费 → `serviceType=商家` → 走门店开票能力
   - **无需新增任务类型**，金额拆分逻辑复用现有 `AmountSplitInfoUtils`

2. **换开支持**
   - 取消费用发票标记不支持退款红冲（`canRefund=false`）
   - 支持用户发起换开（`canReplace=true`）

3. **安全校验**
   - `SAFE-01`：提交开票时校验订单不存在「已完成」状态的发票任务（防并存）
   - `SAFE-02`：`ccf == 0` 时直接返回「取消费用为0，无法开票」
   - `SAFE-03`：评估 `invoiceOrderRefundNotify` 影响
     - 外卖侧调红冲通知时，平台判断是否存在发票
     - 若取消费用发票已开具，需确认是否会被错误冲销（**待与外卖侧确认**）

**Success Criteria**:
1. 有责取消发票任务创建成功，分发路径与正常订单一致
2. 取消费用发票不支持退款红冲，支持换开
3. 0元取消费用场景返回明确错误提示
4. 与外卖侧确认红冲通知的影响范围

---

## 待确认事项

| # | 问题 | 负责方 | 优先级 |
|---|------|--------|--------|
| 1 | 外卖侧 `invoiceOrderRefundNotify` 红冲通知是否会影响取消费用发票 | 外卖订单 | 高 |
| 2 | `idp-user-web` 依赖的 `invoice-customer-extension-api` 版本需从 `0.6.28` 升级到最新（`0.6.41`） | 发票侧 | 高 |
| 3 | 外卖TSP售后接口（`waimai-aftersale-starlight-api`）的具体版本号和调用方式 | 外卖TSP | 中 |
| 4 | 灰度放量策略：先门店白名单→城市灰度→全量 | 发票侧+外卖侧 | 中 |

---

## 改造文件清单

### idp-user-web
| 文件 | 改造类型 | 说明 |
|------|---------|------|
| `service/InvoiceService.java` | 修改 | 放开已取消订单校验，支持取消费用开票金额查询 |
| `service/CollectionService.java` | 修改（可选） | 如有订单状态前置校验需移除 |
| `pom.xml` | 修改 | 升级 `invoice-customer-extension-api` 到 `0.6.41` |

### invoice-customer-extension
| 文件 | 改造类型 | 说明 |
|------|---------|------|
| `integration/waimai/WaimaiAfterSaleClient.java` | 新增 | 封装TSP售后查询接口 |
| `util/DutyCancelUtil.java` | 新增 | 有责取消判断、灰度控制工具类 |
| `service/impl/InvoiceApplyServiceImpl.java` | 修改 | 金额查询逻辑新增有责取消分支 |
| `invoice-customer-extension-service/pom.xml` | 修改 | 引入 `waimai-aftersale-starlight-api` |
| `invoice-customer-extension-api/` | 修改（可选） | 如需对外暴露新接口 |
