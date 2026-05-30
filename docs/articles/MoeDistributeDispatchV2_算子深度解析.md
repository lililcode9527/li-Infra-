# MoeDistributeDispatchV2 算子深度解析：通算融合架构、RDMA 全互联与双缓冲同步机制

## 1. 算子概述：什么是 MoeDistributeDispatchV2

MoeDistributeDispatchV2 是华为昇腾 Ascend NPU 上面向 MoE（Mixture of Experts）架构专家并行推理场景的核心通算融合算子，负责在 EP（Expert Parallelism）域内完成 Token 数据的分布式分发——将每个 Token 按其动态路由选择的目标专家，跨设备发送至对应专家所在的 NPU 卡上。

它与配套的 MoeDistributeCombineV2（或 MoeDistributeCombineAddRmsNorm）算子共同构成了 MoE 专家并行推理的完整通信闭环：Dispatch 负责"分发"，Combine 负责"回收"。

### 核心功能

算子的功能可形式化描述为：对 Token 数据进行可选量化后，执行 EP 域 AllToAllV 通信（如有 TP 域则追加 AllGatherV），完成跨卡分发并输出 Combine 阶段所需的元数据信息。

- 无 TP 通信域时：`expandXOut = AllToAllV(X)`
- 有 TP 通信域时：`expandXOut = AllGatherV(AllToAllV(X))`

量化场景（quantMode=1~4）下，先对输入做量化处理再通信，Combine 阶段再反量化还原。

### 相较 V1 算子的演进

相较于原版 MoeDistributeDispatch，V2 做了两项关键变更：

1. **辅助信息重构**：原 `expandIdx`（shape `BS*K`）替换为 `assistInfoForCombineOut`（shape `A*128`），输出更详细的 Token 分布信息，帮助 CombineV2 算子高效完成全卡同步
2. **通信算法显式配置**：新增 `commAlg` 入参替代 `HCCL_INTRA_PCIE_ENABLE` 和 `HCCL_INTRA_ROCE_ENABLE` 环境变量，用户可直接选择 fullmesh/hierarchy 等通信模式

### 产品支持

| 产品 | 支持 |
|------|------|
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练/推理系列 | √ |
| Atlas A2 训练/推理系列 | √ |

---

## 2. MoE 通信瓶颈与通算融合的必要性

### 2.1 传统 AllToAllV 的效率缺陷

MoE 动态路由机制下，每个 Token 选择 topK 个专家，目标专家呈离散分布，导致：

- **数据分发不均**：不同专家接收的 Token 数差异巨大，只能依赖低效的 AllToAllV 通信
- **元数据同步开销**：获取收发信息需要前置 AllGather 收集路由表，再在 Host 侧完成同步，引入额外通信开销与 Stream 同步延迟

### 2.2 推理场景的双重挑战

推理场景 Token 数据量通常较小，引发：

- **Host Bound 问题**：传统 Host 驱动通信需构造子图并调度，下发时延随 EP 规模线性增长
- **RDMA 同步开销**：RDMA 通信前后同步过程引入额外 RTT 时延

### 2.3 通算融合的核心突破

MoeDistributeDispatchV2/CombineV2 将通信与计算深度融合，实现了两项关键突破：

1. **Host 逻辑下沉至 Device**：将路由计算等 Host 侧逻辑下沉至 NPU Device 侧（AIV/AICPU），彻底消除 Host 与 Device 间的同步开销
2. **通算流水并行**：实现 Combine 操作中部分计算与 AllToAllV 通信的流水并行，有效掩盖计算与通信耗时

---

## 3. AIV+AICPU 融合架构：RDMA 全互联方案

### 3.1 架构设计理念

V2 算子基于昇腾硬件的 AIV（AI Vector Core）+ AICPU（AI CPU）融合架构构建 RDMA 全互联（Fullmesh）方案。核心思想是让 NPU Device 侧自治完成通信调度，不依赖 Host 侧干预。

### 3.2 三阶段处理流程

**预处理阶段（AIV 执行）**

1. 获取每个 Token 的路由信息（expertIds 矩阵）
2. 依照专家索引对 Token 进行重排（MoePermine），将发往同一目标 rank 的数据在 GM 内存中连续存放
3. 实现单次通信完成目标 rank 上所有专家的数据发送，显著减少 RDMA 下发时延

**通信驱动阶段（AICPU 执行）**

1. AIV 将数据在共享内存中的地址、长度信息通过 GM 消息区传递给同处 Device 侧的 AICPU
2. AICPU 直接驱动 RDMA 通信，彻底摒弃传统需要 Host 侧构造子图和调度 RDMA 任务的流程
3. 消除 Host 侧处理耗时和调度带来的额外时延

**通信等待与后处理阶段（AIV 执行）**

1. 通信环节，AIV 轮询数据接收 Flag，确保所有 rank 的 Token 数据全部接收完成，消除 RDMA 同步时延
2. AIV 将共享内存中的数据按专家汇总搬出，为后续 FFN 层计算提供数据准备

### 3.3 技术价值闭环

这三个阶段形成了完整的低延迟处理闭环：

- **通信计算融合**：将通信准备与计算任务深度融合
- **设备侧自治**：减少 Host 侧干预，消除 Host→Device 同步开销
- **端到端优化**：从数据预处理到后处理的全流程性能提升

---

## 4. Dispatch 实现方案详解

### 4.1 整体流水线架构

MoeDistributeDispatchV2 的实现构建了三阶段紧密协作的数据分发流水线：

1. **索引计算与 Token 重排**：基于动态专家分配结果，计算数据分发索引，通过内存重排优化通信数据布局
2. **数据发送与接收同步**：利用高效批量通信接口实现多目标节点并行分发，内置设备侧同步机制
3. **发送后处理**：对接收数据进行结构化重组，生成下游计算所需的元数据

### 4.2 索引计算与 Token 重排

#### 发送状态矩阵设计

构建 `sendStatus` 矩阵（维度 `worldSize × STATUS_ENTRY_COUNT`），布局如下：

- **计数区**（前 `FLAG_OFFSET=24` 个元素）：记录发送 Token 数量，`sendStatus(i,j)` 表示本地卡向第 i 张卡第 j 个专家发送的 Token 数
- **标志区**（第 `FLAG_OFFSET` 个元素）：同步标志位，用于后续接收同步

约束条件：`localExpertNum ≤ FLAG_OFFSET`

#### 索引计算流程

遍历 `expertIds` 矩阵（维度 `BS × K`），对每个元素 `expertIds(i,j)`：

1. 目标专家索引 = `expertIds(i,j)`
2. 目标 rank 索引 = `expertIds(i,j) / localExpertNum`（整除）
3. 专家在目标 rank 上的局部索引 = `expertIds(i,j) % localExpertNum`
4. `expandIdx(i,j) = sendStatus(目标rank索引, 专家局部索引)`
5. 更新 `sendStatus(目标rank索引, 专家局部索引)++`

#### Token 重排机制

构建专家 Token 数量前缀和数组 `expertCumsum`，重排位置由三个参数确定：

1. 目标 rank 索引
2. 前置专家 Token 总数（通过 `expertCumsum` 差值计算）
3. Token 在目标专家中的序号（`expandIdx`）

### 4.3 数据发送与接收同步

#### BatchWrite 通信接口

采用 HCCL 的 BatchWrite 接口进行数据发送，接口特性：

- 输入为 GM 指针，指向结构体数组（每个结构体含 localBuf、remoteBuf、count、dataType、remoteRankId）
- 无内置同步机制，每次下发存在固定时间开销（1-2us）
- 设计要求：最小化下发次数，在算子侧实现接收同步

#### 窗口分配策略

将 `WindowsIn` 和 `WindowsOut` 平均划分为 `worldSize` 个窗口，每个窗口对应一个 rank。窗口内数据结构：

- Token 数量数组 + FLAG1（对应 sendStatus 矩阵相应行）
- Tokens 数据（重排后的 Token 数据）
- FLAG2（位置与 Token 数量相关，动态确定）

#### 分核双循环等待策略

接收同步采用分核双循环机制：

1. **Rank 分配**：将所有 rank 平均分配给每个核
2. **第一层循环**：轮询 FLAG1 值，直到刷新为特定值，确认 Token 数量数组接收完成
3. **数据处理**：对 Token 数量数组求和，确定 FLAG2 位置
4. **第二层循环**：轮询 FLAG2 值，直到刷新为特定值，确认全部数据接收完成

### 4.4 发送后处理

数据接收完成后，执行两步处理：

1. **数据重排**：依据元数据将实际数据内容重新排列，确保同一专家的 Token 在 GM 内存中顺序连续
2. **元数据计算**：生成 `epRecvCount` 和 `expertTokenNum` 两个关键输出

对接收的元数据矩阵（维度 `w × e`）执行转置+前缀和操作，使用 Add+GatherMask+Adds 接口组合高效实现，避免标量操作的性能损失。

处理流程：

1. 行向累加（Add）：按顺序将上一行数据加到当前行，完成列方向前缀和预处理
2. 矩阵转置（GatherMask）：一次性转置整列数据
3. 行前缀和（Adds）：将上一行最后一个元素加到当前行，完成最终前缀和计算

---

## 5. 算子间数据同步：双缓冲机制

### 5.1 数据竞争问题

在分布式专家并行架构中，不同计算节点处理速度存在差异：

- **计算负载不均衡**：不同 rank 处理的数据量可能差异显著
- **异步执行时序**：快 rank 可能先进入 Combine阶段，而慢 rank 的 Dispatch 后处理尚未完成
- **数据完整性风险**：Combine 操作可能覆盖尚未完成处理的 Dispatch 数据

如果不做同步，会出现三类严重问题：

1. **写脏**：快 rank 的 Combine 数据覆盖慢 rank 尚未处理的 Dispatch 数据，导致计算精度异常
2. **标志位踩踏**：标志位和 Token 数据互相覆盖，标志位进入无效状态，产生死锁风险
3. **数据不一致**：读取到部分旧数据+部分新数据的混合结果

### 5.2 双缓冲架构

为解决数据竞争，采用双缓冲机制：

- 将 `WindowIn`（接收缓冲区）和 `WindowOut`（发送缓冲区）均划分为两个等大的存储块
- 在 `WindowIn` 第一块末端（偏移 1MB 处）设置 `bufferChosen` 标志位

### 5.3 缓冲区选择逻辑

- Dispatch 和 Combine 算子初始化时读取 `bufferChosen` 标志位
- 标志位为 0：使用第一组缓冲区（WindowIn/Out Block 0）
- 标志位为 1：使用第二组缓冲区（WindowIn/Out Block 1）
- 算子执行完成前翻转标志位：`bufferChosen = bufferChosen ^ 1`

### 5.4 同步保证原理

双缓冲的正确性基于以下时序特性：

1. 每张卡都必须收到所有其他卡的通信 Flag 才能结束
2. 某卡的第 N 个 EP 算子未结束时，其他卡的第 N+1 个 EP 算子必定还未结束（因为无法收到当前卡第 N+1 个的通信 Flag）
3. 其他卡的第 N+2 个 EP 算子开始时，当前卡第 N 个 EP 算子必定已结束
4. 第 N+2 个和第 N 个 EP 算子的数据缓冲区可安全重用

这一方案通过缓冲区轮转实现同步，在保证数据正确性的同时避免了全卡同步的性能开销。

---

## 6. 通信算法选择：Fullmesh 与 Hierarchy

### 6.1 commAlg 参数

`commAlg` 属性控制通信亲和内存布局算法，各平台支持情况如下：

**Atlas A2 平台**：

| commAlg | 说明 | 适用场景 |
|---------|------|---------|
| nullptr / "" | 依 HCCL 环境变量选择算法（不推荐） | 兼容旧版本 |
| fullmesh | Token 数据直接通过 RDMA 发往目标专家所在卡 | EP 规模 2~384 |
| hierarchy | Token 经跨机、机内两次发送，机内使用 HCCS | EP 规模 16/32/64 |

**Atlas A3 / Ascend 950 平台**：

| commAlg | 说明 |
|---------|------|
| "" / fullmesh_v1 | 默认值，使能 fullmesh_v1 模板 |
| fullmesh_v2 | 使能 fullmesh_v2 模板（仅 tpWorldSize=1，不支持非均匀 BS/xActiveMask/特殊专家） |
| hierarchy | 使能 ROCE 分层直驱能力，需设置 `HCCL_LOGIC_SUPERPOD_ID` 环境变量 |

### 6.2 Fullmesh 模式详解

Fullmesh 模式下，Token 数据通过 RDMA 直接发送到目标专家所在卡。核心优势：

- **通信路径最短**：一次 RDMA 直传，无中间转发
- **下发次数最少**：AIV 预处理将同一 rank 的所有专家数据连续排布，实现单次 BatchWrite 完成全卡发送
- **适用范围广**：EP 规模从 2 到 384 都可使用

Fullmesh 模式的核心实现位于 `moe_distribute_dispatch_v2_full_mesh.h`，关键常量定义：

```cpp
constexpr uint8_t BUFFER_NUM = 2;          // 双缓冲
constexpr uint32_t STATE_OFFSET = 32U;     // 状态空间偏移
constexpr uint64_t WIN_STATE_OFFSET = 384UL * 1024UL;  // 64 + 320 KB 状态区
constexpr uint64_t FLAG_FIELD_OFFSET = 768UL * 1024UL; // 384 * 2 KB 标志区
```

### 6.3 Hierarchy 模式详解

Hierarchy 模式专为大规模多机场景优化，核心思想是分层通信：

1. **跨机阶段**：不同 server 同号卡之间使用 RDMA 通信（高带宽、低延迟的跨机链路）
2. **机内阶段**：server 内使用 HCCS（High-speed Cache Coherent Switch System）通信

优势：避免 RDMA 在机内卡间的开销，HCCS 的机内带宽更优；劣势：两次通信带来额外延迟，仅支持特定 EP 规模。

---

## 7. 量化模式与数据类型

### 7.1 quantMode 参数详解

| quantMode | 说明 | 通信数据类型 | 适用平台 |
|-----------|------|-------------|---------|
| 0 | 非量化 | FP16/BF16 | A2/A3/950 |
| 1 | 静态量化 | INT8 | 950 |
| 2 | pertoken 动态量化 | INT8/FP8 | A2/A3/950 |
| 3 | pertile 动态量化 | FP8 | 950 |
| 4 | MX 动态量化 | FP8/FP4 | 950 |

### 7.2 量化计算公式

**静态量化（quantMode=1）**：

```
xFp32 = CastToFp32(X) × scales
quantOut = Cast(xFp32, dstType)
expandXOut = AllToAllV(quantOut)
```

**pertoken 动态量化（quantMode=2）**：

```
xFp32 = CastToFp32(X) × scales
dynamicScales = dstTypeMax / Max(Abs(xFp32))
quantOut = CastToInt8(xFp32 × dynamicScales)
expandXOut = AllToAllV(quantOut)
dynamicScalesOut = AllToAllV(1.0 / dynamicScales)  ← Combine 阶段反量化使用
```

**pertile 动态量化（quantMode=3）**：

与 pertoken 类似，但按 tile（128 元素分组）计算动态缩放因子，输出 `dynamicScalesOut` shape 为 `(A, Ceil(H, 128))`。

**MX 量化（quantMode=4）**：

基于 MX（Microscaling）格式，使用共享指数：

```
sharedExp = Floor(log₂(max(x))) - emax
dynamicScales = 2^sharedExp
quantOut = CastToFp8(X / dynamicScales)
expandXOut = AllToAllV(quantOut)
dynamicScalesOut = AllToAllV(1.0 / dynamicScales)
```

其中 `emax` 表示该数据类型最大正规数对应的指数值。

---

## 8. 参数体系与约束

### 8.1 核心输入参数

| 参数 | 类型 | 说明 |
|------|------|------|
| x | Tensor `(BS, H)` | 本卡发送的 Token 数据，支持 FP16/BF16/FP8/HF8/FP4 |
| expertIds | Tensor `(BS, K)` | 每个 Token 的 topK 个专家索引，INT32 |
| scalesOptional | Tensor | 量化平滑参数（可选） |
| xActiveMaskOptional | Tensor `(BS,)` 或 `(BS, K)` | Token 是否参与通信（可选） |
| expertScalesOptional | Tensor `(BS, K)` | 每个 Token 的 topK 专家权重（可选） |

### 8.2 核心属性参数

| 参数 | 说明 | 关键约束 |
|------|------|---------|
| groupEp | EP 通信域名称 | 字符串长度 [1, 128)，不能与 groupTp 相同 |
| epWorldSize | EP 通信域大小 | A2: 2~384；A3/950: 2~768 |
| epRankId | EP 域本卡 ID | 取值 [0, epWorldSize)，各卡不重复 |
| moeExpertNum | MoE 专家数量 | 需满足 moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0 |
| groupTp | TP 通信域名称 | 可选，与 groupEp 不同 |
| tpWorldSize | TP 通信域大小 | 0/1 表示无 TP，2 表示有 TP |
| quantMode | 量化模式 | A2 支持 0/2；A3 支持 0/2；950 支持 0/1/2/3/4 |
| commAlg | 通信算法 | fullmesh/hierarchy/fullmesh_v1/fullmesh_v2 |
| globalBS | EP 域全局 BS | 各卡 BS 一致时 = BS × epWorldSize 或 0；不一致时 = maxBS × epWorldSize |

### 8.3 核心输出参数

| 参数 | shape | 说明 |
|------|-------|------|
| expandXOut | `(max(tpWorldSize,1) × A, H)` | 扩展过的 Token 特征 |
| dynamicScalesOut | 1D/2D | 动态量化缩放参数（quantMode>0 时有输出） |
| assistInfoForCombineOut | `(A × 128,)` | Token 分布信息（传给 CombineV2） |
| expertTokenNumsOut | `(localExpertNum,)` | 每个专家收到的 Token 数 |
| epRecvCountsOut | 1D | 从 EP 域各卡接收的 Token 数 |
| tpRecvCountsOut | 1D | 从 TP 域各卡接收的 Token 数（有 TP 时） |
| expandScalesOut | 1D | 本卡输出 Token 权重 |

### 8.4 HCCL_BUFFSIZE 约束

HCCL_BUFFSIZE 环境变量控制单个通信域占用内存大小（单位 MB），默认 200MB：

**Atlas A2 fullmesh**：

```
HCCL_BUFFSIZE >= BS × epWorldSize × min(localExpertNum, K) × H × 4B + 4MB
```

**Atlas A2 hierarchy**：

```
HCCL_BUFFSIZE >= (moeExpertNum + epWorldSize/4) × Align512(maxBS × (H×2 + 16×Align8(K))) + 8MB
```

**Atlas A3 / Ascend 950**：

```
HCCL_BUFFSIZE >= 2 × (localExpertNum × maxBS × epWorldSize × Align512(Align32(2×H) + 64) + (K + sharedExpertNum) × maxBS × Align512(2×H))
```

---

## 9. Combine 算子协同工作

### 9.1 Dispatch-Combine 配套使用

V2 算子必须与 CombineV2 系列算子配套使用，两者共享同一 EP 通信域和 TP 通信域（或都不使用 TP），通信域中不允许有其他算子。

关键数据传递关系：

- Dispatch 的 `assistInfoForCombineOut` → Combine 的 `assistInfoForCombine`
- Dispatch 的 `epRecvCountsOut` → Combine 的 `epSendCounts`
- Dispatch 的 `tpRecvCountsOut` → Combine 的 `tpSendCounts`
- Dispatch 的 `expandScalesOut` → Combine 的 `expertScalesOptional`

### 9.2 Combine 的核心流程

Combine 算子完成 MoE 推理的"回收"阶段，包含三个关键步骤：

**Token 重排（ReorderToken）**

- 将发送至同一目标卡的 Token 在 GM 内存中连续存放
- 基于 Dispatch 传递的 sendCounts 前缀和矩阵确定各目标卡的 Token 分发数量
- 按照 rank 分区将 Token 数据有序追加至 WindowsOut 缓冲区相应区域

**数据发送与接收同步**

- 采用 BatchWrite 接口进行卡间通信
- 窗口分配：WindowsIn/Out 等分为 worldSize 个窗口
- 同步机制：尾部 Flag 标记 + 分核循环等待

**加权求和（Sum）**

- 将 K 倍于原始输入的专家输出 Token 整合还原
- 基于 expertIds 和 expandIdx 定位各专家输出在缓冲区中的地址

Token 地址公式：

```
TokenAddr(i,j) = windowInGM + rankSizeOnWin × rank + expertWindowOffset(expertId) × H + expandIdx(i,j) × H
```

---

## 10. 算子变体与演进

### 10.1 版本演进

| 版本 | 关键变更 |
|------|---------|
| V1（MoeDistributeDispatch） | 基础版，使用 expandIdx 输出，依赖 HCCL 环境变量控制通信算法 |
| V2 | expandIdx 替换为 assistInfoForCombineOut，新增 commAlg 参数 |
| V3 | 新增特殊专家支持（零专家/copy专家/常量专家），支持 copyExpertNum、constExpertNum 参数 |
| V4 | 进一步扩展功能，保持 V3 的特殊专家支持 |

### 10.2 特殊专家类型（V3 引入）

| 类型 | 计算公式 | 说明 |
|------|---------|------|
| 零专家 | `Moe(oriX) = 0` | 输出恒为 0，用于路由到无效专家的 Token |
| copy专家 | `Moe(oriX) = oriX` | 输出等于输入，等价于跳过专家计算 |
| 常量专家 | `Moe(oriX) = α₁×oriX + α₂×V` | 线性变换，α₁/α₂/V 为固定参数 |

### 10.3 配套 Combine 算子

| Combine 版本 | 配套 Dispatch 版本 | 特殊功能 |
|--------------|-------------------|---------|
| CombineV2 | DispatchV2 | 基础加权求和 |
| CombineV3 | DispatchV3 | 支持特殊专家 |
| CombineAddRmsNorm | DispatchV2/V3 | Combine 与 RMSNorm 融合，省去一次额外 Kernel 调度 |

---

## 11. 调用示例与编程范式

### 11.1 两段式接口调用

aclnn 接口采用两段式调用范式：

```cpp
// 第一段：获取 workspace 大小和执行器
aclnnMoeDistributeDispatchV2GetWorkspaceSize(
    x, expertIds, scalesOptional, xActiveMaskOptional,
    expertScalesOptional,
    groupEp, epWorldSize, epRankId, moeExpertNum,
    groupTp, tpWorldSize, tpRankId,
    expertShardType, sharedExpertNum, sharedExpertRankNum,
    quantMode, globalBS, expertTokenNumsType, commAlg,
    expandXOut, dynamicScalesOut, assistInfoForCombineOut,
    expertTokenNumsOut, epRecvCountsOut, tpRecvCountsOut,
    expandScalesOut,
    &workspaceSize, &executor);

// 申请 workspace
aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

// 第二段：执行计算
aclnnMoeDistributeDispatchV2(workspaceAddr, workspaceSize, executor, stream);
aclrtSynchronizeStreamWithTimeout(stream, 10000);
```

Dispatch 与 Combine 必须在同一个线程中顺序执行，不能交叉。

### 11.2 多卡并行执行

典型部署中，每张卡启动一个线程执行 Dispatch+Combine 流程：

```cpp
// 初始化 EP 通信域
HcclCommInitAll(EP_WORLD_SIZE, devicesEp, commsEp);

// 每卡一个线程
for (rankId = 0; rankId < DEV_NUM; rankId++) {
    threads[rankId] = new std::thread(&launchOneThread, std::ref(args[rankId]));
}
for (rankId = 0; rankId < DEV_NUM; rankId++) {
    threads[rankId]->join();
}
```

---

## 12. 关键源码文件索引

| 文件 | 核心内容 |
|------|---------|
| `docs/MoeDistributeDispatch-Combine算子设计介绍.md` | 设计哲学、背景分析、AIV+AICPU 架构介绍 |
| `README.md` | 完整算子规格、参数说明、约束条件 |
| `docs/aclnnMoeDistributeDispatchV2.md` | aclnn API 文档、函数原型、调用示例 |
| `op_host/moe_distribute_dispatch_v2_tiling.cpp/h` | Tiling 切分逻辑、配置计算 |
| `op_host/moe_distribute_dispatch_v2_infershape.cpp` | 输出 shape 推断 |
| `op_kernel/moe_distribute_dispatch_v2.h` | 主 Kernel 入口（A2 non-fullmesh） |
| `op_kernel/moe_distribute_dispatch_v2_full_mesh.h` | Fullmesh Kernel 主流程（A2/A3） |
| `op_kernel/moe_distribute_dispatch_v2_quant.h` | 量化计算逻辑 |
| `op_kernel/moe_distribute_dispatch_v2_tiling.h` | Tiling 数据结构定义 |
| `op_api/aclnn_moe_distribute_dispatch_v2.cpp` | aclnn 接口实现 |

---

## 13. 总结

MoeDistributeDispatchV2 的设计哲学是**让通信不再是计算的瓶颈**：

- 通过 AIV+AICPU 融合架构将通信调度下沉至 Device 侧自治执行，消除 Host 介入带来的同步和调度延迟
- 通过 Token 重排和窗口连续排布实现单次 BatchWrite 完成全卡发送，将 RDMA 下发次数从 N×K 降至 N（N=worldSize）
- 通过分核双循环 Flag 轮询替代传统 RDMA 同步，消除 RTT 等待开销
- 通过双缓冲轮转机制实现算子间隐式同步，避免全卡 Barrier 的性能开销
- 通过量化模式（quantMode 0~4）在通信前压缩数据量，降低通信带宽需求

整套方案从通信架构（Device 自治 RDMA）到数据布局（窗口连续排布）到同步策略（双缓冲轮转+Flag 轮询）到数据压缩（可选量化），每一层都在为最小化通信延迟服务。这种"通算融合"思路将 MoE 专家并行推理的通信瓶颈转化为计算流水的一部分，是昇腾 NPU 上大规模 MoE 推理的核心性能保障。