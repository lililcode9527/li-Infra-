# MoeDistributeCombineV2 算子深度解析：通算融合回收机制、Staggered Send 与双缓冲同步

## 1. 算子定位与功能概述

MoeDistributeCombineV2 是 MoeDistributeDispatchV2 的对称算子，负责将各专家 FFN 的输出按路由权重加权合并还原——Dispatch 发出去，Combine 收回来，两者共同构成 MoE 专家并行推理的完整通信闭环。

### 核心功能

算子的功能可形式化描述为：对经专家 FFN 处理后的 Token 数据执行逆向 AllToAllV 通信（如有 TP 域则前置 ReduceScatterV），然后按路由权重加权求和，还原为原始 Token 序列的输出。

- 无 TP 通信域时：`xOut = Sum(expertScales × AllToAllV(expandX) + expertScales × sharedExpertX)`
- 有 TP 通信域时：`rsOut = ReduceScatterV(expandX)`, `ataOut = AllToAllV(rsOut)`, `xOut = Sum(expertScales × ataOut + expertScales × sharedExpertX)`

### 相较 V1 算子的演进

相较于原版 MoeDistributeCombine，V2 做了三项关键变更：

1. **辅助信息重构**：原 `expandIdx`（shape `BS*K`）替换为 `assistInfoForCombine`（shape `A*128`），提供更详细的 Token 分布三元组（epRankId, tokenIndex, topK），帮助 Combine 算子高效完成全卡同步
2. **共享专家输入新增**：新增 `sharedExpertXOptional` 输入，支持用户在 `sharedExpertNum=0` 时直接传入共享专家计算结果
3. **通信算法显式配置**：新增 `commAlg` 入参替代 `HCCL_INTRA_PCIE_ENABLE` 和 `HCCL_INTRA_ROCE_ENABLE` 环境变量

### 产品支持

| 产品 | 支持 |
|------|------|
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练/推理系列 | √ |
| Atlas A2 训练/推理系列 | √ |
| Atlas 200I/500 A2 推理 | × |
| Atlas 推理系列 | × |
| Atlas 训练系列 | × |

> 算子后续有 V3（新增特殊专家）和 V4（新增性能监控）版本，本文聚焦 V2 版本。

---

## 2. 数学公式与数据流

### 无 TP 域

$$
xOut[i] = \sum_{k=0}^{K-1} expertScales[i,k] \times expandX[i,k] + sharedExpertX[i]
$$

其中 `expandX` 为各专家 FFN 处理后的 Token 特征矩阵，`expertScales` 为路由权重，`sharedExpertX` 为共享专家贡献（可选）。

### 有 TP 域

$$
\begin{aligned}
rsOut &= \text{ReduceScatterV}(expandX) \\
ataOut &= \text{AllToAllV}(rsOut) \\
xOut &= \sum_k expertScales_k \times ataOut_k + sharedExpert_k
\end{aligned}
$$

TP 域下先做 ReduceScatterV（将 TP 维度的数据聚合到单卡），再做 EP 域 AllToAllV 逆向回收。

### Dispatch → Combine 数据流闭环

| Dispatch 输出 | Combine 输入 | 作用 |
|---|---|---|
| `expandXOut` | `expandX` | 各专家 FFN 输出的扩展 Token |
| `assistInfoForCombineOut` | `assistInfoForCombine` | Token 重排映射三元组 |
| `epRecvCountsOut` | `epSendCounts` | EP 域各卡接收的 Token 数量（反向即发送数） |
| `tpRecvCountsOut` | `tpSendCountsOptional` | TP 域各卡接收的 Token 数量 |
| `expandScalesOut` | `expandScalesOptional` | 本卡输出 Token 权重（hierarchy 模式必传） |

---

## 3. 核心流水线详解

### 3.1 Combine 主流水线

MoeDistributeCombineV2 的 `Process()` 方法构建了五阶段紧密协作的回收流水线：

```
ReduceScatterTrans → BuffInit → SetWaitTpStatusAndDispatch → AlltoAllBuffInitAndMaskCal → LocalWindowCopy
```

1. **ReduceScatterTrans**（仅 TP=2）：将 TP 对端数据从 GM 拷入窗口，按核分配处理
2. **BuffInit**：初始化 UB 缓冲区（tokenBuf、rowTmpFloatBuf、mulBuf 等），为后续计算分配本地存储
3. **SetWaitTpStatusAndDispatch**：写 TP 状态标志并等待对端完成 TP ReduceScatter
4. **AlltoAllBuffInitAndMaskCal**：初始化 AlltoAll 通信缓冲区，计算 Token/Expert 活跃掩码
5. **LocalWindowCopy**：核心加权求和阶段——从窗口读取各专家输出，反量化（如有），按路由权值累加还原

### 3.2 ExpertAlltoAllDispatchCopyAdd：数据写入窗口

Combine 的第一步是将本卡 FFN 处理后的 Token 数据从 GM 拷贝到 WinOut 缓冲区，准备逆向 AllToAllV 通信。关键实现：

- **TP 合并**（TP=2 场景）：将本卡和对端 TP 卡的数据合并写入同一窗口段
- **量化写入**（quantMode > 0）：调用 `quantInst_.QuantProcess()` 将 FP16/BF16 数据量化为 INT8/MXFP8 后写入窗口，节省通信带宽
- **Staggered Send**：采用 `(loop + epRankId_) % sendCntNum_` 偏移策略避免多核同时写同一窗口段，详见第 5 节
- **status flag 设置**：每个 Token 写入后在窗口状态区设置 FLAG 标志，通知接收方数据到达

### 3.3 WaitDispatch：等待数据到达

Combine 从其他卡接收数据时需要等待所有 topK + shared expert 的 Token 全部到达：

```
flagRcvCount_ = axisK_ + sharedExpertNum_
```

等待机制：
- 轮询窗口状态区（`selfDataStatusGMTensor_`），每个期望接收的 Token 对应一个 32 字节的状态槽位
- 使用 `STATE_OFFSET`（32 字节步长）遍历状态数组
- 检查状态值是否为期望的值（由 Dispatch 写入），标志数据到达
- 支持可选的 `performanceInfoOptional` 记录：在开始等待时记录时间戳，等待完成后计算等待时长写入 performanceInfo Tensor

### 3.4 ProcessExpert：加权求和核心

`ProcessExpert()` 是 Combine 的计算核心，遍历每个 Token 的 topK 专家输出并加权累加：

```cpp
// 对每个 Token
for (tokenIndex in [startTokenId_, endTokenId_)) {
    for (topkId in [0, axisK_)) {
        // 从 assistInfoForCombine 解析三元组 (epRankId, tokenIndex, topK)
        // 定位该 Token 在 WinIn 中的地址
        // 读取专家输出数据
        // 反量化（如需要）
        // 按路由权值 expertScales 加权累加
        ProcessMoeExpert(tokenIndexOffset, topkId, scaleVal);
    }
    // 加上 sharedExpertX（如有）
    // 如有 AddRmsNorm 融合，执行残差加 + RMSNorm
}
```

---

## 4. 地址定位与窗口组织

### TokenAddr 地址公式

Combine 在 WinIn 中定位某 Token 的地址：

```
TokenAddr(i,j) = WinInBase
              + rank × rankSizeWin              // rank 偏段起始
              + expertOffset(expertId) × H      // 该专家在窗口内的偏移
              + expandIdx(i,j) × H              // Token 在专家内的序号 × 隐层维度
```

其中：
- `WinInBase`：WinIn 的 GM 基地址（含双缓冲偏移 `winDataSizeOffsetEp_`）
- `rankSizeWin`：每个 rank 在 WinIn 中占用的段长（`hAlignWinSize_` 对齐到 512 字节）
- `rank`：Token 来源的 EP rank ID（从 assistInfoForCombine 三元组解析）
- `expertOffset(expertId)`：专家在 rank 段内的偏移量（通过 epSendCounts 前缀和计算）
- `expandIdx(i,j)`：该 Token 在专家内序号（从 assistInfoForCombine 三元组解析）

### assistInfoForCombine 三元组结构

V2 用 `assistInfoForCombine`（shape `(A*128,)`）替代了 V1 的 `expandIdx`（shape `(BS*K,)`），每个 Token 对应一个三元组：

```
triplet(expandIdx) = {epRankId, tokenIndex, topK}   // EXPAND_IDX_INFO = 3
```

Combine 通过遍历 `expertIds(i,k)` 得到目标专家 ID，再通过 `assistInfoForCombine` 解析出该 Token 数据在 WinIn 缓冲区中的精确位置。

### 窗口排布示意图

```
WinIn (Combine 读取方向)
┌────────────┬────────────┬────────────┬────────────┐
│  Rank 0    │  Rank 1    │  Rank 2    │  Rank 3    │
├────────────┼────────────┼────────────┼────────────┤
│ E0   E1    │ E0   E1    │ E0   E1    │  ...       │
│ T0 T1 T2   │ T0 T1      │ T0         │            │
│ ──FLAG──   │ ──FLAG──   │ ──FLAG──   │            │
└────────────┴────────────┴────────────┴────────────┘

状态区 (COMBINE_STATE_WIN_OFFSET = 818KB)
┌────────────────────────────────────────────────────┐
│ Dispatch 状态区 (前 64KB: COMBINE_STATE_OFFSET)     │
│ Combine 状态区 (818KB 起始)                         │
└────────────────────────────────────────────────────┘
```

- 每个 rank 段内按 localExpertNum 分区，每个专家分区存放该专家接收的 Token 数据
- `hAlignWinSize_ = Ceil(H × sizeof(ExpandXType), 512) × 512`：窗口内 Token 行按 512 字节对齐

---

## 5. 核间分派与 Staggered Send 模式

### SplitCoreCal 核分配策略

Combine 在写入窗口时需要将 `selfSendCnt_`（本卡需发送的 Token 总数）平均分配给所有 AIV 核：

```cpp
sendCntNum_ = selfSendCnt_ / aivNum_;
remainderRankNum = selfSendCnt_ % aivNum_;

startTokenId_ = sendCntNum_ * coreIdx_;
if (coreIdx_ < remainderRankNum) {
    sendCntNum_++;
    startTokenId_ += coreIdx_;
} else {
    startTokenId_ += remainderRankNum;
}
endTokenId_ = startTokenId_ + sendCntNum_;
```

每个核处理 `[startTokenId_, endTokenId_)` 范围内的 Token，将它们写入 WinOut 缓冲区对应的 rank 段。

### Staggered Send：无竞争窗口写入

多核并行写入同一窗口时，如果不做协调，多个核可能同时写同一 rank 段的同一位置，造成数据竞争。Combine 采用 **Staggered Send**（交错发送）策略解决这一问题：

```
目标 rank = (loop + epRankId_) % sendCntNum_
```

- 每个核在写入循环中使用 `(loop + epRankId_) % sendCntNum_` 计算目标 rank 偏移
- 不同核的 `epRankId_` 不同，使得同一时刻各核写入的 rank 段互不重叠
- 等价于"循环左移"发送顺序，将竞争窗口从同一段分散到不同段

### moeQueue_ 双缓冲流水

Combine 使用 `TQueBind<VECIN, VECOUT, BUFFER_NUM=2>` 的 `moeQueue_` 实现 Token 数据的乒乓式流水搬移：

- 核从 GM 读取一个 Token 到 `moeQueue_` 的 Buffer 0
- 同时在 Buffer 1 上进行上一 Token 的加权计算
- 计算完成后将 Buffer 1 的结果写回 GM
- 两缓冲交替使用，确保数据搬移与计算始终并行

---

## 6. 双缓冲同步机制

### 与 Dispatch 共享双缓冲

Combine 与 Dispatch 共享同一套 WinIn/WinOut 双缓冲，这是算子间隐式同步的核心机制。

**缓冲区划分**：

- `WindowIn` 和 `WindowOut` 各被划分为两个等大的存储块（Block 0 和 Block 1）
- 在 `WindowIn` 第一块末端（偏移 1MB 处）设置 `bufferChosen` 标志位
- 算子初始化时根据 `dataState_`（由 `InitWinState` 返回）确定使用哪块缓冲区

**翻转规则**：

- Dispatch 和 Combine 初始化时读取 `bufferChosen` 标志位
- 标志位为 0：使用 Block 0（`winDataSizeOffsetEp_ = 0`）
- 标志位为 1：使用 Block 1（`winDataSizeOffsetEp_ = totalWinSizeEp / 2`）
- 算子执行完成前翻转标志位：`bufferChosen = bufferChosen ^ 1`

### Combine 使用的状态区偏移

Combine 在窗口状态区中使用与 Dispatch 不同的偏移位置，避免状态信息冲突：

| 常量 | 值 | 说明 |
|------|-----|------|
| `STATE_SIZE` | 1MB | 每个 rank 的窗口状态区大小 |
| `DISPATCH_STATE_WIN_OFFSET` | 768KB | Dispatch 在状态区使用的偏移 |
| `COMBINE_STATE_WIN_OFFSET` | 818KB | Combine 在状态区使用的偏移（留约 50KB 给 Dispatch） |
| `COMBINE_STATE_OFFSET` | 64KB | Combine 本卡状态在窗口内的偏移（前部预留给 Dispatch） |
| `WIN_STATE_OFFSET` | 384KB | 状态区按缓冲区划分的半区大小 |
| `WIN_ADDR_ALIGN` | 512B | 窗口地址对齐粒度 |
| `UB_ALIGN` | 32B | UB 缓冲区对齐粒度 |

Combine 读取状态的地址计算：

```cpp
selfDataStatusGMTensor_ = statusDataSpaceGm_ + COMBINE_STATE_WIN_OFFSET + coreIdx_ * WIN_ADDR_ALIGN
```

- `COMBINE_STATE_WIN_OFFSET`（818KB）确保 Combine 的状态读取不会覆盖 Dispatch 正在写入的状态
- `COMBINE_STATE_OFFSET`（64KB）用于本卡内部的状态偏移计算

### 同步保证原理

双缓冲的正确性基于以下时序推理：

1. 每张卡都必须收到所有其他卡的通信 Flag 才能结束当前算子
2. 某卡的第 N 个 EP 算子未结束时，其他卡的第 N+1 个 EP 算子必定还未结束（因为无法收到当前卡第 N+1 个的通信 Flag）
3. 其他卡的第 N+2 个 EP 算子开始时，当前卡第 N 个 EP 算子必定已结束
4. 第 N+2 个和第 N 个 EP 算子的数据缓冲区可安全重用（Block 0 ↔ Block 1 轮转）

---

## 7. 通信量化模式（Combine 侧）

### commQuantMode 参数

Combine V2 API 通过 `commQuantMode` 属性控制通信量化方式：

| commQuantMode | 说明 | 适用平台 |
|---|---|---|
| 0 | 无量化（FP16/BF16 直接通信） | A2/A3/950 |
| 2 | INT8 量化通信 | A2 hierarchy / A3（仅 tpWorldSize<2）/ 950 |
| 3 | MXFP8 E5M2 量化通信 | 950PR/950DT |
| 4 | MXFP8 E4M3 量化通信 | 950PR/950DT |

### Int8DequantProcess：标准反量化流程

当 `commQuantMode=2`（INT8）时，Combine 侧需要将接收到的 INT8 数据反量化还原为 FP16/BF16 后再进行加权累加：

```
1. Cast(scaleDivFloatTensor_, scaleDivTensor_, CAST_NONE)    // 量化 scale 从 ExpandXType → float
2. Cast(fp16CastTensor_, castLocalTensor_, CAST_NONE)        // int8 → half
3. Cast(absFloatTensor_, fp16CastTensor_, CAST_NONE)         // half → float
4. Brcb(scaleDupLocalTensor_, scaleDivFloatTensor_)           // scale 广播至 H 维度
5. Mul(absFloatTensor_, absFloatTensor_, scaleDupLocalTensor_) // float × scale = 反量化
6. Cast(outLocal, absFloatTensor_, CAST_RINT)                 // float → XType 输出
```

量化 scale 的存储位置：紧跟在 INT8 数据之后，偏移 `hAlign32Size_ / INT8_DIVIVE`（INT8 数据占用 ExpandXType 大小的 1/4）。

### Int8DequantProcessA5：A5 MicroAPI 融合反量化+加权累加

在 A5（3510）架构上，Combine 量化模块将反量化、权值缩放和累加三个操作融合为一条 MicroAPI 寄存器级流水线，避免多次中间数据搬移：

```cpp
__VEC_SCOPE__ {
    // 1. 加载：Deinterleave B32 拆包 scale，Unpack B8 拆包 int8 token
    DataCopy<float, DIST_DINTLV_B32>(dyScaleFp32_1, dyScaleFp32_2, dyScaleFp32Ptr);
    DataCopy<int8_t, DIST_UNPACK_B8>(tokenSrcReg, tokenPtr);

    // 2. 类型转换链：int8 → half → fp32（两路并行拆包）
    Cast<half, int8_t>(tokenHalfReg, tokenSrcReg);
    Cast<float, half>(tokFp32_1, tokenHalfReg);   // ZERO layout
    Cast<float, half>(tokFp32_2, tokenHalfReg);   // ONE layout (deinterleave)

    // 3. 反量化：scale × token_fp32
    Mul(deqFp32_1, dyScaleFp32_1, tokFp32_1);
    Mul(deqFp32_2, dyScaleFp32_2, tokFp32_2);

    // 4. 权值缩放：dequant_fp32 × expertScale
    Muls(sumLocal_1, deqFp32_1, scaleVal);
    Muls(sumLocal_2, deqFp32_2, scaleVal);

    // 5. 累加：sumFinal += weighted_token
    Add(sumFinal_1, sumFinal_1, sumLocal_1);
    Add(sumFinal_2, sumFinal_2, sumLocal_2);

    // 6. 存储：Interleave B32 写回累加结果
    DataCopy<float, DIST_INTLV_B32>(sumFinalDstPtr, sumFinal_1, sumFinal_2);
}
```

**关键优化点**：

- **Deinterleave/Interleave B32**：利用 A5 MicroAPI 的 `DIST_DINTLV_B32` 加载模式将 256-bit 数据拆为两路 128-bit，`DIST_INTLV_B32` 存储模式将两路合并写回——寄存器级并行处理
- **Unpack B8**：INT8 数据直接从 8-bit 拆包为向量寄存器格式，省去中间 Cast 开销
- **四步融合**：反量化→权值缩放→累加→写回，全程寄存器操作无 UB 中间缓冲

A5 特殊对齐要求：`SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL>` 饱和模式控制——A5 不支持 MX 饱和，需显式关闭。

### DeQuantMxFp8：MXFP8 反量化融合

A5 还支持 MXFP8（E5M2/E4M3）的反量化融合，使用 `fp8_e8m0_t` 共享指数作为量化 scale：

```
1. ShiftLefts(e8m0_scale → fp32)            // e8m0 左移 23 位（到 fp32 指数字段）
2. DataCopy<E2B_B32>(scale广播)              // 4B → 32B 广播（8倍扩展）
3. DataCopy<UNPACK_B8>(fp8 token加载)        // FP8 数据拆包
4. Cast<float, fp8>(token → fp32)            // 两路拆包转换
5. Mul(token_fp32 × scale_fp32)              // 反量化
6. Muls(result × expertScale)                // 权值缩放
7. Add(sumFinal += weighted)                 // 累加
8. DataCopy<INTLV_B32>(写回)                 // 合包写回
```

---

## 8. Layered (RDMA+IPC) Combine 架构

### MoeDistributeCombineV2Layered 类

当 EP 规模超过单机范围时（worldSize > 16），Combine 使用 `MoeDistributeCombineV2Layered` 类实现两层级通信架构——RDMA 跨 server + IPC 机内。

**模板参数**：

```cpp
template <typename ExpandXType, typename ExpandIdxType, typename ExpandXTransType>
```

- `ExpandXType`：输入输出数据类型（FP16/BF16）
- `ExpandIdxType`：索引类型（int32）
- `ExpandXTransType`：传输数据类型——与 `ExpandXType` 相同表示无量化，`int8_t` 表示动态量化

**关键常量**：

| 常量 | 值 | 说明 |
|------|-----|------|
| `SERVER_RANK_SIZE` | 16 | 每个 server 包含 16 个 rank |
| `RDMA_DATA_SIZE` | 400MB | RDMA 缓冲区大小 |
| `IPC_DATA_OFFSET` | 4MB | IPC 共享内存数据区偏移 |
| `STATE_SPACE_SIZE` | 1MB | 状态空间大小 |
| `BUFFER_NUM` | 2 | 双缓冲数量 |
| `GM2IPC_SYNC_FLAG` | 12345 | GM→IPC 同步标志值 |
| `RDMA_TOKEN_ARRIVED_FLAG` | 123 | RDMA Token 到达标志 |
| `RDMA_TOKEN_END_FLAG` | 321 | RDMA Token 结束标志 |
| `WEIGHT_VALUE_NUM` | 16 | Token 附加信息数量（scale 占 32B = 16 × sizeof(bf16/fp16)) |

### 流水线阶段

```
GM2IPC → WaitIPC → SumToWindow → AlltoAllServerDispatch → WaitDispatch → SumToServer
```

**GM2IPC**：将 expandX 中的 Token 数据和对应的路由权值写入 IPC 共享内存（本 server 内的共享区域）。Token 格式为 `data(H) + weight(32B)`，按 rank 分段连续排布。

**WaitIPC**：通过 `GM2IPC_SYNC_FLAG` + `magicValue_` 值同步所有 server 内 rank 的 GM2IPC 完成状态。先写入标志，再轮询等待同 server 其他 rank 完成，最后清除标志。

**SumToWindow**：从 IPC 共享内存读取 Token，按 server 内的 topK 分组加权累加（`ServerInAdd`），将累加结果写入 RDMA 输出窗口对应的目标 server 段。支持动态量化路径（`DynamicQuantProcess`）。

**AlltoAllServerDispatch**：使用 `AIVRDMAPostSend` RDMA 发送将窗口数据从本 server 传到目标 server。每个核负责一个目标 server，通过 `WaitStatusAndDispatch` 等待源端数据到达标志后触发 RDMA WRITE。

**WaitDispatch**：等待 RDMA 数据到达本 server 的 WinIn 缓冲区。轮询 `RDMA_TOKEN_ARRIVED_FLAG + magicValue_` 标志。

**SumToServer**：从 WinIn 缓冲区读取接收到的 Token 数据，反量化（如需要），按路由权值加权累加写入最终输出 `expandOutGlobal_`。

### magicValue_ 递增同步

Layered 模式使用递增的 `magicValue_` 计数器实现多轮同步，避免标志值冲突：

- 初始化时从 IPC 共享内存读取 `magicValue_`（位于 `MAGIC_OFFSET = 2MB - 32×32`）
- 每轮操作使用 `flagValue = BASE_FLAG + magicValue_` 作为同步标志
- `RDMADataSwitch()` 在一轮结束时递增 magicValue_ 并翻转 bufferId_：

```cpp
magicGlobal_.SetValue(MAGIC_OFFSET / sizeof(uint64_t), magicValue_ + 1);  // 递增
bufferIdGlobal_(0) = bufferId_ ^ 1;                                       // 翻转缓冲区
```

### DynamicQuantProcess：Layered 动态量化

在 Layered 模式下，Combine 对 server 内累加后的 Token 数据做动态 INT8 量化后再 RDMA 传输，节省跨 server 通信带宽：

**FP16 路径**：

```
Cast(sumHalf, sumFloat, CAST_RINT)          // fp32 → fp16
Abs(absScale, sumHalf)                       // 取绝对值
BlockReduceMax(reduceMax, absScale)           // 每 16 元素取最大值
Muls(scaleData, reduceMax, 1/127)            // scale = dmax / 127
Brcb(absScale, scaleData)                    // scale 广播至 H 维度
Div(sumHalf, sumHalf, absScale)              // token / scale = 量化数据
Cast(int8Data, sumHalf, CAST_RINT)           // fp16 → int8
DataCopy(window, int8Data + scale)           // 写入 int8 数据 + scale 值
```

**BF16 路径**：

```
Abs(castInFloat, sumFloat)                   // fp32 绝对值
BlockReduceMax(reduceMaxFloat, castInFloat)  // 每 8 元素取最大值
Muls(reduceMaxFloat, 1/127)                  // scale
Brcb(castInFloat, reduceMaxFloat)            // 广播
Div(sumFloat, castInFloat)                   // 量化
Cast(bf16Scale, reduceMaxFloat, CAST_RINT)   // fp32 → bf16 scale
Cast(halfData, sumFloat, CAST_RINT)          // fp32 → bf16 data
Cast(int8Data, halfData, CAST_RINT)          // bf16 → int8
DataCopy(window, int8Data + bf16Scale)       // 写入
```

量化粒度：FP16 每 16 元素一个 scale（`scale_granu_=16`），BF16 每 8 元素一个 scale（`scale_granu_=8`）。

---

## 9. Elastic 弹性伸缩

### MoeDistributeElastic 集成

Combine V2 kernel 集成了 `MoeDistributeElastic` 模块，支持动态缩卡（Scaling Down）场景——当集群中某卡故障需要临时减少 EP 规模时，通过 `elasticInfoOptional` tensor 传递重映射信息。

### elasticInfoOptional 格式

`elasticInfoOptional` 为 INT32 类型 Tensor，格式如下：

```
[0] isScalingDown    // 是否正在缩卡（0=正常, 1=缩卡）
[1+] remap info      // rank 重映射表
```

- `isScalingDownFlag_`：从 `elasticInfoOptional.GetValue(0)` 读取
- `elasticInst_.InitElasticInfo()`：初始化弹性信息，包括 epWorldSize、sharedExpertRankNum、moeExpertNum 的重映射参数

### 平台支持

- **A2**：支持弹性伸缩，可传入有效的 `elasticInfoOptional`
- **A3/950**：当前传 null pointer（reserved 参数）

---

## 10. 参数体系（V2 API）

### 核心输入参数

| 参数 | 类型 | Shape | 说明 |
|------|------|-------|------|
| expandX | FP16/BF16 | `(max(tpWorldSize,1)×A, H)` | 各专家 FFN 处理后的 Token 特征 |
| expertIds | INT32 | `(BS, K)` | 每个 Token 的 topK 专家索引 |
| assistInfoForCombine | INT32 | `(A×128,)` | Token 重排映射三元组（来自 Dispatch） |
| epSendCounts | INT32 | 1D | EP 域各卡发送 Token 数量（来自 Dispatch 的 epRecvCounts） |
| expertScales | FLOAT32 | `(BS, K)` | topK 路由权值 |
| tpSendCountsOptional | INT32 | `(tpWorldSize,)` | TP 域发送数量（仅 TP=2 时有效） |
| xActiveMaskOptional | BOOL | `(BS,)` 或 `(BS, K)` | Token 是否参与通信 |
| expandScalesOptional | FLOAT32 | `(A,)` | 通信量化 scale（A2 hierarchy 必传） |
| sharedExpertXOptional | FP16/BF16 | `(BS, H)` | 共享专家计算结果（可选） |
| elasticInfoOptional | INT32 | 1D | 弹性伸缩信息（A2 支持，A3/950 传 null） |

### 核心属性参数

| 参数 | 说明 | 关键约束 |
|------|------|---------|
| groupEp | EP 通信域名称 | 字符串长度 [1, 128)，与 groupTp 不同 |
| epWorldSize | EP 通信域大小 | A2 fullmesh: 2~384；A2 hierarchy: 16/32/64；A3: [2,768]；950: [2,768] |
| epRankId | EP 域本卡 ID | [0, epWorldSize)，各卡不重复 |
| moeExpertNum | MoE 专家总数 | 需满足 moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0 |
| groupTp | TP 通信域名称 | 可选，与 groupEp 不同 |
| tpWorldSize | TP 域大小 | 0/1=无 TP，2=有 TP |
| tpRankId | TP 域本卡 ID | [0, tpWorldSize) |
| expertShardType | 专家分片方式 | 0=按卡均匀分片 |
| sharedExpertNum | 共享专家数量 | A2: 不支持共享专家；A3: [0,4]；950: [0,4] |
| sharedExpertRankNum | 共享专家部署卡数 | 默认 0 |
| globalBS | EP 域全局 BS | 各卡 BS 一致时 = BS×epWorldSize 或 0 |
| outDtype | 输出数据类型 | 默认 0（与 expandX 同类型） |
| commQuantMode | 通信量化模式 | A2: 0/2（仅 hierarchy 支持量化）；A3: 0/2；950: 0/2/3/4 |
| groupListType | 组列表类型 | reserved |
| commAlg | 通信算法 | A2: nullptr/""/"fullmesh"/"hierarchy"；A3/950: 不支持（传 null） |

### 核心输出参数

| 参数 | Shape | 说明 |
|------|-------|------|
| xOut | `(BS, H)` | 加权求和后的还原输出 |

### 各平台参数差异

**Atlas A2**：
- 不支持共享专家场景、TP 域、弹性伸缩（A3/950 不支持）、const expert
- `commAlg` 支持 nullptr/""/"fullmesh"/"hierarchy"
- hierarchy 模式必须传 `expandScalesOptional`
- `xActiveMaskOptional`：fullmesh 支持 1D `(BS,)`；hierarchy 不支持（传 null）
- `epWorldSize`：fullmesh 支持 2,3,4,5,6,7,8,16,32,64,128,192,256,384；hierarchy 支持 16,32,64
- `moeExpertNum`：fullmesh (0,1024]；hierarchy (0,512]
- `commQuantMode=2` 仅 hierarchy 支持，需 driver >= 25.0.RC1.1

**Atlas A3**：
- 双 DIE 卡，"本卡"指单 DIE
- 不支持 `commAlg`（传 null）
- `H` 范围 [1024, 8192]；`BS` 范围 [1, 512]
- `epWorldSize` 支持 8,16,32,64,128,144,256,288
- `sharedExpertNum` 范围 [0, 4]
- `commQuantMode`：0 或 2（INT8 仅 tpWorldSize<2 时支持）

**Ascend 950PR/950DT**：
- 不支持 TP 域
- 不支持 `commAlg`（传 null）
- 不支持 `expandScalesOptional`
- `H` 范围 [1024, 8192]；`BS` 范围 (0, 256]（fullmesh_v2/hierarchy）或 (0, 512]（fullmesh_v1/""）
- `epWorldSize` 范围 [2, 768]
- `commQuantMode` 支持 0, 2（INT8）, 3（MXFP8 E5M2）, 4（MXFP8 E4M3）

---

## 11. Dispatch-Combine 数据流闭环

### 执行顺序要求

Dispatch → FFN → Combine 三步必须顺序执行，不能交叉：

1. **Dispatch** 完成后，输出 `expandXOut`、`assistInfoForCombineOut`、`epRecvCountsOut` 等
2. **FFN 层** 使用 `expandXOut` 作为输入，产生各专家的输出
3. **Combine** 使用 FFN 输出 + Dispatch 传递的元数据完成加权还原

Dispatch 和 Combine 必须在同一个线程中顺序执行，每张卡一个线程。

### 通信域隔离规则

- Dispatch 和 Combine 共享同一个 EP 通信域（`groupEp`）
- 如有 TP 域，也共享同一个 TP 通信域（`groupTp`）
- 通信域中不允许有其他算子
- 各卡的 `groupEp`、`epWorldSize`、`moeExpertNum`、`groupTp`、`tpWorldSize`、`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`、`globalBS`、`commAlg`、`HCCL_BUFFSIZE` 必须一致且与 Dispatch 匹配

---

## 12. 算子变体与演进

### 版本演进

| 版本 | 关键变更 |
|------|---------|
| V1（MoeDistributeCombine） | 基础版，使用 expandIdx 输入，依赖 HCCL 环境变量 |
| V2 | expandIdx 替换为 assistInfoForCombine，新增 sharedExpertXOptional、commAlg 参数 |
| V3 | 新增特殊专家支持（零专家/copy专家/常量专家） |
| V4 | 新增性能监控参数 performanceInfoOptional |

> 本文聚焦 V2 版本。V3/V4 的特殊专家和性能监控特性不在本文展开范围。

### V2 配套 Combine 版本

| Combine 变体 | 配套 Dispatch 版本 | 特殊功能 |
|---|---|---|
| CombineV2 | DispatchV2 | 基础加权求和 |
| CombineAddRmsNorm | DispatchV2 | Combine 与 Add+RMSNorm 融合，省去一次额外 Kernel 调度 |

---

## 13. 调用示例

### V2 API 两段式调用

aclnn 接口采用两段式调用范式：

```cpp
// 第一段：获取 workspace 大小和执行器
aclnnMoeDistributeCombineV2GetWorkspaceSize(
    expandX, expertIds, assistInfoForCombine, epSendCounts, expertScales,
    tpSendCountsOptional, xActiveMaskOptional,
    activationScaleOptional, weightScaleOptional, groupListOptional,
    expandScalesOptional, sharedExpertXOptional,
    groupEp, epWorldSize, epRankId, moeExpertNum,
    groupTp, tpWorldSize, tpRankId,
    expertShardType, sharedExpertNum, sharedExpertRankNum,
    globalBS, outDtype, commQuantMode, groupListType, commAlg,
    xOut,
    &workspaceSize, &executor);

// 申请 workspace
aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

// 第二段：执行计算
aclnnMoeDistributeCombineV2(workspaceAddr, workspaceSize, executor, stream);
aclrtSynchronizeStreamWithTimeout(stream, 10000);
```

### Dispatch → FFN → Combine 完整流程

```cpp
// 1. Dispatch：分发 Token
aclnnMoeDistributeDispatchV2GetWorkspaceSize(x, expertIds, ..., &wsSize1, &executor1);
aclrtMalloc(&ws1, wsSize1, ACL_MEM_MALLOC_HUGE_FIRST);
aclnnMoeDistributeDispatchV2(ws1, wsSize1, executor1, stream);

// 2. FFN：各专家前向计算
// expertOutput = FFN(expandXOut)  // 用户自行实现

// 3. Combine：加权回收
aclnnMoeDistributeCombineV2GetWorkspaceSize(expandX, expertIds, assistInfo, ..., &wsSize2, &executor2);
aclrtMalloc(&ws2, wsSize2, ACL_MEM_MALLOC_HUGE_FIRST);
aclnnMoeDistributeCombineV2(ws2, wsSize2, executor2, stream);
aclrtSynchronizeStreamWithTimeout(stream, 10000);
```

---

## 14. 关键源码文件索引

| 文件 | 核心内容 |
|------|---------|
| `moe_distribute_combine_v2.h` | Combine 主 Kernel 类模板——Process 流水线、ProcessExpert 加权求和、WaitDispatch 等待逻辑 |
| `moe_distribute_combine_v2_quant.h` | 量化/反量化模块——Int8DequantProcess、Int8DequantProcessA5（MicroAPI 融合）、DeQuantMxFp8 |
| `moe_distribute_combine_v2_layered.h` | Layered (RDMA+IPC) 实现——GM2IPC/WaitIPC/SumToWindow/AlltoAllServerDispatch 流水线 |
| `moe_distribute_combine_a2.h` | A2 架构 Kernel 实现 |
| `moe_distribute_combine_a2_layered.h` | A2 架构 Layered Kernel |
| `moe_distribute_combine_a2_layered_aicpu.h` | A2 架构 AICPU Layered Kernel |
| `arch35/moe_distribute_combine_arch35.h` | A5（3510）架构 Kernel 实现 |
| `moe_distribute_combine_v2_tiling.h` | Tiling 数据结构定义 |
| `op_host/moe_distribute_combine_v2_tiling.cpp` | Tiling 切分逻辑 |
| `op_host/moe_distribute_combine_v2_infershape.cpp` | 输出 shape 推断 |
| `op_api/aclnn_moe_distribute_combine_v2.cpp` | aclnn V2 API 实现 |
| `README.md` | 完整算子规格、参数说明、约束条件 |
| `docs/aclnnMoeDistributeCombineV2.md` | V2 API 文档 |

---

## 15. 总结

MoeDistributeCombineV2 的设计哲学是**让回收不再是通信的瓶颈**：

- **Device 自治 RDMA 回收**：AIV 直接驱动 RDMA 写回数据，消除 Host 介入的同步和调度延迟
- **Staggered Send 无竞争写入**：`(loop + epRankId_) % sendCntNum_` 策略让多核并行写入窗口时互不冲突，避免数据竞争和性能退化
- **双缓冲隐式同步**：通过 bufferChosen 翻转 + COMBINE_STATE_WIN_OFFSET 偏移隔离，Dispatch 和 Combine 交替使用两块缓冲区，无需全卡 Barrier
- **通信量化压缩**：commQuantMode 支持 INT8/MXFP8 量化通信，A5 MicroAPI 融合反量化+权值累加+写回，将三步操作压缩为一条寄存器级流水线
- **弹性伸缩动态适配**：elasticInfoOptional 支持动态缩卡场景的 rank 重映射，保障集群容错性

整套方案从回收架构（Device 自治 RDMA）到写入策略（Staggered Send）到同步机制（双缓冲偏移隔离）到数据压缩（通信量化+MicroAPI 融合），每一层都在为最小化回收延迟服务。与 DispatchV2 的"分发半程"对称，CombineV2 完成了"回收半程"，两者共同将 MoE 专家并行推理的通信瓶颈转化为计算流水的一部分。