# MoeDistributeCombineV2 算子深度解析：MoE 推理的回收半程

## 1. 算子定位：MoE 推理闭环的"回收"半程

在 MoE 模型的推理流程中，Dispatch 负责"把 Token 分发到专家"（发射），Combine 则负责"把各专家处理完的结果收回来"（回收）。两者构成一个完整闭环：Dispatch 发出去，Combine 收回来。

CombineV2 是 DispatchV2 的对称算子，将各专家 FFN 的输出按权重加权还原为最终输出。整个流程为：

```
Token → Dispatch → 各专家 FFN → Combine → 输出
```

## 2. 核心原理与数学公式

### 2.1 无 TP 域

$$
xOut = \sum_{k=1}^{K} \text{expertScales}_k \times \text{expertOutput}_k + \text{sharedExpertX}
$$

### 2.2 有 TP 域

$$
rsOut = \text{ReduceScatterV}(expandX) \\
ataOut = \text{AllToAllV}(rsOut) \\
xOut = \sum_{k=1}^{K} \text{expertScales}_k \times \text{ataOut}_k + \text{sharedExpertX}
$$

## 3. 三步核心流程

Combine 的计算流程可拆为三步：

### 3.1 逆向 AllToAll—— 把结果收回来

Dispatch 阶段的 AllToAllV 是"分出去"（散），Combine 中的 AllToAllV 方向相反（收），将各卡上的专家输出汇聚回来。

因为 Dispatch 已经按专家分配好了，Combine 只需做逆向交换即可。

### 3.2 Token 重排

在逆向 AllToAll 之后，各卡收到来自不同专家的 Token，但这些 Token 的顺序与原始顺序不对齐。Combine 必须依据 Dispatch 产出的 `assistInfoForCombine`，把各 Token 重新排回与原始 Token 对应的顺序。

**assistInfo 中记录了哪些 Token 在 Dispatch 时被送到哪些专家/卡上，这个信息被复用来恢复排序。

### 3.3 加权求和

每个 Token 对应 K 个专家，其输出为 `expandX[i, k]`，路由权重（softmax值）为 `expertScales[i,k]`，加权求和即：

$$
xOut[i] = \sum_{k=1}^{K} \text{expertScales}[i, k] \times \text{expandX}[i, k] + \text{sharedExpert}[i]
$$

如果有 shared expert (sharedExpertX），额外加上。

## 4. Combine 的详细步骤

### 4.1 Token 重排（ReorderToken）

Combine 在 Fullmesh 模式下的 Token 重排，依赖 Dispatch 传递的辅助信息。Dispatch 输出的 `assistInfoForCombineOut` 记录了每个 Token 被哪个专家处理、在通信域中的分布顺序。Combine 解析这份映射，把 AllToAllV 返回的数据重排到原始 Token 序。

### 4.2 窗口数据流

Combine 的关键数据流：```
expandX (专家处理后的结果) → AllToAllV(逆向) → Reorder → Weighted Sum → xOut
```

### 4.3 通信模式

| 场景  | 通信流 |
|---|---|
| 纯 EP域 AllToAllV(逆向)→无TP域 Combine 最简单  |
| EP域+TP域 ReduceScatter+AllToAllV→Weighted Sum  |

## 5. 关键参数解析

### 5.1 输入参数

| 参数 | 类型/形状 | 说明 |
|---|---|---|
| expandX | `(max(tpWorldSize, A, H)` | 经各专家 FFN 处理后的输出 |
| assistInfoForCombine | `(A*128,) int32` | 由 Dispatch 输出的定位信息 |
| epSendCounts | `(epWorldSize×localExpertNum)` | 来自 Dispatch 的 epRecvCounts |
| expertScales | `(BS, K)` | 路由权重，softmax 后的 gate 输出 |
| sharedExpertX (可选) | `(BS, H)` | 共享专家结果 |

### 5.2 输出

- `xOut`，形状 `(BS, H)`，类型与 expandX 一致。

### 5.3 关键约束

- **必须**与 DispatchV2 配对使用，且 ep 通信域相同。
- 两算子的 HCCL_BUFFSIZE 一致。
- `expertScales` 必须与 Dispatch 中的 路由决策一致。

## 6. 与 Dispatch 的对应关系

Combine 和 Dispatch 在通信域、HCCL Buffer 大小、AIV/AICPU 配置上完全对应：

| 维度 | Dispatch | Combine |
|---|---|---|
| 方向 | 分发出去 | 回收回来 |
| 通信类型 | AllToAllV (Send) | AllToAllV (Reverse) |
| 缓冲 | epRecvCounts → 发送字节数 | epSendCounts → 接收大小 |
| 后处理 | 计算辅助信息 | 加权求和 |
| 数据依赖 | x → output → expandX | expandX → xOut |

两者虽然方向相反，但依赖同一组 HCCL buffer，共享双缓冲同步机制。由于 Combine 是回收阶段，如果 Dispatch 在 AIV 流程后处理结束，Combine 则是对 AICPU 反传处理后的回收。

## 7. 增益分析

CombineV2 相对旧版 Combine 的提升主要体现在以下几个方面：

### 7.1 通信开销减少

传统 Combine 分三步：Host 侧编排+MPI→host → device数据拷贝，原Combine V1.0 由 Host 控制 AllToAllV，Host 参与调度。V2 采用 AIV+AICPU 端控制 RDMA 通信，省去 Host 侧的编排环节，减少拷贝开销和调度开销。

### 7.2 避免Host侧调度

V1 所有 AllToAll 通信由 Host 发出，延迟与调度开销过大；V2 依赖 设备 AICPU+AIV 自组织 RDMA 数据传输，免除 Host 中间干预，减少了一次 Host→Device的拷贝与等待。

### 7.3 双缓冲机制

通过 Buffer 0/Buffer 1乒乓操作（Ping-Pong Buffer），使相邻两次 Dispatch-Combine 循环交替使用两块共享内存，减少跨步的时间浪费。

### 7.4 整体系统级的收益

在典型 DeepSeek V3 模型结构 (256 专家) 下，整个推理使用 Dispatch + Combine 系列算子，可省掉 Host 侧传统调度（约 30% 的 AllToAll 路径时间），吞吐提升可达20~40%。

## 8. 完整流程示例

以 DeepSeek-V3 为例，一个Token从一个MoE Layer 的完整推理步骤：

1. **Dispatch**: 每个卡上每个Token经 Top-K 路由得到 K 条路径，计算 Token x Expert 矩阵，并筛选目标专家。
2. **AIV+AICPU**: 将 Token 按各自路由对应的专家卡发送出去，采用 RDMA AllToAll 指令。
3. **Expert FFN: All** received Tokens 進る所在卡上的所有 Expert 计算。
4. **Combine**: 将 Expert 输出通过逆向 AllToAllV 归集回来。
5. **加权求和**: 得到 xOut,每个Token的K个輸出用路由权重加权求和返回至最终输出。

## 9. 双缓冲同步与安全性

Combine 遵守和 Dispatch 一样双缓冲体系: 一块缓冲区用做下一次 Dispatch 的 receive buffer, 另一块就是正在 Forward 中的，对应本层 Combine 的提供方。Buffer 交换的前提是 Buffer 0 中的数据已经用完、Buffer 1 空闲,否则流水线将中断。

总结来说 Combine V2 的意义在于：

1. 消灭 Host 角色，设备上直接 RDMA 通信 → lower latency
2. Double-buffer 读写 Separation -> No cross-run ping-pong
3. 对通信边界使用 AIV/AICPU 的 PDMA全互联，不再需要Host协调。

## 10. 代码调用方式

```cpp
// Combine的调用与 Dispatch 相对应
aclnnMoeDistributeCombineV2GetWorkspaceSize(&workspaceSize, ...);
aclnnMoeDistributeCombineV2(workspace, workspaceSize, executor, stream);
```

且，Combine 的输入参数需要与 Dispatch 结果对齐：assistInfo与epSendCounts 必须来自于该 token 单次 Dispatch 的输出。