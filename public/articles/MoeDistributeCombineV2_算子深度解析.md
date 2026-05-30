# MoeDistributeCombineV2 算子深度解析：MoE 推理的"回收"半程

## 1. 算子概述

MoeDistributeCombineV2 是 DispatchV2 的对称算子，负责将各专家处理后的结果加权合并，还原成与原始 Token 对应的输出。Dispatch 将 Token 分发出去，Combine 将它们收回来。

**完整闭环**: Dispatch(分发) → 专家 FFN 计算 → Combine(回收)

## 2. 计算公式

**无 TP 域:**

```
xOut = Σ(k=1..K) expertScales[i,k] × expertOutput[i,k] + sharedExpertX
```

**有 TP 域:**

```
rsOut = ReduceScatterV(expandX)
ataOut = AllToAllV(rsOut)
xOut = Σ scales × ataOut + sharedExpert
```

## 3. 与 Dispatch 的数据传递关系

| Combine 输入 | 来自 Dispatch 的输出 |
|---|---|
| expandX | Dispatch 输出 → FFN 计算 → Combine 输入 |
| assistInfo | Dispatch 的 assistInfoForCombineOut |
| epSendCounts | Dispatch 的 epRecvCounts → epSendCounts |
| expertScales | 可来自 Dispatch expandScalesOut 或模型路由 |
| sharedExpertX | 由外部推理框架提供 |

## 4. Combine 的三步核心流程

### 4.1 逆向 AllToAll

AllToAllV 将各卡上的专家输出结果收回到原 Token 所在的卡。方向与 Dispatch 相反：Dispatch 是"散"，Combine 是"聚"。

### 4.2 Token 重排

使用来自 Dispatch 输出的 assistInfo 重组 Token 顺序，恢复 Dispatch 之前的 Token 顺序。

### 4.3 加权求和

```
xOut[i] = Σ_k expertScales[i,k] × expandX_reordered[i,k]
```

如有共享专家(sharedExpert)，还需加上共享专家的加权输出。

## 5. 参数详解

| 参数 | 类型 | 说明 |
|---|---|---|
| expandX | (A,H) | 专家 FFN 处理后的 token 数据 |
| assistInfo | (A*128) | 来自 Dispatch 的位置信息 |
| epSendCounts | 1D | Dispatch 的 epRecvCounts 输出 |
| expertScales | (BS, K) | 路由 softmax 概率 |
| sharedExpertX (可选) | (BS, H) | 共享专家输出 |
| xOut | (BS, H) | 加权后的最终输出 |

## 6. TP 域下 ReduceScatterV → AllToAllV

有 TP域 时的流程: ReduceScatter → AllToAll → weightedSum。

ReduceScatter 将分布在各 TP 卡上的数据先在 TP域内聚合，然后 AllToAll 交换跨卡的 expert 分布。

## 7. 与 V1 的演进对比

| 特性 | V1 | V2 |
|---|---|---|
| 辅助数据 | expandIdx | assistInfo (A*128) |
| 同步方式 | Host调度 | AIV+AICPU 设备端驱动 RDMA |
| 通信 | Host端调度 | 设备端 RDMA 直驱 |
| 缓冲同步 | Host侧同步 | 设备侧双缓冲 |

## 8. 双缓冲同步机制

Combine 与 Dispatch 使用相同的双缓冲机制：
- Combine 也使用 Buffer 0 / Buffer 1 交替，在写出时切换 buffer 防止读写竞争。
- 数据一致性依靠 bufferChosen bit 标识当前使用哪一半空间。

## 9. 调用示例

```cpp
// 1. Dispatch
aclnnMoeDistributeDispatchV2(...); // 算子一次调用
// 2. Expert FFN 计算
// 3. Combine
aclnnMoeDistributeCombineV2(...);
```

## 10. HCCL_BUFFSIZE 要求

与 Dispatch 的 HCCL_BUFFSIZE 计算公式相同，Combine 和 Dispatch 共享相同的 HCCL_BUFFSIZE。

## 11. 小结

CombineV2 与 DispatchV2 构成 MoE 推理的全闭环。Dispatch 发出去，Combine 收回来，二者缺一不可。