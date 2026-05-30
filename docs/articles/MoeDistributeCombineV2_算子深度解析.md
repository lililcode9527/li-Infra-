# MoeDistributeCombineV2 算子深度解析：MoE 推理的回收半程

## 1. 算子定位

CombineV2 是 DispatchV2 的对称算子。Dispatch 把 Token "发出去"，Combine 把结果"收回来"。两者构成 MoE 推理闭环：Dispatch → FFN → Combine → 输出。

## 2. 数学公式

### 无 TP 域

$$xOut[i] = \sum_{k=0}^{K-1} \text{expertScales}[i,k] \times \text{expertOutput}[i,k] + \text{sharedExpert}[i]$$

### 有 TP 域

$$
\begin{aligned}
rsOut &= \text{ReduceScatterV}(expandX) \\
ataOut &= \text{AllToAllV}(rsOut) \\
xOut &= \sum_k \text{expertScales}_k \times ataOut_k + \text{shared}
\end{aligned}
$$

## 3. Token 重排原理

AllToAllV 收回来的 Token 按专家编号排列，需要重排回原始 Token 序号。

地址计算公式：

```
TokenAddr = WinInBase
           + rank * rankSize  // 找到对应 rank 的窗口
           + expertOffset(expertId) * H   // 专家在窗口内的偏移
           + expandIdx * H    // Token 在专家内的序号 × hidden size
```

重排时，Combine 按 `assistInfo` 中的映射恢复原始 Token 序号顺序。assistInfo 的每条记录包含 Token 原始序号和 WinIn 内地址映射。

## 4. 窗口布局

WinIn/WinOut 被均分为 worldSize 个窗口，每个 rank 占一段。每个 rank 窗口内按专家再分小区：

```
WinIn
┌────────┬────────┬────────┬────────┐
│ rank 0 │ rank 1 │ rank 2 │ rank 3 │
│  E0 E1 │ E0 E1  │ E0 E1  │ ...
└───────────────────────────────────────┘
```

## 5. 双缓冲同步机制

Combine 与 Dispatch 共享同一对缓冲区（Buffer A / Buffer B），靠 bufferChosen bit 交替使用：

- Buffer A 被 Dispatch 写入，Combine 读取 Buffer B
- 下一帧互换
- 翻转点：每个 EP cycle 结束时 bufferChosen ^= 1

这确保读写永不冲突。

## 6. Combine 的三步核心流程

| 步骤 | 操作 | 说明 |
|---|---|---|
| 1 | 逆向 AllToAllV | 把各卡的专家结果收集回原始卡 |
| 2 | Token Reorder | 用 assistInfo 把 Token 重排到原始序列 |
| 3 | Weighted Sum | expertScales × expertOutput + sharedExpert |

## 7. 加权求和

对于每个 Token i，最终输出：

$$
xOut[i] = \sum_{k=0}^{K-1} \text{expertScales}[i,k] \times \text{expandX}[i,k]
$$

若有共享专家，则追加：

$$
xOut[i] += \text{sharedExpertX}[i]
$$

## 8. 增益分析

| 项目 | V1 | V2 |
|---|---|---|
| 调度 | Host编排 | Device端 AIV+AICPU 驱动 |
| 通信 | Host → AllToAll | RDMA BatchWrite |
| 同步 | Host barier | 双缓冲 ping-pong |
| 延迟 | 含 Host 开销 100+us | ~1-2us |
| 吞吐提升 | — | 20-40% |

核心增益点：

- **消除 Host 侧瓶颈**：Device 端 AIV+AICPU 完成全程，Host 不参与
- **双缓冲避免读写冲突**：读写交替使用 Buffer A/B
- **BatchWrite 批量通信**：所有 Token 一次提交，减少调度轮次

## 9. 调用示例

```cpp
// Dispatch
aclnnMoeDistributeDispatchV2(...);
// FFN compute
aclnnMoeDistributeCombineV2(...);
```

## 10. 与 Dispatch 的数据流闭环

| Dispatch 输出 | Combine 输入 | 用途 |
|---|---|---|
| expandXOut | expandX | FFN输出 |
| assistInfoForCombineOut | assistInfo | Token 重排映射 |
| epRecvCountsOut | epSendCounts | 通信量参数 |

## 11. 小结

CombineV2 与 DispatchV2 构成 MoE 推理闭环。Dispatch "发出去"，Combine "收回来"。二者共享同一 HCCL Buffersize，必须配对使用。

| | Dispatch | Combine |
|---|---|---|
| 方向 | 散出去 | 收回来 |
| 关键输入 | x, expertIds | expandX, assistInfo, expertScales |
| 关键输出 | expandXOut, assistInfo, epRecvCounts | xOut |