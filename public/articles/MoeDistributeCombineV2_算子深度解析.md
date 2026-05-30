# MoeDistributeCombineV2 算子深度解析：MoE 推理的回收半程

## 1. 算子定位

CombineV2 是 DispatchV2 的对称算子，负责将各专家 FFN 的输出按路由权重加权合并还原。Dispatch 发出去，Combine 收回来——两者构成闭环。

## 2. 数学公式

**无 TP 域**

$$xOut = \sum_{k=0}^{K-1} expertScales[i,k] \times expandX[i,k] + sharedExpert[i]$$

**有 TP 域**

$$
\begin{aligned}
rsOut &= \text{ReduceScatterV}(expandX) \\
ataOut &= \text{AllToAllV}(rsOut) \\
xOut &= \sum_k expertScales_k \times ataOut_k + shared
\end{aligned}
$$

## 3. 三步核心流程

### 3.1 逆向 AllToAll

AllToAllV 逆向回收各卡上经专家 FFN 处理后的数据。方向与 Dispatch 相反：Dispatch 是散出去，Combine 是收回。

### 3.2 Token 重排

AllToAll 收回来的 Token 按专家编号排列，需要重排恢复原始 Token 顺序。`assistInfo` 提供映射关系：

```
TokenAddr = WinInGM + rankSize × rank + expertOffset × H + expandIdx × H
```

其中：
- `WinInGM`：WinIn 的 GM 基地址
- `rankSize`：每 rank 在 WinIn 中的段长
- `rank`：Token 来源的 rank ID
- `expertOffset`：专家在窗口内的偏移
- `expandIdx`：该 Token 在专家内序号

### 3.3 加权求和

每个 Token 加权求和：

$$
xOut[i] = \sum_{k=0}^{K-1} expertScales[i,k] \times expandX[i,k] + sharedExpertX[i]
$$

## 4. 地址定位公式

Combine 在 WinIn 中定位某 Token 的地址：

```
addr = WinInBase
      + rank × rankSizeWin      // rank 偏段起始
      + expertOffset(expertId)    // 该专家在窗口内的偏移
      + expandIdx × H             // Token 在专家内的序号 × 隐层维度
```

## 5. 窗口 (Window) 组织

WinIn / WinOut 均分 `worldSize` 段，每个段属于一个 rank，段内按 expert 分区。在 Combine 读取时，按 rank 段找到对应的 Token 数据。

```
WinIn
┌────────┬────────┬────────┬────────┐
│ Rank 0 │ Rank1  │ Rank2  │ Rank 3│
├────────┼────────┼────────┼────────┤
│ E0  E1 │ E0 E1 │ E0  E1 │ ...
└──────────────────────────────────────┘
```

## 6. 双缓冲同步

Combine 与 Dispatch 共享同一套 WinIn/WinOut双缓冲。每帧交替使用 buffer 0/1，靠 `bufferChosen` bit 标识当前使用哪块。Dispatch-Combine 交叉使用两块缓冲区保证读写互不冲突。

**翻转规则**：
- Combine 完成后将 bufferChosen ^= 1
- 下一次 Dispatch 时使用新缓冲区

## 7. 与 Dispatch 的数据流闭环

| Dispatch 输出 | Combine 输入 | 作用 |
|---|---|---|
| expandXOut | expandX | 各专家 FFN 输出 |
| assistInfoForCombineOut | assistInfo | Token 重排映射 |
| epRecvCountsOut | epSendCounts | 各卡 Token 数量 |

## 8. 增益分析

| 改进项 | V1 (Host 调度) | V2 (AIV+AICPU) |
|---|---|---|
| 调度方式 | Host编排 | Device 端 AIV/AICPU 自驱 |
| 同步机制 | Host-barrier | 双缓冲 ping-pong |
| 时延 | O(n) RTT | 1–2 μs AICPU 驱轮 |
| 通信聚合 | 每次 1/N 效率 | batchWrite 一次性批量 |

V2 的关键增益：

- **消除 Host 瓶颈**：AIV+AICPU 在 device 侧完成完整通信调度，Host 不介入
- **双缓冲并行**：读写交替不冲突，吞吐接近理论峰值
- **RDMA 通信**：一个 BatchWrite 下发全部数据，减少软件开销

## 9. 调用代码

```cpp
// 1. Dispatch
aclnnMoeDistributeDispatchV2(&workspace, ...);

// 2. 执行专家 FFN
// expertOutput = FFN(expandX)

// 3. Combine
aclnnMoeDistributeCombineV2(&workspace, ...);
```

## 10. 参数总结

| 参数 | 形状 | 作用 |
|---|---|---|
| expandX | (max(tp,1)×A, H) | 各卡 FFN 处理后的 token |
| assistInfoForCombine | (A×128,) | token 重排映射 |
| epSendCounts | 1D | 来自 Dispatch 的 epRecvCounts |
| expertScales | (BS, K) | topK 路由权 |

输出 `xOut` 形状为 `(BS, H)` 。

## 11. 小结

CombineV2 是 MoE 推理闭环的回收端。它与 DispatchV2 对称，前者"发出去"，Combine "收回来"。二者配合完成：Dispatch→FFN→Combine 三步循环。

| | Dispatch | Combine |
|---|---|---|
| 方向 | 散出去 | 收回来 |
| 输入 | x, expertIds | expandX, assistInfo, epSendCounts |
| 输出 | expandXOut, assistInfo | xOut |