# MoeDistributeCombineV2 算子深度解析：MoE 推理的回收半程

## 1. 算子定位：MoE 推理闭环的回收端

在 MoE（混合专家）推理流程中，Dispatch 负责"发出去"，Combine 负责"收回来"——把各专家 FFN 处理后的输出按权重加权还原为最终结果。CombineV2 与 DispatchV2 成对出现，形成完整闭环。

## 2. 数学公式

**无 TP 域：**

$$xOut = \sum_{k=1}^{K} \text{expertScales}_k \times \text{expertOutput}_k + \text{sharedExpert}$$

**有 TP 域：**

$$
rsOut = ReduceScatterV(expandX) \\
ataOut = AllToAllV(rsOut) \\
xOut = \sum_{k} scales_k \times ataOut_k
$$

## 3. Combine 的三步核心流程

### 3.1 逆向 AllToAll

各专家 FFN 的输出汇总回来。逆向的 AllToAll 通信，把离散在各卡上的专家输出集中回本卡。

### 3.2 Token 重排

利用 Dispatch 传来的 `assistInfo` 和 `epSendCounts` ，将 AllToAll 后的数据重排到正确位置。

### 3.3 加权求和

每个 Token 有 K 条专家路径，各条路径乘上 softmax 权重（expertScales）做加权，得到最终输出 xOut。

## 4. Token 重排（ReorderToken）

Combine 的 Token 重排通过 Dispatch 产出的 assistInfoForCombine 完成。该信息编码了 Token 在分布（Dispatch）时的原始位置，Combine 按此映射重排。

**核心公式**：

```
TokenAddr(i) = windowInGM + rankSize × rank + expertWindowOffset(expertId) + expandIdx[i] × axisH
```

## 5. 加权求和公式

$$
xOut_i = \sum_{k=0}^{K-1} expertScales[i,k] \times expandX[i,k]
$$

如果有共享专家，加 $sharedExpert[i]$ 。 `expandX` 是经过专家 FFN 后的输出；expertScales 则来自路由的 softmax 结果。

## 6. 与 Dispatch 的闭环关系

| | Dispatch | Combine |
|---|---|---|
| 方向 | 把 Token 发出去 | 把结果收回来 |
| 通信 | AllToAllV(正向) | AllToAllV(逆向) |
| 缺一方不成闭环 | 必须配对 | 必须配对 |
| 共用 HCCL Buffer | 双缓冲 | 双缓冲 (A/B 交替) |

Combine 的全部输入都是从 Dispatch 接收过来的中间产物：
- `assistInfo →` by Dispatch 输出
- epSendCounts → Dispatch 的 epRecvCountsOut
- expandX → FFN 处理后的数据

## 7. 增益分析

### 7.1 相比传统 Host 端调度的收益

| 方面 | 传统方案 | V2 方案 |
|---|---|---|
| Host/Device | Host编排通信 | Device 端 RDMA 直驱 |
| 通信延迟 | Host串行处理 | AIV+AICPU 件驱动，延迟减1～2个 RTT |
| 同步开销 | Host 全局应登射 | 双缓冲乒乓交替，延迟 O(1) |
| 推理吞吐 | ~1000 token/s | ~1.5-2x 提升 |

### 7.2 关键增益

- **通信延迟缩短**：RDMA全互联，避免 Host 仲裁；
- **计算通信重叠**：Combine 加权计算和 AllToAll 通信流水；
- **风险减缓**：双缓冲 Flip避免写读竞争。

## 8. 参数与输入输出对照表

| 参数 | 形状 | 说明 |
|---|---|---|
| expandX | (max(tp,1)×A, H) | Expert FFN 输出结果 |
| assistInfo | (A×128) | Dispatch 产出的映射信息 |
| expertScales | (BS, K) | 路由概率（softmax） |
| epSendCounts | 1D | Dispatch 的 epRecvCounts |
| sharedExpertX | (BS, H) | 共享专家结果 (可选) |

输出 xOut: (BS, H)

## 9. HCCL_BUFFSIZE 和约束

- 两算子共享同一个 HCCL_BUFFSIZE，必须同时适配两种算子。
- 通信域不允许有其他算子。
- 需保证 HCCL_BUFFSIZE > 按公式 ≥（$BS×ep × min(localExpertNum,K)×H×2$ B）。