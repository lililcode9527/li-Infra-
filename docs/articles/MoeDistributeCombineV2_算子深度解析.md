# MoeDistributeCombineV2 算子深度解析：MoE 推理的回收半程

## 1. 算子概述

MoeDistributeCombineV2 是 MoE 推理闭环的"回收"算子。与 Dispatch（分发出去）形成对称：Dispatch 分发 Token 到各专家，Combine 则把各专家的输出收回来，加权合并还原成最终输出。

如果把 Dispatch 比作"把信件送出去"，Combine 就是"收件人收到后再把回信收回来、汇总后合并输出"。

## 2. 计算公式

### 无 TP 域（纯 EP 通信）

$$
xOut = \sum_{k=0}^{K-1} expertScales_{i,k} \times expertOutput_{i,k} + sharedExpertX
$$

### 有 TP 域

$$
rsOut = ReduceScatterV(expandX) \\
ataOut = AllToAllV(rsOut) \\
xOut = \sum_k expertScales \times ataOut + sharedExpertX
$$

## 3. Combine 的三步核心流程

### 3.1 逆向 AllToAll

Dispatch 发出 Token → 各专家处理后 → Combine 将各卡的结果收集回原始卡的行中。通信方向与 Dispatch 相反，是逆向的 AllToAllV。

### 3.2 Token 重排

利用 assistInfo 对 AllToAll 后的输出做重排，恢复到 Dispatch 之前原始 Token 顺序。

### 3.3 加权求和

对每条 Token，把 K 个专家的输出乘以路由权重 expertScales 后求和，加上共享专家输出，得到最终输出。

```
xOut[i] = Σ(expertScales[i,k] × ataOut[i,k]) + sharedExpertX[i]（如有）
```

## 4. 输入输出关系

| Combine 输入 | 来源 |
|---|---|
| expandX | Expert FFN 处理后的输出，或 Dispatch 的 expandXOut 经过 expert FFN |
| assistInfo | Dispatch 输出的 assistInfoForCombineOut |
| epSendCounts | Dispatch 的 epRecvCounts |
| expertScales | 路由 softmax 后的权重 |
| sharedExpertX (可选) | 共享专家输出 |

| 输出 | 说明 |
|---|---|
| xOut | 最终输出，维度 (BS, H) |

## 5. 与 Dispatch 的闭环

Combine 的输入全都来自 Dispatch 的输出，经过 FFN 后再输入 Combine，形成闭环：

Dispatch → expandXOut → Expert FFN → Combine

```
           Token → Dispatch → FFN → Combine → 输出
          ──────────────────────────────
```

Combine 不能单独存在，需与 Dispatch 配对。

## 6. 关键参数

| 参数 | Shape | 说明 |
|---|---|---|
| expandX | (A, H) | 经专家FFN 处理后的数据 |
| assistInfo | (A×128) | 来自 Dispatch 输出，包含 Token 重排映射 |
| epSendCounts | 1D | 来自 Dispatch 的 epRecvCountsOut，标识本卡向其他卡发送的 Token 数量 |
| expertScales | (BS, K) | 路由权重矩阵 |
| sharedExpertX | (BS, H) | 共享专家输出（可选） |

## 7. TP域通信

若模型使用了 TP(张量并行), 须在 Combine 之前先做 ReduceScatterV，再 AllToAll，简化数据流。

```
expandX → ReduceScatter → AllToAllV → weighted sum → xOut
```

## 8. V1 vs V2 进化

|  | V1 | V2 |
|---|---|---|
| 辅助数据 | expandIdx (BS*K) | assistInfo (A×128) 更丰富 |
| 驱动模型 | Host 侧编排 | Device 侧 RDMA 直驱 |
| 同步机制 | Host 点检 | 双缓冲硬件轮换 |

## 9. HCCL_BUFFSIZE

Combine 与 Dispatch 共享 HCCL_BUFFSIZE，要求一致。

## 10. 双缓冲同步

Combine 同样采用双缓冲同步：Buffer0 / Buffer1 轮转，以 bufferChosen 位标识，通过 buffer 紧密周转，避免与 Dispatch 的缓冲竞争。

## 11. 小结

Combine 是 Dispatch 的闭环半程。Dispatch 分散出去，Combine 把专家结果收回来，形成 "发送 → FFN 计算 → 收回 → 加权合并" 的闭环。两者不可拆分使用。