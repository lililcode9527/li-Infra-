# Lightning Indexer 算子深度解析：Tiling 切分、流水线优化与 Cube/Vector 并行

## 1. 算子概述：什么是 Lightning Indexer

Lightning Indexer 是华为 Ascend NPU 上针对 DeepSeek 系列 DSA（Dynamic Sparse Attention）架构中 **Indexer 阶段**的高性能算子实现。它的核心功能是：给定 Query Q、Key K 和分组权重 W，计算稀疏注意力中需要关注的 KV 位置索引。

### 计算流程

Lightning Indexer 的计算分为三个阶段：

1. **Q×K^T MatMul（Cube 计算）**：对每个 query token，计算其与所有 key token 的注意力分数矩阵 S = Q @ K^T，形状为 [G×S1, S2]，其中 G = N1/N2 是 GQA 组比
2. **加权 ReLU ReduceSum（Vector 计算）**：对每行分数应用 ReLU 激活，乘以对应组的权重标量 W[g]，然后在 G 维度上做 reduce-sum，得到每行一个长度为 S2 的分数向量
3. **TopK 选择（Vector 计算）**：从 S2 维度的分数向量中选出 top `sparseCount` 个位置，输出它们的索引和可选的分数值

```
score_row = Σ_g (ReLU(S[g,:]) × W[g])     # 加权聚合
output = TopK(score_row, sparseCount)       # 稀疏选择
```

这正是 DeepSeek DSA 中 "Compressor/Indexer" 的硬件实现——用权重做分组门控（ReLU + 缩放），再用 TopK 精准定位需要 attend 的 KV 位置，后续才对选出的位置做真正的稀疏注意力计算。

**输出**：`sparse_indices` [B,S1,N2,TopK]（或 TND 布局下 [T,N2,TopK]）以及可选的 `sparse_values`。

---

## 2. Tiling 切分策略

### 2.1 Tiling 数据结构

LITilingData 控制整个算子的计算分区，核心字段包括：

| 字段 | 说明 |
|------|------|
| `bSize` | Batch 维度 |
| `n2Size` | KV 头数（当前固定为 1） |
| `gSize` | GQA 组比 G = N1/N2 |
| `s1Size` | Query 序列长度 |
| `s2Size` | KV 序列长度 |
| `sparseCount` | TopK 数量（≤2048 基础路径，或 1024 倍数最大 8192） |
| `usedCoreNum` | 使用的 AI Core 数量 |
| `blockSize` | Page Attention 块大小 |
| `sparseMode` | 0=无因果掩码，3=因果掩码 |
| `returnValue` | 是否输出分数值 |

### 2.2 基础块划分

Tiling 采用**基础块（Basic Block）策略**，将完整的 [G×S1, S2] 计算分解为可重复的小块：

**Arch35（Ascend 910B/910_93/950）基础块大小**：

| 维度 | topK ≤ 2048 | topK > 2048 |
|------|------------|-------------|
| M（G×S1） | 4×G（如 G=64 时 M=256） | 2×G（如 G=64 时 M=128） |
| S1 | 4 | 2 |
| S2 | 128 | 128 |
| Head Dim | 128（固定约束） | 128 |

**splitMFlag 优化**：当 G=64 且 topK≤2048 时，M 基础块为 256，但 L0 空间有限。此时 splitMFlag 将 M=256 的块拆成两个 M=128 的子块，分别做 fixpipe 输出，让两个 AIV 核各处理一半，避免跨核同步开销。

**Arch22（Ascend 910B 原版）基础块大小**：

| 维度 | topK ≤ 2048 | topK > 2048 |
|------|------------|-------------|
| M | 8×G | 动态计算（8192/sparseCount × 2 × G） |
| S1 | 8 | 动态计算 |
| S2 | 512 | 512 |

### 2.3 计算循环嵌套顺序

整个算子的循环结构为 **B×N2 → G×S1 → S2**：

- 外层循环遍历 Batch × KV 头
- 中层循环按 G×S1 基础块分步，每个块独立完成一次完整的 Cube+Vector 流水线
- 内层循环按 S2 基础块迭代，Query 在 S2 维度复用，Key 按 S2 基础块逐段加载

这种嵌套保证了 Query 数据在 S2 维度上的最大化复用，减少 GM→L1 的搬运次数。

---

## 3. 流水线优化：如何榨取 Cube 算力

### 3.1 五缓冲 L1 流水线

Arch35 Cube 端的 L1 缓冲布局为 **3 Key + 2 Query**，这是精心设计的五缓冲策略：

- **Key L1（3 缓冲）**：三缓冲允许 MTE2（GM→L1 数据搬运）在当前 Key 正被 MTE1（L1→L0 搬运）和 MatMul 消费时，同时加载下一个 Key tile。三缓冲比双缓冲多一个缓冲位，使得数据搬运和计算完全解耦，Key 搬运永远不会阻塞 MatMul
- **Query L1（2 缓冲）**：Query 在 S2 维度上完全复用——只在首次 S2 迭代时加载，后续 S2 块直接从 L1 读取。双缓冲足够：一个缓冲供当前 MatMul 消费，另一个供下次迭代预加载

```
时间轴：
       ┌─ MTE2 加载 Key[0] ─┬─ MTE2 加载 Key[1] ─┬─ MTE2 加载 Key[2] ─┬─ ...
Key L1:  [buf0]               [buf1]               [buf2]               循环轮转
       ┌──── MTE1→L0 + MatMul(Key[0]) ────┬── MTE1→L0 + MatMul(Key[1]) ──┬── ...
```

### 3.2 双缓冲 L0 流水线

L0A、L0B、L0C 各配 2 缓冲：

- **L0A/L0B 双缓冲**：当前 MatMul 在消费 L0A[0]/L0B[0] 的同时，LoadData 正在从 L1 把下一个 tile 搬进 L0A[1]/L0B[1]
- **L0C 双缓冲**：当前 MatMul 写入 L0C[0]，同时 Fixpipe 正在把 L0C[1] 的结果输出到 UB

这种 L0 双缓冲使得 MatMul → Fixpipe → LoadData 形成完整的软件流水线，Cube 算力利用率接近 100%。

### 3.3 Fixpipe 到 UB 的关键优化

Arch35 的 Fixpipe 输出目标为 UB（而非 GM），这有两个重要优化：

**1. dual_dst_ctl 分半写入**：Fixpipe 将 M 维度拆成两半（M/2 行），同时写入 UB 的两个半区。两个 AIV 核各自消费自己的一半，无需竞争同一块 UB 空间。

**2. Bank-Conflict-Free 布局**：Fixpipe 的 dstStride 设置为 `UB_BANK_DEPTH_STRIDE / sizeof(float) = 512/4 = 128`。Ascend UB 的存储结构为 2-bank × 8-group × 32B-block，stride=128 确保每一行分数落在不同的 bank group 上，Vector 读取时 G 维度并行访问不会产生 bank conflict。

对于 N ≤ 64（float 元素），使用单个 ND block 搬运；N ∈ (64, 128] 时拆成 2 个 ND block（nSize=64），遵守 256B bank 宽度约束。

### 3.4 Arch22 的 Cube 流水线深度解析

Arch22 的 Cube 流水线与 Arch35 有几个关键不同，这些差异源于硬件代际的约束：

**1. Fixpipe 输出到 GM workspace（而非 UB）**

Arch22 使用 `DataCopyCO12DstParams` 将 L0C 结果输出到 mm1ResGm workspace，参数配置：
- `nz2ndEn=true`：将 Cube 的 Fractal Z 格式（L0C 的 NZ 排布）转换为 ND（行优先）格式，方便 Vector 从 GM 按行读取。NZ→ND 转换由 Fixpipe 硬件自动完成
- `reluPre=1`：**硬件级 ReLU**——在 Fixpipe 管线中内置 ReLU 激活，L0C 的负值在输出到 GM 前直接被截断为 0。这意味着 Vector 端收到的数据已经是 ReLU 后的，DoScale 中只需做乘权和累加
- `dstStride = actualSingleProcessSInnerSizeAlign`：GM 中相邻行之间的 stride，对齐到 32B
- `srcStride = CeilAlign(s1gL0RealSize, 16)`：L0C 中相邻行的 stride，对齐到 16（BLOCK_CUBE_SIZE）
- `unitFlag = 0b11`：Fixpipe 三级流水完整执行

mm1ResGm workspace 使用双缓冲（`runInfo.loop % 2`），每个 AI Core 分配 `2 × mBaseSizeAlign × s2BaseSize × sizeof(float)` 空间。Cube 在奇偶循环交替写入两个缓冲区，Vector 交替读取。

**2. Nd2Nz 格式转换的细节**

Arch22 的 KeyNd2Nz 函数将 Key 从 GM 的 ND 格式转换为 L1 的 NZ 格式，有特别的设计：

- 按照两个 L0 分型（S2_BASIC_BLOCK_L0=128）在 L1 上排布 Key 数据，方便后续 MTE1 从 L1→L0 按分型加载
- `dstNzC0Stride` 的计算考虑了 L1 中两个分型的对齐：如果 Key 数据跨两个分型（s2L1RealSize > S2_BASIC_BLOCK_L0），则前半部分 stride=S2_BASIC_BLOCK_L0，后半部分 stride=CeilAlign(s2L1RealSize - S2_BASIC_BLOCK_L0, 16)
- 这种双分型排布使得 LoadKeyToL0b 可以按分型精确提取数据，无需跨分型读取

QueryNd2Nz 更简单：单分型排布，`dstNzC0Stride = CeilAlign(s1gL1RealSize, 16)`。

**3. LoadData3D vs LoadData2D 的混合使用**

Arch22 Cube 端混合使用 LoadData3D（Query L1→L0A）和 LoadData2D（Key L1→L0B）：

- **Query 用 LoadData3DV2**：3D 加载允许从 L1 的 NZ 格式中按滑窗方式提取子块，支持 `mStartPt`（行起始偏移）和 `padList[3]=255`（尾部填充不影响滑窗结果）等参数。Query 的 L1 排布比较复杂（`l1H = CeilDiv(s1gL1RealSize, 16)`，`l1W = 16`），需要 3D 加载来精确提取不同偏移位置的 Query tile
- **Key 用 LoadData2D**：Key 的 L1 排布按两个 L0 分型设计，每个分型内是连续的 NZ 格式，2D 加载足够精确提取。`repeatTimes = CeilDiv(s2L0RealSize, 16) × CeilDiv(headDim, 16)`

**4. 小矩阵的 PipeBarrier<PIPE_M> 保护**

在 ComputeL0c（Mmad）中有一个特殊判断：

```cpp
if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
    PipeBarrier<PIPE_M>();
}
```

当 MatMul 的 M×N 很小（如 M=16, N=128 时 (1×8)=8 < 10），MatMul 计算太快，L0C 结果可能还没写完就开始 Fixpipe，导致数据不一致。此时强制插入 PIPE_M 屏障，确保 Mmad 完成后再进入 Fixpipe。这是 Arch22 流水线的一个关键保护点——小矩阵场景下，软件流水线的自动重叠反而会造成数据竞争。

**5. Page Attention 模式**

Arch22 支持 Page Attention（PA）模式，通过 `KeyNd2NzForPA` 函数从 blockTable 索引 Key：

- 通过 `blkTableGm.GetValue(bIdx × maxBlockNumPerBatch + s2BlkId)` 获取物理 block 编号
- 计算物理偏移：`blockId × kCacheBlockSize × kHeadNum × headDim + s2BlkOffset × headDim`
- 按 block 边界切分搬运大小（`s2Mte2Size = min(remaining, blockSize - offset)`）
- 这使得 Lightning Indexer 可以在 KV Cache 使用分块存储（如 vLLM 的 PagedAttention）时正常工作

---

## 4. Cube 与 Vector 并行：双核协同如何跑满硬件

### 4.1 Arch35 的 1:2 Cube-Vector 并行架构

Ascend 910B/910_93 的每个 AI Core 包含 1 个 AIC（Cube）和 2 个 AIV（Vector）单元。Lightning Indexer 的 Arch35 设计采用 `KERNEL_TYPE_MIX_AIC_1_2` 模式，Cube 和两个 Vector 核并发执行。

**同步机制：QLI_SYNC_MODE4 四事件同步**

QLI_SYNC_MODE4 使用 4 组事件（event 0-3），配合偏移量 `AIV0_AIV1_OFFSET=16` 区分两个 AIV 核的事件通道：

```
时序图（单个 S2 基础块循环）：

Cube 端：
  1. 等待 Vector 释放 UB 缓冲：CrossCoreWaitFlag(CROSS_VC_EVENT + loop%2)
  2. 执行 MatMul + Fixpipe → UB[loop%2]
  3. 通知 Vector 数据就绪：CrossCrossSetFlag(CROSS_CV_EVENT + loop%2)

AIV0 端（处理 S1 上半）：
  1. 等待 Cube 写完：CrossCoreWaitFlag(CROSS_CV_EVENT + loop%2)
  2. 从 UB[(loop-1)%2] 读取 mm1Res，执行加权 ReLU ReduceSum + TopK
  3. 通知 Cube 可以复用缓冲：CrossCoreSetFlag(CROSS_VC_EVENT + loop%2)

AIV1 端（处理 S1 下半）：
  同 AIV0，但事件偏移 +16，处理 CeilDiv(curS1ProcNum, 2) 下半行
```

**Ping-Pong 双缓冲 UB**：Cube 写 mm1Res 到 UB[loop%2]，Vector 从 UB[(loop-1)%2] 读取上一轮数据。两者在不同缓冲区上同时工作，零等待。

**S1 维度拆分**：两个 AIV 核按 blockId%2 分工——AIV0 处理上半行 `CeilDiv(curS1ProcNum, 2)`，AIV1 处理下半行。各自写入 scoreGm workspace 的对应偏移位置，无需合并。

### 4.2 Arch22 的 Cube-Vector 顺序握手与 FIA_SYNC_MODE2

Arch22 采用完全不同的同步策略——Cube 和 Vector 不是并发执行，而是严格的顺序握手。这是由 Ascend 910B 原版的硬件约束决定的：Cube 的 Fixpipe 输出目标是 GM（而非 UB），数据必须经过 GM workspace 中转，因此无法实现 Ping-Pong UB 并发。

**FIA_SYNC_MODE2 两事件同步机制**

Arch22 使用 `FIA_SYNC_MODE2`（两事件同步），同步流程如下：

```
每个 S2 基础块循环的时序：

AIC（Cube）端：
  1. 等待上一轮 Vector 完成：CrossCoreWaitFlag(syncV1C1)  ← 确保上一轮 mm1Res GM 缓冲已被 Vector 消费
  2. 执行 MatMul + Fixpipe → mm1ResGm[loop%2]             ← 写入 GM workspace 双缓冲
  3. 通知 Vector 数据就绪：CrossCoreSetFlag<PIPE_FIX>(syncC1V1)

AIV（Vector）端：
  1. 等待 Cube 写完：CrossCoreWaitFlag(syncC1V1)
  2. 从 mm1ResGm[loop%2] 读取 → 加权计算 + TopK
  3. 通知 Cube 可以复用缓冲：CrossCoreSetFlag<PIPE_MTE2>(syncV1C1)
```

**关键差异：GM workspace 双缓冲 vs UB Ping-Pong**

Arch22 的 mm1ResGm workspace 采用双缓冲布局：

```
GM workspace: |Core0_mm1ResDB0|Core0_mm1ResDB1|Core1_mm1ResDB0|...|Core23_mm1ResDB0|Core23_mm1ResDB1|
              |vec1ResGm（LD中间结果）|vec1ParamGm（LD参数）|
```

每个 AI Core 分配 `2 × mBaseSizeAlign × s2BaseSize × sizeof(float)` 的 mm1Res 空间（双缓冲），Cube 在奇偶循环交替写入两个缓冲区，Vector 交替读取。但由于数据经过 GM（读写延迟约数百个 cycle），Cube 必须等 Vector 完全消费完上一个缓冲才能开始下一轮 MatMul，这与 Arch35 的 UB Ping-Pong 零等待模式形成鲜明对比。

**Fixpipe 硬件级 ReLU：nz2nd + reluPre**

Arch22 的 Fixpipe 使用 `DataCopyCO12DstParams` 参数：

- `nz2ndEn=true`：将 L0C 的 NZ（Fractal Z 形）格式转换为 ND（行优先）格式写入 GM
- `reluPre=1`：**硬件级 ReLU**——Fixpipe 管线内置 ReLU 功能，在 L0C→GM 转换过程中直接对输出值做 ReLU，省去了 Vector 端的一步 ReLU 计算
- `unitFlag=0b11`：3-unit flag 模式，确保 Fixpipe 三级流水完整执行

这意味着 Arch22 的 Vector 端收到的 mm1Res 数据**已经是 ReLU 后的结果**，DoScale 函数中只需做乘权和累加，无需再做 ReLU 判断。而 Arch35 的 Fixpipe 输出到 UB 时没有内置 ReLU（ReLU 由 Vector 端的 MicroAPI MulAddDst 融合完成），这是因为 UB 共享模式下 Vector 可以更灵活地控制计算流程。

**S2 跨 Core 切分与 LD（Lightning Decode）阶段**

Arch22 的 S2 基础块为 512，当 S2 序列长度超过一个 Core 能处理的能力时（如 S2=4096），需要在多个 Core 之间切分 S2 维度。每个 Core 只计算自己负责的那段 S2 范围内的部分 TopK 结果，最终需要跨 Core 合并。

SplitCore 函数负责将总计算量分配到各 AI Core，核心逻辑：

1. 计算总基础块数量 `GetTotalBaseBlockNum()`：遍历所有 Batch × KV头 × G×S1 × S2 基础块，考虑因果掩码对 S2 可见范围的影响
2. 将总块数均匀分配到各 Core，前面的 Core 多处理一个块
3. 如果某个 Core 负责的 S2 范围不是从 S2=0 开始的（即 `s2Start != 0`），则标记 `isLD=true`，表示该 Core 需要参与 LD 合并

LD 合并的工作原理：S2 被切分后，前面的 Core 拥有 S2 前段的 TopK 结果，后面的 Core 拥有后段的结果。每个 Vector 核将自己的 TopK 中间结果写入 vec1ResGm workspace（格式：`[aic, s1_cube, 头尾, idx/value, K]`），同时将 LD 参数写入 vec1ParamGm（16 个 int64 字段包含 needFd、s2AcSeq、s2Start、s2End、isS2End、bn2idx、s1Idx、S1ProcNum、indiceOutOffset 等）。

ProcessLD 函数在所有 ProcessMain 完成后执行：

1. 从 vec1ParamGm 读取 needFd 标志，找到需要合并的 S1 行
2. 依次从 vec1ResGm 搬入各 Core 的 TopK 结果，每积累 4 个 list 就用 MrgSort4 四路合并
3. 遍历后续 Core 的 needFd 直到遇到 isS2End=true（S2 结尾）
4. 最终从合并后的结果中 Extract 出 top sparseCount 个索引，写入输出 GM

---

## 5. Vector 计算细节：如何最大化 Vector 利用率

### 5.1 Arch35 MicroAPI 寄存器级编程

Arch35 的 Vector 端使用 MicroAPI 进行寄存器级编程，核心函数 `BatchMulWeightAndReduceSum` 的优化策略：

**Unroll-2 G 维度循环**：每次迭代处理 2 个 G 组，最大化寄存器利用率。

**融合计算流水线**（每组 2 个 G）：

1. 从 UB 加载 2 行 QK 分数（128 floats/行），bank-aligned stride 读取
2. `BroadcastLane`：从寄存器提取权重标量，广播到所有 S2 lane
3. `MulAddDst`：融合操作 `ReLU(QK) × weight + accumulate`，一条指令完成激活+乘法+累加
4. G 组循环结束后，将 float 结果转换为 uint16 sortable key

**Float → Sortable Key 转换**：使用 XOR 变换将 float 映射为 uint16，保持浮点排序顺序。这使得后续 TopK 可以用高效的 16-bit 基数选择算法。

**Fp32 → Bf16 双路转换**：使用 DeInterleave + Cast 的 EVEN/ODD layout traits，将 fp32 交错拆分为两组 bf16，2 倍吞吐。

### 5.2 Histogram-Based TopK（Arch35）

Arch35 的 TopK 使用 `LiTopKVF`——基于直方图的基数选择算法，专为 uint16 sortable key 设计：

**单 trunk 路径**（S2 ≤ trunkLen）：

1. 将每个 uint16 输入拆为高 8 位和低 8 位
2. 对每个 8 位段构建直方图（256 bins）
3. 利用直方图做基数选择，找到 topK 阈值
4. 用阈值 gather 出 topK 索引

只需 **2 次直方图遍历**（各 8 bit）即可完成 16-bit TopK，远快于比较排序。

**多 trunk 合并路径**（S2 > trunkLen）：

1. 第 1 trunk：LiTopKVF 选出 topK 候选，存入 hisIndexLocal[0]
2. 后续 trunk：LiTopKVF 选出当前 trunk 的 topK，然后 LiTopKGatherVF 与之前的候选合并
3. hisIndexLocal[0/1] 双缓冲 ping-pong 合并
4. 最终 trunk：将 uint16 索引转换为 int32 输出

### 5.3 Arch22 的 Vector 计算：DoScale + DoReduce + MrgSort TopK

Arch22 的 Vector 端使用传统 AscendC API 级编程，与 Arch35 的 MicroAPI 寄存器级编程有显著差异。核心分为三个阶段：**DoScale（加权计算）→ DoReduce（G 维度规约）→ TopK（排序选择）**。

**DoScale：分步加权而非融合**

DoScale 函数执行 `ReLU(QK) × weight` 计算，但分为多步而非一条融合指令：

1. **Cast 权重到 float**：如果权重是 bf16/fp16，先用 `Cast` 转为 float，保证计算精度
2. **Broadcast 权重**：用 `Brcb` 将 `groupInner` 个权重标量从 `[G, 1]` 广播到 `[G, 8]`（对齐到 B32_BLOCK_ALIGN_NUM=8），方便后续向量乘法
3. **分组乘法**：对每个 G 组（groupInner=16），做 `Mul(mmOutUb[i×s2BaseSize], tmpBuff[i×8], countPerRepeat, repeatTimes)`。外层循环 `outerGidx` 控制累加策略：
   - `outerGidx==0`（第一组）：`Mul(reduceCacheBuf, mmOutUb, tmpBuff)`——乘权后直接写入 reduce 缓存
   - `outerGidx>0`（后续组）：`Mul(mmOutUb, mmOutUb, tmpBuff)` + `Add(reduceCacheBuf, mmOutUb, reduceCacheBuf)`——先在 mmOutUb 中乘权，再与 reduce 缓存累加
4. **PipeBarrier<PIPE_V>**：每步之间插入管线屏障，确保 Vector 计算的顺序性

**groupInner=16 的切分因子**

groupInner 固定为 16，这是 UB 空间的硬约束。每次 DoScale 处理 16 个 G 组，需要 `groupInner × s2BaseSize = 16 × 512 = 8192` 个 float 的 mmOutUb 空间，加上同样大小的权重空间。对于 G=64（如 DeepSeek-V3），需要 `CeilDiv(64, 16) = 4` 次 outerG 循环才能完成全部 G 组的加权规约。

**Bank-Conflict 规约优化**

reduceCacheBuf 使用 `REDUCE_BANK_CONFLICT_OFFSETS = 256 bytes = 64 floats` 的偏移量来避免 bank conflict。Ascend UB 的存储结构为 2-bank × 8-group × 32B-block（总深度 512B），256B 的偏移确保连续的 G 组数据不落在同一个 bank 上。

**DoReduce：二分法规约**

DoReduce 函数对 G 维度做规约（将 4 个 `groupInner×s2BaseSize` 的部分结果累加成一行 s2BaseSize 的分数向量），使用**二分法加法树**：

1. 计算最近的不超过 rNum 的 2 的幂次 `dichotomizeAddPow`
2. 先将超出 2^幂部分的行加到对应位置（`Add(src, src, src[dichotomizeAddPow × aNum])`）
3. 然后逐层二分合并（`while (nowRows > 2)`：`Add(src, src, src[nowRows × aNum])`）
4. 最终两行合并：`Add(dst, src, src[aNum])`

例如 G=64、groupInner=16 时，rNum=4，规约过程为：

```
4 行 → 合并前 2 行 → 2 行 → 合并为 1 行（最终分数向量）
```

这种二分法保证了 Vector Add 操作的最大并行度——每次 Add 的两半大小相等，Vector ALU 利用率最高。

**MrgSort-based TopK：排序与合并的复杂策略**

Arch22 的 TopK 实现基于 AscendC 的 `Sort32` + `MrgSort4` 原语，根据 S1 大小和 sparseCount 是否超过 2048 采用不同策略：

**路径 A：s1BaseSize > 4 或 sparseCount > 2048（SortAll + MergeSort）**

每段 S2 先做 `SortAll`：先用 Sort32 对 32B 基础块做全排序，再用 MrgSort4 多轮四路合并直到完整有序。然后用 `MergeSort` 将当前段的排序结果与 globalTopkUb（之前积累的全局 topK）合并，保留 topK 个。

**路径 B：s1BaseSize ≤ 4 且 sparseCount ≤ 2048（缓存策略）**

这是 Arch22 的巧妙优化——利用 UB 空间缓存最多 4 个已排序的 S2 基础块（`SortedBasicBlock_`）：

1. 每段 S2 先用 `Sort<float, true>` 做排序，存入 SortedBasicBlock_ 的缓存位置 `globalTopkUbCacheIdx`
2. 当缓存满 4 块或 S2 结束时，进行精排：
   - 前 4 块直接 `MrgBasicBlock` 合并到 globalTopkUb
   - 后续块先 `MrgBasicBlock` 合并缓存中的数据，再 `SparseTopK` 与 globalTopkUb 做增量合并

这种缓存策略减少了 MrgSort 的调用次数——每 4 个 S2 基础块才做一次全局合并，而非每段都合并。对于 S2=512×N 的场景，MrgSort 调用次数从 N 次降低到 N/4 次。

**MrgSort4 四路合并原理**

MrgSort4 是 AscendC 的硬件排序原语，最多同时合并 4 个有序序列：

- `validBit` 控制有效路数：0b0011=2路，0b0111=3路，0b1111=4路
- `elementLengths` 指定每路长度（value + index 的交错排列）
- `ifExhaustedSuspension` 控制路耗尽时是否挂起

在 LD 阶段，每积累 4 个 Core 的 TopK list 就做一次四路 MrgSort4 合并，将 4×2048 合并为 1×2048，然后继续积累后续 Core 的数据。

**LD 阶段的完整流程**

```
Core0(S2=0-511) → TopK → vec1ResGm[0]
Core1(S2=512-1023) → TopK → vec1ResGm[1]   ← needFd=1
Core2(S2=1024-1535) → TopK → vec1ResGm[2]  ← needFd=1
Core3(S2=1536-2047) → TopK → vec1ResGm[3]  ← needFd=1, isS2End=1

LD 核（Core0 的 AIV）处理流程：
  1. 搬入 Core0 尾部结果（从 vec1ResGm + 2×BASE_TOPK 偏移）
  2. 读取 Core1 的 needFd=1，搬入 Core1 头部结果
  3. 积累 4 个 list → MrgSort4 四路合并 → 缩减为 1 个 list
  4. 继续读取 Core2、Core3 的结果
  5. 最终合并不足 4 个的剩余 list
  6. Extract 出 top sparseCount 索引 → 写入输出 GM
```

---

## 6. 内存层级与数据流全景

### 6.1 Arch35 数据流

```
GM → L1 (MTE2 Nd2Nz)
     ├─ Query L1 ×2 缓冲（S2 维度复用）
     └─ Key L1 ×3 缓冲（S2 基础块逐段加载）

L1 → L0A/L0B (MTE1 LoadData2D)
     ├─ L0A ×2 缓冲（Query tile）
     └─ L0B ×2 缓冲（Key tile）

L0A × L0B → L0C (Mmad Cube MatMul)
     └─ L0C ×2 缓冲

L0C → UB (Fixpipe, dual_dst_ctl=1, bank-conflict-free)
     └─ mm1ResUB ×2 缓冲 ← Cube/Vector ping-pong 共享

UB → Vector Compute (MicroAPI)
     ├─ 加权 ReLU ReduceSum → uint16 sortable key
     └─ scoreGm workspace（跨 S2 块累积）

scoreGm → UB → LiTopKVF → GM output
     └─ TopK 索引（int32）+ 可选分数值（fp16/bf16）
```

### 6.2 Arch22 数据流

```
GM → L1 → L0A/L0B → L0C → GM workspace (Fixpipe, nz2nd, builtin ReLU)
     └─ mm1ResGm ×2 缓冲

GM workspace → UB → Vector (DoScale + DoReduce) → scoreGm
     └─ MrgSort4-based TopK → GM output

跨 Core → LD 合并（splitCoreInfo.isLD）
```

### 6.3 Workspace 分配

**Arch35**：`libApiWorkSpaceSize + s1BaseSize × CeilDiv(s2Size, s2BaseSize) × s2BaseSize × sizeof(uint16_t) × aicNum`

**Arch22**：`libApiWorkSpaceSize + mm1ResSize × DOUBLE_BUFFER × aicNum + LD_decode_data + LD_decode_params`

---

## 7. Cube/Vector 利用率最大化：六大核心机制总结

| 机制 | 原理 | 效果 |
|------|------|------|
| **五缓冲 L1 流水线** | 3 Key + 2 Query，MTE2 数据搬运与 MatMul 计算完全解耦 | Cube 永远不会因等待数据而空转 |
| **双缓冲 L0 流水线** | L0A/L0B/L0C 各 2 缓冲，LoadData/Mmad/Fixpipe 三级流水 | Cube 持续计算，无空闲周期 |
| **Ping-Pong UB + 1:2 并行** | Cube 写 UB[loop%2]，2 个 AIV 读 UB[(loop-1)%2]，QLI_SYNC_MODE4 四事件同步 | Cube 和 Vector 零等待并发，硬件利用率翻倍 |
| **Bank-Conflict-Free UB 布局** | Fixpipe dstStride=128，每行分数落在不同 bank group | Vector G 维度并行读取无 bank 冲突 |
| **MicroAPI 融合计算** | ReLU+MulAddDst 一条指令，Unroll-2 G 组，BroadcastLane 权重广播 | Vector ALU 利用率最大化，寄存器级吞吐 |
| **Histogram-Based uint16 TopK** | 2 次 8-bit 直方图遍历完成 16-bit 基数选择，XOR 变换保持排序 | TopK 吞吐远超比较排序，Vector 快速释放缓冲 |

### 与传统实现的对比

| 特性 | Lightning Indexer（Arch35） | 传统 Flash Attention |
|------|---------------------------|---------------------|
| Cube-Vector 关系 | 1:2 并行并发，ping-pong UB | 顺序执行，Cube 写 GM → Vector 读 GM |
| Cube 输出目标 | UB（shared with Vector） | GM workspace |
| 流水线深度 | 5-buffer L1 + 2-buffer L0 + 2-buffer UB | 通常 2-buffer L1 + 2-buffer L0 |
| TopK 算法 | Histogram-based uint16 radix（2 pass） | MrgSort4（多次比较合并） |
| 权重计算 | MicroAPI 融合 ReLU+MulAdd | 分步 DoScale + DoReduce |
| 跨 Core 合并 | 不需要（单 Core 处理完整 S2） | 需要 LD 阶段跨 Core 合并 |

---

## 8. 算子变体与衍生

Lightning Indexer 系列还包含几个衍生算子：

- **lightning_indexer_grad**：Indexer 的反向传播算子，用于训练中梯度计算
- **dense_lightning_indexer_softmax_lse**：稠密注意力版本，同时输出 softmax LSE（log-sum-exp），用于 DSA 的 attention 计算阶段
- **dense_lightning_indexer_grad_kl_loss**：softmax_lse 的反向传播 + KL 散度损失
- **quant_lightning_indexer**：量化版本，支持 FP8/INT8 输入的 Indexer 计算
- **lightning_indexer_v2_metadata**：V2 版本的元数据生成，支持 AICPU 模式

---

## 9. 关键源码文件索引

| 文件 | 核心内容 |
|------|---------|
| `op_host/lightning_indexer_tiling.cpp/h` | Tiling 切分逻辑、基础块大小计算 |
| `op_kernel/lightning_indexer.cpp` | 主 Kernel 入口、循环结构 |
| `op_kernel/lightning_indexer_common.h` | 公共常量定义、缓冲大小 |
| `op_kernel/arch35/lightning_indexer_kernel.h` | Arch35 Kernel 主流程 |
| `op_kernel/arch35/lightning_indexer_service_cube.h` | Arch35 Cube 五缓冲流水线 + Fixpipe |
| `op_kernel/arch35/lightning_indexer_service_vector.h` | Arch35 Vector MicroAPI 加权计算 |
| `op_kernel/arch35/vf/lightning_indexer_vector1.h` | MicroAPI MulWeightAndReduceSum 寄存器级实现 |
| `op_kernel/arch35/vf/lightning_indexer_topk.h` | Histogram-based LiTopKVF |
| `op_kernel/arch22/lightning_indexer_kernel.h` | Arch22 Kernel 主流程 |
| `op_kernel/arch22/lightning_indexer_service_cube.h` | Arch22 Cube + GM Fixpipe |
| `op_kernel/arch22/lightning_indexer_service_vector.h` | Arch22 Vector + MrgSort TopK |

---

## 10. 总结

Lightning Indexer 的设计哲学是**让 Cube 和 Vector 永不空闲**：

- Cube 端通过五缓冲 L1 和双缓冲 L0 流水线，使 MatMul 持续运行，数据搬运永远不阻塞计算
- Vector 端通过 MicroAPI 寄存器级融合计算和 Histogram-based TopK，将加权聚合和稀疏选择的吞吐最大化
- Cube 和 Vector 通过 Ping-Pong UB 双缓冲和 QLI_SYNC_MODE4 四事件同步实现真正的并发执行，各自在不同缓冲区上同时工作
- 整体设计从内存布局（bank-conflict-free）到算法选择（uint16 radix TopK）到编程模式（MicroAPI register-level），每一层都在为最大化硬件利用率服务

这种 "算子级全栈优化" 的思路，是 Ascend NPU 上高性能算子设计的典范——不是简单地让 Cube 或 Vector 分别跑快，而是让它们**同时跑快且互不等待**，这才是利用率的真正极限。