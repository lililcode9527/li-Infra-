# DeepSeek V3.1 / V4 架构与优化深度分析

> 基于 vllm 和 vllm-ascend 源码深入分析  
> 分析日期：2026-05-27

---

## 1. 架构演进总览

### 1.1 版本演进路线

```
                        DeepSeek V2
                             │
                    MLA + MoE + EP
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         DeepSeek V3    DeepSeek R1    DeepSeek VL2
              │              │
         MTP + FP8     Reasoning Parser
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
   V3.1      V3.2      (工具调用演进)
 (去type)  (DSML格式)
              │
              │
    ┌─────────┼─────────┐
    ▼                   ▼
DeepSeek V4         DeepSeek V4-Flash
    │
 MegaMoE + Compressor
 Indexer + MHC/HC
 Hash MoE + FP4
 Attention Sink
 O-proj 低秩压缩
```

### 1.2 V3 与 V4 架构对比

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        DeepSeek V3 架构                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   Input → Embed ──────────────────────────────────────────────────────┐      ║
║              │                                                        │      ║
║   ┌──────────▼─────────── Decoder Layer (×N) ───────────────┐        │      ║
║   │  ┌─────────────────────────────────────────────────────┐ │        │      ║
║   │  │  RMSNorm → MLA Attention → Post-Process            │ │        │      ║
║   │  │    │                                                  │ │        │      ║
║   │  │    ├── fused_qkv_a_proj (合并 Q_A + KV_A)            │ │        │      ║
║   │  │    ├── q_a → LN → q_b → Q                           │ │        │      ║
║   │  │    ├── kv_a → kv_lora → LN → kv_b → K_nope, V       │ │        │      ║
║   │  │    └── k_pe → RoPE → K_rope                         │ │        │      ║
║   │  └─────────────────────────────────────────────────────┘ │        │      ║
║   │  ┌─────────────────────────────────────────────────────┐ │        │      ║
║   │  │  RMSNorm → MoE (Grouped Top-K + Shared Expert)      │ │        │      ║
║   │  └─────────────────────────────────────────────────────┘ │        │      ║
║   └──────────────────────────────────────────────────────────┘        │      ║
║                                                                       │      ║
║   ┌── RMSNorm ──► LM Head ──► Logits ◄───────────────────────────────┘      ║
║                                                                              ║
║   关键特征：                                                                 ║
║   • 标准 Pre-Norm (RMSNorm)                                                  ║
║   • Fused QKV-A Projection (单次 GEMM 完成 Q_A+KV_A)                        ║
║   • 单层 o_proj                                                              ║
║   • Grouped Top-K MoE 路由 (softmax)                                        ║
║   • FP8 线性层量化                                                           ║
║   • KV Cache = (kv_lora_rank + qk_rope_head_dim) × seq_len                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║                        DeepSeek V4 架构                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   Input → Embed → [hc_mult 分支展开] ────────────────────────────────────┐   ║
║                                      │                                   │   ║
║   ┌──────────────────────────────────▼──── Decoder Layer (×N) ────┐     │   ║
║   │  ┌─ MHC Pre ─────────────────────────────────────────────────┐ │     │   ║
║   │  │  hc_mult 路残差流 → RMSNorm → Linear → sigmoid + Sinkhorn │ │     │   ║
║   │  │  输出: pre_mix, post_mix, comb_mix, layer_input            │ │     │   ║
║   │  └────────────────────────────────────────────────────────────┘ │     │   ║
║   │  ┌─ Attention Block ──────────────────────────────────────────┐ │     │   ║
║   │  │  fused_wqa_wkv (合并 WQA + WKV)                            │ │     │   ║
║   │  │  ├── qr → q_norm → wq_b → Q (padded_heads)                │ │     │   ║
║   │  │  ├── kv → kv_norm → _fused_qnorm_rope_kv_insert            │ │     │   ║
║   │  │  │      → FlashMLA Sparse Attention                        │ │     │   ║
║   │  │  ├── o → inv_rope + FP8 quant → wo_a (einsum) → wo_b      │ │     │   ║
║   │  │  └── Compressor (C4/C128) + Indexer (C4 only)             │ │     │   ║
║   │  └────────────────────────────────────────────────────────────┘ │     │   ║
║   │  ┌─ MHC Post ────────────────────────────────────────────────┐ │     │   ║
║   │  │  out_j = post_mix_j × x + Σ_i comb_mix_ij × residual_i    │ │     │   ║
║   │  └────────────────────────────────────────────────────────────┘ │     │   ║
║   │  ┌─ MHC Pre (FFN) → MoE (Hash/SqrtSoftPlus) → MHC Post ───────┐ │     │   ║
║   │  │  • 前 num_hash_layers 层: Hash 查表路由                     │ │     │   ║
║   │  │  • 其余层: sqrtsoftplus 路由                                │ │     │   ║
║   │  │  • MegaMoE: FP4 专家 + DeepGEMM                            │ │     │   ║
║   │  │  • SwiGLU + Clamping                                        │ │     │   ║
║   │  └────────────────────────────────────────────────────────────┘ │     │   ║
║   └─────────────────────────────────────────────────────────────────┘     │   ║
║                                                                           │   ║
║   ┌── HC Head ──► RMSNorm → Linear → sigmoid → Σ gate_i × residual_i     │   ║
║   │    → RMSNorm → LM Head → Logits ◄─────────────────────────────────────┘   ║
║                                                                              ║
║   关键特征：                                                                 ║
║   • MHC (Multi-Head Composition) 多分支残差 + Sinkhorn 归一化                ║
║   • 分离的 wq_a + wkv 投影 (非 fused_qkv_a)                                  ║
║   • O-proj 低秩 (wo_a + wo_b)                                                ║
║   • Attention Sink (可学习, -inf 初始化)                                     ║
║   • Compressor: C4 (4x) 和 C128 (128x) KV Cache 压缩                        ║
║   • Hash MoE: 前 N 层免 router 计算                                         ║
║   • MegaMoE: FP4 专家 + SM100 DeepGEMM                                      ║
║   • Compress-aware RoPE: 不同压缩比层不同 theta                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 1.3 V3 vs V4 对比表

| 特性 | V3/V3.1/V3.2 | V4 |
|------|-------------|-----|
| **Q 投影** | fused_qkv_a_proj（Q_A+KV_A 合并） | wq_a + wkv 分离 |
| **O 投影** | 单层 o_proj | wo_a (BMM低秩) + wo_b |
| **KV Cache 压缩** | 无 | Compressor: C4(4x) / C128(128x) |
| **归一化** | Pre-Norm RMSNorm | MHC (Multi-Head Composition) |
| **Attention Sink** | 无 | 可学习 -inf sink |
| **MoE 路由** | Grouped Top-K + softmax | Hash MoE (前N层) + sqrtsoftplus |
| **专家精度** | FP8/BF16 | FP4 (MegaMoE) 或 FP8 |
| **激活函数** | SiLU × gate (SwiGLU) | SiLU × gate + Clamping |
| **RoPE** | YaRN / Llama-4 scaling | Compress-aware (不同层不同θ) |
| **Indexer** | V3.2: C4 层 | V4: C4 层集成 Compressor |
| **MTP** | eh_proj (融合 e+h) | e_proj + h_proj (分离, FP8) |
| **多流并行** | 无 | 3-路 parallel (QKV, Compressor, Indexer) |

---

## 2. MLA 低秩注意力

### 2.1 核心思想

MLA 是 DeepSeek 系列最核心的创新，通过**低秩压缩**将 KV Cache 从 `2 × n_heads × head_dim` 降至 `kv_lora_rank + rope_dim`（通常从 ~32K 降至 ~576 字节/每token），减少 **~10-50 倍** KV Cache 显存。

```
传统 MHA:
  Q, K, V = Linear(hidden)            → 维度: (n_heads × head_dim)
  KV Cache = 2 × n_heads × head_dim   → 如 2 × 128 × 128 = 32768 字节/token

MLA:
  KV 低秩投影: kv_a = Linear(hidden, kv_lora_rank)     → 压缩到 512 维
             K_nope, V = Linear(kv_a, head_dim + ...)  → 解压到完整维度
             K_rope = RoPE(hidden[rope_dim:])          → 旋转部分不压缩
  KV Cache = kv_lora_rank + rope_dim  → 如 512 + 64 = 576 字节/token
  压缩比 ≈ 32768 / 576 ≈ 57x
```

### 2.2 V3 MLA 投影流程

```
                    hidden_states [T, hidden_size]
                              │
                    ┌─────────▼──────────┐
                    │  fused_qkv_a_proj   │  ← 合并 Q_A + KV_A 为一个 GEMM
                    │  (MergedColumnPL)   │
                    └──┬──────────────┬──┘
                       │              │
                  q_a  │              │  kv_a
          [T, q_lora_rank]            │  [T, kv_lora_rank + rope_dim]
                       │              │
                 ┌─────▼─────┐  ┌─────▼──────────────────┐
                 │ q_a_layernorm│  │ kv_lora   │  k_pe     │
                 │   (RMSNorm) │  │ [T, 512]  │  [T, 64]  │
                 └─────┬─────┘  └──┬─────────┴─────┬─────┘
                       │           │               │
                 ┌─────▼─────┐  ┌──▼──────────┐  ┌─▼────┐
                 │  q_b_proj  │  │ kv_a_layernorm│  │ RoPE │
                 │            │  │   (RMSNorm)  │  │      │
                 └─────┬─────┘  └──┬──────────┘  └──┬───┘
                       │           │                  │
                  Q [T, H, D]  ┌───▼────────────┐  K_rope
                               │   kv_b_proj     │  [T, H, 64]
                               │                 │
                               └──┬──────┬──────┘
                                  │      │
                             K_nope [H, 128]  V [H, 128]
                                  │
                     ┌────────────▼────────────┐
                     │  K = concat(K_nope, K_rope) │
                     └─────────────────────────┘
                              │
                     ┌────────▼────────┐
                     │  Attention(Q,K,V)│
                     └────────┬────────┘
                              │
                     ┌────────▼────────┐
                     │    o_proj        │
                     └────────┬────────┘
                              │
                         output [T, hidden_size]
```

### 2.3 关键优化：Fused QKV-A Projection

```python
# deepseek_v2.py: DeepSeekV2FusedQkvAProjLinear
class DeepSeekV2FusedQkvAProjLinear(MergedColumnParallelLinear):
    """将 q_a_proj 和 kv_a_proj_with_mqa 融合为单个 GEMM
    
    权重布局:
      [q_lora_rank, kv_lora_rank + rope_dim]
        ↑                    ↑
       Q_A 部分          KV_A 部分
    
    小 batch 优化:
      当 num_tokens <= 16 时, 使用 dsv3_fused_a_gemm 自定义 CUDA kernel
      支持 SM90 (H100) 和 SM100 (B200) 的专用 min-latency kernel
    """
```

### 2.4 V4 MLA 投影流程

```
                    hidden_states [T, hidden_size]
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
    ┌─────▼─────┐     ┌──────▼──────┐    ┌───────▼────────┐
    │   wq_a    │     │ fused_wqa_wkv│    │  compressor     │
    │ (Q_lora)  │     │  (合并投影)   │    │  (C4/C128层)    │
    └─────┬─────┘     └──┬───────────┘    └───────┬────────┘
          │              │                        │
    qr [T, q_lora]   kv [T, head_dim]        kv_score
          │              │                  → Compressor
    ┌─────▼─────┐  ┌─────▼──────────────┐
    │  q_norm   │  │  kv_norm           │
    │ (RMSNorm) │  │  (RMSNorm)         │
    └─────┬─────┘  └─────┬──────────────┘
          │              │
    ┌─────▼─────┐  ┌─────▼──────────────────────────────────┐
    │   wq_b    │  │  _fused_qnorm_rope_kv_insert (CUDA)    │
    │           │  │  Q: RMSNorm(无权重) + GPT-J RoPE       │
    └─────┬─────┘  │  KV: GPT-J RoPE + UE8M0 FP8 quant +    │
          │        │       paged cache insert                │
    Q [T, H, 512]  └────────────────────────────────────────┘
          │                       │
    ┌─────▼───────────────────────▼─────┐
    │  FlashMLA Sparse Attention        │
    │  (Q 被 pad 到 64 或 128 heads)    │
    │  + attn_sink (可学习 -inf)        │
    └─────────────────┬─────────────────┘
                      │
                o [T, H, 512]
                      │
    ┌─────────────────▼─────────────────┐
    │  fused_inv_rope_fp8_quant (Triton) │
    │  • Inverse GPT-J RoPE (rope 部分) │
    │  • Block-scaled FP8 量化 (block=128)│
    │  • SM90: FP32 scale, SM100: UE8M0  │
    └─────────────────┬─────────────────┘
                      │
                o_fp8 [G, T, D], o_scale [G, T, D/128]
                      │
    ┌─────────────────▼─────────────────┐
    │  deepseek_v4_fp8_einsum (CustomOp) │
    │  equation: "bhr,hdr->bhd"         │
    │  o_fp8 @ wo_a_fp8 → z [T, G, o_lora] │
    └─────────────────┬─────────────────┘
                      │
    ┌─────────────────▼─────────────────┐
    │  wo_b                             │
    │  z.flatten(1) → hidden_size       │
    └─────────────────┬─────────────────┘
                      │
                output [T, hidden_size]
```

### 2.5 V4 关键变化

**1. 分离 WQA 和 WKV**（不再 fused_qkv_a）：
- V3: `fused_qkv_a_proj` 同时产生 Q_A 和 KV_A
- V4: `fused_wqa_wkv` 是合并的线性层但只产生 `[q_lora_rank, head_dim]`，其中 `head_dim` 部分是已经投影好的 KV（无需 kv_b_proj 解压）

**2. O 投影低秩化**：
```
# V3: 单层
o_proj: [n_heads × head_dim] → hidden_size

# V4: 双层低秩 + 分组
wo_a: [heads_per_group × head_dim] → [o_lora_rank]  # BMM 操作
wo_b: [n_groups × o_lora_rank] → hidden_size
```

**3. Attention Sink**：
```python
# model.py: L710-L714
# 初始化为 -inf 的注意力 sink，防止注意力浪费在无效位置
# 自动 padding 到 64 或 128 heads 以兼容 FlashMLA
self.attn_sink = nn.Parameter(
    torch.full((padded_heads,), -float("inf"), dtype=torch.float32),
    requires_grad=False,
)
```

**4. Head Padding**：
```python
# FlashMLA sparse kernel 仅支持 64 或 128 heads
if num_heads <= 64:    padded_heads = 64
elif num_heads <= 128: padded_heads = 128
```

---

## 3. MoE 混合专家

### 3.1 V3 MoE 架构

```
                         hidden_states [T, H]
                                │
                    ┌───────────▼───────────┐
                    │      Router Gate       │
                    │   Linear(H, n_experts) │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Grouped Top-K 选择    │
                    │  1. 将专家分为 n_group 组 │
                    │  2. 组间 topk_group 选组 │
                    │  3. 组内 top_k 选专家    │
                    │  4. norm_topk_prob 归一化│
                    └───────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                  ▼
    ┌─────────────────┐  ┌──────────┐  ┌──────────────┐
    │  Routed Expert 0 │  │  ...     │  │ Shared Expert │
    │  (SwiGLU MLP)   │  │          │  │  (所有token)   │
    └────────┬────────┘  └────┬─────┘  └──────┬───────┘
             │                │                │
             └────────┬───────┘                │
                      │                        │
              ┌───────▼────────┐               │
              │ Weighted Sum    │               │
              │ (按 routing 权重)│               │
              └───────┬────────┘               │
                      │                        │
                      └───────────┬────────────┘
                                  │
                          output [T, H]
```

### 3.2 V4 MoE 架构

#### 3.2.1 Hash MoE（前 num_hash_layers 层）

```
                          token_id [T]
                               │
                    ┌──────────▼──────────┐
                    │   tid2eid 查找表     │
                    │   [vocab_size, top_k]│
                    │   → 直接路由到专家    │
                    └─────────────────────┘
                               │
                    无需计算 router logits！
                    零额外计算开销
```

**代码实现**：
```python
# model.py: L506-L519
is_hash_moe = extract_layer_index(prefix) < config.num_hash_layers
if is_hash_moe:
    self.gate.tid2eid = nn.Parameter(...)  # [vocab_size, top_k]
```

#### 3.2.2 SqrtSoftPlus 路由（深层）

```python
# 评分函数对比
V3:    softmax(router_logits)         → 尖锐分布
V4:    sqrt(softplus(router_logits))  → 更平滑、更稳定
```

#### 3.2.3 MegaMoE（DeepGEMM FP4）

```
╔══════════════════════════════════════════════════════════════════╗
║                    MegaMoE 计算流程                               ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   输入: hidden_states [T, H] (bf16)                              ║
║                                                                  ║
║   1. Router → topk_indices, topk_weights                        ║
║                                                                  ║
║   2. 专家权重 (FP4, uint8 打包)                                  ║
║      w13: [n_experts, 2×intermediate, hidden//2]                 ║
║      w13_scale: [n_experts, 2×intermediate, hidden//32] (UE8M0) ║
║      w2: [n_experts, hidden, intermediate//2]                    ║
║      w2_scale: [n_experts, hidden, intermediate//32] (UE8M0)    ║
║                                                                  ║
║   3. deep_gemm.fp8_fp4_mega_moe kernel (SM100)                  ║
║      • FP4 权重 × FP8 激活 → bf16 输出                          ║
║      • 对称缓冲区复用 (symm_buffer_cache)                         ║
║                                                                  ║
║   4. SwiGLU + Clamping                                           ║
║      gate = sigmoid(w13_gate @ x)                                ║
║      up = w13_up @ x                                             ║
║      act = clamp(gate * up, max=swiglu_limit)                    ║
║      output = w2 @ act                                           ║
║                                                                  ║
║   约束:                                                           ║
║   • 仅 SM100 (Blackwell) GPU                                    ║
║   • 仅 scoring_func="sqrtsoftplus"                              ║
║   • 仅 expert_dtype="fp4"                                       ║
║   • hidden_size 和 intermediate_size 必须是 128 的倍数            ║
║   • 需要 --enable-expert-parallel                               ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 4. KV Cache 压缩

### 4.1 设计原理

V4 的 Compressor 是全新的 KV Cache 压缩机制，通过时间维度的压缩进一步减少 KV Cache：

```
压缩比策略（分层配置）:
  compress_ratios = [1, 1, 1, ..., 4, 4, 4, ..., 128, 128, 128]
                     ↑               ↑                ↑
                  浅层不压缩      中层 C4 压缩      深层 C128 压缩
                  (保留完整信息)  (4x, 保留稀疏注意力)  (128x, 极致压缩)
```

### 4.2 Compressor 架构

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    Compressor 架构与数据流                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  输入: kv_score = fused_wkv_wgate(hidden_states)                            ║
║        → 拆分为 kv [T, coff*head_dim] 和 score [T, coff*head_dim]          ║
║                                                                              ║
║  参数:                                                                       ║
║  • compress_ratio ∈ {4, 128}                                                ║
║  • coff = 1 + (compress_ratio == 4)    # C4→2, C128→1                      ║
║  • OVERLAP = (compress_ratio == 4)     # C4→True, C128→False               ║
║  • 窗口大小 = (1 + OVERLAP) × compress_ratio                                ║
║              C4: 2×4=8 tokens, C128: 1×128=128 tokens                       ║
║  • APE: [compress_ratio, coff*head_dim] (可学习绝对位置编码)                 ║
║                                                                              ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │ 步骤 1: State Cache 保存 (Triton kernel)                              │  ║
║  │                                                                        │  ║
║  │  CompressorStateCache (float32, 每层独立)                              │  ║
║  │  state_dim = 2 × coff × head_dim  (kv_state + score_state)            │  ║
║  │  block_size = C4:8, C128:32                                           │  ║
║  │                                                                        │  ║
║  │  每个新 token:                                                         │  ║
║  │    kv_state[slot] = kv                                                │  ║
║  │    score_state[slot] = score + APE[position % compress_ratio]          │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │ 步骤 2: 融合压缩 (Triton kernel, 仅当 position 在压缩边界)            │  ║
║  │                                                                        │  ║
║  │  for i in range(compress_ratio):                                       │  ║
║  │      kv_i, score_i = state_cache[slot - i]                            │  ║
║  │                                                                        │  ║
║  │  # Softmax 加权                                                        │  ║
║  │  weights = softmax([score_0, ..., score_{CR-1}])                       │  ║
║  │  compressed_kv = Σ kv_i × weights[i]                                   │  ║
║  │                                                                        │  ║
║  │  # RMSNorm (fp32)                                                      │  ║
║  │  compressed_kv = RMSNorm(compressed_kv)                                │  ║
║  │                                                                        │  ║
║  │  # RoPE (GPT-J style, interleaved pairs)                               │  ║
║  │  对最后 rope_head_dim 维应用 RoPE                                      │  ║
║  │                                                                        │  ║
║  │  # FP8 量化 (UE8M0 block-scaled) + KV Cache 写入                       │  ║
║  │  nope 部分: FP8 + UE8M0 scale (block=64, 7 blocks)                    │  ║
║  │  rope 部分: bf16 直通                                                 │  ║
║  │  缓存布局 per token: [448B fp8, 128B bf16, 8B scales] = 584B          │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │ 步骤 3: OVERLAP 机制 (仅 C4)                                           │  ║
║  │                                                                        │  ║
║  │  C4 的 OVERLAP=True:                                                   │  ║
║  │    每个压缩周期实际处理 2×compress_ratio = 8 个 token                  │  ║
║  │    产生 1 个压缩 token 但需要前一个周期的历史                            │  ║
║  │    head_offset = compress_ratio 区分新旧条目                            │  ║
║  │                                                                        │  ║
║  │  C128 的 OVERLAP=False:                                                │  ║
║  │    每个压缩周期处理 128 个 token                                        │  ║
║  │    无重叠，直接压缩                                                      │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 4.3 C4 vs C128 对比

| 特性 | C4 | C128 |
|------|-----|------|
| 压缩比 | 4x | 128x |
| coff (系数) | 2 | 1 |
| OVERLAP | True | False |
| 窗口大小 | 8 tokens | 128 tokens |
| State Cache block_size | 8 | 32 |
| Indexer | ✅ 支持 | ❌ 无 |
| Triton num_warps | 4 | 1 |
| Quant block | 64 | 128 |
| head_dim 支持 | 512 | 128 或 512 |
| RoPE theta | config.compress_rope_theta | config.compress_rope_theta |

### 4.4 三个融合 Kernel 变体

| Kernel | head_dim | 量化格式 | 用途 |
|--------|----------|----------|------|
| `_fused_kv_compress_norm_rope_insert_sparse_attn` | 512 | nope=FP8, rope=bf16 | MLA sparse attention |
| `_fused_kv_compress_norm_rope_insert_indexer_attn` | 128 | 全 FP8 | Indexer |
| `_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn` | 128 | MXFP4 | Indexer (FP4 cache) |

所有三个 kernel 共享相同的流水线：**Gather State → Softmax 压缩 → RMSNorm → RoPE → 量化 → Cache 写入**。

---

## 5. 稀疏注意力与 Indexer

### 5.1 Indexer 架构

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    V4 Indexer 架构 (仅 C4 层)                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  参数:                                                                       ║
║  • index_n_heads = 64, index_head_dim = 128                                  ║
║  • index_topk: 选择的稀疏 token 数量                                         ║
║  • index_rope_dim = qk_rope_head_dim = 64                                    ║
║                                                                              ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │ Q 侧处理                                                               │  ║
║  │                                                                        │  ║
║  │  qr [T, q_lora_rank]                                                   │  ║
║  │    → wq_b: q_lora_rank → index_n_heads × index_head_dim                │  ║
║  │    → reshape [T, 64, 128]                                              │  ║
║  │    → fused_indexer_q_rope_quant:                                       │  ║
║  │        • GPT-J RoPE (最后 rope_dim 维)                                 │  ║
║  │        • FP8: per-token-per-head 单 scalar 量化                        │  ║
║  │        • MXFP4: 32 元素 block E2M1 量化                                │  ║
║  │        • 权重折叠: q_scale → weights (FP8) 或 分离 (MXFP4)              │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │ K 侧处理                                                               │  ║
║  │                                                                        │  ║
║  │  hidden_states [T, hidden_size]                                        │  ║
║  │    → weights_proj: hidden_size → index_n_heads                         │  ║
║  │    → Indexer Compressor (独立的 Compressor, head_dim=128)              │  ║
║  │       • 与 MLA Compressor 共享相同的融合 kernel                        │  ║
║  │       • 写入 DeepseekV4IndexerCache (132 bytes/token)                  │  ║
║  │                                                                        │  ║
║  │  K_cache 布局:                                                         │  ║
║  │    head_dim = 128(fp8) + 4(fp32_scale) = 132 bytes                    │  ║
║  │    alignment = 576                                                     │  ║
║  │    可选 MXFP4: head_dim = 64(packed uint8) + 4(ue8m0) = 68 bytes      │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │ SparseAttnIndexer 前向                                                  │  ║
║  │                                                                        │  ║
║  │  scores = Q_fp8 @ K_fp8 + weights                                     │  ║
║  │  topk_indices = argtopk(scores, index_topk)                           │  ║
║  │                                                                        │  ║
║  │  结果用于:                                                              │  ║
║  │  • FlashMLA sparse attention 的 token 选择                            │  ║
║  │  • 只对 topk_indices 中的 token 计算完整注意力                          │  ║
║  │  • + SWA (Sliding Window Attention) 窗口内的 token                    │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  IndexCache (可选优化):                                                      ║
║  • 复用之前的 topk 选择结果，减少重算                                        ║
║  • pattern 示例: "SSSS" → 每 5 层中前 4 层 Skip                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 5.2 预填充索引组合

```
FlashMLA Sparse Prefill 需要的 token 组合:

  combined_indices = topk_indices ∪ SWA_window_indices

  其中:
  • topk_indices: Indexer 选择的 top index_topk 个 token (压缩后的)
  • SWA_window_indices: 最近 window_size 个 token (滑动窗口)

  对齐要求: combined 总数对齐到 128 (FlashMLA 要求)
```

---

## 6. MHC/HC 多头组合

### 6.1 MHC 核心思想

MHC 是 V4 对 Transformer 残差连接的根本性改造，用多分支 Sinkhorn 归一化替代标准 Pre-Norm：

```
传统 Pre-Norm:
  x → RMSNorm → Attention → x + attn_out → RMSNorm → FFN → x + ffn_out

MHC (Multi-Head Composition):
  x [T, H] → 展开为 hc_mult 个分支 [T, hc_mult, H]
  ↓
  每个子层前:
    MHC Pre: RMSNorm(hc_mult 分支) → Linear → sigmoid + Sinkhorn
            → 输出 pre_mix (层输入混合), post_mix (层后混合), comb_mix (残差混合)
  ↓
  子层: Attention 或 FFN
  ↓
  MHC Post: out_j = post_mix_j × layer_out + Σ_i comb_mix_ij × residual_i
  ↓
  最终:
    HC Head: RMSNorm → Linear → sigmoid → Σ gate_i × branch_i → 单路输出
```

### 6.2 MHC Pre 流程

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         MHC Pre 计算流程                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  输入: residual [T, hc_mult, hidden_size]  (hc_mult 个并行残差流)           ║
║                                                                              ║
║  步骤 1: RMSNorm + 线性投影                                                  ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │  x = residual.flatten() → [T, hc_mult × hidden_size] (fp32)          │  ║
║  │  mixes = x @ hc_fn.T                                                 │  ║
║  │  mixes = mixes × rsqrt(mean(x²) + rms_eps)                           │  ║
║  │  mixes → [T, hc_mult + hc_mult + hc_mult²]                          │  ║
║  │           ←pre_logits→ ←post_logits→ ←comb_logits→                  │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  步骤 2: pre_mix (sigmoid gate)                                              ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │  pre_logits [T, hc_mult] → × hc_scale[0] + hc_base[:hc_mult]         │  ║
║  │  pre_mix = sigmoid(pre_logits) + hc_pre_eps                          │  ║
║  │  layer_input = Σ_i pre_mix_i × residual_i → [T, hidden_size]         │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  步骤 3: post_mix (sigmoid gate × scale)                                     ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │  post_logits [T, hc_mult] → × hc_scale[1] + hc_base[hc_mult:2*hc_mult]│  ║
║  │  post_mix = sigmoid(post_logits) × hc_post_mult_value                │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  步骤 4: comb_mix (Sinkhorn 归一化)                                          ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │  comb_logits [T, hc_mult, hc_mult]                                    │  ║
║  │                                                                        │  ║
║  │  # Sinkhorn-Knopp 迭代 (将矩阵投影到双随机流形)                         │  ║
║  │  comb = softmax(comb_logits, dim=-1) + hc_sinkhorn_eps                │  ║
║  │  comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)                 │  ║
║  │  for _ in range(sinkhorn_repeat - 1):                                 │  ║
║  │      comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)  # 行归一化 │  ║
║  │      comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)  # 列归一化 │  ║
║  │                                                                        │  ║
║  │  结果: 近似双随机矩阵 (每行和≈1, 每列和≈1)                              │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  返回值: (post_mix, comb_mix, layer_input)                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 6.3 MHC Post 流程

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         MHC Post 计算流程                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  输入:                                                                       ║
║  • x [T, hidden_size]: 子层 (Attention/FFN) 的输出                          ║
║  • residual [T, hc_mult, hidden_size]: 进入子层前的 hc_mult 个残差流        ║
║  • post_mix [T, hc_mult, 1]: MHC Pre 产生的后混合权重                       ║
║  • comb_mix [T, hc_mult, hc_mult]: Sinkhorn 归一化的组合混合矩阵            ║
║                                                                              ║
║  计算:                                                                       ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │  # 残差流混合 (einsum: ...ij,...ih->...jh)                            │  ║
║  │  mixed_residual = Σ_i comb_mix[:,j,i] × residual[:,i,:]              │  ║
║  │                  = einsum("tij,tih->tjh", comb_mix, residual)         │  ║
║  │                                                                        │  ║
║  │  # 层输出缩放                                                          │  ║
║  │  post_term = post_mix × x.unsqueeze(-2)                              │  ║
║  │                                                                        │  ║
║  │  # 最终输出                                                            │  ║
║  │  out = mixed_residual + post_term → [T, hc_mult, hidden_size]         │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  注意: 输出仍是 hc_mult 路，在模型出口由 HC Head 合并为单路                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 6.4 HC Head（模型出口）

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         HC Head 计算流程                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  输入: hidden_states [T, hc_mult, hidden_size]                              ║
║                                                                              ║
║  步骤 1: RMSNorm                                                            ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │  x = hidden_states.flatten(1) → [T, hc_mult × hidden_size] (fp32)    │  ║
║  │  rsqrt = 1/sqrt(mean(x², dim=-1) + norm_eps)                         │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  步骤 2: Gate 计算                                                           ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │  mixes = (x @ hc_head_fn.T) × rsqrt  → [T, hc_mult]                  │  ║
║  │  gates = sigmoid(mixes × hc_head_scale + hc_head_base) + hc_eps     │  ║
║  │                                                                        │  ║
║  │  hc_head_fn: [hc_mult, hc_mult × hidden_size]  (可学习)               │  ║
║  │  hc_head_base: [hc_mult]  (可学习 bias)                               │  ║
║  │  hc_head_scale: [1]  (可学习 scale)                                   │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  步骤 3: 加权合并                                                             ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │  output = Σ_i gates[i] × hidden_states[:, i, :]                      │  ║
║  │  output → [T, hidden_size]  (单路)                                   │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  步骤 4: 最终 Norm + LM Head                                                 ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │  output = RMSNorm(output)                                             │  ║
║  │  logits = lm_head(output)                                             │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 6.5 Sinkhorn 迭代实现

```python
# 三处等价的实现:

# 1. CUDA tilelang (生产路径, GPU)
def sinkhorn_tilelang(comb, hc_sinkhorn_eps, sinkhorn_repeat):
    # comb = softmax(comb, dim=-1) + eps
    row_max = reduce_max(comb, dim=1)
    comb = exp(comb - row_max)
    comb = comb / sum(comb, dim=1) + hc_sinkhorn_eps
    # comb = comb / (sum(comb, dim=-2) + eps)
    comb = comb / (sum(comb, dim=0) + hc_sinkhorn_eps)
    # 迭代 sinkhorn_repeat - 1 次
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (sum(comb, dim=1) + eps)  # 行归一化
        comb = comb / (sum(comb, dim=0) + eps)  # 列归一化
    return comb

# 2. PyTorch 回退 (HIP/ROCm)
def sinkhorn_torch(comb_logits, hc_sinkhorn_eps, sinkhorn_repeat):
    comb = softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    return comb

# 3. Ascend NPU 自定义算子 (npu_hc_pre)
# 在 NPU 上通过 C++ 自定义算子实现，与上述逻辑等价
```

### 6.6 融合 Post-Pre

```
MHCFusedPostPreOp = MHCPostOp → MHCPreOp (融合为一个 kernel)

目的: 减少 kernel launch 开销
输入: x, residual, post_mix, comb_mix, fn_next, hc_scale_next, hc_base_next, ...
输出: (residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur)
```

---

## 7. MTP 推测解码

### 7.1 V3 vs V4 MTP 对比

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        V3 MTP                                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  输入: embedding(token_{t+1}) + hidden_states[t]                            ║
║                                                                              ║
║  ┌──────────────────────────────────────────────┐                          ║
║  │  eh_proj (融合): [H+H] → [2H]               │                          ║
║  │  e_part = eh_proj(emb)[:H]   (FP8 quant)    │                          ║
║  │  h_part = eh_proj(emb)[H:]   (FP8 quant)    │                          ║
║  └──────────────────────────────────────────────┘                          ║
║                                                                              ║
║  e_part → enorm → e_out                                                      ║
║  h_part → hnorm → h_out                                                      ║
║  combined = e_out + h_out                                                    ║
║  → mtp_block (标准 DeepseekV2DecoderLayer)                                  ║
║  → shared_head → logits                                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║                        V4 MTP                                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  输入: embedding(token_{t+1}) + _mtp_hidden_buffer (pre-hc_head 多分支残差) ║
║                                                                              ║
║  ┌──────────────────────────────────────────────┐                          ║
║  │  e_proj: emb → hidden (FP8 quant)            │                          ║
║  │  h_proj: hidden → hidden (FP8 quant)         │                          ║
║  └──────────────────────────────────────────────┘                          ║
║                                                                              ║
║  e_part → enorm → e_out                                                      ║
║  h_part → hnorm → h_out                                                      ║
║  combined = e_out + h_out                                                    ║
║  → mtp_block (DeepseekV4DecoderLayer, 含 MHC)                               ║
║  → HC Head (hc_head_fn/scale/base) → logits                                 ║
║                                                                              ║
║  关键差异:                                                                    ║
║  • 分离 e_proj/h_proj (V3 融合为 eh_proj)                                   ║
║  • V4 使用 HC Head 替代 shared_head                                         ║
║  • V4 的 mtp_block 是 DeepseekV4DecoderLayer (含 MHC 前后处理)              ║
║  • V4 使用 _mtp_hidden_buffer (多分支残差流) 而非 单路 hidden_states        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 7.2 多步推测流程

```
主模型 forward:
  tokens[t] → Model → hidden_states (pre-hc_head) → _mtp_hidden_buffer
                    → HC Head → logits[t]

MTP draft (step 1):
  token[t+1]_emb + _mtp_hidden_buffer
    → MTP Layer 0 → logits[t+1]
    → 采样 → token[t+1]

MTP draft (step 2):
  token[t+2]_emb + MTP_Layer_0_hidden
    → MTP Layer 1 → logits[t+2]
    → 采样 → token[t+2]

... 最多 spec_k 步

验证:
  主模型用 tokens[t:t+spec_k+1] 做一次 forward
  → 对比 draft tokens，接受匹配的部分
```

---

## 8. 量化方案

### 8.1 V3 量化

```
线性层: FP8 block-scaled (Fp8Config)
MoE 专家: FP8 block-scaled + float32 scale
```

### 8.2 V4 量化全景

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    DeepSeek V4 量化方案                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  expert_dtype = "fp4" (MegaMoE 路径):                                        ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │ 线性层: FP8 block-scaled + UE8M0 scale                                │  ║
║  │                                                                        │  ║
║  │ MoE w13_weight: [E, 2×inter, hidden//2]  uint8 (打包 FP4)            │  ║
║  │      w13_scale: [E, 2×inter, hidden//32]  uint8 (UE8M0)             │  ║
║  │ MoE w2_weight:  [E, hidden, inter//2]     uint8 (打包 FP4)           │  ║
║  │      w2_scale:  [E, hidden, inter//32]    uint8 (UE8M0)             │  ║
║  │                                                                        │  ║
║  │ moe_quant_algo: "NVFP4" (ModelOpt NVFP4) 或默认                       │  ║
║  │                                                                        │  ║
║  │ 权重变换 (finalize_weights):                                           │  ║
║  │   deep_gemm.transform_sf_into_required_layout(scale)                   │  ║
║  │   deep_gemm.transform_weights_for_mega_moe(weight)                     │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  expert_dtype = "fp8" (标准路径):                                            ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │ 线性层: FP8 block-scaled + float32 scale                              │  ║
║  │ MoE 专家: FP8 block-scaled + float32 scale                           │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  Compressor 量化:                                                            ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │ Nope 部分: FP8 + UE8M0 scale (block=64, 7 blocks + 1 pad)            │  ║
║  │ Rope 部分: bf16 直通                                                  │  ║
║  │ 缓存布局: [448B fp8, 128B bf16, 8B scales] = 584B per compressed tok │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  Indexer 量化:                                                               ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │ FP8:   per-token-per-head 单 scalar 量化                              │  ║
║  │ MXFP4: 32 元素 block E2M1 量化 + UE8M0 scale                          │  ║
║  │                                                                        │  ║
║  │ MXFP4 格式: E2M1 (指数2位 + 尾数1位), 每字节打包2个                    │  ║
║  │ 可表示范围: ±6.0 (max), ±2^-14 (min subnormal)                        │  ║
║  │ scale = 2^ceil(log2(block_amax / 6.0))                                │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  O-proj 量化:                                                                ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │ fused_inv_rope_fp8_quant:                                              │  ║
║  │   • Inverse RoPE + block-scaled FP8 (block=128)                       │  ║
║  │   • SM90: FP32 scale, SM100: INT32 packed UE8M0                       │  ║
║  │ wo_a: FP8 weight + UE8M0 scale                                        │  ║
║  │ fp8_einsum: "bhr,hdr->bhd" with recipe (1,128,128) or (1,1,128)      │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 8.3 UE8M0 格式

```
UE8M0: Unsigned Exponent 8-bit, Mantissa 0-bit
  格式: [E7 E6 E5 E4 E3 E2 E1 E0]
  值 = 2^(exponent - 127)
  
  优点:
  • 2 的幂次方 scale，反量化只需整数加法
  • 8-bit 存储，紧凑
  • 硬件友好
```

---

## 9. 多流并行调度

### 9.1 三路并行 GEMM 调度

```
╔══════════════════════════════════════════════════════════════════════════════╗
║              V4 Attention 多流并行调度                                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  阶段 1: 并行 GEMM (attn_gemm_parallel_execute)                              ║
║                                                                              ║
║  ┌──────────────────┐  ┌─────────────────────┐  ┌────────────────────────┐ ║
║  │ 默认流 (Stream 0) │  │ Aux Stream 1        │  │ Aux Stream 2           │ ║
║  │                  │  │                     │  │                        │ ║
║  │ fused_wqa_wkv    │  │ compressor.          │  │ indexer.               │ ║
║  │ (主 GEMM)        │  │ fused_wkv_wgate      │  │ weights_proj           │ ║
║  │                  │  │ (Compressor KV+Gate) │  │ (Indexer 权重)         │ ║
║  │ → qr_kv          │  │ → kv_score           │  │ → indexer_weights       │ ║
║  └────────┬─────────┘  └──────────┬──────────┘  └───────────┬────────────┘ ║
║           │                       │                          │              ║
║           └───────────────────────┼──────────────────────────┘              ║
║                                   │ 同步 (events)                           ║
║                                   ▼                                         ║
║  阶段 2: 三路并行 (当 indexer + compressor 都存在)                            ║
║                                                                              ║
║  ┌──────────────────┐  ┌─────────────────────┐  ┌────────────────────────┐ ║
║  │ 默认流 (Stream 0) │  │ Aux Stream 0        │  │ Aux Stream 1           │ ║
║  │                  │  │                     │  │                        │ ║
║  │ wq_b → Q         │  │ indexer.forward()   │  │ compressor.forward()   │ ║
║  │ ↓                │  │ (Indexer Q量化+      │  │ (State Cache保存+      │ ║
║  │ _fused_qnorm     │  │  SparseAttnIndexer)  │  │  融合压缩量化+          │ ║
║  │ _rope_kv_insert  │  │                     │  │  KV Cache写入)          │ ║
║  │ ↓                │  │                     │  │                        │ ║
║  │ FlashMLA Sparse  │  │                     │  │                        │ ║
║  │ Attention        │  │                     │  │                        │ ║
║  └──────────────────┘  └─────────────────────┘  └────────────────────────┘ ║
║                                                                              ║
║  阶段 2b: 两路并行 (当仅 compressor 存在, 无 indexer)                         ║
║                                                                              ║
║  ┌──────────────────┐  ┌─────────────────────┐                              ║
║  │ 默认流 (Stream 0) │  │ Aux Stream 0        │                              ║
║  │ wq_b + kv_insert  │  │ compressor.forward()│                              ║
║  │ + FlashMLA        │  │                     │                              ║
║  └──────────────────┘  └─────────────────────┘                              ║
║                                                                              ║
║  阶段 2c: 单流 (SWA-only 层, 无 compressor/indexer)                          ║
║                                                                              ║
║  ┌──────────────────┐                                                       ║
║  │ wq_b → kv_insert  │                                                       ║
║  │ → FlashMLA        │                                                       ║
║  └──────────────────┘                                                       ║
║                                                                              ║
║  多流阈值:                                                                    ║
║  enable_parallel = num_tokens <= VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD      ║
║  (小 batch 时启用，大 batch 时 GPU 已饱和无需多流)                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 9.2 CUDA 融合 Kernel

```
╔══════════════════════════════════════════════════════════════════════════════╗
║        CUDA Kernel: fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  融合操作:                                                                    ║
║    1. Q 侧: RMSNorm (无权重) + GPT-J RoPE → in-place 写回 Q                 ║
║    2. KV 侧: GPT-J RoPE + UE8M0 FP8 量化 + paged KV cache insert            ║
║                                                                              ║
║  线程组织:                                                                    ║
║    Grid: ceil(num_tokens × (num_heads_q + 1) / 8) blocks                    ║
║    Block: 256 threads (8 warps)                                              ║
║    每个 warp: 处理一个 (token, head_slot) 对                                ║
║      head_slot < num_heads_q  → Q 分支                                      ║
║      head_slot == num_heads_q → KV 分支                                     ║
║                                                                              ║
║  硬编码常量:                                                                  ║
║    head_dim=512, rope_dim=64, nope_dim=448                                   ║
║    quant_block=64, n_quant_blocks=7                                          ║
║    scale_bytes_per_token=8, token_data_bytes=576                             ║
║                                                                              ║
║  KV Cache 布局 (per paged block):                                            ║
║    [0, bs×576):      token data (448 fp8 + 128 bf16)                        ║
║    [bs×576, bs×584): UE8M0 scales (7 real + 1 pad per token)                ║
║                                                                              ║
║  Reduced Grid 优化:                                                           ║
║    当 num_tokens >= 1024 时，每 token 一个 block，warp 循环所有 head_slot    ║
║    减少 grid launch 开销                                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 10. Ascend 平台适配

### 10.1 Ascend 与 NVIDIA 对比

```
╔══════════════════════════════════════════════════════════════════════════════╗
║               Ascend vs NVIDIA 组件映射                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌──────────────────────┬──────────────────────┬─────────────────────────┐ ║
║  │ 组件                 │ NVIDIA               │ Ascend (vllm-ascend)    │ ║
║  ├──────────────────────┼──────────────────────┼─────────────────────────┤ ║
║  │ MLA Attention        │ FlashMLA/TRT-LLM MLA │ AscendDeepseekSparse    │ ║
║  │                      │                      │ Attention (DSA)         │ ║
║  │ RoPE                 │ GPT-J (CUDA kernel)  │ npu_rotary_mul          │ ║
║  │                      │                      │ (interleave mode)       │ ║
║  │ MHC Pre/Post         │ Tilelang CUDA kernel │ npu_hc_pre/npu_hc_post  │ ║
║  │                      │                      │ (NPU 自定义算子)        │ ║
║  │ KV Cache Backend     │ FlashMLA Backend     │ AscendDSABackend        │ ║
║  │ KV Transfer          │ Mooncake (P2P)       │ Mooncake Hybrid (Ascend)│ ║
║  │ Graph Capture        │ CUDA Graph           │ ACL Graph               │ ║
║  │ Quant Type           │ float8_e4m3fn        │ A5:float8_e4m3fn,       │ ║
║  │                      │                      │ 其他:int8               │ ║
║  └──────────────────────┴──────────────────────┴─────────────────────────┘ ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 10.2 Ascend 特有优化

#### 10.2.1 ComplexExpRotaryEmbedding

```python
# Ascend 专用 RoPE: 使用 npu_rotary_mul 原生算子
class ComplexExpRotaryEmbedding:
    """
    全局静态缓存 (RopeGlobalState):
      • 跨层共享预计算的 cos/sin 表
      • 避免重复计算
    
    运行时缓冲:
      • 为每个 group 分配 max_num_batched_tokens 大小的 cos/sin 缓冲
    
    多 group 支持:
      • 同一层可有多个 rope group (如 "default" 和 "c4")
      • 适配不同 compress_ratio 层使用不同 rope_theta
    """
    def forward(self, x, positions):
        return torch_npu.npu_rotary_mul(
            x, cos, sin, rotary_mode="interleave"
        )
```

#### 10.2.2 Hadamard Transform

```python
# 激活值旋转: 增强数值稳定性
def rotate_activation(x):
    """
    x: [T, hidden_size]
    使用 Hadamard 正交矩阵对激活值做旋转
    scale = hidden_size ** -0.5
    """
    dim = x.shape[-1]
    dim_padded = next_power_of_2(dim)
    hadamard_matrix = scipy.linalg.hadamard(dim_padded)
    return F.linear(x, hadamard_matrix[:dim, :dim]) * (dim ** -0.5)
```

#### 10.2.3 Patch 系统

```
vllm-ascend 通过 monkey-patch 修改 vllm 行为:

┌────────────────────────────────────────────────────────────────┐
│ Patch 文件                   │ 目的                           │
├──────────────────────────────┼────────────────────────────────┤
│ patch_deepseek_mtp           │ 为 GLM5 + rotary quant 扩展   │
│                              │ MTP 层，添加 rot.weight        │
│ patch_deepseek_compressor    │ 替换 CompressorStateCache 为   │
│                              │ Ascend 后端版本                │
│ patch_deepseek_v4_tool_parser│ 支持流式工具调用增量参数发送    │
│ patch_kv_cache_utils         │ 调整 KV cache block size       │
│                              │ 和分组策略                     │
│ patch_kv_cache_interface     │ 扩展 MLA spec 支持 DSA         │
│                              │ 和 Sparse C8                   │
│ patch_speculative_config     │ 映射 MTP speculative config    │
│ patch_weight_utils           │ 增强权重加载 (remap            │
│                              │ kv_scale_name)                 │
└──────────────────────────────┴────────────────────────────────┘
```

#### 10.2.4 Ascend V4 Model 关键差异

```python
# 1. hidden_states 多分支展开 (DeepseekV4Model.forward)
hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
# [T, H] → [T, hc_mult, H]

# 2. HC Pre/Post 使用 NPU 自定义算子
hc_pre  → torch.ops._C_ascend.npu_hc_pre(...)
hc_post → torch.ops._C_ascend.npu_hc_post(...)

# 3. HC Head 是纯 PyTorch (非 NPU 算子)
hc_head → sigmoid(RMSNorm(x) @ hc_fn * scale + base) × x

# 4. MTP 缓冲
_mtp_hidden_buffer[:num_tokens] = hidden_states.flatten(1)  # pre-hc_head 多分支残差

# 5. 跳过 rotary_emb.inv_freq 加载 (使用自己的 ComplexExpRotaryEmbedding)
if "rotary_emb.inv_freq" in name:
    continue
```

---

## 11. 性能优化总结

### 11.1 内存优化

| 技术 | 适用模型 | 效果 | 原理 |
|------|---------|------|------|
| MLA KV 低秩压缩 | V2/V3/V4 | KV Cache ↓ ~10-50x | KV 投影到 kv_lora_rank=512 维 |
| Compressor C4 | V4 | KV Cache 再 ↓ 4x | 时间维度 softmax 加权压缩 |
| Compressor C128 | V4 | KV Cache 再 ↓ 128x | 大窗口时间维度压缩 |
| FP8 KV Cache | V3/V4 | KV 精度减半 | UE8M0 block-scaled FP8 |
| IndexCache | V3.2/V4 | 减少 topk 重算 | 跨层复用 topk 选择 |
| State Cache 复用 | V4 | 避免重复存储 | CompressorStateCache 统一管理 |

### 11.2 计算优化

| 技术 | 说明 | 加速比来源 |
|------|------|-----------|
| Fused QKV-A Proj (V3) | Q_A + KV_A 合并 GEMM | 减少 kernel launch |
| Min-Latency GEMM (V3) | ≤16 tokens 专用 CUDA kernel | 小 batch 场景优化 |
| FlashMLA (V4) | 高性能 MLA sparse attention | 专用 attention kernel |
| DeepGEMM MegaMoE (V4) | FP4 专家 + SM100 | 4-bit 权重 × 8-bit 激活 |
| Hash MoE (V4) | 查表替代 router | 免 router 计算 |
| Fused Compress Kernel (V4) | 5 操作融合为 1 kernel | 减少显存往返 |
| Fused QNorm+RoPE+KV Insert (V4) | Q 和 KV 端融合 | 减少 kernel launch |
| Multi-Stream Overlap (V4) | QKV + Compressor + Indexer 并行 | GPU 利用率提升 |

### 11.3 分布式优化

| 技术 | 说明 |
|------|------|
| Expert Parallel (EP) | 专家分布到多 GPU |
| EPLB | 冗余专家 + 动态重映射负载均衡 |
| Sequence Parallel MoE | MoE 输入序列并行 |
| Pipeline Parallel (PP) | 层间流水线 |
| PD Separation | Prefill-Decode 分离部署 |
| Mooncake KV Transfer | 跨节点 KV Cache P2P |
| FlashComm v1 (Ascend) | all_gather 恢复全量 token |

### 11.4 关键环境变量

```bash
# DeepGEMM MegaMoE (NVIDIA SM100)
--kernel-config moe_backend=deep_gemm_mega_moe

# V4 TRT-LLM MLA
VLLM_DEEPSEEK_V4_USE_TRTLLM_MLA=1

# Multi-Stream GEMM 阈值
VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=<int>

# Ascend DSV4 Patch
VLLM_ASCEND_APPLY_DSV4_PATCH=1

# Ascend MLA Prefill Optimization
VLLM_ASCEND_ENABLE_MLAPO=1

# Indexer FP4 Cache
--attention-config use_fp4_indexer_cache=true
```

---

## 12. 关键文件索引

### 12.1 vllm 文件

**V3/V3.1/V3.2 核心:**
- [deepseek_v2.py](file:///D:/trae-workspace/github/vllm/vllm/model_executor/models/deepseek_v2.py) — 统一模型实现
- [deepseek_mtp.py](file:///D:/trae-workspace/github/vllm/vllm/model_executor/models/deepseek_mtp.py) — MTP 推测解码

**V4 核心:**
- [models/deepseek_v4/nvidia/model.py](file:///D:/trae-workspace/github/vllm/vllm/models/deepseek_v4/nvidia/model.py) — NVIDIA V4 主模型
- [models/deepseek_v4/nvidia/mtp.py](file:///D:/trae-workspace/github/vllm/vllm/models/deepseek_v4/nvidia/mtp.py) — V4 MTP
- [models/deepseek_v4/compressor.py](file:///D:/trae-workspace/github/vllm/vllm/models/deepseek_v4/compressor.py) — KV Cache 压缩器
- [models/deepseek_v4/quant_config.py](file:///D:/trae-workspace/github/vllm/vllm/models/deepseek_v4/quant_config.py) — FP8/FP4 量化配置

**V4 Attention 算子:**
- [models/deepseek_v4/nvidia/ops/attention.py](file:///D:/trae-workspace/github/vllm/vllm/models/deepseek_v4/nvidia/ops/attention.py) — MLA Wrapper + Indexer
- [models/deepseek_v4/nvidia/flashmla.py](file:///D:/trae-workspace/github/vllm/vllm/models/deepseek_v4/nvidia/flashmla.py) — FlashMLA
- [models/deepseek_v4/common/ops/fused_qk_rmsnorm.py](file:///D:/trae-workspace/github/vllm/vllm/models/deepseek_v4/common/ops/fused_qk_rmsnorm.py) — 融合 QK RMSNorm
- [models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py](file:///D:/trae-workspace/github/vllm/vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py) — 逆 RoPE + FP8 量化
- [models/deepseek_v4/common/ops/fused_compress_quant_cache.py](file:///D:/trae-workspace/github/vllm/vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py) — 融合压缩量化缓存
- [models/deepseek_v4/common/ops/fused_indexer_q.py](file:///D:/trae-workspace/github/vllm/vllm/models/deepseek_v4/common/ops/fused_indexer_q.py) — 融合 Indexer Q
- [models/deepseek_v4/common/ops/cache_utils.py](file:///D:/trae-workspace/github/vllm/vllm/models/deepseek_v4/common/ops/cache_utils.py) — KV 缓存工具

**MHC/HC:**
- [model_executor/layers/mhc.py](file:///D:/trae-workspace/github/vllm/vllm/model_executor/layers/mhc.py) — MHC Pre/Post/Head 算子
- [_tilelang_ops.py](file:///D:/trae-workspace/github/vllm/vllm/_tilelang_ops.py) — Sinkhorn tilelang CUDA 实现
- [model_executor/kernels/mhc/torch.py](file:///D:/trae-workspace/github/vllm/vllm/model_executor/kernels/mhc/torch.py) — Sinkhorn PyTorch 回退

**CUDA Kernel:**
- [csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu](file:///D:/trae-workspace/github/vllm/csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu) — V4 融合 kernel

**配置:**
- [transformers_utils/configs/deepseek_v4.py](file:///D:/trae-workspace/github/vllm/vllm/transformers_utils/configs/deepseek_v4.py) — DeepseekV4Config

### 12.2 vllm-ascend 文件

**模型实现:**
- [models/deepseek_v4.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/models/deepseek_v4.py) — Ascend V4 完整实现 (1334行)
- [models/deepseek_v4_mtp.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/models/deepseek_v4_mtp.py) — Ascend V4 MTP

**Ascend 算子:**
- [ops/dsa.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/ops/dsa.py) — AscendDeepseekSparseAttention
- [ops/rope_dsv4.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/ops/rope_dsv4.py) — ComplexExpRotaryEmbedding
- [attention/dsa_v1.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/attention/dsa_v1.py) — DSA v1 注意力后端

**Patch 系统:**
- [patch/__init__.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/patch/__init__.py) — Patch 系统总览
- [patch/worker/patch_deepseek_mtp.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/patch/worker/patch_deepseek_mtp.py)
- [patch/worker/patch_deepseek_compressor.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/patch/worker/patch_deepseek_compressor.py)
- [patch/platform/patch_kv_cache_utils.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/patch/platform/patch_kv_cache_utils.py)

---

> **文档生成说明**: 本文档基于 vllm 和 vllm-ascend 源码深入分析生成，覆盖了 200+ 个相关文件的代码阅读，包含完整的架构图、数据流图和代码级技术细节。重点涵盖 DeepSeek V3.1/V4 的 MLA 注意力、MoE 路由、KV Cache 压缩、MHC 多分支残差、MTP 推测解码、量化方案、多流并行调度以及华为 Ascend NPU 平台适配。
