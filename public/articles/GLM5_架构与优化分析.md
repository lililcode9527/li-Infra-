# GLM-5 / GLM-4.x 架构与优化深度分析

> 基于 vllm 和 vllm-ascend 源码深入分析  
> 分析日期：2026-05-27

---

## 1. GLM 系列演进总览

### 1.1 版本演进路线

```
                    ChatGLM / GLM-2 / GLM-3
                    标准 Dense Transformer
                    架构: Pre-Norm + MHA + SwiGLU
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
          GLM-4           GLM-4V         ChatGLM3 多模态
          Dense 模型     多模态视觉       (早期视觉探索)
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
GLM-4.5     GLM-4.6    GLM-4.7         ← MoE 架构
(MoE)       (MoE)      (MoE + MLA)
    │          │          │
    └──────────┼──────────┘
               │
    ┌──────────▼──────────┐
    ▼                     ▼
GLM-5                  GLM-5.1           ← DSA 架构 (复用 DeepSeek V2)
(MoE + MLA + DSA)      (MoE + MLA + DSA)
    │                     │
    └──────────┬──────────┘
               │
    ┌──────────┼──────────┬──────────────┐
    ▼          ▼          ▼              ▼
GLM-ASR     GLM-OCR    GLM-4.1V     AutoGLM-Phone
语音识别     OCR识别    多模态 VLM    手机 Agent
```

### 1.2 全系列架构对比

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                      GLM 系列架构全景对比                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌──────────────┬─────────────┬──────────────┬──────────────┬────────────┐ ║
║  │ 特性          │ GLM-4.5/4.6 │ GLM-4.7      │ GLM-4.7-Flash │ GLM-5/5.1  │ ║
║  ├──────────────┼─────────────┼──────────────┼──────────────┼────────────┤ ║
║  │ 注意力        │ MHA + QK-Norm│ MHA + QK-Norm│ MLA (DSv2)   │ MLA (DSv2) │ ║
║  │ RoPE          │ 标准 GPT-J   │ 标准 GPT-J   │ DS YaRN      │ DS YaRN    │ ║
║  │ MoE 路由      │ Grouped TK  │ Grouped TK   │ Grouped TK   │ Grouped TK │ ║
║  │ 评分函数      │ sigmoid      │ sigmoid      │ sigmoid      │ sigmoid    │ ║
║  │ 路由缩放      │ √            │ √            │ √            │ √          │ ║
║  │ 共享专家      │ 1            │ 1            │ 1            │ 1          │ ║
║  │ MLP 激活      │ SiLU         │ SiLU         │ SiLU         │ SiLU       │ ║
║  │ Partial RoPE  │ 0.5          │ 0.5          │ N/A (MLA)    │ N/A (MLA)  │ ║
║  │ DSA/SparseAttn│ ✗            │ ✗            │ √ (V3.2)     │ √ (V3.2)   │ ║
║  │ Rotary Quant  │ ✗            │ ✗            │ ✗            │ √          │ ║
║  │ MTP           │ √ (融合eh)   │ √ (融合eh)   │ √ (融合eh)   │ √ (DS MTP) │ ║
║  │ 工具解析      │ GLM45 XML    │ GLM47 XML    │ GLM47 XML    │ GLM47 XML  │ ║
║  │ 推理解析      │ GLM45        │ GLM45        │ GLM45        │ GLM45      │ ║
║  │ vocab_size    │ ~65024       │ ~65024       │ ~65024       │ ~65024     │ ║
║  │ 多模态        │ ✗            │ ✗            │ ✗            │ ✗          │ ║
║  └──────────────┴─────────────┴──────────────┴──────────────┴────────────┘ ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 1.3 模型注册与架构类

```python
# vllm 模型注册表 (registry.py)
"GlmForCausalLM"          → glm.py       → GlmForCausalLM          # GLM-4
"Glm4ForCausalLM"         → glm4.py      → Glm4ForCausalLM         # GLM-4-0414
"Glm4MoeForCausalLM"      → glm4_moe.py  → Glm4MoeForCausalLM      # GLM-4.5/4.6/4.7
"Glm4MoeLiteForCausalLM"  → glm4_moe_lite.py → Glm4MoeLiteForCausalLM  # GLM-4.7-Flash
"GlmMoeDsaForCausalLM"    → deepseek_v2.py   → GlmMoeDsaForCausalLM    # GLM-5/5.1
```

---

## 2. GLM-5 架构总览

### 2.1 核心架构

GLM-5 是智谱 AI 最新的旗舰模型，采用 **MoE + MLA + DSA（DeepSeek Sparse Attention）** 架构，面向复杂系统工程和长周期 Agent 任务。

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        GLM-5 架构全景                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   Input → Embed ──────────────────────────────────────────────────────┐      ║
║              │                                                        │      ║
║   ┌──────────▼─────────── Decoder Layer (×N) ───────────────┐        │      ║
║   │  ┌─────────────────────────────────────────────────────┐ │        │      ║
║   │  │  RMSNorm → MLA Attention (DeepSeek V2 架构)        │ │        │      ║
║   │  │    │                                                  │ │        │      ║
║   │  │    ├── fused_qkv_a_proj (合并 Q_A + KV_A)            │ │        │      ║
║   │  │    ├── q_a → LN → q_b → Q                           │ │        │      ║
║   │  │    ├── kv_a → kv_lora → LN → kv_b → K_nope, V       │ │        │      ║
║   │  │    ├── k_pe → RoPE → K_rope                         │ │        │      ║
║   │  │    └── DSA (V3.2 Sparse Attention + Indexer)        │ │        │      ║
║   │  └─────────────────────────────────────────────────────┘ │        │      ║
║   │  ┌─────────────────────────────────────────────────────┐ │        │      ║
║   │  │  RMSNorm → MoE (Grouped Top-K + Shared Expert)      │ │        │      ║
║   │  │    │                                                  │ │        │      ║
║   │  │    ├── Router: sigmoid + e_score_correction_bias     │ │        │      ║
║   │  │    ├── Grouped Top-K 选择                            │ │        │      ║
║   │  │    ├── Routed Experts (SwiGLU)                       │ │        │      ║
║   │  │    ├── Shared Expert (1×, 所有 token)               │ │        │      ║
║   │  │    └── routed_scaling_factor 缩放                    │ │        │      ║
║   │  └─────────────────────────────────────────────────────┘ │        │      ║
║   └──────────────────────────────────────────────────────────┘        │      ║
║                                                                       │      ║
║   ┌── RMSNorm ──► LM Head ──► Logits ◄───────────────────────────────┘      ║
║                                                                              ║
║   关键特征：                                                                 ║
║   • 继承 DeepSeek V2 架构 (GlmMoeDsaForCausalLM)                            ║
║   • MLA 低秩 KV 压缩 (kv_lora_rank + rope_dim)                              ║
║   • V3.2 级别 DSA 稀疏注意力 (Indexer + IndexCache)                         ║
║   • sigmoid 评分 + e_score_correction_bias                                  ║
║   • Rotary Quantization (rot.weight) - GLM-5 专有特性                       ║
║   • MTP 推测解码 (DeepSeek 风格, MTP start layer = num_hidden_layers)       ║
║   • FP8/W8A8/W4A8 量化支持                                                  ║
║   • 支持 transformers 5.2.0+                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 2.2 GLM-5 与 DeepSeek V3 的关系

```
GLM-5 不是独立的模型架构，而是通过继承 DeepSeek V2 架构实现的：

class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
    pass  # 完全复用 DeepSeek V2 架构

关键差异（通过 HF config 配置而非代码）：
  • model_type = "glm_moe_dsa" (vs "deepseek_v3" 或 "deepseek_v2")
  • scoring_func = "sigmoid" (vs "softmax")
  • 有 e_score_correction_bias 参数
  • 支持 Rotary Quantization
  • 工具解析用 GLM47 XML 格式 (vs DeepSeek JSON/DSML)
  • 推理解析用 GLM45
```

### 2.3 GLM-5.1 参数

```python
# benchmarks/kernels/benchmark_fused_moe_lora_one_shot.py
"glm5_1": dict(
    K=6144,              # hidden_size
    N_per_slice=2048,    # moe_intermediate_size
    E=256,               # n_routed_experts
    top_k=8,             # num_experts_per_tok
    # 约 2T 总参数量级（基于 MoE 架构推算）
)
```

---

## 3. GLM-4.x MoE 架构

### 3.1 GLM-4.5/4.6/4.7 架构

GLM-4.x 系列使用**标准 MHA + MoE** 架构（非 MLA），是智谱最早的 MoE 模型系列。

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    GLM-4.5/4.6/4.7 架构                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   Input → Embed ──────────────────────────────────────────────────────┐      ║
║              │                                                        │      ║
║   ┌──────────▼─────────── Decoder Layer (×N) ───────────────┐        │      ║
║   │  ┌─────────────────────────────────────────────────────┐ │        │      ║
║   │  │  RMSNorm → MHA Attention (标准 GQA)                 │ │        │      ║
║   │  │    │                                                  │ │        │      ║
║   │  │    ├── qkv_proj (QKV 合并投影)                       │ │        │      ║
║   │  │    ├── QK-Norm (可选, RMSNorm per head)              │ │        │      ║
║   │  │    ├── Partial RoPE (partial_rotary_factor=0.5)      │ │        │      ║
║   │  │    └── o_proj                                        │ │        │      ║
║   │  └─────────────────────────────────────────────────────┘ │        │      ║
║   │  ┌─────────────────────────────────────────────────────┐ │        │      ║
║   │  │  RMSNorm → MoE / Dense MLP                          │ │        │      ║
║   │  │    │                                                  │ │        │      ║
║   │  │    ├── MoE 层: Grouped Top-K + Shared Expert         │ │        │      ║
║   │  │    │   • Router: nn.Linear (float32)                  │ │        │      ║
║   │  │    │   • sigmoid 评分 + e_score_correction_bias       │ │        │      ║
║   │  │    │   • routed_scaling_factor (路由输出缩放)        │ │        │      ║
║   │  │    │   • apply_routed_scale_to_output=True           │ │        │      ║
║   │  │    ├── Dense 层: SwiGLU MLP (前 k 层和每隔 M 层)    │ │        │      ║
║   │  │    └── first_k_dense_replace + moe_layer_freq        │ │        │      ║
║   │  └─────────────────────────────────────────────────────┘ │        │      ║
║   └──────────────────────────────────────────────────────────┘        │      ║
║                                                                       │      ║
║   ┌── RMSNorm (残差融合) ──► LM Head ──► Logits ◄────────────────────┘      ║
║                                                                              ║
║   关键特征：                                                                 ║
║   • 标准 MHA + GQA (非 MLA, KV Cache 无低秩压缩)                            ║
║   • Partial RoPE (仅前 50% 维度旋转)                                        ║
║   • QK-Norm: 每 head 的 Q/K RMSNorm                                         ║
║   • sigmoid 评分 + e_score_correction_bias (与 DeepSeek noaux_tc 类似)      ║
║   • routed_scaling_factor 控制路由专家的贡献度                               ║
║   • 混合 Dense/MoE 层 (first_k_dense_replace + moe_layer_freq)              ║
║   • EPLB 支持 (冗余专家 + 动态重映射)                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 3.2 MHA Attention 数据流

```
                    hidden_states [T, hidden_size]
                              │
                    ┌─────────▼──────────┐
                    │    qkv_proj         │
                    │ (QKVParallelLinear) │
                    └─────────┬──────────┘
                              │
                    qkv [T, q_size + 2×kv_size]
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         Q [T, Hq, D]   K [T, Hkv, D]   V [T, Hkv, D]
              │               │
         ┌────▼────┐    ┌────▼────┐
         │ QK-Norm │    │ QK-Norm │  (可选: use_qk_norm=True)
         │ (RMSNorm│    │ (RMSNorm│
         │ per head)│    │ per head)│
         └────┬────┘    └────┬────┘
              │               │
         ┌────▼───────────────▼────┐
         │  Rotary Embedding       │
         │  (partial_rotary=0.5)   │
         │  仅前 50% dims 做 RoPE  │
         └────────────┬────────────┘
                      │
              ┌───────▼───────┐
              │  Attention     │
              │  (GQA, flash)  │
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │    o_proj      │
              └───────┬───────┘
                      │
                 output [T, hidden_size]
```

### 3.3 MoE 路由详解

```python
# glm4_moe.py: Glm4MoE 类
class Glm4MoE(nn.Module):
    """
    关键参数:
      n_routed_experts: 路由专家数 (如 128)
      n_shared_experts: 共享专家数 (1)
      num_experts_per_tok: 每 token 激活专家数 (top_k)
      n_group: 专家组数
      topk_group: 每组选择的组数
      scoring_func: "sigmoid"
      routed_scaling_factor: 路由输出缩放
      norm_topk_prob: 是否归一化 top-k 概率
      e_score_correction_bias: 专家评分修正偏置

    路由流程:
      1. gate = nn.Linear(hidden, n_routed_experts, dtype=float32)
      2. router_logits = gate(hidden_states.float())
      3. FusedMoE 内部:
         a. sigmoid(router_logits) + e_score_correction_bias
         b. Grouped Top-K 选择 (组间→组内)
         c. 归一化 top-k 概率
         d. 专家计算 + 加权求和
         e. × routed_scaling_factor
      4. + shared_expert 输出
    """
```

---

## 4. GLM-4.7-Flash

### 4.1 架构定位

GLM-4.7-Flash 是 GLM-4.7 的轻量版，引入 MLA（Multi-head Latent Attention）以大幅降低 KV Cache 显存占用。

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    GLM-4.7-Flash 架构                                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  架构 = GLM-4.x 的 MoE + DeepSeek V2 的 MLA                                  ║
║                                                                              ║
║  核心变化 (相对于 GLM-4.7):                                                  ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │  注意力: MHA → MLA (DeepSeekV2Attention / DeepSeekV2MLAAttention)   │  ║
║  │  KV Cache: 标准 → 低秩压缩 (kv_lora_rank=512)                       │  ║
║  │  RoPE: Partial RoPE → DeepSeek YaRN RoPE                            │  ║
║  │  QK-Norm: 不需要 (MLA 内部已处理)                                    │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  继承关系:                                                                   ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │  Glm4MoeLiteDecoderLayer                                              │  ║
║  │    ├── self_attn: DeepseekV2Attention (或 DeepseekV2MLAAttention)    │  ║
║  │    ├── mlp: Glm4MoeLite (继承 Glm4MoE)                               │  ║
║  │    ├── input_layernorm: RMSNorm                                       │  ║
║  │    └── post_attention_layernorm: RMSNorm                              │  ║
║  │                                                                        │  ║
║  │  Glm4MoeLite → Glm4MoE (完全继承, 无修改)                             │  ║
║  │  Glm4MoeLiteAttention → DeepseekV2Attention (完全继承)                │  ║
║  │  Glm4MoeLiteMLAAttention → DeepseekV2MLAAttention (完全继承)          │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  支持 Indexer (V3.2 稀疏注意力):                                            ║
║    if hasattr(config, "index_topk"):                                         ║
║        topk_indices_buffer = torch.empty(max_batch, index_topk)              ║
║        → 传递给 DeepseekV2Attention                                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 4.2 MLA vs MHA 对比

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  GLM-4.7 (标准 MHA)                    GLM-4.7-Flash (MLA)                   ║
║  ─────────────────                     ────────────────                       ║
║                                                                              ║
║  Q,K,V 直接投影:                       Q,K,V 低秩投影:                       ║
║  hidden → [Q, K, V]                    hidden → kv_a (压缩) → kv_b (解压)    ║
║  Q_dim = n_heads × head_dim            kv_cache_dim = kv_lora_rank + rope    ║
║  KV Cache = 2 × n_kv × head_dim        KV Cache = kv_lora_rank + rope_dim    ║
║  ≈ 2 × 8 × 128 = 2048 bytes/tok       ≈ 512 + 64 = 576 bytes/tok            ║
║                                                                              ║
║  压缩比: 1x (无压缩)                   压缩比: ~3.5x (MLA 低秩压缩)           ║
║                                                                              ║
║  RoPE: partial_rotary_factor=0.5       RoPE: full dims, YaRN scaling         ║
║  QK-Norm: 可选                         QK-Norm: 不需要                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 5. DSA 稀疏注意力

### 5.1 DSA 架构集成

GLM-5 直接继承 `DeepseekV2ForCausalLM`，这意味着它复用了 DeepSeek V2/V3 的全部架构特性：

```
╔══════════════════════════════════════════════════════════════════════════════╗
║              GLM-5 DSA 架构 = DeepSeek V2/V3 架构 + GLM 特性                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  继承自 DeepSeek V2 的组件:                                                  ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │  ✓ MLA 低秩注意力 (Q_A + KV_A 投影, kv_lora_rank, q_lora_rank)       │  ║
║  │  ✓ Fused QKV-A Projection (合并 GEMM)                                │  ║
║  │  ✓ YaRN RoPE (支持 Llama-4 scaling)                                  │  ║
║  │  ✓ Grouped Top-K MoE 路由                                             │  ║
║  │  ✓ EPLB (专家并行负载均衡)                                             │  ║
║  │  ✓ FP8 量化 (block-scaled)                                            │  ║
║  │  ✓ V3.2 Indexer + DSA 稀疏注意力                                     │  ║
║  │  ✓ IndexCache (跨层复用 topk)                                        │  ║
║  │  ✓ DeepSeek 风格 MTP (eh_proj 融合)                                  │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  GLM 特有的差异化配置:                                                        ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │  ✓ scoring_func = "sigmoid" (vs DeepSeek 的 "softmax")                │  ║
║  │  ✓ e_score_correction_bias (专家评分修正)                              │  ║
║  │  ✓ routed_scaling_factor (路由输出缩放)                               │  ║
║  │  ✓ Rotary Quantization (rot.weight, MTP 层额外参数)                   │  ║
║  │  ✓ model_type = "glm_moe_dsa"                                        │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 5.2 V3.2 级别稀疏注意力

GLM-5 支持与 DeepSeek V3.2 同级别的 DSA 稀疏注意力：

```
Indexer 配置:
  index_n_heads: 64        # Indexer 专用 head 数
  index_head_dim: 128      # Indexer head 维度
  index_topk: 选择的 token 数
  index_rope_dim: 64       # RoPE 维度

IndexCache 配置 (可选):
  use_index_cache: True    # 跨层复用 topk 选择
  index_topk_freq: 1       # 每 N 层重新计算
  index_topk_pattern: "SSSS"  # Skip/Compute 模式

工作流程:
  1. Indexer Q: qr → wq_b → RoPE → FP8 quant
  2. Indexer K: hidden → weights_proj + Compressor
  3. SparseAttnIndexer: Q_fp8 @ K_fp8 → topk_indices
  4. FlashMLA: 只对 topk_indices + SWA_window 计算完整注意力
```

### 5.3 MLA 后端选择

GLM-5 在 vllm 上游的后端选择逻辑如下：

```
MLA 后端选择优先级（CUDA 平台）:
  ┌──────────────────────────────────────────────────────────────────┐
  │  Blackwell (SM 10.x):                                            │
  │  1. FLASHINFER_MLA  2. TOKENSPEED_MLA  3. CUTLASS_MLA          │
  │  4. FLASH_ATTN_MLA  5. FLASHMLA        6. TRITON_MLA            │
  │  7. FLASHINFER_MLA_SPARSE / FLASHMLA_SPARSE                     │
  │                                                                   │
  │  Ampere/Hopper (SM 8.x-9.x):                                     │
  │  1. FLASH_ATTN_MLA  2. FLASHMLA        3. FLASHINFER_MLA        │
  │  4. TRITON_MLA      5. FLASHMLA_SPARSE                           │
  │                                                                   │
  │  Sparse MLA 特殊规则:                                            │
  │  - FP8 KV cache: 优先 FLASHINFER_MLA_SPARSE                      │
  │  - BF16 KV cache: num_heads≤16 → FLASHINFER_MLA_SPARSE          │
  │                   否则 → FLASHMLA_SPARSE                          │
  └──────────────────────────────────────────────────────────────────┘
```

---

## 6. Rotary 量化

### 6.1 设计目的

Rotary Quantization 是 GLM-5 特有的优化技术，对 MTP 层中的 `previous_hidden_states` 应用**额外的可学习线性变换**，实现以下目的：

1. **特征增强**: 对主模型的 hidden states 做可学习的旋转变换，增强 MTP 层的特征提取能力
2. **量化友好**: 旋转后数值分布更均匀，量化误差更小
3. **训练稳定**: 作为可学习参数参与训练，自动适应模型分布

### 6.2 完整实现

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    GLM-5 Rotary Quantization 数据流                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  标准 MTP (DeepSeek V3):                                                     ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │  inputs_embeds → enorm → eh_proj([emb, prev_hidden]) → mtp_block     │  ║
║  │  previous_hidden_states → hnorm → eh_proj(...)                        │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  GLM-5 MTP (with Rotary Quantization):                                      ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │  inputs_embeds → enorm → eh_proj([emb, prev_hidden_rot]) → mtp_block │  ║
║  │  previous_hidden_states → rot.weight → hnorm → ...                    │  ║
║  │                          ↑                                            ║
║  │                    rot.weight: [H, H] (可学习线性变换)                  │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  配置方式:                                                                   ║
║  quant_description = {"is_rot_used": True}                                  ║
║  target_model_type = "glm_moe_dsa"                                          ║
║  两者同时满足时才创建 rot.weight                                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 6.3 Ascend 实现

```python
# vllm-ascend: patch/worker/patch_deepseek_mtp.py

class AscendDeepSeekMultiTokenPredictorLayer(DeepSeekMultiTokenPredictorLayer):
    def __init__(self, vllm_config, prefix):
        super().__init__(vllm_config, prefix)
        # 检查是否启用 rotary quant
        quant_desc = vllm_config.quant_config.quant_description
        self.is_rot_used = quant_desc.get("is_rot_used", False)
        self.target_model_type = vllm_config.speculative_config...model_type
        
        # 为 GLM-5 创建 rot.weight
        if self.is_rot_used and self.target_model_type == "glm_moe_dsa":
            self.rot = nn.Linear(
                self.config.hidden_size, 
                self.config.hidden_size, 
                bias=False
            )

    def forward(self, input_ids, positions, previous_hidden_states, ...):
        inputs_embeds = torch.where(positions == 0, 0, inputs_embeds)
        inputs_embeds = self.enorm(inputs_embeds)
        
        # GLM-5 专有: 对 previous_hidden_states 做旋转
        if self.is_rot_used and self.target_model_type == "glm_moe_dsa":
            previous_hidden_states = self.rot(previous_hidden_states)
        
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )
        # ... 后续与标准 MTP 相同
```

### 6.4 权重名称映射

```python
# Ascend MTP 权重名称重写
# rot.weight 在 checkpoint 中的名称: model.layers.{spec_layer}.rot.weight
# 通过 _rewrite_spec_layer_name 处理

def get_spec_layer_idx_from_weight_name(config, weight_name):
    """解析 MTP 层索引，支持 rot.weight"""
    if hasattr(config, "num_nextn_predict_layers"):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            prefix = f"model.layers.{layer_idx + i}."
            if weight_name.startswith(prefix):
                return layer_idx + i
    return None
```

---

## 7. MTP 推测解码

### 7.1 GLM-4.x MTP

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    GLM-4.x MTP 架构                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Glm4MoeMultiTokenPredictorLayer:                                            ║
║                                                                              ║
║  输入: input_ids, positions, previous_hidden_states                          ║
║                                                                              ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │  inputs_embeds = embed_tokens(input_ids)                              │  ║
║  │  inputs_embeds[positions == 0] = 0  # 屏蔽 padding 位置               │  ║
║  │                                                                        │  ║
║  │  inputs_embeds = enorm(inputs_embeds)                                 │  ║
║  │  previous_hidden_states = hnorm(previous_hidden_states)                │  ║
║  │                                                                        │  ║
║  │  # 融合 e+h 投影 (与 DeepSeek V3 相同)                                │  ║
║  │  hidden_states = eh_proj(                                              │  ║
║  │      concat([inputs_embeds, previous_hidden_states], dim=-1)          │  ║
║  │  )  # [2H] → [H]                                                      │  ║
║  │                                                                        │  ║
║  │  # 标准 Decoder Layer (使用 Glm4MoeDecoderLayer)                       │  ║
║  │  hidden_states, residual = mtp_block(positions, hidden_states)         │  ║
║  │  hidden_states = residual + hidden_states                              │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  特征:                                                                       ║
║  • eh_proj: 融合 embedding + hidden 投影 (H+H→H)                            ║
║  • SharedHead: RMSNorm → LM Head (共享主模型 LM Head)                       ║
║  • mtp_block: 标准 Glm4MoeDecoderLayer (含 MoE)                             ║
║  • 多步循环: num_nextn_predict_layers 个 MTP 层                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 7.2 GLM-5 MTP

```
GLM-5 的 MTP 完全复用 DeepSeek V2 的 MTP 架构:

GlmMoeDsaForCausalLM → DeepseekV2ForCausalLM → DeepSeekMTP

  + Ascend 上额外的 rot.weight (Rotary Quantization)
  + DeepSeekMultiTokenPredictorLayer (标准 DS MTP)
  + eh_proj: 融合 e+h 投影
  + mtp_block: DeepseekV2DecoderLayer (含 MLA + MoE)
  + shared_head: 共享 LM Head
```

### 7.3 MTP FP8 量化

MTP 层的 FP8 量化通过通用的 `per_token_group_quant_fp8` 函数实现，支持多后端：

```python
# vllm/model_executor/layers/quantization/utils/fp8_utils.py
def per_token_group_quant_fp8(
    x: torch.Tensor, group_size: int, eps: float = 1e-10,
    dtype: torch.dtype | None = None,
    column_major_scales: bool = False,
    tma_aligned_scales: bool = False,
    out_q: torch.Tensor | None = None,
    use_ue8m0: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token-group FP8 量化
    支持: DeepGEMM, XPU, Triton fallback
    DeepSeek 风格: group_size=(1, 128)
    """
```

### 7.4 Ascend MTP RoPE 特殊处理

```python
# vllm-ascend/ops/rotary_embedding.py
class AscendRotaryEmbedding(RotaryEmbedding):
    def forward_oot(self, positions, query, key, offsets=None, ...):
        is_draft_model = _EXTRA_CTX.is_draft_model
        flash_comm_v1_enabled = _EXTRA_CTX.flash_comm_v1_enabled
        # MTP draft model + FlashComm1 → all_gather positions
        if is_draft_model and self.use_mtp and flash_comm_v1_enabled:
            positions = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                positions.contiguous(), True)
        return torch.ops.vllm.npu_rotary_embedding(
            positions, query, key, self.cos_sin_cache,
            self.head_size, self.rotary_dim, is_neox_style)
```

---

## 8. 工具调用与推理

### 8.1 工具调用解析器

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    GLM 系列工具调用解析器                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  模型 → 解析器映射:                                                          ║
║  ┌────────────────┬─────────────────┬──────────────────────────────────┐  ║
║  │ 模型            │ 工具解析器       │ 格式                             │  ║
║  ├────────────────┼─────────────────┼──────────────────────────────────┤  ║
║  │ GLM-4.5/4.6    │ GLM45 XML       │ <tool_calls> XML 格式             │  ║
║  │ GLM-4.7/GLM-5  │ GLM47 XML       │ <tool_calls> XML 格式 (支持零参数)│  ║
║  │ GLM-4           │ GLM4 JSON       │ JSON 格式                         │  ║
║  └────────────────┴─────────────────┴──────────────────────────────────┘  ║
║                                                                              ║
║  GLM47 解析器特性:                                                           ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │  • XML 格式工具调用 (与 DeepSeek V3.2 DSML 类似)                      │  ║
║  │  • 支持零参数工具调用 (不需要 "arguments" 字段)                        │  ║
║  │  • 流式解析: 增量字符串拼接 + 状态机                                  │  ║
║  │  • Rust 高性能实现: rust/src/tool-parser/src/glm_xml/glm47_moe.rs    │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 8.2 推理（Thinking）解析器

```
模型 → 推理解析器映射:
  GLM-4.5/4.6/4.7/GLM-5 → GLM45 解析器

解析模式:
  glmmoe45_format: 
    与 DeepSeek R1 类似，使用 thinking/response 标记
    具体标记格式由 tokenizer 定义
```

### 8.3 Rust 注册

```rust
// rust/src/chat/src/parser/tool/mod.rs
factory
    .register_pattern("glm-5", names::GLM47)    // GLM-5 用 GLM47 解析器
    .register_pattern("glm-4.7", names::GLM47)
    .register_pattern("glm-4.6", names::GLM45)
    .register_pattern("glm-4.5", names::GLM45)

// rust/src/chat/src/parser/reasoning/mod.rs
factory
    .register_pattern("glm-5", names::GLM45)     // GLM-5 推理用 GLM45
    .register_pattern("glm-4.7", names::GLM45)
    .register_pattern("glm-4.6", names::GLM45)
```

---

## 9. 量化方案

### 9.1 量化全景

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    GLM 系列量化方案                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  W4A8 (权重 4-bit, 激活 8-bit):                                             ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │  GLM-5-w4a8:  1× A3 (16卡) 或 1× A2 (8卡)                           │  ║
║  │  GLM-5.1-w4a8: 同上                                                   │  ║
║  │  量化工具: msmodelslim (Ascend 官方) 或 Eco-Tech 预量化               │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  W8A8 (权重 8-bit, 激活 8-bit):                                             ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │  GLM-5-w8a8:  1× A3 (16卡)                                           │  ║
║  │  GLM-5.1-w8a8: 同上                                                   │  ║
║  │  GLM-4.7-w8a8: 1× A3 (16卡) 或 1× A2 (8卡)                          │  ║
║  │  量化工具: msmodelslim 或 Modelers_Park/Eco-Tech 预量化               │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  BF16 (全精度):                                                              ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │  GLM-5-bf16:  2× A3 (32卡)                                           │  ║
║  │  GLM-4.5/4.6/4.7-bf16: 2× A3 或 2× A2                               │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  Rotary Quantization (GLM-5 专有):                                           ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │  对 MTP 层的 previous_hidden_states 做可学习旋转 + 量化                │  ║
║  │  rot.weight: [H, H] 矩阵，配合 W8A8/W4A8 使用                         │  ║
║  │  quant_description = {"is_rot_used": True}                            │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 9.2 W8A8 实现

```python
# vllm-ascend/quantization/methods/w8a8_dynamic.py

@register_scheme("W8A8_DYNAMIC", "linear")
class AscendW8A8DynamicLinearMethod(AscendLinearScheme):
    def apply(self, layer, x, bias=None, tp_rank=None):
        # Step 1: 动态量化激活 → int8
        quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x)
        # Step 2: int8 矩阵乘法
        output = torch_npu.npu_weight_quant_batchmatmul(
            quantized_x, layer.weight, pertoken_scale, layer.weight_scale
        )
        if bias is not None:
            output = output + bias
        return output
```

---

## 10. Ascend 计算优化

### 10.1 SFA 稀疏注意力

GLM-5/5.1 (DSA 架构) 在 Ascend 上使用专属 SFA 后端：

```python
# vllm-ascend/attention/sfa_v1.py
class AscendSFAImpl:
    def __init__(self, ...):
        if self.vllm_config.model_config.hf_config.model_type in ["glm_moe_dsa"]:
            self.is_rope_neox_style = False      # GLM DSA 不使用 neox-style RoPE
            self.use_torch_npu_lightning_indexer = True  # Lightning Indexer 加速
```

关键优化点：
- **Lightning Indexer**：加速稀疏注意力索引查找
- **Sparse C8 Indexer**：Hadamard 变换矩阵的稀疏 C8 量化索引器
- **MLAPO 融合**：W8A8 场景下融合 Q/K/V 预处理步骤
- **DSA-CP**：支持上下文并行 (Context Parallel)

### 10.2 GQA-C8 权重加载

GLM-4.x MoE 的 C8 量化权重加载做了 Monkey Patch：

```python
# vllm-ascend/patch/worker/patch_gqa_c8.py
Glm4MoeForCausalLM.load_weights = lambda self, weights: _patched_causal_lm_load_weights(...)
# 拦截 C8 scale 参数，直接加载到对应的 scale 参数
```

### 10.3 量化模型 Slim 配置

```
# vllm-ascend/quantization/modelslim_config.py

glm_moe_dsa:   gate_up_proj, experts(gate/up/down), fused_qkv_a_proj(q_a/kv_a_with_mqa)
glm4_moe:      qkv_proj(q/k/v), gate_up_proj, experts
glm4_moe_lite: gate_up_proj, experts, fused_qkv_a_proj
glm4v_moe:     qkv_proj, gate_up_proj, experts
```

### 10.4 xlite 推理引擎

GLM-4.x MoE 有专门的 xlite 适配：

```python
# vllm-ascend/xlite/xlite.py
class Glm4MoeXliteModel:
    # 配置 partial_rotary_factor、dense layers、routed/shared experts
    # 支持 ScoringFuncSigmoid 路由评分
    # 初始化 routed experts (re_up_gate, re_down) + shared experts (se_up_gate, se_down)
    # 支持 NZ 格式权重和量化 scale 权重
```

---

## 11. Ascend 通信优化

### 11.1 MoE 通信方法

vllm-ascend 实现了 4 种 MoE 通信方法：

| 方法 | 适用场景 |
|------|----------|
| AllGather | 默认，`npu_moe_init_routing_v2` + `npu_moe_token_unpermute` |
| AlltoAll | EP>1 且 `npu_grouped_matmul` 可用 |
| MC2 | 计算-通信并行 |
| **Fused MC2** | 融合 dispatch+ffn+combine |

FusedMC2 融合算子：

```python
# vllm-ascend/ops/fused_moe/moe_comm_method.py
# 条件: EP<=32, W8A8, 非 MTP, 非 dynamic EPLB
dispatch_ffn_combine(...)           # Prefill 节点
dispatch_gmm_combine_decode(...)    # Decode 节点专用
```

### 11.2 FlashComm

```bash
# GLM-5 PD 分离的 prefill 节点启用
VLLM_ASCEND_ENABLE_FLASHCOMM1=1      # TP 场景下减少通信开销
VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE  # O-matrix TP 分组
```

### 11.3 MoE 融合流水线

```
Prepare → Token Dispatch → MLP Compute → Token Combine

FusedExpertsResult:
  routed_out, before_dispatch_evt, before_gmm2_evt, before_combine_evt
  group_list_type, expert_tokens, swiglu_limit  # 动态 EPLB 支持
```

---

## 12. MoE 深度优化

### 12.1 Shared Expert 优化

```bash
--additional-config '{"enable_shared_expert_dp": true, "multistream_overlap_shared_expert": true}'
```

- `enable_shared_expert_dp`：Shared Expert 数据并行
- `multistream_overlap_shared_expert`：多流重叠 Shared Expert 计算

### 12.2 GLM-4.7-Flash FSE

ROCm 平台支持 AITER fusion shared experts：

```python
# vllm/model_executor/models/glm4_moe_lite.py
# 将 shared expert 权重合并到 routed expert 末尾
if envs.VLLM_AITER_ENABLE_FUSED_SHARED_EXPERTS:
    ...
```

### 12.3 MoE Layer Frequency

GLM-4.7-Flash 支持稀疏 MoE 层插入频率：

```python
moe_layer_freq: int  # 每隔 N 层插入一个 MoE 层
```

---

## 13. EPLB 负载均衡

### 13.1 EPLB 生命周期

```
热收集间隔 → 算法执行间隔 → 逐层权重更新（num_moe_layers 步）
```

```python
# vllm-ascend/eplb/eplb_updator.py
class EplbUpdator:
    def forward_before(self):
        # 获取 EPLB 进程计算的更新信息，执行 D2D 权重传输
    def forward_end(self):
        # 计算 MoE 负载，唤醒 EPLB worker 进程
    def compute_and_set_moe_load(self):
        # 跨 rank 收集 expert 负载数据
```

### 13.2 环境变量

```bash
DYNAMIC_EPLB=true  # 启用动态 EPLB
```

### 13.3 冗余专家

```python
n_physical_experts = n_logical_experts + n_redundant_experts
# 冗余专家是对热门逻辑专家的额外副本，实现推理时负载均衡
```

### 13.4 通信后端

| 后端 | 特点 |
|------|------|
| NCCL P2P | GPU 直连 |
| Gloo Staged | CPU 中转 |
| PyNccl | Python NCCL 封装 |
| NIXL | RDMA READ 传输 |

---

## 14. SFA 稀疏注意力

### 14.1 与 DeepSeek V3.2 关系

GLM-5 的 DSA 直接继承 DeepSeek V3.2 的 SFA 实现：

```
GlmMoeDsaForCausalLM → DeepseekV2ForCausalLM → V3.2 SparseAttention
```

### 14.2 Indexer 配置

```python
index_n_heads: 64        # Indexer 专用 head 数
index_head_dim: 128      # Indexer head 维度
index_topk: 选择的 token 数
index_rope_dim: 64       # RoPE 维度

# IndexCache (可选)
use_index_cache: True    # 跨层复用 topk 选择
index_topk_freq: 1       # 每 N 层重新计算
index_topk_pattern: "SSSS"  # Skip/Compute 模式
```

### 14.3 MLA 后端选择优先级

```
Blackwell (SM 10.x):
  FLASHINFER_MLA → TOKENSPEED_MLA → CUTLASS_MLA → FLASH_ATTN_MLA → FLASHMLA → TRITON_MLA

Ampere/Hopper (SM 8.x-9.x):
  FLASH_ATTN_MLA → FLASHMLA → FLASHINFER_MLA → TRITON_MLA
```

---

## 15. 多模态扩展

### 15.1 GLM-4V / GLM-4.1V

| 模型 | 视觉编码器 | 特点 |
|------|-----------|------|
| GLM-4V | EVA2-CLIP | 基础视觉 |
| GLM-4.1V | Glm4vVisionTransformer | 3D RoPE, MRoPE |
| GLM-4v-MoE | Glm4vVisionTransformer | MoE 语言模型 + 视觉 |
| AutoGLM-Phone-9B | Glm4vVisionTransformer | 手机 Agent |

### 15.2 GLM-OCR

```python
# vllm/model_executor/models/glm_ocr.py
class GlmOcrForConditionalGeneration(Glm4vForConditionalGeneration):
    # 自定义 GlmOcrVisionTransformer
    # 使用 GlmOcrVisionAttention (Q/K RMSNorm)
    # 自定义 patch embed 和 merger
```

### 15.3 GLM-ASR (音频)

```python
# vllm/model_executor/models/glmasr.py
class GlmAsrForConditionalGeneration:
    # GlmAsrEncoder: 卷积 + Transformer + RoPE
    # GlmAsrMultiModalProjector: 两层 MLP
    # 支持 Whisper 兼容的特征提取
```

---

## 16. 部署配置参考

### 16.1 GLM-5 部署 (vllm-ascend)

| 配置 | 硬件 | 参数 |
|------|------|------|
| w4a8 单机 | 1× A3 (16卡) | `dp1 tp16`, max-model-len=200000 |
| w8a8 单机 | 1× A3 (16卡) | `dp1 tp16`, max-model-len=40960 |
| w4a8 单机 | 1× A2 (8卡) | `dp1 tp8`, max-model-len=32768 |
| bf16 多机 | 2× A3 (32卡) | `dp2 tp16`, max-model-len=8192 |
| w8a8 多机 | 2× A3 (32卡) | `dp2 tp16`, max-model-len=200000 |
| PD 分离 (2P4D) | 多机 w8a8 | Prefill `dp2 tp16`, Decode `dp16 tp4` |

优化开关：
```bash
VLLM_ASCEND_ENABLE_FLASHCOMM1=1   # Prefill 节点
VLLM_ASCEND_ENABLE_FUSED_MC2=1    # Prefill + Decode
VLLM_ASCEND_ENABLE_MLAPO=1        # Decode 节点
VLLM_ASCEND_BALANCE_SCHEDULING=1  # 负载均衡调度
```

### 16.2 GLM-4.x 部署

| 配置 | 硬件 | 参数 |
|------|------|------|
| 单机 A3 | 1× A3 (16卡) | `dp2 tp8`, W8A8, MTP spec decode (3 tokens) |
| 单机 A2 | 1× A2 (8卡) | `dp1 tp8`, W8A8 |
| 多机 (2×A2) | 2× A2 | `dp2 tp8`, max-model-len=140000 |
| PD 分离 (2P1D) | 多机 | Prefill `dp2 tp8`, Decode `dp8 tp4` |

### 16.3 MTP 部署

GLM-5 使用 MTP 时需要先调整权重：

```bash
python adjust_weight.py  # 将 embed_tokens 和 lm_head 映射到 MTP 层
```

---

## 17. 性能优化总结

### 17.1 精度基准

| 模型 | 数据集 | 得分 |
|------|--------|------|
| GLM-4.7 | GPQA | 84.85 |
| GLM-4.7 | MATH500 | 98.8 |
| GLM-5.1 | GPQA | 85.35 |
| GLM-5.1 | AIME 2025 | 90 |

### 17.2 优化全景

| 类别 | 优化 | 模型 |
|------|------|------|
| 计算 | SFA 稀疏注意力 + Lightning Indexer | GLM-5 |
| 计算 | MLAPO Q/K/V 融合 | GLM-5 (W8A8) |
| 计算 | Shared Expert 多流重叠 | GLM-4.x / GLM-5 |
| 通信 | Fused MC2 (dispatch+ffn+combine) | GLM-5 |
| 通信 | FlashComm1/2 TP 通信优化 | GLM-5 (PD 分离) |
| 内存 | W4A8/W8A8 量化 | GLM-4.x / GLM-5 |
| 内存 | MLA 低秩 KV Cache 压缩 | GLM-4.7-Flash / GLM-5 |
| 调度 | EPLB 动态负载均衡 | GLM-4.x / GLM-5 |
| 调度 | PD 分离 + Mooncake KV 传输 | GLM-4.7 |
| 推测 | MTP 多令牌预测 | GLM-4.x / GLM-5 |

---

## 18. 关键文件索引

### vllm (上游)

| 文件 | 用途 |
|------|------|
| [glm4_moe.py](file:///D:/trae-workspace/github/vllm/vllm/model_executor/models/glm4_moe.py) | GLM-4.5/4.6/4.7 MoE 模型 |
| [glm4_moe_lite.py](file:///D:/trae-workspace/github/vllm/vllm/model_executor/models/glm4_moe_lite.py) | GLM-4.7-Flash (MLA) |
| [deepseek_v2.py](file:///D:/trae-workspace/github/vllm/vllm/model_executor/models/deepseek_v2.py) | GLM-5 DSA (GlmMoeDsaForCausalLM) |
| [glm4_moe_mtp.py](file:///D:/trae-workspace/github/vllm/vllm/model_executor/models/glm4_moe_mtp.py) | GLM-4.x MTP |
| [glm4v.py](file:///D:/trae-workspace/github/vllm/vllm/model_executor/models/glm4v.py) | GLM-4V 多模态 |
| [glm4_1v.py](file:///D:/trae-workspace/github/vllm/vllm/model_executor/models/glm4_1v.py) | GLM-4.1V / GLM-4v-MoE |
| [glm_ocr.py](file:///D:/trae-workspace/github/vllm/vllm/model_executor/models/glm_ocr.py) | GLM-OCR |
| [glmasr.py](file:///D:/trae-workspace/github/vllm/vllm/model_executor/models/glmasr.py) | GLM-ASR 音频 |
| [eplb_state.py](file:///D:/trae-workspace/github/vllm/vllm/distributed/eplb/eplb_state.py) | EPLB 核心 |
| [eplb_communicator.py](file:///D:/trae-workspace/github/vllm/vllm/distributed/eplb/eplb_communicator.py) | EPLB 通信后端 |

### vllm-ascend (昇腾适配)

| 文件 | 用途 |
|------|------|
| [sfa_v1.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/attention/sfa_v1.py) | SFA 稀疏注意力 |
| [patch_gqa_c8.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/patch/worker/patch_gqa_c8.py) | GLM-4 C8 权重加载 |
| [patch_deepseek_mtp.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/patch/worker/patch_deepseek_mtp.py) | GLM-5 MTP (rot.weight) |
| [moe_comm_method.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py) | MoE 通信方法 |
| [modelslim_config.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/quantization/modelslim_config.py) | GLM 量化层映射 |
| [xlite.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/xlite/xlite.py) | Glm4MoeXliteModel |
| [eplb_updator.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/eplb/eplb_updator.py) | EPLB 更新器 |
| [envs.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/envs.py) | FlashComm/FusedMC2 开关 |
| [GLM5.md](file:///D:/trae-workspace/github/vllm-ascend/docs/source/tutorials/models/GLM5.md) | GLM-5 部署指南 |
| [GLM4.x.md](file:///D:/trae-workspace/github/vllm-ascend/docs/source/tutorials/models/GLM4.x.md) | GLM-4.x 部署指南 |