# Qwen3.5 系列架构与优化深度分析

> 基于 vllm 和 vllm-ascend 源码的完整技术解读  
> 涵盖 Qwen3.5 Dense、Qwen3.5-MoE、Qwen3-Next 架构及华为 Ascend NPU 全栈优化

---

## 目录

1. [模型体系概览](#1-模型体系概览)
2. [Qwen3.5 Dense 架构](#2-qwen35-dense-架构)
3. [Qwen3.5-MoE 架构](#3-qwen35-moe-架构)
4. [GDN 线性注意力](#4-gdn-gated-deltanet-线性注意力)
5. [Qwen3-Next 前代架构](#5-qwen3-next-前代架构)
6. [MTP 多令牌预测](#6-mtp-多令牌预测)
7. [Qwen3.5-Omni 全模态架构](#7-qwen35-omni-全模态架构)
8. [训练基础设施与 Qwen3.6](#8-训练基础设施与-qwen36)
9. [量化方案](#9-量化方案)
10. [NPU 融合算子](#10-npu-融合算子详解)
11. [Ascend 平台优化](#11-ascend-平台全栈优化)
12. [推理优化技术](#12-推理优化技术)
13. [架构总结与对比](#13-架构总结与对比)

---

## 1. 模型体系概览

### 1.1 模型矩阵

Qwen3.5 系列是 Qwen 家族的下一代大语言模型，核心架构创新是引入了 **GDN (Gated DeltaNet)** 线性注意力机制，形成混合注意力架构。

```
                    ┌─────────────────────────────────────┐
                    │           Qwen 模型家族               │
                    └─────────────────────────────────────┘
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          │                            │                            │
          ▼                            ▼                            ▼
   ┌─────────────┐            ┌──────────────┐            ┌──────────────┐
   │   Qwen3      │            │  Qwen3-Next   │            │  Qwen3.5      │
   │ (Full Attn)  │            │ (Hybrid Attn) │            │ (Hybrid Attn) │
   │              │            │               │            │               │
   │ • GQA        │            │ • GDN + GQA   │            │ • GDN + GQA   │
   │ • QK-Norm    │            │ • Layer Scale  │            │ • GemmaRMSNorm │
   │ • RoPE       │            │ • Gate Output  │            │ • Gate Output  │
   │ • Dense/     │            │ • Dense/       │            │ • Dense/       │
   │   MoE 变体   │            │   MoE 变体     │            │   MoE 变体     │
   └─────────────┘            └──────────────┘            └──────────────┘
```

### 1.2 发布背景与战略定位

Qwen3.5 于 **2026 年 2 月 16 日** 由阿里巴巴通义千问团队正式发布，采用 **三波发布策略**：

| 发布时间 | 模型 | 定位 |
|----------|------|------|
| 2026-02-16 | Qwen3.5-397B-A17B | 旗舰 MoE，最强开源推理 |
| 2026-02-24 | 27B / 35B-A3B / 122B-A10B | 中型系列，兼顾性能与效率 |
| 2026-03-02 | 0.8B / 1.5B / 4B / 9B / 14B | 边缘端系列，手机/笔记本可运行 |
| 2026-03-30 | Qwen3.5-Omni | 全模态旗舰，视频分析 SOTA |

阿里官方将 Qwen3.5 定义为 **"为 Agentic AI 时代而生"** 的模型——从「聊天式 AI」向「自主执行任务的 Agent AI」全面转型。这一代的核心叙事转变是：

- **Gated Delta Networks** 首次在生产规模模型中站稳脚跟，替代了大部分 Transformer 注意力层
- **Early Fusion** 多模态进入实用阶段——文本和视觉从 Token 级别就共享表示空间
- **0.8B 的模型能处理视频**——边缘端多模态从理论走向硬件现实

> 上一代 30B 模型的能力，这一代 9B 装下 —— 成为可量化的工程事实。

### 1.3 完整模型矩阵

Qwen3.5 系列覆盖从 0.8B 到 397B 的完整谱系，全系列 **Apache 2.0 开源**：

| 模型 | 总参数 | 激活参数 | 架构 | 上下文 | 最小显存(BF16) |
|------|--------|----------|------|--------|----------------|
| Qwen3.5-0.8B | 0.8B | 0.8B | Dense | 256K→1M | 2 GB |
| Qwen3.5-1.5B | 1.5B | 1.5B | Dense | 256K→1M | 4 GB |
| Qwen3.5-4B | 4B | 4B | Dense | 256K→1M | 10 GB |
| Qwen3.5-9B | 9B | 9B | Dense | 256K→1M | 20 GB |
| Qwen3.5-14B | 14B | 14B | Dense | 256K→1M | 30 GB |
| Qwen3.5-27B | 27B | 27B | Dense（混合注意力） | 256K→1M | 55 GB |
| Qwen3.5-35B-A3B | 35B | 3B | MoE + GDN | 256K→1M | 8 GB（激活） |
| Qwen3.5-122B-A10B | 122B | 10B | MoE + GDN | 256K→1M | 22 GB（激活） |
| Qwen3.5-397B-A17B | 397B | 17B | MoE + GDN | 256K→1M | 40 GB（激活） |
| Qwen3.5-Omni-Plus | 30B | 3B | MoE + Thinker-Talker | 256K | 60 GB |
| Qwen3.5-Omni-Flash | — | — | MoE + Thinker-Talker | 256K | 更小 |

#### Qwen3.5-397B-A17B 旗舰关键参数

| 参数 | 值 |
|------|-----|
| 总参数量 | 397B（3970 亿） |
| 每 token 激活参数 | 17B（约 4.3%） |
| 隐藏层维度 | 4096 |
| Token 嵌入维度 | 248320（Padded） |
| 层数 | 60 |
| 隐藏层结构 | 15 × (3 × GDN → MoE → 1 × Full Attention → MoE) |
| GDN 头数 | V=64, QK=16, head_dim=128 |
| Full Attention 头数 | Q=32, KV=2, head_dim=256 |
| MoE 专家总数 | 512 |
| 激活专家/token | 10 路由 + 1 共享 |
| 专家中间层维度 | 1024 |
| 原生上下文长度 | 256K tokens（可扩展至 1M） |
| 支持语言 | 201 种语言与方言 |
| 词表大小 | 250K |

#### Qwen3.5-27B Dense 关键参数

| 参数 | 值 |
|------|-----|
| 参数量 | 27B |
| 隐藏层维度 | 5120 |
| Token 嵌入维度 | 248320（Padded） |
| 层数 | 64 |
| 隐藏层结构 | 16 × (3 × GDN → FFN → 1 × Full Attention → FFN) |
| GDN 头数 | V=48, QK=16, head_dim=128 |
| Full Attention 头数 | Q=24, KV=4, head_dim=256 |
| FFN 中间层维度 | 17408 |

### 1.4 基准性能

#### 旗舰模型 vs 竞品

Qwen3.5 在多项评测中达到或超越同级闭源模型：

| 基准 | Qwen3.5 | Qwen3-Max | GPT-5.2 | Gemini 3 Pro |
|------|---------|-----------|---------|-------------|
| MMLU-Pro | **89.8** | 89.5 | — | — |
| GPQA Diamond（推理） | **81.7** | — | 80.1 | — |
| IFBench（视觉指令遵循） | **76.5** | — | 75.4 | — |
| MathVision（视觉数学） | **88.6** | — | — | — |
| HumanEval（代码） | 领先 | — | — | 超越 |
| OmniDocBench（文档理解） | **87.7** | — | 78.2 | — |

#### 小模型逆袭：Qwen3.5-9B vs 前代 120B

| 基准 | Qwen3.5-9B (9B) | Qwen3-30B (30B) | GPT-oss-120B (120B) |
|------|-----------------|-----------------|---------------------|
| GPQA Diamond | **81.7** | — | 80.1 |
| MMU-Pro（视觉推理） | **70.1** | — | 59.7 |
| Video-MME（视频理解） | **84.5** | — | — |
| HMMT（数学） | **83.2** | — | — |
| OmniDocBench | **87.7** | — | 78.2 |

#### 推理效率提升

| 上下文长度 | Qwen3.5 vs Qwen3-Max 吞吐量提升 | vs 传统 Transformer |
|------------|-------------------------------|---------------------|
| 32K | **8.6×** | — |
| 256K | **19.0×** | — |
| 500K | — | 计算量仅 3~4×（非 100×） |

### 1.5 架构对比

| 特性 | Qwen3 | Qwen3-Next | Qwen3.5 Dense | Qwen3.5 MoE |
|------|-------|------------|---------------|-------------|
| 注意力类型 | 100% Full Attention | 75% GDN + 25% GQA | 75% GDN + 25% GQA | 75% GDN + 25% GQA |
| 线性注意力 | 无 | GDN | GDN | GDN |
| MoE | 可选(部分层) | 可选(交错) | 无 | **全层 MoE** |
| 专家数 | 128 | **512** | - | **256** |
| Top-K | 8 | **10** | - | 8 |
| 归一化 | RMSNorm | RMSNorm + Layer Scale | **GemmaRMSNorm** | **GemmaRMSNorm** |
| 注意力输出门控 | 无 | 可选(sigmoid) | 可选(sigmoid) | 可选(sigmoid) |
| 共享专家 | 有 | 有(sigmoid gate) | - | 有(sigmoid gate) |
| MTP | 无 | 有 | 有 | 有 |

### 1.6 源码文件索引
|------|------|
| [vllm/model_executor/models/qwen3.py](file:///D:/trae-workspace/github/vllm/vllm/model_executor/models/qwen3.py) | Qwen3 密集模型 |
| [vllm/model_executor/models/qwen3_moe.py](file:///D:/trae-workspace/github/vllm/vllm/model_executor/models/qwen3_moe.py) | Qwen3 MoE 模型 |
| [vllm/model_executor/models/qwen3_next.py](file:///D:/trae-workspace/github/vllm/vllm/model_executor/models/qwen3_next.py) | Qwen3-Next (Qwen3.5 前身) |
| [vllm/model_executor/models/qwen3_5.py](file:///D:/trae-workspace/github/vllm/vllm/model_executor/models/qwen3_5.py) | Qwen3.5 Dense + MoE |
| [vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py](file:///D:/trae-workspace/github/vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py) | GDN 线性注意力核心实现 (1542行) |
| [vllm/v1/attention/backends/gdn_attn.py](file:///D:/trae-workspace/github/vllm/vllm/v1/attention/backends/gdn_attn.py) | GDN 注意力后端 |
| [vllm/model_executor/models/qwen3_5_mtp.py](file:///D:/trae-workspace/github/vllm/vllm/model_executor/models/qwen3_5_mtp.py) | Qwen3.5 MTP |
| [vllm/model_executor/models/qwen3_next_mtp.py](file:///D:/trae-workspace/github/vllm/vllm/model_executor/models/qwen3_next_mtp.py) | Qwen3-Next MTP |
| [vllm/transformers_utils/configs/qwen3_5.py](file:///D:/trae-workspace/github/vllm/vllm/transformers_utils/configs/qwen3_5.py) | Qwen3.5 配置类 |
| [vllm/transformers_utils/configs/qwen3_5_moe.py](file:///D:/trae-workspace/github/vllm/vllm/transformers_utils/configs/qwen3_5_moe.py) | Qwen3.5-MoE 配置类 |
| [vllm/transformers_utils/configs/qwen3_next.py](file:///D:/trae-workspace/github/vllm/vllm/transformers_utils/configs/qwen3_next.py) | Qwen3-Next 配置类 |

### 1.7 vllm-ascend 适配文件

| 文件 | 用途 |
|------|------|
| [vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py) | Qwen3.5 注意力层 NPU 适配 |
| [vllm-ascend/vllm_ascend/patch/worker/patch_qwen3vl.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3vl.py) | Qwen3-VL + Qwen3-MoE 注意力适配 |
| [vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_dflash.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_dflash.py) | DFlash KV 缓存预计算优化 |
| [vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_next_mtp.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_next_mtp.py) | Qwen3-Next MTP 适配 |
| [vllm-ascend/vllm_ascend/ops/triton/fused_gdn_gating.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/ops/triton/fused_gdn_gating.py) | 融合 GDN 门控 Triton 内核 |
| [vllm-ascend/vllm_ascend/attention/attention_v1.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/attention/attention_v1.py) | NPU 注意力后端 (含 C8 量化) |

---

## 2. Qwen3.5 Dense 架构

### 2.1 整体结构

Qwen3.5 Dense 采用混合注意力架构：约 **75% 的层使用 GDN 线性注意力**，**25% 的层使用标准全注意力**，通过 `full_attention_interval=4` 交错排列。

```
Qwen3.5 Dense Decoder Layer 结构:

                    ┌─────────────┐
                    │ hidden_states│
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  输入 RMSNorm │  (GemmaRMSNorm: weight + 1.0)
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
   ┌─────────────────────┐   ┌──────────────────────┐
   │  Full Attention 层   │   │  GDN Linear Attention │
   │  (每隔4层出现一次)    │   │  (其余所有层)          │
   │                     │   │                       │
   │  Qwen3NextAttention │   │  QwenGatedDeltaNet-   │
   │  • QKV 投影          │   │  Attention            │
   │  • QK-Norm          │   │  • in_proj_qkvz       │
   │  • RoPE             │   │  • in_proj_ba         │
   │  • Attention        │   │  • conv1d             │
   │  • Gate Output      │   │  • ChunkGatedDeltaRule│
   │  • O 投影            │   │  • RMSNormGated       │
   └─────────────────────┘   └──────────────────────┘
              │                         │
              └────────────┬────────────┘
                           │
                    ┌──────▼──────┐
                    │  残差连接    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  输出 RMSNorm │  (GemmaRMSNorm)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Dense MLP  │
                    │  • gate_proj│
                    │  • up_proj  │
                    │  • SiluAndMul│
                    │  • down_proj│
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  残差连接    │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │ (可选)                   │
              ▼                         │
   ┌─────────────────────┐              │
   │  Layer Scale        │              │
   │  output = output *   │              │
   │    (scale + 1.0)    │              │
   └─────────────────────┘              │
                           │           │
                           ▼           ▼
                    ┌──────▼──────┐
                    │ hidden_states│
                    └─────────────┘
```

### 2.2 原生多模态

Qwen3.5 是 Qwen 家族首个真正意义上的**原生多模态模型**，从预训练第一天起就同时处理文本和图像 token，而非传统的"先训练语言模型再接视觉 Adapter"的后融合方案。

#### Early Fusion 核心优势

```
传统 Late Fusion:                  Qwen3.5 Early Fusion:

文本预训练 → 语言模型               文本 Token + 图像 Token
     ↓                                    ↓
视觉 Adapter → 对齐层              统一 Transformer 处理
     ↓                                    ↓
融合推理                          端到端多模态推理

问题:                               优势:
• 视觉模块加上后语言能力退化          • 训练效率接近纯文本的 100%
• 两个独立系统沟通成本高              • 真正跨模态推理而非跨模态翻译
• 多模态训练效率低                   • 不需要中间 Adapter 层
```

#### MRoPE（多模态旋转位置编码）

Qwen3.5 引入的 **MRoPE** 同时处理时间、高度、宽度三个维度的位置信息，实现文本与视觉位置信息的统一编码：

```python
# Qwen3_5TextRotaryEmbedding 初始化关键参数
partial_rotary_factor = 0.25  # 只对 Q/K 向量的 1/4 维度进行旋转编码
# 剩余 3/4 维度不编码位置信息，用于传递多模态语义信息

# mrope_section: 定义不同维度的旋转频率分配
# 例如 [16, 24, 24] 表示 3 个分段，分别对应 t, h, w 维度
self.mrope_section = config.mrope_section
```

**关键设计**：

- `partial_rotary_factor=0.25`：只对 Q/K 向量的 **25%** 维度施加 RoPE，剩余维度保留原始语义
- `mrope_interleaved`：控制 3D 位置编码的维度交错方式
- 与 Qwen-VL 系列相比，MRoPE 更适配原生多模态需求，不需要额外的视觉编码器分支

#### 多模态能力指标

| 能力 | 表现 |
|------|------|
| 视频支持 | 可分析长达 **2 小时** 视频，精度秒级 |
| 视觉数学 MathVision | **88.6** 分（同级最佳） |
| 视觉指令遵循 IFBench | **76.5**，超越多个闭源模型 |
| GUI Agent | 支持查看 UI 截图并生成对应 HTML/CSS 代码 |
| MCP 协议 | 原生支持 Model Context Protocol |

### 2.3 标准注意力层

**Qwen3NextAttention** 继承自 `nn.Module`，支持注意力输出门控：

```python
# qwen3_next.py:L207-L322
class Qwen3NextAttention(nn.Module):
    def __init__(self, ...):
        # QKV 并行投影
        self.qkv_proj = QKVParallelLinear(
            hidden_size, self.head_dim, self.total_num_heads,
            self.total_num_kv_heads, bias=config.attention_bias,
        )
        # 逐头 QK-Norm (使用 GemmaRMSNorm)
        self.q_norm = GemmaRMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(head_dim, eps=config.rms_norm_eps)
        # 可选的注意力输出门控
        self.attn_output_gate = getattr(config, "attn_output_gate", True)

    def forward(self, positions, output, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        if self.attn_output_gate:
            # QKV 投影输出额外的门控头
            q_gate, k, v = qkv.split([q_size * 2, kv_size, kv_size], dim=-1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
        else:
            q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)       # 门控机制
            attn_output = attn_output * gate

        output[:], _ = self.o_proj(attn_output)
```

**注意力输出门控** 是 Qwen3.5 的一个重要特性：在 QKV 投影时额外输出一个门控向量，经过 sigmoid 后与注意力输出逐元素相乘。这相当于让模型学习每个 token 的注意力输出应该保留多少。

### 2.4 GemmaRMSNorm

Qwen3.5 使用 **GemmaRMSNorm** 而非标准 RMSNorm，关键区别是权重加 1：

```python
# GemmaRMSNorm: weight = 1.0 + original_weight
# 标准 RMSNorm: weight = original_weight

# 效果: GemmaRMSNorm 初始化时 weight 从 1.0 附近开始，
# 提供更稳定的训练初始化和更好的梯度流
```

在 Ascend 上有专用融合算子：

```python
# vllm-ascend ops/layernorm.py
class AscendGemmaRMSNorm(CustomOp):
    def forward_npu(self, x, weight):
        # weight 已经在 __init__ 中处理为 (1.0 + weight)
        return torch.ops._C_ascend.npu_gemma_rms_norm(x, weight, self.eps)
```

### 2.5 Layer Scale

可选的层缩放机制，为注意力和 FFN 输出添加可学习的乘法门控：

```python
# qwen3_next.py:L386-L460
self.layer_scale = getattr(config, "layer_scale", False)
if self.layer_scale:
    self.attn_layer_scale = nn.Parameter(torch.zeros(1, 1, hidden_size))
    self.ffn_layer_scale = nn.Parameter(torch.zeros(1, 1, hidden_size))

# 前向传播中:
hidden_states = hidden_states + attn_output * (self.attn_layer_scale + 1)
hidden_states = hidden_states + ffn_output * (self.ffn_layer_scale + 1)
```

初始化时 scale 参数为 0，所以 `(scale + 1) = 1`，不影响初始行为。训练过程中逐渐学习最佳缩放比例。

---

## 3. Qwen3.5-MoE 架构

### 3.1 整体结构

Qwen3.5-MoE 与 Dense 版本共享相同的注意力架构（GDN + Full Attention 混合），区别在于：

- **所有层都使用 MoE** 替代 Dense MLP
- 256 个专家，每个 token 选择 top-8
- 带共享专家和 sigmoid 门控

```
Qwen3.5-MoE Decoder Layer:

                    ┌─────────────┐
                    │ hidden_states│
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │ (同 Qwen3.5 Dense)       │
              │ Full Attention / GDN     │
              └────────────┬────────────┘
                           │
                    ┌──────▼──────┐
                    │  输出 RMSNorm │
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │  Qwen3NextSparseMoeBlock│
              │                         │
              │  ┌───────────────────┐  │
              │  │ Router (gate)      │  │
              │  │ Linear(hidden->256)│  │
              │  └─────────┬─────────┘  │
              │            │            │
              │  ┌─────────▼─────────┐  │
              │  │ FusedMoE          │  │
              │  │ • 256 experts     │  │
              │  │ • top_k=8         │  │
              │  │ • sigmoid routing │  │
              │  └─────────┬─────────┘  │
              │            │            │
              │  ┌─────────▼─────────┐  │
              │  │ Shared Expert     │  │
              │  │ • expert_gate     │  │
              │  │ • sigmoid(gate)   │  │
              │  │ • SiluAndMul      │  │
              │  └─────────┬─────────┘  │
              │            │            │
              │   shared + routed       │
              └────────────┬────────────┘
                           │
                    ┌──────▼──────┐
                    │ hidden_states│
                    └─────────────┘
```

### 3.2 MoE 层实现

```python
# qwen3_next.py:L87-L204
class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, ...):
        # 路由器
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)
        # 共享专家门控
        self.shared_expert_gate = ReplicatedLinear(hidden_size, 1, bias=False)
        # 可选的共享专家
        self.shared_expert = Qwen3NextMLP(...)
        # FusedMoE 路由专家
        self.experts = FusedMoE(
            num_experts=num_experts,
            top_k=top_k,
            scoring_func="sigmoid",      # 使用 sigmoid 评分
            renormalize=renormalize,
            ...
        )

    def forward(self, hidden_states):
        # 路由器 logits
        router_logits, _ = self.gate(hidden_states)
        # 路由专家前向
        routed_out = self.experts(hidden_states=hidden_states, router_logits=router_logits)

        # 共享专家
        if self.shared_expert is not None:
            shared_out = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
                shared_out = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_out
            routed_out = routed_out + shared_out

        return routed_out
```

### 3.3 与 Qwen3-Next 差异

| 特性 | Qwen3-Next | Qwen3.5-MoE |
|------|-----------|-------------|
| MoE 层分布 | 交错(部分 Dense, 部分 MoE) | **全部 MoE** |
| 专家数 | 512 | 256 |
| Top-K | 10 | 8 |
| FusedMoE 类型 | FusedMoE | **Qwen3_5_MoeMixtureOfExperts** (mixin) |
| 配置类 | Qwen3NextConfig | **Qwen3_5MoeTextConfig** |

### 3.4 MoE Mixin

Qwen3.5-MoE 引入了 `Qwen3_5_MoeMixtureOfExperts` mixin 类来管理 MoE 超参数：

```python
# qwen3_5.py
class Qwen3_5_MoeMixtureOfExperts:
    """Mixin 提供 MoE 超参数的标准化访问"""
    @property
    def num_experts(self):
        return self.config.num_experts           # 256

    @property
    def top_k(self):
        return self.config.num_experts_per_tok   # 8

    @property
    def intermediate_size_per_expert(self):
        return self.config.moe_intermediate_size

    @property
    def shared_expert_intermediate_size(self):
        return self.config.shared_expert_intermediate_size
```

---

## 4. GDN (Gated DeltaNet) 线性注意力

### 4.1 什么是 GDN

GDN (Gated DeltaNet) 是 Qwen3.5 最核心的技术创新。它是一种 **线性注意力机制**，计算复杂度为 O(N) 而非传统注意力的 O(N²)，借鉴了 Mamba/SSM 的设计思想，同时加入了门控机制。

**核心思想**：通过一个可学习的线性递归状态来维护长程依赖，而非计算全量注意力矩阵。

```
传统 Attention:               GDN Linear Attention:

Q × K^T → N×N 矩阵           递归状态 S_t = g_t × S_{t-1} + v_t × k_t^T
↓ softmax                     输出: o_t = S_t × q_t
↓ × V                         
O(N²) 复杂度                   O(N) 复杂度
```

### 4.2 GDN 架构详解

```
QwenGatedDeltaNetAttention 数据流:

                          ┌─────────────┐
                          │ hidden_states│
                          └──────┬──────┘
                                 │
                    ┌────────────▼────────────┐
                    │   in_proj_qkvz           │
                    │   投影到 [Q, K, V, Z]     │
                    │   qkvz = Linear(hidden)  │
                    └────────────┬────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
          ▼                      ▼                      ▼
   ┌─────────────┐      ┌──────────────┐      ┌──────────────┐
   │ Q, K, V     │      │ Z (门控)      │      │ B, A (SSM参数)│
   │ 投影后经过   │      │ 用于输出门控   │      │ in_proj_ba    │
   │ causal conv1d│      │               │      │ 投影后经过     │
   └──────┬──────┘      └──────┬───────┘      │ causal conv1d  │
          │                    │               └──────┬───────┘
          ▼                    │                      │
   ┌──────────────┐           │               ┌──────▼───────┐
   │ SiLU 激活     │           │               │ A: 状态衰减   │
   │ (作用于 K, V) │           │               │ B: 输入门控   │
   └──────┬───────┘           │               └──────┬───────┘
          │                    │                      │
          ▼                    │                      ▼
   ┌──────────────┐           │        ┌──────────────────────────┐
   │ ChunkGated   │           │        │ GDN 门控计算 (融合内核)    │
   │ DeltaRule    │           │        │ g = -exp(A_log) *        │
   │ (分块递归)    │           │        │     softplus(a + dt_bias) │
   └──────┬───────┘           │        │ beta = sigmoid(b)         │
          │                    │        └──────────────────────────┘
          │                    │                      │
          └────────────────────┼──────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  RMSNormGated        │
                    │  output = RMSNorm(   │
                    │    delta_out) * z    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  out_proj (O 投影)   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  (可选) beta 门控    │
                    │  output *= beta     │
                    └─────────────────────┘
```

### 4.3 关键投影

```python
# qwen_gdn_linear_attn.py
# Qwen3.5 使用非交错布局 (gqa_interleaved_layout=False)

# in_proj_qkvz: 分离的 [Q, K, V, Z] 权重
# 布局: [num_q_heads * head_dim, num_k_heads * head_k_dim,
#         num_v_heads * head_v_dim, num_v_heads * head_v_dim]
self.in_proj_qkvz = MergedColumnParallelLinear(
    hidden_size,
    [num_q_heads * head_dim,      # Q
     num_k_heads * head_k_dim,    # K
     num_v_heads * head_v_dim,    # V
     num_v_heads * head_v_dim],   # Z (与 V 相同维度)
    bias=False,
)

# in_proj_ba: [B, A] 门控向量
# B: 输入门控 (input-dependent gate)
# A: 状态衰减参数
self.in_proj_ba = MergedColumnParallelLinear(
    hidden_size,
    [num_v_heads * head_v_dim,    # B
     num_v_heads * head_v_dim],   # A
    bias=False,
)
```

**布局对比**：

```
Qwen3-Next (交错 GQA):        Qwen3.5 (非交错):
[q1,k1,v1,q2,k2,v2,...]       [q1,q2,...,k1,k2,...,v1,v2,...]
需要复杂的交错/去交错           加载和 TP 分片更简单
```

### 4.4 因果卷积

GDN 在每个 Q/K/V 投影后应用深度可分离的因果卷积：

```python
# conv1d: 深度可分离，kernel_size 通常为 4
self.conv1d = ColumnParallelLinear(
    # 对每个 head 应用独立的 1D 因果卷积
    # 输入: [batch, seq_len, channels]
    # 输出: [batch, seq_len, channels] (同维度)
)
```

**作用**：因果卷积在局部窗口内混合信息，增强模型的局部建模能力，弥补线性注意力在局部依赖上的不足。

### 4.5 GDN 递归核心

```
GDN 递归公式:

S_t = g_t × S_{t-1} + v_t ⊗ k_t       (状态更新)
o_t = S_t × q_t                         (输出计算)

其中:
  g_t = -exp(A_log) × softplus(a_t + dt_bias)   (门控衰减率)
  a_t = A 投影后的值
  dt_bias = 可学习的偏置参数

最终输出:
  output = RMSNorm(o_t) × z_t × beta_t
  z_t = Z 投影 (输出门控)
  beta_t = sigmoid(b_t) (B 投影门控)
```

### 4.6 分块计算

Prefill 阶段使用分块计算以提高效率：

```python
# ChunkGatedDeltaRule - 两种后端实现
class ChunkGatedDeltaRule(CustomOp):
    def forward_cuda(self, q, k, v, g, beta, ...):
        # FlashInfer 后端 (SM90/SM100)
        return flashinfer_chunk_gated_delta_rule(q, k, v, g, beta, ...)

    def forward_native(self, q, k, v, g, beta, ...):
        # Triton / FLA 后端
        return triton_chunk_gated_delta_rule(q, k, v, g, beta, ...)
```

**分块策略**：
- 将长序列切分为固定大小的 chunk (默认 64)
- 每个 chunk 内使用并行扫描计算
- chunk 之间通过状态传递保持依赖

```
序列: [t0, t1, ..., t63 | t64, ..., t127 | t128, ...]
       ←── Chunk 0 ──→   ←── Chunk 1 ──→   ←── Chunk 2 ──→
       
状态流: S_init → [Chunk 0 并行] → S_64 → [Chunk 1 并行] → S_128 → ...
```

### 4.7 Decode 阶段优化

单 token 解码时使用融合的递归更新：

```python
# fused_sigmoid_gating_delta_rule_update
# 单次内核完成: conv1d_state_update + 递归 + 门控
# 避免多次内核启动的开销
```

**打包递归解码** (FLA Packed Recurrent Decode)：

```python
# 当 enable_packed_recurrent_decode=True 时
# 将多个请求的 decode 打包到一个批次中处理
# 使用 fused_recurrent_gated_delta_rule_packed_decode
```

### 4.8 融合 GDN 门控内核

在 Ascend 上有专用的 Triton 融合内核：

```python
# vllm-ascend ops/triton/fused_gdn_gating.py
@triton.jit
def fused_gdn_gating_kernel(g, beta_output, A_log, a, b, dt_bias, ...):
    """
    融合计算:
    1. g = -exp(A_log) * softplus(a + dt_bias)    # 状态衰减率
    2. beta_output = sigmoid(b)                     # 输出门控
    """
    pid = tl.program_id(0)
    # 加载 A_log, a, b, dt_bias
    # 计算 g 和 beta
    # 存储结果
```

### 4.9 GDN 后端选择

| 后端 | 平台 | Prefill | Decode |
|------|------|---------|--------|
| FlashInfer | CUDA SM90+ | chunk_gated_delta_rule | fused_sigmoid_gating_delta_rule_update |
| Triton/FLA | CUDA 通用 | triton chunk | triton fused |
| AITER Triton | ROCm | 融合 reshape+conv+recurrent | 融合 decode |
| XPU GDN | Intel XPU | 原生 XPU | 原生 XPU |
| CPU GDN | CPU | 注册的 CPU 操作 | 注册的 CPU 操作 |

---

## 5. Qwen3-Next 前代架构

### 5.1 架构概述

Qwen3-Next 是 Qwen3.5 的前身，共享 GDN 混合注意力架构，但有以下关键差异：

```
Qwen3-Next 层结构:

Layer 0:  [Full Attention] + [Dense MLP]
Layer 1:  [GDN Linear Attn] + [Dense MLP]
Layer 2:  [GDN Linear Attn] + [Dense MLP]
Layer 3:  [Full Attention] + [Dense MLP]
Layer 4:  [GDN Linear Attn] + [MoE]
Layer 5:  [GDN Linear Attn] + [MoE]
Layer 6:  [GDN Linear Attn] + [Dense MLP]
Layer 7:  [Full Attention] + [MoE]
...
```

### 5.2 关键差异

| 特性 | Qwen3-Next | Qwen3.5 |
|------|-----------|---------|
| 注意力布局 | **交错 GQA** (gqa_interleaved_layout=True) | 非交错 |
| MoE 分布 | Dense MLP + MoE 交错 | 纯 Dense / 全 MoE |
| 归一化 | RMSNorm + Layer Scale | **GemmaRMSNorm** |
| 专家数 | **512** | 256 |
| Top-K | **10** | 8 |
| TP 处理 | 复杂的交错/去交错逻辑 | 简化的直接分片 |

### 5.3 交错 GQA 布局

Qwen3-Next 的 GDN 投影使用交错 GQA 布局，需要额外的处理：

```python
# Qwen3-Next: gqa_interleaved_layout=True
# in_proj_qkvz 输出布局:
# [q1,k1,v1,q2,k2,v2,q3,k3,v3,...]  # 所有 head 的交错

# 需要先 reshape 再分块:
# qkv = qkv.reshape(batch, seq_len, num_heads, head_dim_total)
# q = qkv[..., :head_dim]
# k = qkv[..., head_dim:head_dim+head_k_dim]
# v = qkv[..., head_dim+head_k_dim:]
```

**Qwen3.5 简化**：使用非交错布局，各投影独立排列，权重加载和 TP 分片都更简单。

---

## 6. MTP 多令牌预测

### 6.1 概述

Qwen3.5 支持通过 MTP (Multi-Token Prediction) 进行推测解码，在单次前向传播中预测多个后续 token。

```
MTP 流程:

Target Model:                         Draft Model (MTP):
                                      
Input: [t1, t2, ..., tn]              Input: [t1, t2, ..., tn]
  │                                      │
  ▼                                      ▼
┌──────────┐                        ┌──────────┐
│ 主模型     │                        │ MTP 模块  │
│ 前向传播   │                        │ (轻量级)   │
└────┬─────┘                        └────┬─────┘
     │                                   │
     ▼                                   ▼
hidden_states[n]                   pred_tokens[n+1, n+2, ...]
     │                                   │
     └────────────┬──────────────────────┘
                  │
                  ▼
          ┌──────────────┐
          │  验证 + 接受   │
          │  投机采样      │
          └──────────────┘
```

### 6.2 Qwen3.5 MTP 实现

```python
# qwen3_5_mtp.py
# Qwen3.5 Dense: Qwen3_5MTP
# Qwen3.5 MoE: Qwen3_5MoeMTP

class Qwen3_5MTP(nn.Module):
    """
    多令牌预测器:
    - n_predict = mtp_num_hidden_layers (预测的 token 数)
    - 每个预测头是一个轻量级的 Transformer 层
    - 共享 embedding 和 lm_head
    """
    def __init__(self, vllm_config, prefix):
        self.mtp_layers = nn.ModuleList([
            Qwen3_5DecoderLayer(...)
            for _ in range(config.mtp_num_hidden_layers)
        ])

    def forward(self, input_ids, positions, previous_hidden_states, ...):
        # 逐个预测后续 token
        for i, layer in enumerate(self.mtp_layers):
            hidden_states = layer(positions, hidden_states)
            # 每个位置的 hidden_states 用于预测下一个 token
```

### 6.3 MTP 配置自动检测

```python
# vllm-ascend patch_speculative_config.py
if hf_config.model_type in ("qwen3_5", "qwen3_5_moe"):
    is_moe = hf_config.model_type == "qwen3_5_moe"
    # 自动选择对应的 MTP 架构
    architectures = ["Qwen3_5MoeMTP" if is_moe else "Qwen3_5MTP"]
```

---

## 7. Qwen3.5-Omni 全模态架构

### 7.1 概述

Qwen3.5-Omni 于 **2026 年 3 月 30 日** 压轴发布，是千问系列迄今为止音视频理解能力最强的模型，在 **215 项音视频基准** 上拿到 SOTA。

### 7.2 Thinker-Talker 架构

Qwen3.5-Omni 继承并升级了 Qwen2.5-Omni 的 **Thinker-Talker** 架构：

```
┌─────────────────────────────────────────────────────┐
│              Qwen3.5-Omni 架构                        │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌──────────────────┐    ┌──────────────────┐        │
│  │    Thinker        │    │    Talker         │        │
│  │  (思考模块)        │───▶│  (表达模块)        │        │
│  │                   │    │                    │        │
│  │  • 多模态理解      │    │  • 语音合成        │        │
│  │  • 视频分析        │    │  • 文本生成        │        │
│  │  • 音频理解        │    │  • 流式输出        │        │
│  │  • 推理决策        │    │                    │        │
│  └──────────────────┘    └──────────────────┘        │
│           │                        │                  │
│           │        音频 Token       │                  │
│           └────────────────────────┘                  │
│                                                       │
│  输入: 文本 + 图像 + 视频 + 音频                        │
│  输出: 文本 + 语音（双模态输出）                        │
└─────────────────────────────────────────────────────┘
```

**Thinker** 负责深度理解，处理文本、图像、视频、音频等多模态输入，进行推理和决策。

**Talker** 负责表达输出，将 Thinker 的思考结果转化为自然流畅的语音或文本，支持流式输出。

### 7.3 视频处理能力

| 能力 | 详情 |
|------|------|
| 视频长度 | 可分析长达 **2 小时** 的视频 |
| 时间精度 | 秒级定位 |
| 视频 SOTA | 215 项音视频基准 SOTA |
| 对比优势 | 全面超越 Gemini 3.1 Pro |

### 7.4 Audio-Visual Vibe Coding

Qwen3.5-Omni 引入了视频驱动代码生成能力：

```
用户: 展示一个网页设计稿的视频

Qwen3.5-Omni:
  1. 分析视频中的 UI 设计
  2. 理解布局、组件、交互逻辑
  3. 生成对应的 HTML/CSS/JS 代码
  4. 保持与原设计一致的视觉效果
```

### 7.5 Omni 模型规格

| 模型 | 总参数 | 激活参数 | 架构 | 显存 |
|------|--------|----------|------|------|
| Qwen3.5-Omni-Plus | 30B | 3B | MoE + Thinker-Talker | 60 GB (BF16) |
| Qwen3.5-Omni-Flash | — | — | MoE + Thinker-Talker | 更小 |

---

## 8. 训练基础设施与 Qwen3.6

### 8.1 训练基础设施

Qwen3.5 的训练体系包含多项工程创新：

#### FP8 训练管线

```
传统 BF16 训练:                  Qwen3.5 FP8 训练:
                                  
权重: BF16 (2 bytes/param)       权重: FP8 (1 byte/param)
激活: BF16                       激活: FP8
梯度: BF16                       梯度: BF16 (保持精度)
                                  
显存占用: 基准                    显存占用: ~50%
训练吞吐: 基准                    训练吞吐: ~2×
```

FP8 精度训练大幅降低显存占用和训练成本，使得 397B 参数的旗舰模型训练成为可能。

#### 大规模强化学习环境扩展（RL Environment Scaling）

Qwen3.5 在后训练阶段采用了创新的 RL 训练范式：

```
┌───────────────────────────────────────────────────────┐
│              异步 RL 框架                               │
├───────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐     ┌─────────────┐                   │
│  │ Agent 采样   │     │ 模型更新     │                   │
│  │ (百万级环境) │────▶│ (异步更新)   │                   │
│  └─────────────┘     └─────────────┘                   │
│         │                    │                          │
│         │    环境编排         │   渐进式难度             │
│         ▼                    ▼                          │
│  ┌─────────────────────────────────────┐               │
│  │  跨百万级 Agent 环境进行 RL           │               │
│  │  使用渐进式复杂任务分布               │               │
│  │  解耦采样与更新，提升吞吐和稳定性      │               │
│  └─────────────────────────────────────┘               │
│                                                         │
└───────────────────────────────────────────────────────┘
```

**关键特性**：

- **百万级 Agent 环境**：同时在大量不同任务环境中进行 RL
- **渐进式复杂任务分布**：从简单到复杂逐步提升难度
- **异步 RL 框架**：Agent 动作采样与模型更新解耦，提升训练稳定性和吞吐量
- **多模态训练效率接近 100%**：与纯文本训练相当，业界罕见

### 8.2 Qwen3.6

Qwen3.6 与 Qwen3.5 VL MoE 共享相同架构（`Qwen3_5MoeForConditionalGeneration`），通过相同的 Bridge 支持，无需代码变更。

| 模型 | 总参数 | 激活参数 | 架构 |
|------|--------|----------|------|
| Qwen3.6-35B-A3B | 35B | 3B | MoE + GDN |

**Qwen3.6 关键参数**：

| 参数 | 值 |
|------|-----|
| 专家总数 | 256 |
| 激活专家/token | 8 路由 + 1 共享 |
| 层数 | 40 |
| 隐藏层结构 | 10 × (3 × GDN → 1 × Full Attention) |

---

## 9. 量化方案

### 9.1 支持的量化方案

| 方案 | 权重精度 | 激活精度 | 适用层 |
|------|---------|---------|--------|
| W8A8_DYNAMIC | INT8 | INT8 (动态量化) | 所有 Linear |
| W4A8_DYNAMIC | INT4 | INT8 (动态量化) | 所有 Linear |
| W4A16 | INT4 | FP16/BF16 | 所有 Linear |
| W8A16 | INT8 | FP16/BF16 | 所有 Linear |
| C8 | FP16/BF16 | INT8 (KV Cache) | 注意力层 |
| TurboQuant | FP16 | K8V4/K4V4/K3V4 | 注意力层 |

### 9.2 W8A8 实现

```python
# vllm-ascend quantization/methods/w8a8_dynamic.py
class AscendW8A8DynamicLinearMethod:
    def apply(self, layer, x, bias=None):
        # 动态量化激活
        quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x)
        # 量化矩阵乘法
        output = torch_npu.npu_quant_matmul(
            quantized_x, layer.weight, layer.weight_scale,
            pertoken_scale=pertoken_scale, bias=bias,
            output_dtype=x.dtype,
        )
        return output

    def process_weights_after_loading(self, layer):
        # 权重转置 + Fractal NZ 格式
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight.data = maybe_trans_nz(layer.weight.data)
        layer.weight_scale.data = layer.weight_scale.data.flatten()
```

### 9.3 W4A8 实现

```python
# vllm-ascend quantization/methods/w4a8.py (599行)
class AscendW4A8DynamicLinearMethod:
    """
    两级量化:
    1. Per-channel 量化: 权重从 FP16 -> INT4 (per-channel scale)
    2. Per-group 量化: 进一步压缩为 per-group scale
    """
    def process_weights_after_loading(self, layer):
        # 合并两级 scale: antiquant_scale = scale * per_group_scale
        # 权重打包: 2个INT4 -> 1个INT8
        layer.weight.data = torch_npu.npu_convert_weight_to_int4pack(
            layer.weight.data
        )

    def apply(self, layer, x, bias=None):
        # 使用融合的权重量化批量矩阵乘法
        return torch_npu.npu_weight_quant_batchmatmul(
            x, layer.weight, layer.antiquant_scale, bias=bias
        )
```

### 9.4 MoE W4A8 实现

```python
class AscendW4A8DynamicFusedMoEMethod:
    """
    MoE 层 W4A8 量化:
    - 每个 expert 独立量化
    - 支持动态 EPLB (专家权重在 NPU 间传输)
    - 支持 INT4 打包为 INT32 格式
    """
    def process_weights_after_loading(self, layer):
        # 权重打包为 INT4
        for expert_idx in range(num_experts):
            w13 = layer.w13_weight[expert_idx]
            w2 = layer.w2_weight[expert_idx]
            # npu_quantize 返回 INT4 打包权重
            layer.w13_weight_list.append(torch_npu.npu_quantize(w13))
            layer.w2_weight_list.append(torch_npu.npu_quantize(w2))
```

### 9.5 C8 KV Cache

```python
# vllm-ascend quantization/methods/kv_c8.py
class AscendC8KVCacheAttentionMethod:
    """
    INT8 KV Cache 量化:
    - Per-channel 静态量化
    - K/V 各有独立的 scale 和 offset
    - 通过类手术将 Attention Backend 切换为 C8 实现
    """
    def create_weights(self, layer):
        # 创建量化参数
        layer.k_cache_scale = nn.Parameter(...)
        layer.k_cache_offset = nn.Parameter(...)
        layer.v_cache_scale = nn.Parameter(...)
        layer.v_cache_offset = nn.Parameter(...)
        # 类手术: 替换为 C8 Attention Backend
        layer.impl.__class__ = AscendC8AttentionBackendImpl
```

### 9.6 TurboQuant

专门为 Qwen3 优化的 KV Cache 量化方案：

```python
# TurboQuant 预设
TQ_PRESETS = {
    "turboquant_k8v4":   {"key_quant_bits": 8, "value_quant_bits": 4},
    "turboquant_4bit_nc": {"key_quant_bits": 4, "value_quant_bits": 4, "norm_correction": True},
    "turboquant_k3v4_nc": {"key_quant_bits": 3, "value_quant_bits": 4, "norm_correction": True},
}
# norm_correction: 修正量化引起的方差偏移
# 对 Qwen3-4B 至关重要: 无此修正 GSM8K 下降约 30 分
```

---

## 10. NPU 融合算子详解

### 10.1 融合 QKV+RMSNorm+MRoPE

Qwen3.5 在 Ascend 上最重要的融合优化：

```python
# vllm-ascend patch_qwen3_5.py
class AscendQwen3NextAttention:
    def forward(self, positions, output, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        if "qwen3_5" in self.config.model_type:
            # 融合算子: 单次内核完成以下操作:
            # 1. QKV 张量分割
            # 2. Q/K 的 RMS 归一化 (GemmaRMSNorm)
            # 3. MRoPE 位置编码应用
            # 4. 门控向量提取 (如有 attn_output_gate)
            q, k, v, gate = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
                qkv=qkv,
                q_weight=1.0 + self.q_norm.weight,
                k_weight=1.0 + self.k_norm.weight,
                cos_sin=cos_sin,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_dim,
                eps=self.config.rms_norm_eps,
                mrope_section=self.rotary_emb.mrope_section,
                is_interleaved=self.rotary_emb.mrope_interleaved,
                rope_dim=self.rotary_emb.rotary_dim,
                has_gate=self.attn_output_gate,
            )
```

**优化效果**：将 4 个独立操作（split + rms_norm × 2 + RoPE）融合为 1 个内核，减少 3 次显存往返。

### 10.2 NPU 分页注意力

```python
# vllm-ascend attention/attention_v1.py
class AscendAttentionBackendImpl:
    def forward_decode(self, ...):
        # Decode 阶段: 使用 NPU 原生分页注意力
        output = torch_npu._npu_paged_attention(
            query, key_cache, value_cache,
            num_heads, scale_value,
            block_table=block_table,
            actual_seq_lengths=actual_seq_lengths,
        )

    def forward_prefill(self, ...):
        # Prefill 阶段: 使用融合推理注意力
        output = torch_npu.npu_fused_infer_attention_score(
            query, key, value,
            attn_mask=mask,
            scale_value=scale,
        )
```

### 10.3 NPU RoPE / MRoPE

```python
# vllm-ascend ops/rotary_embedding.py
class AscendMRotaryEmbedding:
    """
    多分辨率 RoPE (MRoPE):
    - 不同维度段使用不同的旋转频率
    - 支持图像+文本多分辨率输入
    """
    def forward(self, positions, query, key):
        if self.use_npu_mrope:
            # NPU 原生 MRoPE (half 精度模式)
            return torch_npu.npu_mrope(
                positions, query, key,
                self.cos_sin_cache,
                mode="half",
            )
        else:
            # Triton MRoPE (大网格回退)
            return triton_mrope(positions, query, key, ...)
```

### 10.4 融合残差 + RMSNorm

```python
# vllm-ascend ops/layernorm.py
class AscendRMSNorm:
    def forward(self, x, residual=None):
        if residual is not None:
            # 融合: residual + x + RMSNorm
            return torch.ops._C_ascend.npu_add_rms_norm_bias(
                x, residual, self.weight, self.eps
            )
        else:
            return torch_npu.npu_rms_norm(x, self.weight, self.eps)
```

### 10.5 Fused MC2

```python
# vllm-ascend ops/fused_moe/moe_comm_method.py
class FusedMC2CommImpl:
    """
    融合 MC2 通信: 将 token dispatch + MLP + token combine
    融合为单个 NPU 内核调用
    """
    def fused_experts(self, input):
        if enable_fused_mc2 == 1:
            # dispatch_ffn_combine: dispatch + FFN + combine
            out = torch.ops._C_ascend.dispatch_ffn_combine(
                x=input.hidden_states,
                weight1=input.weights.w1,
                weight2=input.weights.w2,
                expert_idx=topk_ids,
                ...
            )
        elif enable_fused_mc2 == 2:
            # dispatch_gmm_combine_decode: GMM + combine (decode 优化)
            out = torch.ops._C_ascend.dispatch_gmm_combine_decode(
                x=input.hidden_states,
                expert_ids=topk_ids,
                gmm1_permuted_weight=input.weights.w1,
                gmm2_weight=input.weights.w2,
                ...
            )
```

### 10.6 DFlash KV 预计算

```python
# vllm-ascend patch_qwen3_dflash.py
# Qwen3 DFlash 的优化: 单次前向传播预计算所有层的 KV 缓存

def precompute_and_store_context_kv(self, hidden_states, ...):
    # 优化 1: 融合 KV 投影 -- 所有层单次 GEMM
    all_kv, _ = self.fused_kv_proj(hidden_states)

    # 优化 2: 层优先布局变换 -- [2, L, num_ctx, nkv, hd]
    all_kv = all_kv.view(2, num_layers, num_tokens, num_kv_heads, head_dim)

    # 优化 3: 逐层 K RMS 归一化
    for layer_idx in range(num_layers):
        k[layer_idx] = rms_norm(k[layer_idx], layer.k_norm.weight)

    # 优化 4: 跨所有层融合 RoPE (单次内核)
    fused_rope_across_layers(q_pe, k_pe, positions, cos_sin)

    # 优化 5: 直接逐层缓存写入
    for layer_idx in range(num_layers):
        kv_cache[layer_idx][...] = k[layer_idx], v[layer_idx]
```

---

## 11. Ascend 平台全栈优化

### 11.1 优化全景图

```
Qwen3.5 on Ascend NPU 优化栈:

┌─────────────────────────────────────────────────────────────┐
│                       应用层                                  │
│  MTP 推测解码 | EPLB 负载均衡 | DP 负载均衡调度               │
├─────────────────────────────────────────────────────────────┤
│                       编译层                                  │
│  npugraph_ex 图编译 | Static Kernel | FX 图融合 Pass          │
│  fuse_norm_quant | fuse_qknorm_rope | fuse_muls_add          │
├─────────────────────────────────────────────────────────────┤
│                       算子层                                  │
│  融合 QKV+Norm+RoPE | 融合残差+Norm | 融合 GDN 门控            │
│  Fused MC2 | FIA 注意力 | NPU Paged Attention                 │
├─────────────────────────────────────────────────────────────┤
│                       通信层                                  │
│  FlashComm1 (TP+DP→EP) | FlashComm2 | HCCL AIV 模式          │
├─────────────────────────────────────────────────────────────┤
│                       硬件层                                  │
│  Ascend 910B/C NPU | 可扩展内存 | 任务队列                     │
└─────────────────────────────────────────────────────────────┘
```

### 11.2 编译优化

**npugraph_ex** (默认启用)：

```python
# compiler_interface.py
def npugraph_ex_compile(graph, example_inputs, ...):
    import torchair
    config = torchair.CompilerConfig()
    config.mode = "reduce-overhead"  # 减少图启动开销
    config.debug.run_eagerly = True  # 先 eager 执行优化 FX 图
    npugraph_ex = torchair.get_npu_backend(compiler_config=config)
    return npugraph_ex(graph, example_inputs)
```

**Static Kernel** (可选，适用于 shape 变化不大的场景)：

```python
if enable_static_kernel:
    # 预编译固定形状的算子二进制
    config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
    # 指定支持的 batch size 范围
    config.experimental_config.aclgraph._aclnn_static_shape_kernel_sym_value_range = [
        1, 2, 4, 8, 16, 32, 64, 128
    ]
```

### 11.3 通信优化

**FlashComm1** (序列并行)：

```
传统 TP+DP:
  TP AG → Attention → TP RS → DP AG → MoE → DP RS

FlashComm1 (TP+DP→EP):
  EP AG → Attention → EP RS → MoE
  (将 TP 和 DP 的 all-gather/reduce-scatter 融合为单次 EP 通信)
```

**FlashComm2** (OShard 优化)：

```python
# 将 QKV 并行投影的 AllReduce 替换为 O 投影的 ReduceScatter
# QKV: 无通信 (每 rank 计算自己的分片)
# O: ReduceScatter (合并各 rank 的输出分片)
```

### 11.4 内存优化

```python
# 可扩展内存段
PYTORCH_NPU_ALLOC_CONF = "expandable_segments:True"
# 允许 NPU 内存池动态扩展，减少碎片

# 任务队列
TASK_QUEUE_ENABLE = 1
# 启用 NPU 任务队列，减少驱动开销
```

### 11.5 EPLB

```python
# 与 DeepSeek 相同的 EPLB 机制
# 自动检测 expert 负载不均衡，动态重排 expert 分布

DYNAMIC_EPLB = "true"
# 在每个 MoE 层记录 expert 负载
# 定期触发重排算法
# 通过 D2D (Device-to-Device) 传输 expert 权重
```

### 11.6 测试配置参考

来自 vllm-ascend 实际测试配置的性能数据：

| 模型 | 配置 | 性能 |
|------|------|------|
| Qwen3-8B | TP1 | 1514 tok/s |
| Qwen3-30B-A3B W8A8 | TP2 | - |
| Qwen3-32B W8A8 | TP4, FlashComm | - |
| Qwen3-235B-A22B W8A8 | TP4, DP4, EP, FlashComm1 | - |
| Qwen3.5-27B W8A8 MTP | TP, MTP=3 | - |
| Qwen3.5-397B-A17B W8A8 MTP | TP16, EP, FUSED_MC2, MTP=5 | - |

---

## 12. 推理优化技术

### 12.1 GDN Prefill 预热

```python
# qwen_gdn_linear_attn.py:L991-L1093
def _warmup_prefill_kernels(self, qkv_or_qkvz, v_dim):
    """
    在 V1 分析阶段预热自动调优内核
    避免在内存已分配后 OOM
    使用 T=chunk_size 运行一次 chunk_gated_delta_rule
    足以填充整个自动调优器缓存
    """
    # 模拟 chunk_size=64 的 prefill
    # 触发 FlashInfer/Triton 的自动调优
    chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=64)
```

### 12.2 FSE 权重加载

在 ROCm 平台上，共享专家权重可以直接合并到路由专家中：

```python
# qwen3_next.py:L550-L596
if is_fse and "mlp.shared_expert." in name:
    # 将共享专家权重重映射到融合专家槽位
    name = name.replace("mlp.shared_expert.", f"mlp.experts.{num_routed}.")
    # 共享专家成为第 num_routed 个专家
    # 路由时自动包含共享专家的计算
```

### 12.3 AWQ/Marlin 兼容

```python
# qwen_gdn_linear_attn.py:L538-L554
def maybe_disable_tp(self, quant_config):
    """
    AWQMarlin 要求 output_size_per_partition >= 64
    Qwen3.5 的非交错布局 [num_v_heads]*2 在 TP>=2 时违反此要求
    解决方案: 复制 ba_proj 并在前向传播中切分到本地 TP 秩
    """
    return (current_platform.is_cuda()
            and not self.gqa_interleaved_layout
            and isinstance(quant_config, (AWQMarlinConfig, AutoGPTQConfig, INCConfig)))
```

### 12.4 分块索引预计算

```python
# GDN 注意力后端
# 预计算分块索引以避免 GPU→CPU 同步
# 支持 spec decode + prefill + decode 混合批次
# 管理 conv_state + ssm_state 的 Mamba 风格 KV 缓存
```

### 12.5 推理思考解析

```python
# reasoning/qwen3_reasoning_parser.py
class Qwen3ReasoningParser:
    """
    处理 Qwen3 的思考令牌:
    <think>
    ...思考内容...
     response
    ...最终回答...
    
    <tool_call> 隐式结束思考模式
    """
```

---

## 13. 架构总结与对比

### 13.1 核心创新总结

| 创新 | 描述 | 影响 |
|------|------|------|
| **GDN 线性注意力** | O(N) 复杂度的门控 DeltaNet | 长序列推理效率大幅提升 |
| **混合注意力** | 75% GDN + 25% Full Attention | 兼顾效率与精度 |
| **GemmaRMSNorm** | weight = 1.0 + param | 更稳定的训练初始化 |
| **注意力输出门控** | sigmoid(gate) × attn_output | 动态控制注意力贡献 |
| **Layer Scale** | output *= (scale + 1.0) | 细粒度的残差流控制 |
| **全层 MoE** | Qwen3.5-MoE 所有层使用 MoE | 更大的模型容量 |

### 13.2 与 DeepSeek 对比

| 特性 | DeepSeek V3/V4 | Qwen3.5 |
|------|---------------|---------|
| 注意力类型 | MLA (低秩压缩) | GDN (线性递归) + GQA |
| KV Cache 压缩 | 低秩压缩 (~10-50x) | 线性注意力 (无 KV Cache) |
| 稀疏注意力 | Indexer + Compressor (V4) | 无 |
| MoE 路由 | Grouped Top-K (softmax/sigmoid) | Sigmoid Top-K |
| 共享专家 | 有 | 有 (sigmoid gate) |
| MTP | 有 (eh_proj / e_proj+h_proj) | 有 (独立层) |
| 量化 | FP8/W8A8/MXFP4 | W8A8/W4A8/C8/TurboQuant |
| 特殊结构 | MHC/HC (V4) | Layer Scale, Output Gate |

### 13.3 Ascend 优化对比

| 优化 | DeepSeek | Qwen3.5 |
|------|----------|---------|
| 融合 QKV 处理 | MLAPO (MLA 预处理) | triton_split_qkv_rmsnorm_mrope |
| 注意力后端 | SFA/DSA/MLA | FIA/Paged Attention |
| 稀疏注意力 | Lightning Indexer + SFA | 无 (使用全量注意力) |
| MoE 通信 | MC2/All2All/AllGather | FusedMC2/All2All |
| 多流重叠 | Shared Expert + Gate | Gate + Shared Expert |
| 编译优化 | npugraph_ex | npugraph_ex + Static Kernel |
| KV Cache 量化 | C8 (稀疏索引器) | C8 (逐通道) + TurboQuant |

### 13.4 文件路径速查

| 组件 | 路径 |
|------|------|
| Qwen3.5 Dense 模型 | [qwen3_5.py](file:///D:/trae-workspace/github/vllm/vllm/model_executor/models/qwen3_5.py) |
| Qwen3.5-MoE 模型 | [qwen3_5.py](file:///D:/trae-workspace/github/vllm/vllm/model_executor/models/qwen3_5.py) (同一文件) |
| Qwen3-Next 模型 | [qwen3_next.py](file:///D:/trae-workspace/github/vllm/vllm/model_executor/models/qwen3_next.py) |
| GDN 线性注意力 | [qwen_gdn_linear_attn.py](file:///D:/trae-workspace/github/vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py) |
| GDN 注意力后端 | [gdn_attn.py](file:///D:/trae-workspace/github/vllm/vllm/v1/attention/backends/gdn_attn.py) |
| Qwen3.5 MTP | [qwen3_5_mtp.py](file:///D:/trae-workspace/github/vllm/vllm/model_executor/models/qwen3_5_mtp.py) |
| Qwen3.5 Ascend 适配 | [patch_qwen3_5.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py) |
| DFlash Ascend 优化 | [patch_qwen3_dflash.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_dflash.py) |
| 融合 GDN 门控内核 | [fused_gdn_gating.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/ops/triton/fused_gdn_gating.py) |
| NPU 注意力后端 | [attention_v1.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/attention/attention_v1.py) |
| W8A8 量化 | [w8a8_dynamic.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/quantization/methods/w8a8_dynamic.py) |
| W4A8 量化 | [w4a8.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/quantization/methods/w4a8.py) |
| C8 KV Cache 量化 | [kv_c8.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/quantization/methods/kv_c8.py) |
| 量化配置入口 | [modelslim_config.py](file:///D:/trae-workspace/github/vllm-ascend/vllm_ascend/quantization/modelslim_config.py) |
| Qwen3.5 配置类 | [qwen3_5.py configs](file:///D:/trae-workspace/github/vllm/vllm/transformers_utils/configs/qwen3_5.py) |
| Qwen3.5-MoE 配置类 | [qwen3_5_moe.py configs](file:///D:/trae-workspace/github/vllm/vllm/transformers_utils/configs/qwen3_5_moe.py) |
