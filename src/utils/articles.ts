import type { ArticleMeta } from '../types'

export const articles: ArticleMeta[] = [
  {
    id: 'qwen35',
    title: 'Qwen3.5 系列架构与优化深度分析',
    description:
      '深入剖析 Qwen3.5 混合注意力架构（Gated DeltaNet + Full Attention）、MoE 稀疏专家、原生多模态 Early Fusion、MRoPE、MTP 多令牌预测、Qwen3.5-Omni 全模态架构及华为 Ascend NPU 全栈优化。',
    tags: ['Qwen3.5', 'GDN', 'MoE', '线性注意力', 'Ascend', '多模态'],
    date: '2026-05-28',
    readingTime: 55,
    slug: 'qwen35',
    filePath: '/articles/Qwen3.5_架构与优化分析.md',
    category: 'model-insight',
  },
  {
    id: 'deepseek',
    title: 'DeepSeek V3/V4 架构与优化深度分析',
    description:
      '深入剖析 DeepSeek V3.1 和 V4 的 MLA 低秩注意力、MoE 混合专家、Compressor/Indexer 稀疏注意力、MHC/HC 多头组合、MTP 推测解码及 Ascend NPU 平台适配等关键技术。',
    tags: ['DeepSeek', 'MLA', 'MoE', 'DSA', 'Ascend'],
    date: '2026-05-27',
    readingTime: 45,
    slug: 'deepseek',
    filePath: '/articles/DeepSeek_V3_V4_架构与优化分析.md',
    category: 'model-insight',
  },
  {
    id: 'glm5',
    title: 'GLM-5 / GLM-4.x 架构与优化深度分析',
    description:
      '全面解析 GLM-5 DSA 架构、GLM-4.x MoE 模型、Rotary Quantization、MTP 推测解码、W4A8/W8A8 量化方案、SFA 稀疏注意力、EPLB 负载均衡及华为 Ascend 平台全栈优化。',
    tags: ['GLM-5', 'MoE', 'MLA', 'Ascend', 'Quantization'],
    date: '2026-05-27',
    readingTime: 50,
    slug: 'glm5',
    filePath: '/articles/GLM5_架构与优化分析.md',
    category: 'model-insight',
  },
  {
    id: 'lightning-indexer',
    title: 'Lightning Indexer 算子深度解析：Tiling 切分、流水线优化与 Cube/Vector 并行',
    description:
      '深入剖析 Ascend NPU 上 Lightning Indexer 算子的五缓冲 L1 流水线、双缓冲 L0 流水线、Ping-Pong UB 并发、MicroAPI 融合计算、Histogram-based TopK 等核心优化机制，揭示如何将 Cube 和 Vector 利用率推至极限。',
    tags: ['Lightning Indexer', 'DSA', 'Ascend', 'Tiling', 'Cube/Vector', 'Pipeline'],
    date: '2026-05-29',
    readingTime: 40,
    slug: 'lightning-indexer',
    filePath: '/articles/lightning_indexer_算子优化深度解析.md',
    category: 'operator-opt',
  },
]

export function getArticleBySlug(slug: string): ArticleMeta | undefined {
  return articles.find((a) => a.slug === slug)
}

export function getArticlesByCategory(category: string): ArticleMeta[] {
  return articles.filter((a) => a.category === category)
}
