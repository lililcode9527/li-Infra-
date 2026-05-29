export type Category = 'model-insight' | 'inference-opt' | 'operator-opt' | 'other'

export interface CategoryInfo {
  id: Category
  name: string
  description: string
  icon: string
}

export interface ArticleMeta {
  id: string
  title: string
  description: string
  tags: string[]
  date: string
  readingTime: number
  slug: string
  filePath: string
  category: Category
}

export interface TocItem {
  id: string
  text: string
  level: number
}

export const categories: CategoryInfo[] = [
  {
    id: 'model-insight',
    name: '模型洞察',
    description: '大模型架构设计、注意力机制、MoE 路由策略等深度技术分析',
    icon: 'Brain',
  },
  {
    id: 'inference-opt',
    name: '推理优化',
    description: '量化、KV Cache、投机解码、算子融合等推理加速技术',
    icon: 'Zap',
  },
  {
    id: 'operator-opt',
    name: '算子优化',
    description: 'CUDA/Triton/Ascend 算子开发、融合优化与性能调优实践',
    icon: 'Cpu',
  },
  {
    id: 'other',
    name: '其他技术',
    description: '推理框架源码解读、工程实践、工具链与杂项技术分享',
    icon: 'Code2',
  },
]

export const categoryMap = Object.fromEntries(
  categories.map((c) => [c.id, c]),
) as Record<Category, CategoryInfo>