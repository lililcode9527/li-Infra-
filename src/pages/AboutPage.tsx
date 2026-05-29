import { Cpu, Github } from 'lucide-react'
import { useThemeStore } from '../store/themeStore'
import { articles } from '../utils/articles'
import { categories, categoryMap } from '../types'

const categoryAccents: Record<string, string> = {
  'model-insight': '#d2991d',
  'inference-opt': '#3fb950',
  'other': '#58a6ff',
  'operator-opt': '#f78166',
}

export default function AboutPage() {
  const { theme } = useThemeStore()
  const isDark = theme === 'dark'

  return (
    <div className="mx-auto max-w-3xl px-6 py-20">
      <div className="mb-10 text-center">
        <Cpu size={40} className="mx-auto mb-4 text-amber-500" />
        <div className="mb-3 inline-flex items-center gap-2 rounded-full border px-4 py-1 text-xs font-medium"
          style={{
            borderColor: 'rgba(210,153,29,0.3)',
            color: '#d2991d',
          }}
        >
          AI INFRA
        </div>
        <h1
          className="mb-3 text-3xl font-bold tracking-tight"
          style={{ color: isDark ? '#e6edf3' : '#1f2328' }}
        >
          关于 AI Infra
        </h1>
        <div className="mx-auto mb-6 h-1 w-16 rounded-full bg-gradient-to-r from-amber-500 to-emerald-500" />
        <p
          className="text-lg leading-relaxed"
          style={{ color: isDark ? '#8b949e' : '#656d76' }}
        >
          一个专注于大模型基础设施的技术博客，涵盖模型架构、推理优化、框架源码与硬件适配等核心领域。
        </p>
      </div>

      <div
        className="mb-10 rounded-xl border p-6"
        style={{
          borderColor: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
          backgroundColor: isDark ? '#161b22' : '#f6f8fa',
        }}
      >
        <h2
          className="mb-4 text-lg font-semibold"
          style={{ color: isDark ? '#e6edf3' : '#1f2328' }}
        >
          内容分类
        </h2>
        <div className="grid gap-3 sm:grid-cols-2">
          {categories.map((cat) => {
            const accent = categoryAccents[cat.id] || '#d2991d'
            return (
              <div
                key={cat.id}
                className="rounded-lg p-3"
                style={{
                  backgroundColor: isDark ? '#0d1117' : '#ffffff',
                  borderLeft: `3px solid ${accent}`,
                }}
              >
                <div className="mb-1 text-sm font-medium" style={{ color: accent }}>
                  {cat.name}
                </div>
                <div className="text-xs" style={{ color: isDark ? '#484f58' : '#8b949e' }}>
                  {cat.description}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      <div
        className="mb-10 rounded-xl border p-6"
        style={{
          borderColor: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
          backgroundColor: isDark ? '#161b22' : '#f6f8fa',
        }}
      >
        <h2
          className="mb-4 text-lg font-semibold"
          style={{ color: isDark ? '#e6edf3' : '#1f2328' }}
        >
          文章来源
        </h2>
        <ul className="space-y-2">
          {[
            { name: 'vllm', url: 'https://github.com/vllm-project/vllm', desc: '高性能 LLM 推理引擎' },
            { name: 'vllm-ascend', url: 'https://github.com/vllm-project/vllm-ascend', desc: '华为 Ascend NPU 适配' },
            { name: 'transformers', url: 'https://github.com/huggingface/transformers', desc: 'HuggingFace Transformers' },
          ].map((item) => (
            <li key={item.name} className="flex items-start gap-3">
              <Github size={18} className="mt-0.5 shrink-0" style={{ color: isDark ? '#8b949e' : '#656d76' }} />
              <div>
                <a
                  href={item.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-medium text-amber-500 hover:text-amber-400"
                >
                  {item.name}
                </a>
                <span
                  className="ml-2 text-sm"
                  style={{ color: isDark ? '#484f58' : '#8b949e' }}
                >
                  — {item.desc}
                </span>
              </div>
            </li>
          ))}
        </ul>
      </div>

      <div
        className="rounded-xl border p-6"
        style={{
          borderColor: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
          backgroundColor: isDark ? '#161b22' : '#f6f8fa',
        }}
      >
        <h2
          className="mb-4 text-lg font-semibold"
          style={{ color: isDark ? '#e6edf3' : '#1f2328' }}
        >
          已发布文章
        </h2>
        <div className="space-y-3">
          {articles.map((article) => {
            const cat = categoryMap[article.category]
            return (
              <div
                key={article.id}
                className="flex items-center justify-between rounded-lg p-3"
                style={{
                  backgroundColor: isDark ? '#0d1117' : '#ffffff',
                }}
              >
                <div>
                  <span style={{ color: isDark ? '#c9d1d9' : '#24292f' }}>
                    {article.title}
                  </span>
                  {cat && (
                    <span
                      className="ml-2 rounded-full px-2 py-0.5 text-xs"
                      style={{
                        color: categoryAccents[cat.id] || '#d2991d',
                        backgroundColor: isDark
                          ? `${categoryAccents[cat.id]}15`
                          : `${categoryAccents[cat.id]}10`,
                      }}
                    >
                      {cat.name}
                    </span>
                  )}
                </div>
                <span
                  className="text-sm"
                  style={{ color: isDark ? '#484f58' : '#8b949e' }}
                >
                  {article.readingTime} 分钟
                </span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
