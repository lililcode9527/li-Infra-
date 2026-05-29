import { Link } from 'react-router-dom'
import { useThemeStore } from '../store/themeStore'
import { categories } from '../types'
import { Brain, Zap, Code2, Cpu } from 'lucide-react'

const iconMap: Record<string, typeof Brain> = {
  Brain,
  Zap,
  Code2,
  Cpu,
}

const categoryAccents: Record<string, string> = {
  'model-insight': '#d2991d',
  'inference-opt': '#3fb950',
  'other': '#58a6ff',
  'operator-opt': '#f78166',
}

export default function HeroSection() {
  const { theme } = useThemeStore()
  const isDark = theme === 'dark'

  return (
    <section className="relative overflow-hidden py-28 sm:py-36">
      <div
        className="absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage:
            'radial-gradient(circle at 25% 25%, #d2991d 1px, transparent 1px), radial-gradient(circle at 75% 75%, #3fb950 1px, transparent 1px)',
          backgroundSize: '60px 60px',
        }}
      />
      <div className="relative mx-auto max-w-4xl px-6 text-center">
        <div className="mb-6 inline-flex items-center gap-2 rounded-full border px-4 py-1.5 text-xs font-medium"
          style={{
            borderColor: isDark ? 'rgba(210,153,29,0.3)' : 'rgba(210,153,29,0.3)',
            color: '#d2991d',
          }}
        >
          <Cpu size={14} />
          AI INFRA
        </div>
        <h1
          className="mb-6 text-4xl font-bold tracking-tight sm:text-5xl lg:text-6xl"
          style={{ color: isDark ? '#e6edf3' : '#1f2328' }}
        >
          技术文档
        </h1>
        <div className="mx-auto mb-8 h-1 w-20 rounded-full bg-gradient-to-r from-amber-500 to-emerald-500" />
        <p
          className="mx-auto max-w-xl text-lg leading-relaxed"
          style={{ color: isDark ? '#8b949e' : '#656d76' }}
        >
          基于开源推理框架源码，深入剖析模型架构、推理优化、框架源码与硬件适配的核心技术。
        </p>

        <div className="mx-auto mt-12 grid max-w-2xl grid-cols-2 gap-4 sm:grid-cols-4">
          {categories.map((cat) => {
            const Icon = iconMap[cat.icon] || Brain
            const accent = categoryAccents[cat.id] || '#d2991d'
            return (
              <Link
                key={cat.id}
                to={`/category/${cat.id}`}
                className="flex flex-col items-center gap-2 rounded-xl border p-4 transition-all duration-300 hover:-translate-y-1 hover:shadow-lg"
                style={{
                  borderColor: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
                  backgroundColor: isDark ? '#161b22' : '#f6f8fa',
                }}
              >
                <Icon size={22} style={{ color: accent }} />
                <span className="text-sm font-medium" style={{ color: isDark ? '#e6edf3' : '#1f2328' }}>
                  {cat.name}
                </span>
              </Link>
            )
          })}
        </div>
      </div>
    </section>
  )
}