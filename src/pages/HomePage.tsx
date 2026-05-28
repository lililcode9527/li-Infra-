import { Brain, Zap, Code2, Cpu } from 'lucide-react'
import HeroSection from '../components/HeroSection'
import ArticleCard from '../components/ArticleCard'
import { useThemeStore } from '../store/themeStore'
import { categories } from '../types'
import { getArticlesByCategory } from '../utils/articles'

const iconMap: Record<string, typeof Brain> = {
  Brain,
  Zap,
  Code2,
  Cpu,
}

const categoryAccents: Record<string, string> = {
  'model-insight': '#d2991d',
  'inference-opt': '#3fb950',
  framework: '#58a6ff',
  hardware: '#f78166',
}

export default function HomePage() {
  const { theme } = useThemeStore()
  const isDark = theme === 'dark'

  return (
    <div>
      <HeroSection />

      {categories.map((cat) => {
        const catArticles = getArticlesByCategory(cat.id)
        if (catArticles.length === 0) return null
        const Icon = iconMap[cat.icon] || Brain
        const accent = categoryAccents[cat.id] || '#d2991d'

        return (
          <section key={cat.id} id={`category-${cat.id}`} className="mx-auto max-w-4xl px-6 pb-16">
            <div className="mb-6 flex items-center gap-3">
              <Icon size={22} style={{ color: accent }} />
              <h2
                className="text-xl font-bold tracking-tight"
                style={{ color: isDark ? '#e6edf3' : '#1f2328' }}
              >
                {cat.name}
              </h2>
              <span
                className="text-sm"
                style={{ color: isDark ? '#484f58' : '#8b949e' }}
              >
                {cat.description}
              </span>
            </div>
            <div className="grid gap-6 md:grid-cols-2">
              {catArticles.map((article, index) => (
                <ArticleCard key={article.id} article={article} index={index} />
              ))}
            </div>
          </section>
        )
      })}

      <section className="mx-auto max-w-4xl px-6 pb-20 text-center">
        <div
          className="rounded-xl border p-10"
          style={{
            borderColor: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
            backgroundColor: isDark ? '#161b22' : '#f6f8fa',
          }}
        >
          <p className="text-sm" style={{ color: isDark ? '#484f58' : '#8b949e' }}>
            更多分类文章正在编写中 · 推理优化 · 框架源码 · 硬件适配
          </p>
        </div>
      </section>
    </div>
  )
}
