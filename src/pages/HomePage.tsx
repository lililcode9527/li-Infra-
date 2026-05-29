import { Link } from 'react-router-dom'
import { Brain, Zap, Code2, Cpu, ArrowRight } from 'lucide-react'
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
        const Icon = iconMap[cat.icon] || Brain
        const accent = categoryAccents[cat.id] || '#d2991d'

        return (
          <section key={cat.id} className="mx-auto max-w-4xl px-6 pb-16">
            <Link
              to={`/category/${cat.id}`}
              className="mb-6 flex items-center gap-3 group"
            >
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
              <ArrowRight
                size={16}
                className="ml-auto opacity-0 group-hover:opacity-100 transition-opacity"
                style={{ color: accent }}
              />
            </Link>

            {catArticles.length > 0 ? (
              <div className="grid gap-6 md:grid-cols-2">
                {catArticles.map((article, index) => (
                  <ArticleCard key={article.id} article={article} index={index} />
                ))}
              </div>
            ) : (
              <div
                className="rounded-xl border p-8 text-center"
                style={{
                  borderColor: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
                  backgroundColor: isDark ? '#161b22' : '#f6f8fa',
                }}
              >
                <Icon size={32} className="mx-auto mb-3" style={{ color: isDark ? '#484f58' : '#8b949e' }} />
                <p className="text-sm" style={{ color: isDark ? '#484f58' : '#8b949e' }}>
                  「{cat.name}」分类的文章正在编写中，敬请期待
                </p>
                <Link
                  to={`/category/${cat.id}`}
                  className="mt-3 inline-flex items-center gap-1 text-sm font-medium"
                  style={{ color: accent }}
                >
                  查看分类详情 <ArrowRight size={14} />
                </Link>
              </div>
            )}
          </section>
        )
      })}
    </div>
  )
}