import { useParams, Link } from 'react-router-dom'
import { Brain, Zap, Code2, Cpu, ArrowRight, FolderOpen } from 'lucide-react'
import { useThemeStore } from '../store/themeStore'
import { categoryMap, categories } from '../types'
import { getArticlesByCategory } from '../utils/articles'
import ArticleCard from '../components/ArticleCard'

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

export default function CategoryPage() {
  const { id } = useParams<{ id: string }>()
  const { theme } = useThemeStore()
  const isDark = theme === 'dark'

  const cat = categoryMap[id as keyof typeof categoryMap]
  const articles = id ? getArticlesByCategory(id) : []
  const accent = categoryAccents[id || ''] || '#d2991d'
  const Icon = cat ? iconMap[cat.icon] || Brain : FolderOpen

  if (!cat) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <div className="text-center">
          <FolderOpen size={48} className="mx-auto mb-4" style={{ color: isDark ? '#484f58' : '#8b949e' }} />
          <h2
            className="mb-2 text-2xl font-bold"
            style={{ color: isDark ? '#e6edf3' : '#1f2328' }}
          >
            分类不存在
          </h2>
          <p className="mb-4" style={{ color: isDark ? '#8b949e' : '#656d76' }}>
            您访问的分类不存在或已被移除。
          </p>
          <Link to="/" className="inline-flex items-center gap-1.5 text-amber-500 hover:text-amber-400">
            返回首页 <ArrowRight size={14} />
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-4xl px-6 py-16">
      {/* 分类头部 */}
      <div className="mb-10">
        <Link
          to="/"
          className="mb-6 inline-flex items-center gap-1.5 text-sm transition-colors"
          style={{ color: isDark ? '#8b949e' : '#656d76' }}
        >
          ← 返回首页
        </Link>
        <div className="flex items-center gap-4 mb-4">
          <div
            className="rounded-xl p-3"
            style={{ backgroundColor: isDark ? `${accent}15` : `${accent}10` }}
          >
            <Icon size={28} style={{ color: accent }} />
          </div>
          <div>
            <h1
              className="text-3xl font-bold tracking-tight"
              style={{ color: isDark ? '#e6edf3' : '#1f2328' }}
            >
              {cat.name}
            </h1>
            <p className="text-sm mt-1" style={{ color: isDark ? '#8b949e' : '#656d76' }}>
              {cat.description}
            </p>
          </div>
        </div>
        <div
          className="h-1 w-16 rounded-full"
          style={{ backgroundColor: accent }}
        />
      </div>

      {/* 文章列表或空状态 */}
      {articles.length > 0 ? (
        <div className="grid gap-6 md:grid-cols-2">
          {articles.map((article, index) => (
            <ArticleCard key={article.id} article={article} index={index} />
          ))}
        </div>
      ) : (
        <div
          className="rounded-xl border p-12 text-center"
          style={{
            borderColor: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
            backgroundColor: isDark ? '#161b22' : '#f6f8fa',
          }}
        >
          <Icon size={40} className="mx-auto mb-4" style={{ color: isDark ? '#484f58' : '#8b949e' }} />
          <h3
            className="mb-2 text-lg font-semibold"
            style={{ color: isDark ? '#e6edf3' : '#1f2328' }}
          >
            暂无文章
          </h3>
          <p className="mb-6 text-sm" style={{ color: isDark ? '#8b949e' : '#656d76' }}>
            「{cat.name}」分类的文章正在编写中，敬请期待。
          </p>
          <div className="flex flex-wrap items-center justify-center gap-3">
            {categories.filter((c) => c.id !== id).map((otherCat) => {
              const otherAccent = categoryAccents[otherCat.id] || '#d2991d'
              const OtherIcon = iconMap[otherCat.icon] || Brain
              return (
                <Link
                  key={otherCat.id}
                  to={`/category/${otherCat.id}`}
                  className="inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium transition-all hover:-translate-y-0.5"
                  style={{
                    borderColor: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
                    color: otherAccent,
                    backgroundColor: isDark ? `${otherAccent}08` : `${otherAccent}06`,
                  }}
                >
                  <OtherIcon size={14} />
                  {otherCat.name}
                </Link>
              )
            })}
            <Link
              to="/"
              className="inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium text-amber-500 transition-all hover:-translate-y-0.5"
              style={{
                borderColor: isDark ? 'rgba(210,153,29,0.2)' : 'rgba(210,153,29,0.15)',
                backgroundColor: isDark ? '#d2991d08' : '#d2991d06',
              }}
            >
              ← 返回首页
            </Link>
          </div>
        </div>
      )}
    </div>
  )
}