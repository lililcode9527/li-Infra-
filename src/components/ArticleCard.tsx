import { Link } from 'react-router-dom'
import { Clock, ArrowRight } from 'lucide-react'
import { useThemeStore } from '../store/themeStore'
import { categoryMap } from '../types'
import type { ArticleMeta } from '../types'

const accentColors = ['#d2991d', '#3fb950']

export default function ArticleCard({
  article,
  index,
}: {
  article: ArticleMeta
  index: number
}) {
  const { theme } = useThemeStore()
  const isDark = theme === 'dark'
  const accent = accentColors[index % accentColors.length]
  const cat = categoryMap[article.category]

  return (
    <Link
      to={`/article/${article.slug}`}
      className={`group relative block overflow-hidden rounded-xl border p-6 transition-all duration-300 hover:-translate-y-1 hover:shadow-xl ${
        isDark
          ? 'border-white/[0.06] bg-[#161b22] hover:border-white/[0.1] hover:shadow-amber-500/5'
          : 'border-black/[0.06] bg-[#f6f8fa] hover:border-black/[0.1] hover:shadow-black/5'
      }`}
    >
      <div
        className="absolute left-0 top-0 h-full w-1 transition-all duration-300 group-hover:w-1.5"
        style={{ backgroundColor: accent }}
      />
      <div className="pl-3">
        <div className="mb-3 flex items-center gap-2">
          {cat && (
            <span
              className="rounded-full px-2.5 py-0.5 text-xs font-medium"
              style={{
                backgroundColor: isDark ? `${accent}15` : `${accent}10`,
                color: accent,
              }}
            >
              {cat.name}
            </span>
          )}
          {article.tags.slice(0, 2).map((tag) => (
            <span
              key={tag}
              className="rounded-full px-2 py-0.5 text-xs"
              style={{
                backgroundColor: isDark
                  ? 'rgba(255,255,255,0.04)'
                  : 'rgba(0,0,0,0.04)',
                color: isDark ? '#8b949e' : '#656d76',
              }}
            >
              {tag}
            </span>
          ))}
        </div>
        <h3
          className="mb-2 text-xl font-semibold tracking-tight transition-colors"
          style={{ color: isDark ? '#e6edf3' : '#1f2328' }}
        >
          {article.title}
        </h3>
        <p
          className="mb-4 line-clamp-2 text-sm leading-relaxed"
          style={{ color: isDark ? '#8b949e' : '#656d76' }}
        >
          {article.description}
        </p>
        <div className="flex items-center justify-between">
          <div
            className="flex items-center gap-1.5 text-xs"
            style={{ color: isDark ? '#484f58' : '#8b949e' }}
          >
            <Clock size={14} />
            <span>{article.readingTime} 分钟阅读</span>
          </div>
          <span
            className="flex items-center gap-1 text-sm font-medium transition-colors"
            style={{ color: accent }}
          >
            阅读文章 <ArrowRight size={14} />
          </span>
        </div>
      </div>
    </Link>
  )
}
