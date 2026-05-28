import { useEffect, useState, useCallback } from 'react'
import { useParams, Link } from 'react-router-dom'
import { ArrowLeft, Clock, Tag } from 'lucide-react'
import { useThemeStore } from '../store/themeStore'
import { getArticleBySlug } from '../utils/articles'
import { categoryMap } from '../types'
import ScrollProgress from '../components/ScrollProgress'
import Sidebar from '../components/Sidebar'
import MarkdownRenderer from '../components/MarkdownRenderer'
import BackToTop from '../components/BackToTop'
import type { TocItem } from '../types'

export default function ArticlePage() {
  const { slug } = useParams<{ slug: string }>()
  const { theme } = useThemeStore()
  const isDark = theme === 'dark'
  const [content, setContent] = useState('')
  const [tocItems, setTocItems] = useState<TocItem[]>([])
  const [loading, setLoading] = useState(true)

  const article = getArticleBySlug(slug || '')

  const handleTocReady = useCallback((items: TocItem[]) => {
    setTocItems(items)
  }, [])

  useEffect(() => {
    if (!article) {
      setLoading(false)
      return
    }

    fetch(article.filePath)
      .then((res) => res.text())
      .then((text) => {
        setContent(text)
        setLoading(false)
      })
      .catch(() => {
        setContent('# 加载失败\n\n无法加载文章内容，请稍后重试。')
        setLoading(false)
      })
  }, [article])

  if (!article) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <div className="text-center">
          <h2
            className="mb-2 text-2xl font-bold"
            style={{ color: isDark ? '#e6edf3' : '#1f2328' }}
          >
            文章未找到
          </h2>
          <p className="mb-4" style={{ color: isDark ? '#8b949e' : '#656d76' }}>
            您访问的文章不存在或已被移除。
          </p>
          <Link to="/" className="text-amber-500 hover:text-amber-400">
            返回首页
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="relative">
      <ScrollProgress />
      <div className="mx-auto max-w-7xl px-6">
        <div className="flex">
          <Sidebar items={tocItems} />
          <div className="w-full xl:ml-56 xl:max-w-3xl">
            <div className="mb-8">
              <Link
                to="/"
                className="mb-4 inline-flex items-center gap-1.5 text-sm transition-colors"
                style={{ color: isDark ? '#8b949e' : '#656d76' }}
              >
                <ArrowLeft size={14} />
                <span className="hover:text-amber-500">返回首页</span>
              </Link>
              <h1
                className="mb-4 text-3xl font-bold tracking-tight sm:text-4xl"
                style={{ color: isDark ? '#e6edf3' : '#1f2328' }}
              >
                {article.title}
              </h1>
              <div className="mb-4 flex flex-wrap items-center gap-3">
                {categoryMap[article.category] && (
                  <span
                    className="rounded-full px-3 py-1 text-xs font-semibold"
                    style={{
                      backgroundColor: isDark ? '#d2991d15' : '#d2991d10',
                      color: '#d2991d',
                    }}
                  >
                    {categoryMap[article.category].name}
                  </span>
                )}
                <div
                  className="flex items-center gap-1.5 text-sm"
                  style={{ color: isDark ? '#484f58' : '#8b949e' }}
                >
                  <Clock size={14} />
                  <span>{article.readingTime} 分钟阅读</span>
                </div>
                <div
                  className="flex items-center gap-1.5 text-sm"
                  style={{ color: isDark ? '#484f58' : '#8b949e' }}
                >
                  <Tag size={14} />
                  <span>{article.date}</span>
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                {article.tags.map((tag) => (
                  <span
                    key={tag}
                    className="rounded-full border px-2.5 py-0.5 text-xs font-medium"
                    style={{
                      borderColor: isDark
                        ? 'rgba(210,153,29,0.3)'
                        : 'rgba(210,153,29,0.2)',
                      color: '#d2991d',
                    }}
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>

            {loading ? (
              <div className="animate-pulse space-y-4 py-8">
                {Array.from({ length: 12 }).map((_, i) => (
                  <div
                    key={i}
                    className="h-4 rounded"
                    style={{
                      width: `${60 + Math.random() * 40}%`,
                      backgroundColor: isDark ? '#1c2128' : '#e1e4e8',
                    }}
                  />
                ))}
              </div>
            ) : (
              <div className="pb-20">
                <MarkdownRenderer content={content} onTocReady={handleTocReady} />
              </div>
            )}
          </div>
        </div>
      </div>
      <BackToTop />
    </div>
  )
}
