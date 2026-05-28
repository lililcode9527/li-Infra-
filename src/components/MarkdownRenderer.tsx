import { useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeRaw from 'rehype-raw'
import rehypeHighlight from 'rehype-highlight'
import { useThemeStore } from '../store/themeStore'
import type { TocItem } from '../types'

interface MarkdownRendererProps {
  content: string
  onTocReady?: (items: TocItem[]) => void
}

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\u4e00-\u9fff]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

function extractTocFromMarkdown(markdown: string): TocItem[] {
  const withoutCodeBlocks = markdown.replace(/```[\s\S]*?```/g, '').replace(/```[\s\S]*$/g, '')
  const headingRegex = /^(#{1,6})\s+(.+)$/gm
  const items: TocItem[] = []
  let match: RegExpExecArray | null

  while ((match = headingRegex.exec(withoutCodeBlocks)) !== null) {
    const level = match[1].length
    if (level > 2) continue
    const text = match[2].trim()
    items.push({ id: slugify(text), text, level })
  }

  return items
}

export default function MarkdownRenderer({ content, onTocReady }: MarkdownRendererProps) {
  const { theme } = useThemeStore()
  const isDark = theme === 'dark'

  const tocItems = useMemo(() => {
    const items = extractTocFromMarkdown(content)
    onTocReady?.(items)
    return items
  }, [content, onTocReady])

  return (
    <div className="article-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw, rehypeHighlight]}
        components={{
          h1: ({ children, ...props }) => {
            const text = String(children)
            return (
              <h1
                id={slugify(text)}
                className="mb-6 mt-10 text-3xl font-bold tracking-tight"
                style={{ color: isDark ? '#e6edf3' : '#1f2328' }}
                {...props}
              >
                {children}
              </h1>
            )
          },
          h2: ({ children, ...props }) => {
            const text = String(children)
            return (
              <h2
                id={slugify(text)}
                className="mb-4 mt-8 text-2xl font-semibold tracking-tight"
                style={{ color: isDark ? '#e6edf3' : '#1f2328' }}
                {...props}
              >
                {children}
              </h2>
            )
          },
          h3: ({ children, ...props }) => {
            const text = String(children)
            return (
              <h3
                id={slugify(text)}
                className="mb-3 mt-6 text-xl font-semibold"
                style={{ color: isDark ? '#e6edf3' : '#1f2328' }}
                {...props}
              >
                {children}
              </h3>
            )
          },
          h4: ({ children, ...props }) => {
            const text = String(children)
            return (
              <h4
                id={slugify(text)}
                className="mb-2 mt-4 text-lg font-medium"
                style={{ color: isDark ? '#e6edf3' : '#1f2328' }}
                {...props}
              >
                {children}
              </h4>
            )
          },
          p: ({ children, ...props }) => (
            <p
              className="mb-4 leading-relaxed"
              style={{ color: isDark ? '#c9d1d9' : '#24292f' }}
              {...props}
            >
              {children}
            </p>
          ),
          a: ({ children, href, ...props }) => (
            <a
              href={href}
              className="text-amber-500 underline underline-offset-2 hover:text-amber-400"
              target={href?.startsWith('http') ? '_blank' : undefined}
              rel={href?.startsWith('http') ? 'noopener noreferrer' : undefined}
              {...props}
            >
              {children}
            </a>
          ),
          ul: ({ children, ...props }) => (
            <ul
              className="mb-4 list-disc space-y-1 pl-6"
              style={{ color: isDark ? '#c9d1d9' : '#24292f' }}
              {...props}
            >
              {children}
            </ul>
          ),
          ol: ({ children, ...props }) => (
            <ol
              className="mb-4 list-decimal space-y-1 pl-6"
              style={{ color: isDark ? '#c9d1d9' : '#24292f' }}
              {...props}
            >
              {children}
            </ol>
          ),
          li: ({ children, ...props }) => (
            <li className="leading-relaxed" {...props}>
              {children}
            </li>
          ),
          blockquote: ({ children, ...props }) => (
            <blockquote
              className="mb-4 border-l-4 border-amber-500/50 py-1 pl-4 italic"
              style={{ color: isDark ? '#8b949e' : '#656d76' }}
              {...props}
            >
              {children}
            </blockquote>
          ),
          code: ({ className, children, ...props }) => {
            const isInline = !className
            if (isInline) {
              return (
                <code
                  className="rounded px-1.5 py-0.5 text-sm"
                  style={{
                    backgroundColor: isDark ? '#1c2128' : '#afb8c133',
                    color: isDark ? '#e6edf3' : '#24292f',
                    fontFamily: "'JetBrains Mono', monospace",
                  }}
                  {...props}
                >
                  {children}
                </code>
              )
            }
            return (
              <code
                className={className}
                style={{ fontFamily: "'JetBrains Mono', monospace" }}
                {...props}
              >
                {children}
              </code>
            )
          },
          pre: ({ children, ...props }) => (
            <pre
              className="mb-4 overflow-x-auto rounded-lg p-4 text-sm leading-relaxed"
              style={{
                backgroundColor: isDark ? '#161b22' : '#f6f8fa',
                border: isDark
                  ? '1px solid rgba(255,255,255,0.06)'
                  : '1px solid rgba(0,0,0,0.06)',
              }}
              {...props}
            >
              {children}
            </pre>
          ),
          table: ({ children, ...props }) => (
            <div className="mb-4 overflow-x-auto">
              <table
                className="w-full border-collapse text-sm"
                style={{
                  border: isDark
                    ? '1px solid rgba(255,255,255,0.1)'
                    : '1px solid rgba(0,0,0,0.1)',
                }}
                {...props}
              >
                {children}
              </table>
            </div>
          ),
          th: ({ children, ...props }) => (
            <th
              className="px-4 py-2 text-left font-semibold"
              style={{
                backgroundColor: isDark ? '#1c2128' : '#f0f2f5',
                color: isDark ? '#e6edf3' : '#1f2328',
                border: isDark
                  ? '1px solid rgba(255,255,255,0.1)'
                  : '1px solid rgba(0,0,0,0.1)',
              }}
              {...props}
            >
              {children}
            </th>
          ),
          td: ({ children, ...props }) => (
            <td
              className="px-4 py-2"
              style={{
                color: isDark ? '#c9d1d9' : '#24292f',
                border: isDark
                  ? '1px solid rgba(255,255,255,0.06)'
                  : '1px solid rgba(0,0,0,0.06)',
              }}
              {...props}
            >
              {children}
            </td>
          ),
          tr: ({ children, ...props }) => (
            <tr
              className="even:bg-opacity-50"
              style={{
                backgroundColor: isDark ? 'transparent' : 'transparent',
              }}
              {...props}
            >
              {children}
            </tr>
          ),
          hr: ({ ...props }) => (
            <hr
              className="my-8 border-0"
              style={{
                height: '1px',
                backgroundColor: isDark
                  ? 'rgba(255,255,255,0.06)'
                  : 'rgba(0,0,0,0.06)',
              }}
              {...props}
            />
          ),
          strong: ({ children, ...props }) => (
            <strong
              style={{ color: isDark ? '#e6edf3' : '#1f2328' }}
              {...props}
            >
              {children}
            </strong>
          ),
          img: ({ src, alt, ...props }) => {
            if (src && src.startsWith('http')) {
              return <img src={src} alt={alt || ''} className="mb-4 max-w-full rounded-lg" {...props} />
            }
            return null
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}
