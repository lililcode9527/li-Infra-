import { useEffect, useRef } from 'react'
import { useThemeStore } from '../store/themeStore'
import { useScrollSpy } from '../hooks/useScrollSpy'
import type { TocItem } from '../types'

interface SidebarProps {
  items: TocItem[]
}

export default function Sidebar({ items }: SidebarProps) {
  const { theme } = useThemeStore()
  const isDark = theme === 'dark'
  const ids = items.map((item) => item.id)
  const activeId = useScrollSpy(ids, 100)
  const sidebarRef = useRef<HTMLElement>(null)

  useEffect(() => {
    if (!activeId || !sidebarRef.current) return
    const el = sidebarRef.current.querySelector(`[data-id="${activeId}"]`)
    if (el) {
      el.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
    }
  }, [activeId])

  const handleClick = (id: string) => {
    const el = document.getElementById(id)
    if (el) {
      const top = el.getBoundingClientRect().top + window.scrollY - 90
      window.scrollTo({ top, behavior: 'smooth' })
    }
  }

  if (items.length === 0) return null

  return (
    <aside
      ref={sidebarRef}
      className="fixed top-24 hidden w-56 overflow-y-auto pr-4 xl:block"
      style={{ maxHeight: 'calc(100vh - 120px)' }}
    >
      <h4
        className="mb-3 text-xs font-semibold uppercase tracking-wider"
        style={{ color: isDark ? '#484f58' : '#8b949e' }}
      >
        目录
      </h4>
      <nav className="space-y-0.5">
        {items.map((item) => (
          <button
            key={item.id}
            data-id={item.id}
            onClick={() => handleClick(item.id)}
            className={`block w-full truncate py-1 text-left text-sm transition-colors ${
              activeId === item.id
                ? 'font-medium text-amber-500'
                : isDark
                  ? 'text-[#8b949e] hover:text-[#e6edf3]'
                  : 'text-[#656d76] hover:text-[#1f2328]'
            }`}
            style={{ paddingLeft: `${(item.level - 1) * 12}px` }}
          >
            {item.text}
          </button>
        ))}
      </nav>
    </aside>
  )
}
