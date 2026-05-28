import { Moon, Sun, Cpu } from 'lucide-react'
import { Link, useLocation } from 'react-router-dom'
import { useThemeStore } from '../store/themeStore'

export default function Navbar() {
  const { theme, toggleTheme } = useThemeStore()
  const location = useLocation()

  const isDark = theme === 'dark'

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 h-16 border-b backdrop-blur-xl transition-colors duration-300 ${
        isDark
          ? 'border-white/[0.06] bg-[#0d1117]/80'
          : 'border-black/[0.06] bg-white/80'
      }`}
    >
      <div className="mx-auto flex h-full max-w-7xl items-center justify-between px-6">
        <Link
          to="/"
          className="flex items-center gap-2.5 text-lg font-bold tracking-tight transition-opacity hover:opacity-80"
          style={{ color: isDark ? '#e6edf3' : '#1f2328' }}
        >
          <Cpu size={22} className="text-amber-500" />
          <span>AI Infra</span>
        </Link>

        <div className="flex items-center gap-5">
          <Link
            to="/about"
            className={`text-sm transition-colors ${
              location.pathname === '/about'
                ? 'text-amber-500'
                : isDark
                  ? 'text-[#8b949e] hover:text-[#e6edf3]'
                  : 'text-[#656d76] hover:text-[#1f2328]'
            }`}
          >
            关于
          </Link>
          <button
            onClick={toggleTheme}
            className={`rounded-lg p-2 transition-colors ${
              isDark
                ? 'hover:bg-white/[0.06] text-[#8b949e] hover:text-[#e6edf3]'
                : 'hover:bg-black/[0.04] text-[#656d76] hover:text-[#1f2328]'
            }`}
            aria-label="切换主题"
          >
            {isDark ? <Sun size={18} /> : <Moon size={18} />}
          </button>
        </div>
      </div>
    </nav>
  )
}
