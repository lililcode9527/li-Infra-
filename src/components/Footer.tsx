import { useThemeStore } from '../store/themeStore'

export default function Footer() {
  const { theme } = useThemeStore()
  const isDark = theme === 'dark'

  return (
    <footer
      className={`border-t py-8 transition-colors duration-300 ${
        isDark ? 'border-white/[0.06]' : 'border-black/[0.06]'
      }`}
    >
      <div className="mx-auto max-w-7xl px-6 text-center">
        <p
          className="text-sm"
          style={{ color: isDark ? '#484f58' : '#8b949e' }}
        >
          AI Infra · 大模型基础设施深度技术洞察
        </p>
      </div>
    </footer>
  )
}
