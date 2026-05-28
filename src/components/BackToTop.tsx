import { useState, useEffect } from 'react'
import { ArrowUp } from 'lucide-react'

export default function BackToTop() {
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      setVisible(window.scrollY > 400)
    }
    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return (
    <button
      onClick={scrollToTop}
      className={`fixed bottom-8 right-8 z-40 rounded-full border border-amber-500/30 bg-[#0d1117]/90 p-3 text-amber-500 shadow-lg backdrop-blur-sm transition-all duration-300 hover:border-amber-500 hover:bg-amber-500/10 hover:shadow-amber-500/20 ${
        visible ? 'translate-y-0 opacity-100' : 'translate-y-4 opacity-0 pointer-events-none'
      }`}
      aria-label="返回顶部"
    >
      <ArrowUp size={20} />
    </button>
  )
}
