import { Outlet } from 'react-router-dom'
import { useThemeStore } from '../store/themeStore'
import Navbar from './Navbar'
import Footer from './Footer'

export default function Layout() {
  const { theme } = useThemeStore()
  const isDark = theme === 'dark'

  return (
    <div
      className={`min-h-screen transition-colors duration-300 ${
        isDark ? 'bg-[#0d1117]' : 'bg-[#ffffff]'
      }`}
    >
      <Navbar />
      <main className="pt-16">
        <Outlet />
      </main>
      <Footer />
    </div>
  )
}
