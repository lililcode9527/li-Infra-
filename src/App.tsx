import { HashRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import HomePage from './pages/HomePage'
import ArticlePage from './pages/ArticlePage'
import AboutPage from './pages/AboutPage'
import CategoryPage from './pages/CategoryPage'

export default function App() {
  return (
    <HashRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<HomePage />} />
          <Route path="/article/:slug" element={<ArticlePage />} />
          <Route path="/category/:id" element={<CategoryPage />} />
          <Route path="/about" element={<AboutPage />} />
        </Route>
      </Routes>
    </HashRouter>
  )
}
