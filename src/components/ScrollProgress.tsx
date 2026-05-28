import { useScrollPercentage } from '../hooks/useScrollPercentage'

export default function ScrollProgress() {
  const progress = useScrollPercentage()

  return (
    <div className="fixed top-16 left-0 right-0 z-40 h-[2px]">
      <div
        className="h-full bg-gradient-to-r from-amber-500 via-amber-400 to-emerald-500 transition-[width] duration-150 ease-out"
        style={{ width: `${progress}%` }}
      />
    </div>
  )
}
