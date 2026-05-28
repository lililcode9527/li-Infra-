import { useEffect, useState } from 'react'

export function useScrollSpy(ids: string[], offset = 80) {
  const [activeId, setActiveId] = useState<string>('')

  useEffect(() => {
    if (ids.length === 0) return

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting)
        if (visible.length > 0) {
          setActiveId(visible[0].target.id)
        }
      },
      { rootMargin: `-${offset}px 0px -80% 0px`, threshold: 0 },
    )

    ids.forEach((id) => {
      const el = document.getElementById(id)
      if (el) observer.observe(el)
    })

    return () => observer.disconnect()
  }, [ids, offset])

  return activeId
}
