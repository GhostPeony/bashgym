import type { CSSProperties } from 'react'
import {
  ghostPeonyIconPath,
  type GhostPeonyIconName,
  type GhostPeonyIconTone
} from './ghostPeonyIconAssets'

interface GhostPeonyIconProps {
  name: GhostPeonyIconName
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl' | '2xl' | '3xl'
  tone?: GhostPeonyIconTone
  hue?: number
  active?: boolean
  muted?: boolean
  className?: string
  title?: string
}

const SIZE_CLASS: Record<NonNullable<GhostPeonyIconProps['size']>, string> = {
  xs: 'w-4 h-4',
  sm: 'w-5 h-5',
  md: 'w-6 h-6',
  lg: 'w-8 h-8',
  xl: 'w-10 h-10',
  '2xl': 'w-14 h-14',
  '3xl': 'w-16 h-16'
}

export function GhostPeonyIcon({
  name,
  size = 'md',
  tone = 'neutral',
  hue,
  active = false,
  muted = false,
  className,
  title
}: GhostPeonyIconProps) {
  const style = hue == null ? undefined : ({ '--ghost-peony-icon-hue': hue } as CSSProperties)

  return (
    <span
      className={['ghost-peony-icon', SIZE_CLASS[size], className].filter(Boolean).join(' ')}
      data-tone={tone}
      data-active={active ? 'true' : 'false'}
      data-muted={muted ? 'true' : 'false'}
      style={style}
      title={title}
    >
      <img src={ghostPeonyIconPath(name, tone)} alt="" draggable={false} />
    </span>
  )
}
