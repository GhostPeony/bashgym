import { GhostPeonyIcon } from '../common/GhostPeonyIcon'
import { nodeFlowerIconName, type NodeFlowerVariant } from './nodeFlowerAssets'

export type { NodeFlowerVariant } from './nodeFlowerAssets'

interface NodeFlowerMarkProps {
  variant: NodeFlowerVariant
  active?: boolean
  muted?: boolean
  hue?: number
  tone?: 'neutral' | 'color' | 'accent' | 'node'
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl'
  className?: string
  title?: string
}

export function NodeFlowerMark({
  variant,
  active = false,
  muted = false,
  hue,
  tone = 'node',
  size = 'md',
  className,
  title
}: NodeFlowerMarkProps) {
  return (
    <GhostPeonyIcon
      name={nodeFlowerIconName(variant)}
      size={size}
      tone={tone}
      hue={hue}
      active={active}
      muted={muted}
      className={['node-flower-mark', className].filter(Boolean).join(' ')}
      title={title}
    />
  )
}
