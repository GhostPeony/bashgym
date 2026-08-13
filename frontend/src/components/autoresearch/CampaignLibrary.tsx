import { useState } from 'react'
import { Archive, ArchiveRestore } from 'lucide-react'

import type { CampaignRecord } from '../../stores/campaignStore'
import { Button } from '../common/Button'
import { Modal } from '../common/Modal'
import { campaignStatusTone, type PresentationTone } from './controlRoomModel'

const toneClasses: Record<PresentationTone, string> = {
  success: 'border-status-success/60 bg-status-success/10 text-status-success',
  warning: 'border-status-warning/60 bg-status-warning/10 text-status-warning',
  error: 'border-status-error/60 bg-status-error/10 text-status-error',
  info: 'border-accent/60 bg-accent/10 text-accent-dark',
  neutral: 'border-border-subtle bg-background-secondary text-text-secondary'
}

function readable(value: string): string {
  const words = value.replace(/[_:-]+/g, ' ').trim()
  return words ? words.replace(/\b\w/g, (letter) => letter.toUpperCase()) : 'Unknown'
}

export interface CampaignLibraryProps {
  campaigns: CampaignRecord[]
  onUnarchive: (campaignId: string) => void
  onOpen: (campaignId: string) => void
}

/**
 * Archived-campaign library: a compact command-bar button that opens a modal.
 * Entries are hidden from the campaign selector and canvas auto-materialization;
 * the durable campaign records and their evidence remain untouched and reappear
 * on unarchive.
 */
export function CampaignLibrary({ campaigns, onUnarchive, onOpen }: CampaignLibraryProps) {
  const [open, setOpen] = useState(false)
  if (campaigns.length === 0) return null
  const ordered = [...campaigns].sort(
    (left, right) =>
      right.updated_at.localeCompare(left.updated_at) ||
      left.campaign_id.localeCompare(right.campaign_id)
  )
  return (
    <>
      <Button
        type="button"
        size="sm"
        variant="ghost"
        onClick={() => setOpen(true)}
        leftIcon={<Archive className="h-3.5 w-3.5" aria-hidden="true" />}
        title="Archived campaigns. Durable records and evidence are kept; unarchive any time."
      >
        Library ({campaigns.length})
      </Button>
      <Modal
        isOpen={open}
        onClose={() => setOpen(false)}
        title="Campaign library"
        description="Archived campaigns are hidden from the selector and canvas. Their durable records and evidence are untouched."
        size="lg"
      >
        <ul className="divide-y divide-border-subtle">
          {ordered.map((campaign) => (
            <li
              key={campaign.campaign_id}
              className="flex flex-wrap items-center gap-x-3 gap-y-2 py-3 first:pt-0 last:pb-0"
            >
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm text-text-primary" title={campaign.title}>
                  {campaign.title}
                </p>
                <p
                  className="truncate font-mono text-[10px] text-text-muted"
                  title={campaign.campaign_id}
                >
                  {campaign.campaign_id} · updated {campaign.updated_at.slice(0, 10)}
                </p>
              </div>
              <span
                className={`inline-flex shrink-0 items-center rounded-brutal border px-2 py-0.5 font-mono text-[10px] uppercase tracking-wide ${toneClasses[campaignStatusTone(campaign.status)]}`}
              >
                {readable(campaign.status)}
              </span>
              <Button
                type="button"
                size="sm"
                variant="ghost"
                onClick={() => {
                  onOpen(campaign.campaign_id)
                  setOpen(false)
                }}
                title="Select this campaign without unarchiving it"
              >
                Open
              </Button>
              <Button
                type="button"
                size="sm"
                variant="secondary"
                onClick={() => onUnarchive(campaign.campaign_id)}
                leftIcon={<ArchiveRestore className="h-3.5 w-3.5" aria-hidden="true" />}
                title="Return this campaign to the selector and canvas"
              >
                Unarchive
              </Button>
            </li>
          ))}
        </ul>
      </Modal>
    </>
  )
}
