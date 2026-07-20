import { useEffect, useMemo, useState } from 'react'
import { ChevronRight, FileSearch, Loader2, ShieldCheck } from 'lucide-react'

import type {
  CampaignArtifact,
  CampaignArtifactPreview,
  CampaignEventItem,
} from '../../campaignVisibility'
import { campaignApi } from '../../services/api'
import {
  describeCampaignArtifact,
  describeCampaignEvent,
} from '../../utils/campaignMeaning'
import { Modal } from '../common/Modal'

export type CampaignEvidenceSelection =
  | { kind: 'artifact'; artifact: CampaignArtifact }
  | { kind: 'event'; item: CampaignEventItem }

interface CampaignEvidenceRowProps {
  selection: CampaignEvidenceSelection
  onInspect: (selection: CampaignEvidenceSelection) => void
}

function readable(value: string): string {
  return value.replace(/^campaign:/, '').replaceAll('_', ' ').replaceAll(':', ' · ')
}

function selectionPresentation(selection: CampaignEvidenceSelection) {
  return selection.kind === 'artifact'
    ? describeCampaignArtifact(selection.artifact)
    : describeCampaignEvent(selection.item.event)
}

export function CampaignEvidenceRow({ selection, onInspect }: CampaignEvidenceRowProps) {
  const presentation = selectionPresentation(selection)
  const timestamp = selection.kind === 'artifact'
    ? selection.artifact.created_at
    : selection.item.event.created_at
  return (
    <button
      type="button"
      onClick={() => onInspect(selection)}
      aria-label={`Inspect ${presentation.summary}`}
      className="group flex w-full items-start gap-3 border-2 border-border-subtle bg-background-card px-3 py-2.5 text-left transition-colors hover:border-accent hover:bg-background-secondary focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
    >
      <FileSearch className="mt-0.5 h-4 w-4 shrink-0 text-accent" aria-hidden="true" />
      <span className="min-w-0 flex-1">
        <span className="block text-xs font-semibold leading-5 text-text-primary">{presentation.summary}</span>
        <span className="mt-0.5 block text-[11px] leading-4 text-text-secondary">{presentation.detail}</span>
        <time className="mt-1 block font-mono text-[11px] text-text-muted" dateTime={timestamp}>
          {new Date(timestamp).toLocaleString()}
        </time>
      </span>
      <span className="mt-0.5 flex shrink-0 items-center gap-1 font-mono text-[11px] font-bold uppercase tracking-wide text-accent">
        Inspect <ChevronRight className="h-3.5 w-3.5" aria-hidden="true" />
      </span>
    </button>
  )
}

function DetailRow({ label, value }: { label: string; value: string | number | null | undefined }) {
  return (
    <div className="grid gap-1 border-t border-border-subtle py-2 first:border-t-0 sm:grid-cols-[10rem_minmax(0,1fr)]">
      <dt className="font-mono text-[11px] font-bold uppercase tracking-wide text-text-muted">{label}</dt>
      <dd className="break-words font-mono text-[11px] leading-5 text-text-primary">{value ?? '—'}</dd>
    </div>
  )
}

function formattedPreview(preview: CampaignArtifactPreview): string {
  if (!preview.content) return ''
  if (preview.preview_kind === 'json') {
    try {
      return JSON.stringify(JSON.parse(preview.content), null, 2)
    } catch {
      return preview.content
    }
  }
  if (preview.preview_kind === 'jsonl') {
    return preview.content.split('\n').map((line) => {
      try {
        return JSON.stringify(JSON.parse(line), null, 2)
      } catch {
        return line
      }
    }).join('\n')
  }
  return preview.content
}

interface CampaignEvidenceDetailProps {
  selection: CampaignEvidenceSelection
  preview?: CampaignArtifactPreview | null
  loading: boolean
  error: string | null
}

export function CampaignEvidenceDetail({
  selection,
  preview,
  loading,
  error,
}: CampaignEvidenceDetailProps) {
  const presentation = selectionPresentation(selection)
  if (selection.kind === 'event') {
    const { item } = selection
    const { event } = item
    const summary = event.summary
    return (
      <article className="space-y-4 text-xs" aria-label="Campaign event detail">
        <header className="border-l-4 border-accent pl-4">
          <p className="font-mono text-[11px] font-bold uppercase tracking-widest text-accent">Event detail</p>
          <h3 className="mt-1 font-serif text-xl font-semibold text-text-primary">{presentation.summary}</h3>
          <p className="mt-1 text-sm leading-6 text-text-secondary">{presentation.detail}</p>
        </header>
        <dl className="border-2 border-border bg-background-secondary px-3">
          <DetailRow label="Event ID" value={event.event_id} />
          <DetailRow label="Event type" value={event.event_type} />
          <DetailRow label="Recorded" value={new Date(event.created_at).toLocaleString()} />
          <DetailRow label="Actor" value={`${event.actor_id} · ${readable(event.credential_kind)}`} />
          <DetailRow label="Cursor" value={item.cursor} />
          <DetailRow label="Sequence" value={event.sequence} />
          <DetailRow label="Aggregate version" value={event.aggregate_version} />
        </dl>
        {summary ? (
          <section className="border-2 border-border bg-background-card p-3">
            <h4 className="font-mono text-[11px] font-bold uppercase tracking-widest text-text-primary">Published references</h4>
            <dl className="mt-2">
              {Object.entries(summary).filter(([key]) => key !== 'schema_version').map(([key, value]) => (
                <DetailRow key={key} label={readable(key)} value={String(value)} />
              ))}
            </dl>
          </section>
        ) : null}
      </article>
    )
  }

  const { artifact } = selection
  return (
    <article className="space-y-4 text-xs" aria-label="Campaign artifact detail">
      <header className="border-l-4 border-accent pl-4">
        <p className="font-mono text-[11px] font-bold uppercase tracking-widest text-accent">Sealed evidence</p>
        <h3 className="mt-1 font-serif text-xl font-semibold text-text-primary">{presentation.summary}</h3>
        <p className="mt-1 text-sm leading-6 text-text-secondary">{presentation.detail}</p>
      </header>
      <dl className="border-2 border-border bg-background-secondary px-3">
        <DetailRow label="Artifact ID" value={artifact.artifact_id} />
        <DetailRow label="Schema" value={artifact.schema_name} />
        <DetailRow label="Producer action" value={artifact.producer_action_id} />
        <DetailRow label="Recorded" value={new Date(artifact.created_at).toLocaleString()} />
        <DetailRow label="Size" value={`${artifact.size_bytes.toLocaleString()} bytes`} />
        <DetailRow label="SHA-256" value={artifact.sha256} />
        <DetailRow label="Integrity" value={artifact.sealed && artifact.valid ? 'Sealed and valid' : 'Integrity unavailable'} />
      </dl>
      <section className="border-2 border-border bg-background-card" aria-labelledby="artifact-preview-title">
        <div className="flex items-center justify-between gap-3 border-b-2 border-border px-3 py-2">
          <div>
            <h4 id="artifact-preview-title" className="font-mono text-[11px] font-bold uppercase tracking-widest text-text-primary">Content preview</h4>
            {preview?.integrity_verified ? (
              <p className="mt-0.5 flex items-center gap-1 text-[11px] text-status-success"><ShieldCheck className="h-3.5 w-3.5" />Hash verified before reading</p>
            ) : null}
          </div>
          {preview?.truncated ? <span className="font-mono text-[11px] text-status-warning">Preview truncated · Latest 500 lines shown</span> : null}
        </div>
        <div className="p-3">
          {loading ? (
            <p className="flex items-center gap-2 text-xs text-text-secondary"><Loader2 className="h-4 w-4 animate-spin" />Loading verified preview…</p>
          ) : error ? (
            <p className="text-xs text-status-error" role="alert">{error}</p>
          ) : preview?.preview_kind === 'unavailable' ? (
            <p className="text-xs leading-5 text-text-secondary">{preview.unavailable_reason}</p>
          ) : preview ? (
            <>
              <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words bg-background-secondary p-3 font-mono text-[11px] leading-5 text-text-primary">{formattedPreview(preview)}</pre>
              {preview.redaction_count > 0 ? (
                <p className="mt-2 font-mono text-[11px] text-status-warning">{preview.redaction_count} sensitive values redacted</p>
              ) : null}
            </>
          ) : (
            <p className="text-xs text-text-muted">No preview has been requested.</p>
          )}
        </div>
      </section>
    </article>
  )
}

export function CampaignEvidenceInspector({ selection }: { selection: CampaignEvidenceSelection }) {
  const [preview, setPreview] = useState<CampaignArtifactPreview | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let current = true
    setPreview(null)
    setError(null)
    if (selection.kind !== 'artifact') {
      setLoading(false)
      return () => { current = false }
    }
    setLoading(true)
    void campaignApi.artifactPreview(
      selection.artifact.workspace_id,
      selection.artifact.campaign_id,
      selection.artifact.artifact_id,
    ).then((response) => {
      if (!current) return
      if (response.ok && response.data) setPreview(response.data)
      else setError(response.error || 'Unable to load the verified artifact preview.')
    }).finally(() => {
      if (current) setLoading(false)
    })
    return () => { current = false }
  }, [selection])

  return <CampaignEvidenceDetail selection={selection} preview={preview} loading={loading} error={error} />
}

export function CampaignEvidenceDialog({
  selection,
  onClose,
}: {
  selection: CampaignEvidenceSelection | null
  onClose: () => void
}) {
  const title = useMemo(
    () => selection ? selectionPresentation(selection).summary : 'Campaign evidence',
    [selection],
  )
  return (
    <Modal
      isOpen={selection !== null}
      onClose={onClose}
      title={title}
      description="Inspect the public campaign record and its verified evidence projection."
      size="xl"
    >
      {selection ? <CampaignEvidenceInspector selection={selection} /> : null}
    </Modal>
  )
}
