import { UserRound } from 'lucide-react'
import { useAgentSessionsStore } from '../../stores/agentSessionsStore'
import { MaskedValue } from '../common'

/**
 * Opt-in account info from the local agent-CLI config. Nothing is read from
 * disk until the toggle is enabled; values render masked by default.
 */
export function AccountChip() {
  const { accountOptIn, account, setAccountOptIn } = useAgentSessionsStore()

  return (
    <div className="border-t border-brutal border-border-subtle pt-3 space-y-2">
      <label className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-wider text-text-muted cursor-pointer">
        <input
          type="checkbox"
          checked={accountOptIn}
          onChange={(e) => setAccountOptIn(e.target.checked)}
          className="accent-[var(--accent)]"
        />
        <UserRound className="w-3 h-3" />
        Show account info
      </label>
      {accountOptIn && account && (
        <div className="font-mono text-[10px] text-text-secondary space-y-1 pl-5">
          {account.emailAddress && (
            <div className="flex items-center gap-1.5 min-w-0">
              <span className="text-text-muted flex-shrink-0">account</span>
              <MaskedValue value={account.emailAddress} />
            </div>
          )}
          {account.organizationName && (
            <div className="flex items-center gap-1.5 min-w-0">
              <span className="text-text-muted flex-shrink-0">org</span>
              <MaskedValue value={account.organizationName} />
            </div>
          )}
          <div className="flex items-center gap-1.5 flex-wrap">
            {[account.billingType, account.seatTier, account.userRateLimitTier]
              .filter(Boolean)
              .map((tag) => (
                <span
                  key={String(tag)}
                  className="px-1 border-brutal border-border-subtle rounded-brutal"
                >
                  {String(tag)}
                </span>
              ))}
          </div>
        </div>
      )}
      {accountOptIn && !account && (
        <p className="font-mono text-[10px] text-text-muted pl-5">no account info found</p>
      )}
    </div>
  )
}
