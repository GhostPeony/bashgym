import type { CampaignDetailState } from '../../stores/campaignStore'
import type { CampaignOutcomeViewModel } from '../autoresearch/campaignOutcomeModel'

export function ExperimentJournal({
  detail,
  outcome
}: {
  detail?: CampaignDetailState
  outcome?: CampaignOutcomeViewModel | null
}) {
  const proposals = detail?.proposals || []
  const latest = [...proposals].sort((a, b) => b.updated_at.localeCompare(a.updated_at))[0]
  const study = detail?.studies.find((item) => item.proposal_id === latest?.proposal.proposal_id)
  return (
    <div className="studio-journal" aria-label="Experiment journal">
      <section className="studio-paper studio-hypothesis">
        <div className="studio-section-heading">
          <span className="studio-step">01</span>
          <h2>Hypothesis</h2>
          <span className="studio-tag">
            {latest
              ? latest.validation.valid
                ? 'Validated proposal'
                : 'Needs revision'
              : 'Awaiting a proposal'}
          </span>
        </div>
        <p className="studio-hypothesis-text">
          {latest?.proposal.hypothesis ||
            'What is one change that could improve the starting model?'}
        </p>
        <p>
          {latest
            ? latest.proposal.expected_outcome
            : 'Your agent’s recorded hypothesis will appear here, together with its expected outcome and a way to prove it wrong.'}
        </p>
        {latest && (
          <dl className="studio-facts">
            <dt>Falsification criterion</dt>
            <dd>{latest.proposal.falsification_criterion}</dd>
            <dt>Validation</dt>
            <dd>
              {latest.validation.valid
                ? 'Accepted by the campaign validator'
                : latest.validation.reason_codes.join(', ') || 'Not accepted'}
            </dd>
          </dl>
        )}
      </section>
      <section className="studio-paper">
        <div className="studio-section-heading">
          <span className="studio-step">02</span>
          <h2>The change</h2>
        </div>
        <p>{latest?.proposal.primary_variable || 'No proposed change has been recorded.'}</p>
        {study ? (
          <ol className="studio-stage-list" aria-label="Recorded stage plan">
            {study.stage_plan.items.map((stage, index) => (
              <li
                key={`${stage.stage}-${index}`}
                aria-current={index === study.current_stage_index ? 'step' : undefined}
              >
                <span>{stage.stage.replaceAll('_', ' ')}</span>
                <small>
                  {stage.disposition} · {stage.reason}
                </small>
              </li>
            ))}
          </ol>
        ) : (
          <p className="studio-caption">
            The validated study plan will show which stages run and which are skipped.
          </p>
        )}
      </section>
      <section className="studio-paper">
        <div className="studio-section-heading">
          <span className="studio-step">03</span>
          <h2>Run history</h2>
          <span className="studio-tag">{detail?.attempts.length || 0} recorded</span>
        </div>
        {detail?.error && (
          <p role="alert" className="studio-notice">
            Run history could not be refreshed. {detail.error}
          </p>
        )}
        {detail?.attempts.length ? (
          <div className="studio-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Attempt</th>
                  <th>Stage</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {[...detail.attempts]
                  .sort((a, b) => b.updated_at.localeCompare(a.updated_at))
                  .map((attempt) => (
                    <tr key={attempt.attempt_id}>
                      <td>{attempt.attempt_number}</td>
                      <td>{attempt.stage.replaceAll('_', ' ')}</td>
                      <td>
                        <span
                          className={`studio-tag ${attempt.status === 'failed' ? 'studio-tag-error' : ''}`}
                        >
                          {attempt.status.replaceAll('_', ' ')}
                        </span>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p>No attempts are recorded yet. A prepared campaign waits for an explicit Start.</p>
        )}
      </section>
      <section className="studio-paper">
        <div className="studio-section-heading">
          <span className="studio-step">04</span>
          <h2>Comparison</h2>
        </div>
        {outcome?.metrics.length ? (
          <div className="studio-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Baseline</th>
                  <th>Candidate</th>
                  <th>Change</th>
                </tr>
              </thead>
              <tbody>
                {outcome.metrics.map((metric) => (
                  <tr key={metric.id}>
                    <td>{metric.id}</td>
                    <td>{metric.baseline ?? 'Unavailable'}</td>
                    <td>{metric.candidate ?? 'Unavailable'}</td>
                    <td>{metric.delta ?? 'Unavailable'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p>Comparable evaluation evidence has not been recorded yet.</p>
        )}
        <p className="studio-caption">
          {outcome?.sameEvaluationSuite === true
            ? 'Baseline and candidate use the same evaluation suite.'
            : 'Compare only verified results from the same evaluation suite.'}
        </p>
      </section>
      <section className="studio-paper studio-decision">
        <div className="studio-section-heading">
          <span className="studio-step">05</span>
          <h2>Decision</h2>
        </div>
        <h3>{outcome?.decision || 'Keep, discard, or investigate further.'}</h3>
        <p>
          {outcome?.lifecycleReason ||
            'The recorded decision and its evidence will close this experiment. Human review remains below when a decision needs you.'}
        </p>
      </section>
    </div>
  )
}
