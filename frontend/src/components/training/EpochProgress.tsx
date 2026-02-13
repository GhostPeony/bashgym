interface EpochProgressProps {
  currentEpoch: number
  totalEpochs: number
  currentStep: number
  totalSteps: number
}

export function EpochProgress({
  currentEpoch,
  totalEpochs,
  currentStep,
  totalSteps
}: EpochProgressProps) {
  // Guard against division by zero
  const stepProgress = totalSteps > 0 ? (currentStep / totalSteps) * 100 : 0
  const isWaiting = totalEpochs === 0 && totalSteps === 0

  return (
    <div className="flex flex-col items-center">
      {/* Epoch Display */}
      <div className="card w-full p-6 flex flex-col items-center mb-4">
        {isWaiting ? (
          <>
            <span className="font-brand text-3xl text-text-muted">&mdash;</span>
            <span className="font-mono text-xs uppercase tracking-widest text-text-muted mt-1">Waiting</span>
          </>
        ) : (
          <>
            <span className="font-brand text-3xl text-text-primary">
              {currentEpoch}/{totalEpochs}
            </span>
            <span className="font-mono text-xs uppercase tracking-widest text-text-secondary mt-1">Epochs</span>
          </>
        )}

        {/* Epoch dots */}
        {!isWaiting && totalEpochs > 0 && (
          <div className="flex items-center gap-2 mt-3">
            {Array.from({ length: totalEpochs }, (_, i) => (
              <div
                key={i}
                className={`status-dot ${i < currentEpoch ? 'status-success' : ''}`}
              />
            ))}
          </div>
        )}
      </div>

      {/* Step Progress */}
      <div className="w-full">
        <div className="flex items-center justify-between mb-2">
          <span className="font-mono text-xs uppercase tracking-widest text-text-muted">Steps</span>
          <span className="font-mono text-xs text-text-primary">
            {currentStep.toLocaleString()} / {totalSteps.toLocaleString()}
          </span>
        </div>
        <div className="progress-bar">
          <div
            className="progress-fill"
            style={{ width: `${stepProgress}%` }}
          />
        </div>
        <div className="flex items-center justify-end mt-1">
          <span className="font-mono text-xs text-text-muted">{stepProgress.toFixed(1)}%</span>
        </div>
      </div>
    </div>
  )
}
