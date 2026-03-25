import { useState, useEffect, useCallback } from 'react'
import {
  Zap,
  Loader2,
  CheckCircle,
  Cpu,
  Code,
  Flame,
} from 'lucide-react'
import { routerApi, providersApi, OllamaModel } from '../../services/api'
import { clsx } from 'clsx'

interface StudentModelPickerProps {
  ollamaModels: OllamaModel[]
  ollamaAvailable: boolean
  onRefresh: () => void
}

export function StudentModelPicker({
  ollamaModels,
  ollamaAvailable,
  onRefresh: _onRefresh,
}: StudentModelPickerProps) {
  const [currentStudent, setCurrentStudent] = useState<string | null>(null)
  const [isSettingModel, setIsSettingModel] = useState<string | null>(null)
  const [isWarmingUp, setIsWarmingUp] = useState<string | null>(null)

  const fetchConfig = useCallback(async () => {
    const result = await routerApi.getConfig()
    if (result.ok && result.data?.student_model) {
      setCurrentStudent(result.data.student_model.name)
    }
  }, [])

  useEffect(() => {
    fetchConfig()
  }, [fetchConfig])

  const handleSetStudent = async (modelName: string) => {
    setIsSettingModel(modelName)
    try {
      const result = await routerApi.setStudentProvider('ollama', modelName)
      if (result.ok) {
        setCurrentStudent(modelName)
      }
    } finally {
      setIsSettingModel(null)
    }
  }

  const handleWarmUp = async (modelName: string) => {
    setIsWarmingUp(modelName)
    try {
      await providersApi.warmupOllamaModel(modelName)
    } finally {
      setIsWarmingUp(null)
    }
  }

  if (!ollamaAvailable || ollamaModels.length === 0) return null

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h4 className="font-brand text-sm text-text-primary flex items-center gap-1.5">
          <Zap className="w-3.5 h-3.5 text-accent" />
          Student Model (Inference)
        </h4>
      </div>

      <p className="text-[11px] text-text-muted font-mono">
        Select which local model handles Student inference.
      </p>

      <div className="space-y-1.5">
        {ollamaModels.map(model => {
          const isActive = currentStudent === model.name
          const isSetting = isSettingModel === model.name
          const isWarming = isWarmingUp === model.name

          return (
            <div
              key={model.name}
              className={clsx(
                'flex items-center gap-2 p-2 border-brutal rounded-brutal transition-press',
                isActive
                  ? 'border-accent bg-accent-light/30 shadow-brutal-sm'
                  : 'border-border bg-background-card hover:shadow-brutal-sm'
              )}
            >
              <div className={clsx(
                'w-7 h-7 border-brutal border-border rounded-brutal flex items-center justify-center flex-shrink-0',
                model.is_code_model ? 'bg-accent-light text-accent-dark' : 'bg-background-secondary text-text-muted'
              )}>
                {model.is_code_model ? <Code className="w-3.5 h-3.5" /> : <Cpu className="w-3.5 h-3.5" />}
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5">
                  <span className="text-xs font-mono font-semibold text-text-primary truncate">
                    {model.name}
                  </span>
                  {isActive && (
                    <span className="tag text-[9px] py-0 px-1"><span>active</span></span>
                  )}
                </div>
                <div className="text-[10px] text-text-muted font-mono">
                  {model.size_gb.toFixed(1)}GB · {model.parameter_size} · {model.quantization}
                </div>
              </div>

              <div className="flex items-center gap-1 flex-shrink-0">
                <button
                  onClick={() => handleWarmUp(model.name)}
                  disabled={isWarming}
                  className="btn-icon w-7 h-7 text-text-muted hover:text-accent"
                  title="Pre-load into VRAM"
                >
                  {isWarming ? (
                    <Loader2 className="w-3 h-3 animate-spin" />
                  ) : (
                    <Flame className="w-3 h-3" />
                  )}
                </button>

                {!isActive && (
                  <button
                    onClick={() => handleSetStudent(model.name)}
                    disabled={!!isSettingModel}
                    className="btn-primary text-[10px] py-1 px-2 font-mono"
                  >
                    {isSetting ? (
                      <Loader2 className="w-3 h-3 animate-spin" />
                    ) : (
                      'Use'
                    )}
                  </button>
                )}

                {isActive && (
                  <CheckCircle className="w-4 h-4 text-accent" />
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
