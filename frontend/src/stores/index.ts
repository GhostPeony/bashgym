export { useThemeStore } from './themeStore'
export { useTerminalStore } from './terminalStore'
export type {
  TerminalSession,
  Panel,
  PanelType,
  AttentionState,
  ViewMode as TerminalViewMode,
  AgentStatus,
  CanvasNode,
  CanvasEdge,
  MonitorAutoMode,
  ToolHistoryItem,
  SessionMetrics
} from './terminalStore'
export { useWorkspaceStore } from './workspaceStore'
export type { WorkspaceMeta } from './workspacePersistence'
export { buildWorkspaceSessionIndex } from './workspaceSessionIndex'
export type {
  WorkspaceSessionGroup,
  WorkspaceSessionRecord,
  SessionRuntimeState
} from './workspaceSessionIndex'
export { useCanvasControlStore } from './canvasControlStore'
export type { CanvasControlState } from './canvasControlStore'
export { useFileStore, initializeFileStore } from './fileStore'
export type { FileNode } from './fileStore'
export { useTrainingStore } from './trainingStore'
export { useCampaignStore } from './campaignStore'
export type {
  CampaignStatus,
  CampaignRecord,
  CampaignAttempt,
  CampaignStudy,
  CampaignEvidence,
  CampaignArtifact,
  CampaignComparison,
  CampaignEventItem,
  CampaignMetricValue,
  CampaignDetailState
} from './campaignStore'
export { useCanvasOrchestratorStore } from './canvasOrchestratorStore'
export { useRuntimeStore } from './runtimeStore'
export type { ObservedRuntimeJob } from './runtimeStore'
export { useSkillLabStore } from './skillLabStore'
export { useHFContextStore } from './hfContextStore'
export type {
  TrainingStrategy,
  TrainingProfile,
  TrainingStatus,
  TrainingMetrics,
  TrainingConfig,
  TrainingRun,
  TrainingLog,
  DataSource,
  GrpoMetric
} from './trainingStore'
export { useRouterStore } from './routerStore'
export type { RoutingStrategy, RoutingStats, RoutingDecision } from './routerStore'
export { useTracesStore } from './tracesStore'
export type {
  TraceStatus,
  TraceQualityTier,
  TraceStep,
  QualityMetrics,
  Trace,
  RepoInfo
} from './tracesStore'
export { useUIStore } from './uiStore'
export type { ViewMode, PanelPresentationRequest } from './uiStore'
export { useTutorialStore } from './tutorialStore'
export type { TutorialStep } from './tutorialStore'
export { useAchievementStore } from './achievementStore'
export {
  useAccentStore,
  ACCENT_PRESETS,
  TERMINAL_FG_PRESETS,
  DEFAULT_TERMINAL_FG_HUE,
  BLACK_TERMINAL_FG_HUE,
  WHITE_TERMINAL_FG_HUE,
  isTerminalFgHue,
  getTerminalFgColor,
  getTerminalFgLabel
} from './accentStore'
export type { AccentPreset, TerminalFgPreset } from './accentStore'
export { useOrchestratorStore } from './orchestratorStore'
export type { TaskNode, OrchestratorJob, SpecInput } from './orchestratorStore'
export { useAgentStore } from './agentStore'
export type { ChatMessage as AgentChatMessage } from './agentStore'
export { useAuthStore } from './authStore'
export type { AuthUser } from './authStore'
export { useCascadeStore } from './cascadeStore'
export type { CascadeStatus, CascadeStage, StageStatus } from './cascadeStore'
// agentSessionsStore is deliberately NOT re-exported here: it is Electron-only
// and a barrel export would pull it into the web bundle. Import it directly
// from './agentSessionsStore' inside Electron-gated components.
