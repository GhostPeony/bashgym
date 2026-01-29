import { useState, useCallback, useEffect } from 'react'
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  LineChart,
  Line
} from 'recharts'
import { Settings, RefreshCw, TrendingUp, Clock, CheckCircle, XCircle } from 'lucide-react'
import { useRouterStore, useThemeStore } from '../../stores'
import { routerApi } from '../../services/api'
import { clsx } from 'clsx'

export function RouterDashboard() {
  const { strategy, studentRate, stats, setStudentRate, setStrategy, updateStats } = useRouterStore()
  const { theme } = useThemeStore()
  const [showSettings, setShowSettings] = useState(false)
  const [isRefreshing, setIsRefreshing] = useState(false)

  const handleRefresh = useCallback(async () => {
    if (isRefreshing) return

    setIsRefreshing(true)
    try {
      const result = await routerApi.getStats()
      if (result.ok && result.data) {
        updateStats({
          totalRequests: result.data.total_requests,
          teacherRequests: result.data.teacher_requests,
          studentRequests: result.data.student_requests,
          teacherSuccessRate: result.data.teacher_success_rate,
          studentSuccessRate: result.data.student_success_rate,
          avgTeacherLatency: result.data.avg_teacher_latency,
          avgStudentLatency: result.data.avg_student_latency,
          currentStudentRate: result.data.current_student_rate * 100
        })
      }
    } finally {
      setIsRefreshing(false)
    }
  }, [isRefreshing, updateStats])

  const colors = {
    teacher: theme === 'dark' ? '#76B900' : '#0066CC',
    student: theme === 'dark' ? '#00A6FF' : '#76B900',
    grid: theme === 'dark' ? '#2C2C2E' : '#E5E5EA',
    text: theme === 'dark' ? '#A1A1A6' : '#6E6E73'
  }

  // Traffic split data
  const trafficData = [
    { name: 'Teacher', value: 100 - studentRate, color: colors.teacher },
    { name: 'Student', value: studentRate, color: colors.student }
  ]

  // Performance data from stats (or empty if no data yet)
  const hasStats = stats.totalRequests > 0
  const performanceData = hasStats ? [
    { model: 'Teacher', success: stats.teacherSuccessRate, latency: stats.avgTeacherLatency },
    { model: 'Student', success: stats.studentSuccessRate, latency: stats.avgStudentLatency }
  ] : []

  // Latency history - populated via API when available
  const [latencyHistory, setLatencyHistory] = useState<Array<{ time: string; teacher: number; student: number }>>([])

  // Fetch stats on mount
  useEffect(() => {
    handleRefresh()
  }, [])

  return (
    <div className="h-full p-6 overflow-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-semibold text-text-primary">Router Dashboard</h1>
          <p className="text-sm text-text-secondary mt-1">
            Strategy: <span className="font-medium capitalize">{strategy.replace('_', ' ')}</span>
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            className="btn-ghost p-2"
            onClick={handleRefresh}
            disabled={isRefreshing}
            title="Refresh stats"
          >
            <RefreshCw className={clsx('w-5 h-5', isRefreshing && 'animate-spin')} />
          </button>
          <button onClick={() => setShowSettings(!showSettings)} className="btn-secondary flex items-center gap-2">
            <Settings className="w-4 h-4" />
            Configure
          </button>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-12 gap-4">
        {/* Traffic Split Chart */}
        <div className="col-span-5 card-elevated p-4">
          <h2 className="text-lg font-medium text-text-primary mb-4">Traffic Distribution</h2>
          <div className="h-64 flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={trafficData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={90}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {trafficData.map((entry, index) => (
                    <Cell key={index} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: theme === 'dark' ? '#242424' : '#F5F5F7',
                    border: `1px solid ${colors.grid}`,
                    borderRadius: '8px'
                  }}
                  formatter={(value: number) => [`${value}%`, 'Traffic']}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Legend */}
          <div className="flex items-center justify-center gap-6 mt-4">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: colors.teacher }} />
              <span className="text-sm text-text-secondary">Teacher ({100 - studentRate}%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: colors.student }} />
              <span className="text-sm text-text-secondary">Student ({studentRate}%)</span>
            </div>
          </div>

          {/* Slider Control */}
          <div className="mt-6">
            <label className="block text-sm text-text-muted mb-2">
              Student Traffic Rate: {studentRate}%
            </label>
            <input
              type="range"
              min={0}
              max={100}
              value={studentRate}
              onChange={(e) => setStudentRate(parseInt(e.target.value))}
              className="w-full h-2 bg-background-tertiary rounded-lg appearance-none cursor-pointer accent-primary"
            />
            <div className="flex justify-between text-xs text-text-muted mt-1">
              <span>0% (Teacher only)</span>
              <span>100% (Student only)</span>
            </div>
          </div>
        </div>

        {/* Performance Comparison */}
        <div className="col-span-7 card-elevated p-4">
          <h2 className="text-lg font-medium text-text-primary mb-4">Performance Comparison</h2>
          <div className="h-64">
            {performanceData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={performanceData} layout="vertical" margin={{ left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} horizontal={false} />
                  <XAxis type="number" domain={[0, 100]} stroke={colors.text} fontSize={11} />
                  <YAxis
                    type="category"
                    dataKey="model"
                    stroke={colors.text}
                    fontSize={11}
                    width={60}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: theme === 'dark' ? '#242424' : '#F5F5F7',
                      border: `1px solid ${colors.grid}`,
                      borderRadius: '8px'
                    }}
                    formatter={(value: number) => [`${value}%`, 'Success Rate']}
                  />
                  <Bar
                    dataKey="success"
                    fill={colors.teacher}
                    radius={[0, 4, 4, 0]}
                    barSize={24}
                  />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-text-muted">
                <TrendingUp className="w-12 h-12 mb-3 opacity-30" />
                <p className="text-sm">Router metrics will appear once models are deployed</p>
              </div>
            )}
          </div>
        </div>

        {/* Stats Cards */}
        <div className="col-span-3 card-elevated p-4">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 rounded-lg bg-status-success/10 flex items-center justify-center">
              <CheckCircle className="w-5 h-5 text-status-success" />
            </div>
            <div>
              <p className="text-sm text-text-muted">Teacher Success</p>
              <p className="text-2xl font-semibold text-text-primary">
                {hasStats ? `${stats.teacherSuccessRate}%` : '--'}
              </p>
            </div>
          </div>
        </div>

        <div className="col-span-3 card-elevated p-4">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 rounded-lg bg-accent/10 flex items-center justify-center">
              <TrendingUp className="w-5 h-5 text-accent" />
            </div>
            <div>
              <p className="text-sm text-text-muted">Student Success</p>
              <p className="text-2xl font-semibold text-text-primary">
                {hasStats ? `${stats.studentSuccessRate}%` : '--'}
              </p>
            </div>
          </div>
        </div>

        <div className="col-span-3 card-elevated p-4">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
              <Clock className="w-5 h-5 text-primary" />
            </div>
            <div>
              <p className="text-sm text-text-muted">Avg Teacher Latency</p>
              <p className="text-2xl font-semibold text-text-primary">
                {hasStats ? `${stats.avgTeacherLatency}ms` : '--'}
              </p>
            </div>
          </div>
        </div>

        <div className="col-span-3 card-elevated p-4">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 rounded-lg bg-status-info/10 flex items-center justify-center">
              <Clock className="w-5 h-5 text-status-info" />
            </div>
            <div>
              <p className="text-sm text-text-muted">Avg Student Latency</p>
              <p className="text-2xl font-semibold text-text-primary">
                {hasStats ? `${stats.avgStudentLatency}ms` : '--'}
              </p>
            </div>
          </div>
        </div>

        {/* Latency Over Time */}
        <div className="col-span-12 card-elevated p-4">
          <h2 className="text-lg font-medium text-text-primary mb-4">Latency Over Time</h2>
          <div className="h-48">
            {latencyHistory.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={latencyHistory} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} vertical={false} />
                  <XAxis dataKey="time" stroke={colors.text} fontSize={11} />
                  <YAxis stroke={colors.text} fontSize={11} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: theme === 'dark' ? '#242424' : '#F5F5F7',
                      border: `1px solid ${colors.grid}`,
                      borderRadius: '8px'
                    }}
                    formatter={(value: number) => [`${value}ms`, 'Latency']}
                  />
                  <Line
                    type="monotone"
                    dataKey="teacher"
                    stroke={colors.teacher}
                    strokeWidth={2}
                    dot={{ fill: colors.teacher, r: 4 }}
                    name="Teacher"
                  />
                  <Line
                    type="monotone"
                    dataKey="student"
                    stroke={colors.student}
                    strokeWidth={2}
                    dot={{ fill: colors.student, r: 4 }}
                    name="Student"
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-text-muted">
                <Clock className="w-10 h-10 mb-3 opacity-30" />
                <p className="text-sm">Collecting latency data...</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
