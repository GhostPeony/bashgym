import { Modal } from './Modal'
import { useUIStore } from '../../stores'
import { Keyboard } from 'lucide-react'

interface ShortcutGroup {
  title: string
  shortcuts: { keys: string[]; description: string }[]
}

const shortcutGroups: ShortcutGroup[] = [
  {
    title: 'General',
    shortcuts: [
      { keys: ['Ctrl', 'K'], description: 'Open command palette' },
      { keys: ['Ctrl', ','], description: 'Open settings' },
      { keys: ['Ctrl', 'D'], description: 'Toggle dark/light theme' },
      { keys: ['Ctrl', '?'], description: 'Show keyboard shortcuts' },
      { keys: ['Escape'], description: 'Close overlay/modal' }
    ]
  },
  {
    title: 'Terminal',
    shortcuts: [
      { keys: ['Ctrl', 'N'], description: 'New terminal' },
      { keys: ['Ctrl', 'W'], description: 'Close current panel' },
      { keys: ['Ctrl', 'Shift', 'C'], description: 'Copy selection' },
      { keys: ['Ctrl', 'Shift', 'V'], description: 'Paste' },
      { keys: ['Ctrl', 'L'], description: 'Clear terminal' },
      { keys: ['Ctrl', '1-9'], description: 'Focus terminal 1-9' }
    ]
  },
  {
    title: 'Canvas View',
    shortcuts: [
      { keys: ['1-9'], description: 'Focus session by number' },
      { keys: ['Tab'], description: 'Cycle through sessions' },
      { keys: ['Shift', 'Tab'], description: 'Cycle sessions (reverse)' },
      { keys: ['F'], description: 'Fit view to all nodes' },
      { keys: ['G'], description: 'Toggle grid' },
      { keys: ['M'], description: 'Toggle minimap' },
      { keys: ['Space'], description: 'Pause/resume focused agent' },
      { keys: ['Shift', 'P'], description: 'Pause/resume all agents' }
    ]
  },
  {
    title: 'Navigation',
    shortcuts: [
      { keys: ['Ctrl', 'B'], description: 'Toggle sidebar' },
      { keys: ['Ctrl', 'Shift', 'T'], description: 'Open Training dashboard' },
      { keys: ['Ctrl', 'Shift', 'R'], description: 'Open Router dashboard' },
      { keys: ['Ctrl', 'Shift', 'F'], description: 'Open Factory dashboard' },
      { keys: ['Ctrl', 'Shift', 'E'], description: 'Open Evaluator dashboard' }
    ]
  },
  {
    title: 'Training',
    shortcuts: [
      { keys: ['Space'], description: 'Pause/Resume training (when focused)' },
      { keys: ['Ctrl', 'Enter'], description: 'Start training' },
      { keys: ['Ctrl', 'S'], description: 'Save checkpoint' }
    ]
  }
]

export function KeyboardShortcutsModal() {
  const { isKeyboardShortcutsOpen, setKeyboardShortcutsOpen } = useUIStore()

  return (
    <Modal
      isOpen={isKeyboardShortcutsOpen}
      onClose={() => setKeyboardShortcutsOpen(false)}
      title="Keyboard Shortcuts"
      description="Quick reference for keyboard shortcuts"
      size="lg"
    >
      <div className="grid grid-cols-2 gap-6">
        {shortcutGroups.map((group) => (
          <div key={group.title}>
            <h3 className="text-sm font-semibold text-text-primary mb-3 flex items-center gap-2">
              <Keyboard className="w-4 h-4 text-text-muted" />
              {group.title}
            </h3>
            <div className="space-y-2">
              {group.shortcuts.map((shortcut, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between py-1.5 border-b border-border-subtle last:border-0"
                >
                  <span className="text-sm text-text-secondary">{shortcut.description}</span>
                  <div className="flex items-center gap-1">
                    {shortcut.keys.map((key, kidx) => (
                      <span key={kidx}>
                        <kbd className="px-2 py-1 text-xs rounded bg-background-tertiary border border-border-subtle text-text-muted font-mono">
                          {key}
                        </kbd>
                        {kidx < shortcut.keys.length - 1 && (
                          <span className="text-text-muted mx-0.5">+</span>
                        )}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 pt-4 border-t border-border-subtle">
        <p className="text-xs text-text-muted text-center">
          Press <kbd className="px-1.5 py-0.5 text-xs rounded bg-background-tertiary border border-border-subtle">Ctrl</kbd>
          <span className="mx-1">+</span>
          <kbd className="px-1.5 py-0.5 text-xs rounded bg-background-tertiary border border-border-subtle">?</kbd>
          {' '}anytime to show this dialog
        </p>
      </div>
    </Modal>
  )
}
