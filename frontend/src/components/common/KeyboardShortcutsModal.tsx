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
            <h3 className="font-brand text-sm text-text-primary mb-3 flex items-center gap-2">
              <Keyboard className="w-4 h-4 text-text-muted" />
              {group.title}
            </h3>
            <div className="border-brutal border-border rounded-brutal overflow-hidden">
              {group.shortcuts.map((shortcut, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between px-3 py-2 border-b border-border last:border-b-0 bg-background-card"
                >
                  <span className="text-sm text-text-secondary">{shortcut.description}</span>
                  <div className="flex items-center gap-1">
                    {shortcut.keys.map((key, kidx) => (
                      <span key={kidx}>
                        <kbd className="tag text-[10px] py-0.5 px-1.5">
                          <span>{key}</span>
                        </kbd>
                        {kidx < shortcut.keys.length - 1 && (
                          <span className="text-text-muted mx-0.5 font-mono">+</span>
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

      <div className="section-divider mt-6 mb-4" />
      <p className="text-xs text-text-muted text-center font-mono">
        Press{' '}
        <kbd className="tag text-[10px] py-0.5 px-1.5"><span>Ctrl</span></kbd>
        <span className="mx-1 font-mono">+</span>
        <kbd className="tag text-[10px] py-0.5 px-1.5"><span>?</span></kbd>
        {' '}anytime to show this dialog
      </p>
    </Modal>
  )
}
