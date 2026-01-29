import { useState, useRef } from 'react'
import { X } from 'lucide-react'
import { clsx } from 'clsx'
import { SEED_CATEGORY_TAGS } from './types'

interface TagEditorProps {
  tags: string[]
  onChange: (tags: string[]) => void
}

export function TagEditor({ tags, onChange }: TagEditorProps) {
  const [input, setInput] = useState('')
  const [showSuggestions, setShowSuggestions] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const suggestions = SEED_CATEGORY_TAGS.filter(
    tag => !tags.includes(tag) && tag.toLowerCase().includes(input.toLowerCase())
  )

  const addTag = (tag: string) => {
    const trimmed = tag.trim().toLowerCase()
    if (trimmed && !tags.includes(trimmed)) {
      onChange([...tags, trimmed])
    }
    setInput('')
    setShowSuggestions(false)
    inputRef.current?.focus()
  }

  const removeTag = (tag: string) => {
    onChange(tags.filter(t => t !== tag))
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && input.trim()) {
      e.preventDefault()
      addTag(input)
    } else if (e.key === 'Backspace' && !input && tags.length > 0) {
      removeTag(tags[tags.length - 1])
    }
  }

  return (
    <div className="relative">
      <div className="flex flex-wrap items-center gap-1.5 px-2 py-1.5 bg-background-tertiary rounded-lg border border-border-subtle focus-within:border-primary/50 transition-colors">
        {tags.map(tag => (
          <span
            key={tag}
            className="inline-flex items-center gap-1 px-2 py-0.5 text-xs rounded-full bg-primary/15 text-primary"
          >
            {tag}
            <button
              onClick={() => removeTag(tag)}
              className="hover:text-status-error transition-colors"
            >
              <X className="w-3 h-3" />
            </button>
          </span>
        ))}
        <input
          ref={inputRef}
          value={input}
          onChange={(e) => {
            setInput(e.target.value)
            setShowSuggestions(true)
          }}
          onFocus={() => setShowSuggestions(true)}
          onBlur={() => setTimeout(() => setShowSuggestions(false), 150)}
          onKeyDown={handleKeyDown}
          placeholder={tags.length === 0 ? 'Add tags...' : ''}
          className="flex-1 min-w-[80px] bg-transparent text-xs text-text-primary outline-none placeholder:text-text-muted"
        />
      </div>

      {showSuggestions && input && suggestions.length > 0 && (
        <div className="absolute z-10 mt-1 w-full bg-background-secondary border border-border-subtle rounded-lg shadow-elevated overflow-hidden">
          {suggestions.slice(0, 6).map(tag => (
            <button
              key={tag}
              onMouseDown={(e) => {
                e.preventDefault()
                addTag(tag)
              }}
              className="w-full px-3 py-1.5 text-xs text-left text-text-secondary hover:bg-background-tertiary hover:text-text-primary transition-colors"
            >
              {tag}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
