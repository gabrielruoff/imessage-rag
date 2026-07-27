import { useLayoutEffect, useRef, useState, type KeyboardEvent } from 'react'
import { Textarea } from './ui/textarea'
import { cn } from '@/lib/cn'

/**
 * The message input. Enter sends, Shift+Enter inserts a newline. The textarea
 * auto-grows up to a cap; while a turn is streaming the send button becomes a
 * stop button.
 */
export function Composer({
  onSend,
  onStop,
  isStreaming,
}: {
  onSend: (text: string) => void
  onStop: () => void
  isStreaming: boolean
}) {
  const [value, setValue] = useState('')
  const ref = useRef<HTMLTextAreaElement>(null)

  // Auto-grow: reset then clamp to a max height so long drafts scroll in place.
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [value])

  const submit = () => {
    const text = value.trim()
    if (!text || isStreaming) return
    onSend(text)
    setValue('')
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="border-t border-line bg-paper/80 px-4 py-3 backdrop-blur sm:px-6">
      <div
        className={cn(
          'mx-auto flex max-w-3xl items-end gap-2 rounded-2xl border border-line',
          'bg-surface px-3 py-2 shadow-sm transition-[border-color,box-shadow]',
          'focus-within:border-accent/40 focus-within:shadow-[0_0_0_3px_var(--accent-soft)]',
        )}
      >
        <Textarea
          ref={ref}
          rows={1}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask your message history…"
          className="scroll-slim max-h-[200px] py-1.5"
          aria-label="Message"
        />
        {isStreaming ? (
          <button
            type="button"
            onClick={onStop}
            aria-label="Stop generating"
            className={cn(
              'grid h-9 w-9 shrink-0 place-items-center rounded-xl',
              'border border-line text-muted transition-colors hover:bg-surface-2 hover:text-ink',
            )}
          >
            <span className="h-3 w-3 rounded-[3px] bg-current" />
          </button>
        ) : (
          <button
            type="button"
            onClick={submit}
            disabled={!value.trim()}
            aria-label="Send"
            className={cn(
              'grid h-9 w-9 shrink-0 place-items-center rounded-xl transition-all',
              'bg-accent text-accent-ink hover:brightness-110 active:brightness-95',
              'disabled:bg-surface-2 disabled:text-faint disabled:hover:brightness-100',
            )}
          >
            <svg viewBox="0 0 24 24" className="h-[18px] w-[18px]" aria-hidden>
              <path
                fill="currentColor"
                d="M3.4 20.4l17.45-7.48a1 1 0 0 0 0-1.84L3.4 3.6a1 1 0 0 0-1.39 1.02L3.1 11l9.9 1-9.9 1-1.09 6.38a1 1 0 0 0 1.39 1.02Z"
              />
            </svg>
          </button>
        )}
      </div>
      <p className="mx-auto mt-1.5 max-w-3xl px-1 text-[11px] text-faint">
        Enter to send · Shift+Enter for a new line · runs locally, your corpus
        never leaves this machine
      </p>
    </div>
  )
}
