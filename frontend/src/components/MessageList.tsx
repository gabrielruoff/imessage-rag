import { useEffect, useRef } from 'react'
import type { ChatMessage } from '@/lib/types'
import { MessageBubble } from './MessageBubble'

const PROMPTS = [
  'Who did I text most this spring?',
  'Summarize my thread with Mom over the last month.',
  'What plans did I make but never follow up on?',
  'How has my tone with Alex changed this year?',
]

function EmptyState({ onSelect }: { onSelect: (text: string) => void }) {
  return (
    <div className="mx-auto flex h-full max-w-lg flex-col items-center justify-center px-6 text-center">
      <div className="mb-4 text-[12px] font-medium text-faint">
        Retrieval, not recall
      </div>
      <h2 className="text-balance text-2xl font-semibold tracking-tight text-ink">
        Ask your message history anything.
      </h2>
      <p className="mt-2 text-[14px] leading-relaxed text-muted">
        Your corpus never enters the prompt. The engine searches it, shows you
        what it pulled, and answers from what it found — all on this machine.
      </p>
      <ul className="mt-6 flex flex-wrap justify-center gap-2">
        {PROMPTS.map((p) => (
          <li key={p}>
            <button
              type="button"
              onClick={() => onSelect(p)}
              className="cursor-pointer rounded-full border border-line bg-surface px-3.5 py-1.5 text-[14px] text-muted transition-colors hover:border-accent/50 hover:bg-surface-2 hover:text-ink focus-visible:border-accent/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-soft"
            >
              {p}
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}

export function MessageList({
  messages,
  isStreaming,
  onSend,
}: {
  messages: ChatMessage[]
  isStreaming: boolean
  onSend: (text: string) => void
}) {
  const endRef = useRef<HTMLDivElement>(null)

  // Follow the tail as tokens stream in and as turns are appended.
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'end' })
  }, [messages, isStreaming])

  if (messages.length === 0) {
    return (
      <div className="scroll-slim min-h-0 flex-1 overflow-y-auto">
        <EmptyState onSelect={onSend} />
      </div>
    )
  }

  return (
    <div className="scroll-slim min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto flex max-w-3xl cursor-default flex-col gap-4 px-4 py-6 sm:px-6">
        {messages.map((m) => (
          <MessageBubble key={m.id} message={m} />
        ))}
        <div ref={endRef} className="h-0" />
      </div>
    </div>
  )
}
