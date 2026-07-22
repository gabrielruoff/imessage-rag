import { useEffect, useRef } from 'react'
import type { ChatMessage } from '@/lib/types'
import { MessageBubble } from './MessageBubble'

const PROMPTS = [
  'Who did I text most this spring?',
  'Summarize my thread with Mom over the last month.',
  'What plans did I make but never follow up on?',
  'How has my tone with Alex changed this year?',
]

function EmptyState() {
  return (
    <div className="mx-auto flex h-full max-w-lg flex-col items-center justify-center px-6 text-center">
      <div className="mb-4 font-mono text-[12px] uppercase tracking-[0.2em] text-faint">
        Retrieval, not recall
      </div>
      <h2 className="text-balance text-2xl font-semibold tracking-tight text-ink">
        Ask your message history anything.
      </h2>
      <p className="mt-2 text-[14px] leading-relaxed text-muted">
        Your corpus never enters the prompt. The engine searches it, shows you
        what it pulled, and answers from what it found — all on this machine.
      </p>
      <ul className="mt-6 flex flex-col gap-1.5 text-left">
        {PROMPTS.map((p) => (
          <li
            key={p}
            className="rounded-lg border border-line bg-surface px-3 py-2 font-mono text-[12.5px] text-muted"
          >
            {p}
          </li>
        ))}
      </ul>
    </div>
  )
}

export function MessageList({
  messages,
  isStreaming,
}: {
  messages: ChatMessage[]
  isStreaming: boolean
}) {
  const endRef = useRef<HTMLDivElement>(null)

  // Follow the tail as tokens stream in and as turns are appended.
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'end' })
  }, [messages, isStreaming])

  if (messages.length === 0) {
    return (
      <div className="scroll-slim min-h-0 flex-1 overflow-y-auto">
        <EmptyState />
      </div>
    )
  }

  return (
    <div className="scroll-slim min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto flex max-w-3xl flex-col gap-4 px-4 py-6 sm:px-6">
        {messages.map((m) => (
          <MessageBubble key={m.id} message={m} />
        ))}
        <div ref={endRef} className="h-0" />
      </div>
    </div>
  )
}
