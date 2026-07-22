import { cn } from '@/lib/cn'
import type { ChatMessage } from '@/lib/types'
import { ToolCallNotice } from './ToolCallNotice'

function TypingCaret() {
  return (
    <span
      aria-label="Assistant is typing"
      className="ml-0.5 inline-block h-[1.05em] w-[2px] translate-y-[2px] rounded-full bg-accent align-middle"
      style={{ animation: 'caret-blink 1s steps(1) infinite' }}
    />
  )
}

export function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user'
  const streaming = message.status === 'streaming'
  const errored = message.status === 'error'
  const empty = message.content.length === 0

  return (
    <div
      className={cn('flex w-full', isUser ? 'justify-end' : 'justify-start')}
      style={{ animation: 'msg-in 0.28s ease-out both' }}
    >
      <div className={cn('max-w-[min(680px,88%)]', isUser && 'flex flex-col items-end')}>
        {/* Retrieval receipts stack above the reply as the engine works. */}
        {!isUser && message.toolCalls.length > 0 && (
          <div className="mb-1.5 flex flex-col gap-1">
            {message.toolCalls.map((c, i) => (
              <ToolCallNotice key={`${c.name}-${i}`} call={c} />
            ))}
          </div>
        )}

        <div
          className={cn(
            'whitespace-pre-wrap break-words rounded-2xl px-3.5 py-2.5 text-[15px] leading-[1.55]',
            isUser
              ? 'bg-accent text-accent-ink rounded-br-md'
              : 'bg-surface text-ink border border-line rounded-bl-md',
            errored && 'border-danger/50 bg-danger/5 text-danger',
          )}
        >
          {errored ? (
            <span className="text-danger">
              {message.error ?? 'Something went wrong on this turn.'}
            </span>
          ) : (
            <>
              {empty && streaming ? (
                <span className="text-faint">Thinking</span>
              ) : (
                message.content
              )}
              {streaming && <TypingCaret />}
            </>
          )}
        </div>

        {/* Non-model support block — a distinct, bordered care callout. */}
        {message.epilogue && (
          <div className="mt-2 max-w-full rounded-xl border border-care-line/60 bg-care-bg px-3.5 py-2.5 text-[13px] leading-relaxed text-care-ink">
            <div className="mb-1 font-mono text-[11px] uppercase tracking-wide opacity-70">
              Support
            </div>
            <div className="whitespace-pre-wrap break-words">{message.epilogue}</div>
          </div>
        )}
      </div>
    </div>
  )
}
