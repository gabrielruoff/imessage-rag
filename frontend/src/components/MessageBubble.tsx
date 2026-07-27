import { memo } from 'react'
import { cn } from '@/lib/cn'
import type { ChatMessage } from '@/lib/types'
import { Markdown } from './Markdown'
import { ToolReceipts } from './ToolCallNotice'

function TypingCaret() {
  return (
    <span
      aria-label="Assistant is typing"
      className="ml-0.5 inline-block h-[1.05em] w-[2px] translate-y-[2px] rounded-full bg-accent align-middle"
      style={{ animation: 'caret-blink 1s steps(1) infinite' }}
    />
  )
}

/* The native three-dot "typing" indicator, shown before the first token. */
function TypingDots() {
  return (
    <span aria-label="Assistant is typing" className="inline-flex items-center gap-1 py-1">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="h-[7px] w-[7px] rounded-full bg-faint"
          style={{
            animation: 'typing-dot 1.2s ease-in-out infinite',
            animationDelay: `${i * 150}ms`,
          }}
        />
      ))}
    </span>
  )
}

/* Memoized so a token frame (which replaces only the in-flight message
   object) skips re-rendering — and re-parsing the Markdown of — every
   settled bubble in the list. */
export const MessageBubble = memo(function MessageBubble({
  message,
}: {
  message: ChatMessage
}) {
  const isUser = message.role === 'user'
  const streaming = message.status === 'streaming'
  const errored = message.status === 'error'
  const empty = message.content.length === 0

  return (
    <div
      className={cn(
        'flex w-full',
        isUser ? 'origin-bottom-right justify-end' : 'origin-bottom-left justify-start',
      )}
      style={{ animation: 'msg-in 0.28s ease-out both' }}
    >
      <div className={cn('max-w-[min(680px,88%)]', isUser && 'flex flex-col items-end')}>
        {/* Retrieval receipts cluster above the reply as the engine works. */}
        {!isUser && message.toolCalls.length > 0 && (
          <div className="mb-1.5">
            <ToolReceipts calls={message.toolCalls} streaming={streaming} />
          </div>
        )}

        <div
          className={cn(
            'break-words rounded-[18px] px-3.5 py-2.5 text-[15px] leading-[1.55]',
            isUser
              ? 'whitespace-pre-wrap bg-accent text-accent-ink rounded-br-[6px]'
              : 'bg-surface text-ink border border-line rounded-bl-[6px]',
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
                <TypingDots />
              ) : isUser ? (
                message.content
              ) : streaming ? (
                /* Plain text while tokens stream — re-parsing the whole reply
                   as Markdown per token is O(n²). The done patch swaps in the
                   Markdown render once. */
                <span className="whitespace-pre-wrap">{message.content}</span>
              ) : (
                <Markdown>{message.content}</Markdown>
              )}
              {streaming && !empty && <TypingCaret />}
            </>
          )}
        </div>

        {/* Non-model support block — a distinct, bordered care callout. */}
        {message.epilogue && (
          <div className="mt-2 max-w-full rounded-xl border border-care-line/60 bg-care-bg px-3.5 py-2.5 text-[13px] leading-relaxed text-care-ink">
            <div className="mb-1 text-[11px] font-semibold tracking-wide opacity-70">
              Support
            </div>
            <div className="whitespace-pre-wrap break-words">{message.epilogue}</div>
          </div>
        )}
      </div>
    </div>
  )
})
