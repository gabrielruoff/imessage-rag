import { useCallback, useRef, useState } from 'react'
import { readEventStream, toChatEvent } from '@/lib/sse'
import type { ChatMessage, Turn } from '@/lib/types'

let uid = 0
const nextId = () => `m${++uid}`

interface SendResult {
  /** The conversation id from the `meta` frame (new chats get a fresh one). */
  conversationId: number | null
}

/**
 * Owns the displayed message list and the POST + SSE reader for a turn.
 * `hydrate` seeds the list from persisted history; `send` streams a new turn,
 * mutating the optimistic assistant bubble as frames arrive and resolving with
 * the (possibly newly created) conversation id so the caller can select it.
 */
export function useChatStream() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const hydrate = useCallback((turns: Turn[]) => {
    setMessages(
      turns.map((t) => ({
        id: `turn-${t.id}`,
        role: t.role,
        content: t.content,
        toolCalls: [],
        status: 'done' as const,
      })),
    )
  }, [])

  const clear = useCallback(() => setMessages([]), [])

  const stop = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  const send = useCallback(
    async (
      conversationId: number | null,
      text: string,
      model?: string,
    ): Promise<SendResult> => {
      const assistantId = nextId()
      // Optimistically append the user turn and an empty streaming reply.
      setMessages((prev) => [
        ...prev,
        { id: nextId(), role: 'user', content: text, toolCalls: [], status: 'done' },
        { id: assistantId, role: 'assistant', content: '', toolCalls: [], status: 'streaming' },
      ])
      setIsStreaming(true)

      // Narrow updates to the in-flight assistant bubble.
      const patch = (fn: (m: ChatMessage) => ChatMessage) =>
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? fn(m) : m)),
        )

      const controller = new AbortController()
      abortRef.current = controller
      let resolvedId = conversationId
      let finalReply: string | null = null

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
          body: JSON.stringify({
            conversation_id: conversationId,
            message: text,
            ...(model ? { model } : {}),
          }),
          signal: controller.signal,
        })

        if (!res.ok || !res.body) {
          throw new Error(`Request failed (${res.status} ${res.statusText})`)
        }

        for await (const frame of readEventStream(res.body)) {
          const ev = toChatEvent(frame)
          if (!ev) continue
          switch (ev.type) {
            case 'meta':
              resolvedId = ev.conversation_id
              break
            case 'token':
              patch((m) => ({ ...m, content: m.content + ev.text }))
              break
            case 'tool_call':
              patch((m) => ({
                ...m,
                toolCalls: [
                  ...m.toolCalls,
                  {
                    name: ev.name,
                    arguments: ev.arguments,
                    count: ev.count,
                    tokens_est: ev.tokens_est,
                  },
                ],
              }))
              break
            case 'epilogue':
              patch((m) => ({ ...m, epilogue: ev.text }))
              break
            case 'retry':
              // Transient; the stream continues. Ignored beyond a faint notice
              // the pane derives from status. No state change needed.
              break
            case 'done':
              finalReply = ev.reply
              break
            case 'error':
              patch((m) => ({ ...m, status: 'error', error: ev.message }))
              break
          }
        }

        // `done.reply` is authoritative (buffered wellbeing turns arrive whole).
        patch((m) =>
          m.status === 'error'
            ? m
            : {
                ...m,
                content: finalReply ?? m.content,
                status: 'done',
              },
        )
      } catch (err) {
        const aborted = err instanceof DOMException && err.name === 'AbortError'
        patch((m) => ({
          ...m,
          status: aborted && m.content ? 'done' : 'error',
          error: aborted
            ? m.content
              ? undefined
              : 'Stopped before a reply arrived.'
            : err instanceof Error
              ? err.message
              : 'The stream failed.',
        }))
      } finally {
        setIsStreaming(false)
        abortRef.current = null
      }

      return { conversationId: resolvedId }
    },
    [],
  )

  return { messages, isStreaming, send, hydrate, clear, stop, setMessages }
}
