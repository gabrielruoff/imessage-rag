import { useCallback, useEffect, useRef, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { api, qk } from '@/lib/api'
import { cn } from '@/lib/cn'
import { useChatStream } from '@/hooks/useChatStream'
import { useTheme } from '@/hooks/useTheme'
import { ConversationSidebar } from '@/components/ConversationSidebar'
import { ChatPane } from '@/components/ChatPane'

function ThemeToggle() {
  const { theme, toggle } = useTheme()
  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
      className="grid h-8 w-8 place-items-center rounded-lg text-muted transition-colors hover:bg-surface-2 hover:text-ink"
    >
      {theme === 'dark' ? (
        <svg viewBox="0 0 24 24" className="h-[18px] w-[18px]" aria-hidden>
          <path
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            d="M12 4V2m0 20v-2m8-8h2M2 12h2m13.66 5.66 1.41 1.41M4.93 4.93l1.41 1.41m0 11.32-1.41 1.41M19.07 4.93l-1.41 1.41M12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10Z"
          />
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" className="h-[18px] w-[18px]" aria-hidden>
          <path
            fill="currentColor"
            d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79Z"
          />
        </svg>
      )}
    </button>
  )
}

function HealthDot() {
  const { data } = useQuery({ queryKey: qk.health, queryFn: api.health, refetchInterval: 30_000 })
  const ok = data?.reachable
  const label = data
    ? `${data.backend} · ${data.model}${data.reachable ? '' : ' (unreachable)'}`
    : 'checking backend…'
  return (
    <div className="hidden items-center gap-2 text-[12px] text-muted sm:flex" title={label}>
      <span
        className={cn(
          'h-2 w-2 rounded-full',
          data ? (ok ? 'bg-emerald-500' : 'bg-danger') : 'bg-faint',
        )}
      />
      <span className="max-w-[16rem] truncate text-[11px]">{data?.model ?? '…'}</span>
    </div>
  )
}

export default function App() {
  const qc = useQueryClient()
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const { messages, isStreaming, send, hydrate, clear, stop } = useChatStream()
  // The conversation whose turns are currently displayed. Guards against the
  // load-effect clobbering a turn we just streamed into a brand-new chat.
  const loadedRef = useRef<number | null>(null)

  const { data: conversations = [] } = useQuery({
    queryKey: qk.conversations,
    queryFn: api.listConversations,
  })

  // Load a selected conversation's history when switching TO it (not after a
  // send, which already left its turns in memory).
  useEffect(() => {
    if (selectedId == null || loadedRef.current === selectedId) return
    let cancelled = false
    api
      .getConversation(selectedId)
      .then((detail) => {
        if (cancelled) return
        hydrate(detail.turns)
        loadedRef.current = selectedId
      })
      .catch(() => {
        /* deleted or missing — the list refetch will reconcile */
      })
    return () => {
      cancelled = true
    }
  }, [selectedId, hydrate])

  const onNew = useCallback(() => {
    if (isStreaming) return
    loadedRef.current = null
    setSelectedId(null)
    clear()
  }, [isStreaming, clear])

  const onSelect = useCallback(
    (id: number) => {
      if (isStreaming || id === selectedId) return
      setSelectedId(id)
    },
    [isStreaming, selectedId],
  )

  const onDelete = useCallback(
    async (id: number) => {
      await api.deleteConversation(id)
      if (id === selectedId) {
        loadedRef.current = null
        setSelectedId(null)
        clear()
      }
      qc.invalidateQueries({ queryKey: qk.conversations })
    },
    [selectedId, clear, qc],
  )

  const onSend = useCallback(
    async (text: string) => {
      const { conversationId } = await send(selectedId, text)
      // A brand-new chat: adopt its id WITHOUT re-hydrating (the streamed turn
      // is already displayed), so the load-effect stays a no-op for it.
      if (conversationId != null && conversationId !== selectedId) {
        loadedRef.current = conversationId
        setSelectedId(conversationId)
      }
      // Title / updated_at / turn_count changed — refresh the sidebar.
      qc.invalidateQueries({ queryKey: qk.conversations })
    },
    [send, selectedId, qc],
  )

  return (
    <div className="flex h-full w-full overflow-hidden">
      <ConversationSidebar
        conversations={conversations}
        selectedId={selectedId}
        onSelect={onSelect}
        onNew={onNew}
        onDelete={onDelete}
      />

      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <header className="flex h-12 shrink-0 items-center justify-between border-b border-line bg-paper/80 px-4 backdrop-blur sm:px-6">
          <div className="flex items-baseline gap-2">
            <span className="text-[15px] font-semibold tracking-tight text-ink">
              iMessage RAG
            </span>
            <span className="hidden text-[11px] text-faint sm:inline">
              local · private
            </span>
          </div>
          <div className="flex items-center gap-3">
            <HealthDot />
            <ThemeToggle />
          </div>
        </header>

        <ChatPane
          messages={messages}
          isStreaming={isStreaming}
          onSend={onSend}
          onStop={stop}
        />
      </div>
    </div>
  )
}
