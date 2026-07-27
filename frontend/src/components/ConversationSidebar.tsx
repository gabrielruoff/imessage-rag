import { cn } from '@/lib/cn'
import type { Conversation } from '@/lib/types'

/** Compact relative time for a conversation's last activity. */
function ago(iso: string): string {
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return ''
  const s = Math.max(0, (Date.now() - then) / 1000)
  if (s < 60) return 'now'
  const m = s / 60
  if (m < 60) return `${Math.floor(m)}m`
  const h = m / 60
  if (h < 24) return `${Math.floor(h)}h`
  const d = h / 24
  if (d < 7) return `${Math.floor(d)}d`
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

function PlusIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden>
      <path
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        d="M12 5v14M5 12h14"
      />
    </svg>
  )
}

function TrashIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" aria-hidden>
      <path
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2m-9 0 1 12a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1l1-12"
      />
    </svg>
  )
}

export function ConversationSidebar({
  conversations,
  selectedId,
  onSelect,
  onNew,
  onDelete,
}: {
  conversations: Conversation[]
  selectedId: number | null
  onSelect: (id: number) => void
  onNew: () => void
  onDelete: (id: number) => void
}) {
  return (
    <aside className="hidden w-64 shrink-0 flex-col border-r border-line bg-surface md:flex">
      <div className="px-3 pb-2 pt-3">
        <button
          type="button"
          onClick={onNew}
          className={cn(
            'flex w-full items-center gap-2 rounded-lg border border-line bg-paper px-3 py-2',
            'text-sm font-medium text-ink transition-colors hover:bg-surface-2',
          )}
        >
          <PlusIcon />
          New chat
        </button>
      </div>

      <nav className="scroll-slim min-h-0 flex-1 overflow-y-auto px-2 pb-3">
        {conversations.length === 0 ? (
          <p className="px-3 py-6 text-center text-[13px] text-faint">
            No conversations yet.
          </p>
        ) : (
          <ul className="flex flex-col gap-0.5">
            {conversations.map((c) => {
              const active = c.id === selectedId
              return (
                <li key={c.id} className="group relative">
                  <button
                    type="button"
                    onClick={() => onSelect(c.id)}
                    className={cn(
                      'flex w-full flex-col items-start gap-0.5 rounded-lg px-3 py-2 pr-8 text-left',
                      'transition-colors',
                      active ? 'bg-accent-soft' : 'hover:bg-surface-2',
                    )}
                  >
                    <span
                      className={cn(
                        'w-full truncate text-[13.5px] font-medium',
                        active ? 'text-ink' : 'text-ink/90',
                      )}
                    >
                      {c.title || 'Untitled'}
                    </span>
                    <span className="text-[11px] text-faint">
                      {ago(c.updated_at)} · {c.turn_count} msg
                    </span>
                  </button>
                  <button
                    type="button"
                    aria-label="Delete conversation"
                    onClick={(e) => {
                      e.stopPropagation()
                      onDelete(c.id)
                    }}
                    className={cn(
                      'absolute right-1.5 top-2 grid h-6 w-6 place-items-center rounded-md',
                      'text-faint opacity-0 transition-opacity hover:bg-line hover:text-danger',
                      'group-hover:opacity-100 focus-visible:opacity-100',
                    )}
                  >
                    <TrashIcon />
                  </button>
                </li>
              )
            })}
          </ul>
        )}
      </nav>
    </aside>
  )
}
