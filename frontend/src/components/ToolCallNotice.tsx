import { useState } from 'react'
import type { ToolCall } from '@/lib/types'

type Glyph =
  | 'search'
  | 'thread'
  | 'stats'
  | 'contacts'
  | 'conflict'
  | 'voice'
  | 'trend'
  | 'topics'
  | 'loose'
  | 'default'

/** A friendly verb + monochrome glyph for a tool name — the receipt's headline. */
function describe(name: string): { glyph: Glyph; verb: string } {
  if (name.startsWith('search_')) return { glyph: 'search', verb: 'searched messages' }
  if (name.startsWith('get_conversation') || name.startsWith('get_full_messages'))
    return { glyph: 'thread', verb: 'pulled the thread' }
  if (name.includes('voiceprint') || name.startsWith('draft_') || name.includes('style'))
    return { glyph: 'voice', verb: 'studied a voice' }
  if (name.includes('conflict') || name.includes('apolog'))
    return { glyph: 'conflict', verb: 'weighed a conflict' }
  if (name.includes('trajectory') || name.includes('drift') || name.includes('wellbeing'))
    return { glyph: 'trend', verb: 'traced a trajectory' }
  if (name.includes('contact') || name.includes('directory') || name.includes('roster'))
    return { glyph: 'contacts', verb: 'looked up contacts' }
  if (name.includes('nudge') || name.includes('loose') || name.includes('thread'))
    return { glyph: 'loose', verb: 'checked loose ends' }
  if (name.includes('topic')) return { glyph: 'topics', verb: 'grouped topics' }
  if (name.includes('stats') || name.includes('summary') || name.startsWith('count_'))
    return { glyph: 'stats', verb: 'crunched the numbers' }
  if (name.startsWith('list_chats')) return { glyph: 'thread', verb: 'listed conversations' }
  return { glyph: 'default', verb: 'ran a tool' }
}

/* SF-Symbols-flavored stroke glyphs; drawn, not emoji, so they stay monochrome
   and inherit the capsule's text color. */
const GLYPH_PATHS: Record<Glyph, string> = {
  search: 'M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14Zm5 12 4 4',
  thread: 'M12 4a8 8 0 0 1 8 8 8 8 0 0 1-8 8H4V12a8 8 0 0 1 8-8Z',
  stats: 'M5 20v-9m7 9V4m7 16v-6',
  contacts: 'M12 11a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Zm-7 9a7 7 0 0 1 14 0',
  conflict: 'M12 4v16M6 6l-3 6h6L6 6Zm12 0-3 6h6l-3-6ZM5 6h14',
  voice: 'm4 20 1-4L16 5l3 3L8 19l-4 1Z',
  trend: 'm3 17 6-6 4 4 8-8m-6 0h6v6',
  topics: 'M3 7a1 1 0 0 1 1-1h5l2 2h9a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V7Z',
  loose:
    'M10 14a5 5 0 0 0 7 0l3-3a5 5 0 0 0-7-7l-1 1m2 5a5 5 0 0 0-7 0l-3 3a5 5 0 0 0 7 7l1-1',
  default: 'M6 12h.01M12 12h.01M18 12h.01',
}

function GlyphIcon({ glyph }: { glyph: Glyph }) {
  return (
    <svg viewBox="0 0 24 24" className="h-3.5 w-3.5 shrink-0" aria-hidden>
      <path
        d={GLYPH_PATHS[glyph]}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function Chevron({ up }: { up?: boolean }) {
  return (
    <svg viewBox="0 0 24 24" className="h-3 w-3 shrink-0" aria-hidden>
      <path
        d={up ? 'm6 14 6-6 6 6' : 'm6 10 6 6 6-6'}
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function friendlyCount(count: number | 'ok'): string | null {
  if (count === 'ok') return null
  if (count === 0) return 'no results'
  const n = count.toLocaleString()
  return count === 1 ? '1 result' : `${n} results`
}

const capsule =
  'inline-flex items-center gap-1.5 rounded-full border border-line bg-surface-2/60 ' +
  'py-1 pl-2 pr-2.5 text-[12px] leading-4 text-muted'

function Receipt({
  call,
  index,
  animate,
}: {
  call: ToolCall
  index: number
  animate: boolean
}) {
  const { glyph, verb } = describe(call.name)
  const count = friendlyCount(call.count)
  return (
    <span
      title={call.name}
      className={capsule}
      style={
        animate
          ? { animation: 'msg-in 0.25s ease-out both', animationDelay: `${index * 60}ms` }
          : undefined
      }
    >
      <GlyphIcon glyph={glyph} />
      <span className="text-ink/80">{verb}</span>
      {count && <span className="text-accent">{count}</span>}
    </span>
  )
}

/**
 * The retrieval receipts for one assistant turn. While the turn streams, each
 * receipt appears as its own capsule with a slight stagger; once the turn
 * settles, two or more receipts collapse into a single summary capsule that
 * expands on click.
 */
export function ToolReceipts({
  calls,
  streaming,
}: {
  calls: ToolCall[]
  streaming: boolean
}) {
  const [expanded, setExpanded] = useState(false)
  if (calls.length === 0) return null

  if (streaming || expanded || calls.length === 1) {
    return (
      <div className="flex flex-wrap items-center gap-1.5">
        {calls.map((c, i) => (
          <Receipt key={`${c.name}-${i}`} call={c} index={i} animate={streaming} />
        ))}
        {!streaming && calls.length > 1 && (
          <button
            type="button"
            aria-label="Collapse retrieval details"
            onClick={() => setExpanded(false)}
            className={`${capsule} cursor-pointer px-2 transition-colors hover:bg-surface-2 hover:text-ink`}
          >
            <Chevron up />
          </button>
        )}
      </div>
    )
  }

  const results = calls.reduce(
    (n, c) => n + (typeof c.count === 'number' ? c.count : 0),
    0,
  )
  return (
    <div>
      <button
        type="button"
        onClick={() => setExpanded(true)}
        title={calls.map((c) => c.name).join(', ')}
        className={`${capsule} cursor-pointer transition-colors hover:bg-surface-2 hover:text-ink`}
      >
        <GlyphIcon glyph="search" />
        <span className="text-ink/80">
          {calls.length} lookups
          {results > 0 && ` · ${results.toLocaleString()} results`}
        </span>
        <Chevron />
      </button>
    </div>
  )
}
