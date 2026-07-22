import type { ToolCall } from '@/lib/types'

/** A friendly verb + glyph for a tool name — the receipt's headline. */
function describe(name: string): { glyph: string; verb: string } {
  if (name.startsWith('search_')) return { glyph: '🔍', verb: 'searched messages' }
  if (name.startsWith('get_conversation') || name.startsWith('get_full_messages'))
    return { glyph: '💬', verb: 'pulled the thread' }
  if (name.includes('voiceprint') || name.startsWith('draft_') || name.includes('style'))
    return { glyph: '✍️', verb: 'studied a voice' }
  if (name.includes('conflict') || name.includes('apolog'))
    return { glyph: '⚖️', verb: 'weighed a conflict' }
  if (name.includes('trajectory') || name.includes('drift') || name.includes('wellbeing'))
    return { glyph: '📈', verb: 'traced a trajectory' }
  if (name.includes('contact') || name.includes('directory') || name.includes('roster'))
    return { glyph: '👥', verb: 'looked up contacts' }
  if (name.includes('nudge') || name.includes('loose') || name.includes('thread'))
    return { glyph: '🧵', verb: 'checked loose ends' }
  if (name.includes('topic')) return { glyph: '🗂️', verb: 'grouped topics' }
  return { glyph: '⋯', verb: 'ran a tool' }
}

function friendlyCount(count: number | 'ok'): string | null {
  if (count === 'ok') return null
  if (count === 0) return 'no results'
  const n = count.toLocaleString()
  return count === 1 ? '1 result' : `${n} results`
}

export function ToolCallNotice({ call }: { call: ToolCall }) {
  const { glyph, verb } = describe(call.name)
  const count = friendlyCount(call.count)
  return (
    <div className="flex items-baseline gap-2 rounded-md border-l-2 border-accent/40 bg-surface-2/60 py-1 pl-2.5 pr-2 font-mono text-[12px] leading-5 text-muted">
      <span aria-hidden className="shrink-0 not-italic">
        {glyph}
      </span>
      <span className="min-w-0 flex-1 truncate">
        <span className="text-ink/80">{verb}</span>
        <span className="text-faint"> · {call.name}</span>
      </span>
      {count && (
        <span className="shrink-0 rounded bg-accent-soft px-1.5 py-px text-[11px] text-accent">
          {count}
        </span>
      )}
    </div>
  )
}
