import type { ChatEvent } from './types'

/** A raw parsed SSE frame: an event name and its (possibly multi-line) data. */
interface RawFrame {
  event: string
  data: string
}

const KNOWN_EVENTS = new Set([
  'meta',
  'token',
  'tool_call',
  'retry',
  'epilogue',
  'done',
  'error',
])

/** Parse one `\n\n`-delimited SSE frame block into { event, data }. */
function parseFrame(block: string): RawFrame | null {
  let event = 'message'
  const dataLines: string[] = []
  for (const line of block.split(/\r?\n/)) {
    if (!line || line.startsWith(':')) continue // blank or comment
    const colon = line.indexOf(':')
    const field = colon === -1 ? line : line.slice(0, colon)
    // Per spec, a single leading space after the colon is stripped.
    let value = colon === -1 ? '' : line.slice(colon + 1)
    if (value.startsWith(' ')) value = value.slice(1)
    if (field === 'event') event = value
    else if (field === 'data') dataLines.push(value)
  }
  if (dataLines.length === 0) return null
  return { event, data: dataLines.join('\n') }
}

/**
 * Read a `text/event-stream` body and yield decoded frames as they arrive.
 * Buffers across network chunks so a frame split mid-read is not lost.
 */
export async function* readEventStream(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<RawFrame> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let sep: number
      // Frames end at a blank line (handle both \n\n and \r\n\r\n).
      while ((sep = nextFrameBoundary(buffer)) !== -1) {
        const [block, advance] = splitAt(buffer, sep)
        buffer = buffer.slice(advance)
        const frame = parseFrame(block)
        if (frame) yield frame
      }
    }
    buffer += decoder.decode()
    const tail = buffer.trim()
    if (tail) {
      const frame = parseFrame(tail)
      if (frame) yield frame
    }
  } finally {
    reader.releaseLock()
  }
}

function nextFrameBoundary(buf: string): number {
  const a = buf.indexOf('\n\n')
  const b = buf.indexOf('\r\n\r\n')
  if (a === -1) return b
  if (b === -1) return a
  return Math.min(a, b)
}

function splitAt(buf: string, idx: number): [string, number] {
  const isCrlf = buf.startsWith('\r\n\r\n', idx)
  return [buf.slice(0, idx), idx + (isCrlf ? 4 : 2)]
}

/** Fold a raw frame into a typed ChatEvent; returns null for unknown events. */
export function toChatEvent(frame: RawFrame): ChatEvent | null {
  if (!KNOWN_EVENTS.has(frame.event)) return null
  let payload: Record<string, unknown> = {}
  try {
    const parsed: unknown = JSON.parse(frame.data)
    if (parsed && typeof parsed === 'object') {
      payload = parsed as Record<string, unknown>
    }
  } catch {
    // Malformed data line — surface as an error frame rather than crashing.
    if (frame.event === 'error') return { type: 'error', message: frame.data }
    return null
  }
  // The event name is the discriminant; the payload fills the rest.
  return { type: frame.event, ...payload } as ChatEvent
}
