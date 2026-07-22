/** Wire + client types for the chat UI. Mirrors the fixed FastAPI contract. */

/** A single retrieval tool the engine invoked mid-turn (the "receipt"). */
export interface ToolCall {
  name: string
  arguments: Record<string, unknown>
  /** Numeric result count, or "ok" for tools that don't return rows. */
  count: number | 'ok'
  tokens_est: number
}

/**
 * SSE frames from POST /api/chat. Discriminated on `type`, which is carried by
 * the SSE `event:` line; the JSON `data:` payload supplies the rest. No `any`.
 */
export type ChatEvent =
  | { type: 'meta'; conversation_id: number; model: string }
  | { type: 'token'; text: string }
  | ({ type: 'tool_call' } & ToolCall)
  | { type: 'retry'; error: string }
  | { type: 'epilogue'; text: string }
  | { type: 'done'; reply: string }
  | { type: 'error'; message: string }

/** A message as rendered in the pane (persisted history or a live stream). */
export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  toolCalls: ToolCall[]
  /** Code-emitted, non-model support block rendered as a distinct callout. */
  epilogue?: string
  status: 'streaming' | 'done' | 'error'
  error?: string
}

/** GET /api/conversations row. */
export interface Conversation {
  id: number
  title: string
  created_at: string
  updated_at: string
  turn_count: number
}

/** A persisted turn inside a conversation. */
export interface Turn {
  id: number
  role: 'user' | 'assistant'
  content: string
  created_at: string
}

/** GET /api/conversations/{id}. */
export interface ConversationDetail {
  id: number
  title: string
  created_at: string
  updated_at: string
  turns: Turn[]
}

/** POST /api/conversations. */
export interface ConversationStub {
  id: number
  title: string
  created_at: string
  updated_at: string
}

/** GET /api/health. */
export interface Health {
  ok: boolean
  backend: string
  model: string
  reachable: boolean
}
