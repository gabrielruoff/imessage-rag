import type {
  Conversation,
  ConversationDetail,
  ConversationStub,
  Health,
} from './types'

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`${res.status} ${res.statusText}${body ? ` — ${body}` : ''}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  listConversations: (): Promise<Conversation[]> =>
    fetch('/api/conversations').then((r) => json<Conversation[]>(r)),

  getConversation: (id: number): Promise<ConversationDetail> =>
    fetch(`/api/conversations/${id}`).then((r) => json<ConversationDetail>(r)),

  createConversation: (): Promise<ConversationStub> =>
    fetch('/api/conversations', { method: 'POST' }).then((r) =>
      json<ConversationStub>(r),
    ),

  deleteConversation: async (id: number): Promise<void> => {
    const res = await fetch(`/api/conversations/${id}`, { method: 'DELETE' })
    if (!res.ok && res.status !== 204) {
      throw new Error(`Failed to delete conversation (${res.status})`)
    }
  },

  health: (): Promise<Health> =>
    fetch('/api/health').then((r) => json<Health>(r)),
}

/** Query keys, centralized so invalidation stays consistent. */
export const qk = {
  conversations: ['conversations'] as const,
  conversation: (id: number) => ['conversation', id] as const,
  health: ['health'] as const,
}
