import type { ChatMessage } from '@/lib/types'
import { MessageList } from './MessageList'
import { Composer } from './Composer'

/** The right-hand column: the scrolling transcript plus the composer. */
export function ChatPane({
  messages,
  isStreaming,
  onSend,
  onStop,
}: {
  messages: ChatMessage[]
  isStreaming: boolean
  onSend: (text: string) => void
  onStop: () => void
}) {
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <MessageList messages={messages} isStreaming={isStreaming} onSend={onSend} />
      <Composer onSend={onSend} onStop={onStop} isStreaming={isStreaming} />
    </div>
  )
}
