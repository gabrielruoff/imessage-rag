import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

/* GFM renderer for assistant replies. Visual styling lives in index.css
   under .md-body so it stays token-driven (light/dark for free). */
export function Markdown({ children }: { children: string }) {
  return (
    <div className="md-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          // Wide tables scroll inside the bubble instead of blowing out layout.
          table: ({ node: _node, ...props }) => (
            <div className="overflow-x-auto">
              <table {...props} />
            </div>
          ),
          a: ({ node: _node, ...props }) => (
            <a target="_blank" rel="noreferrer" {...props} />
          ),
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  )
}
