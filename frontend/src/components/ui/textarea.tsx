import { forwardRef, type TextareaHTMLAttributes } from 'react'
import { cn } from '@/lib/cn'

export const Textarea = forwardRef<
  HTMLTextAreaElement,
  TextareaHTMLAttributes<HTMLTextAreaElement>
>(({ className, ...props }, ref) => (
  <textarea
    ref={ref}
    className={cn(
      'w-full resize-none bg-transparent text-ink placeholder:text-faint',
      'outline-none text-[15px] leading-6',
      className,
    )}
    {...props}
  />
))
Textarea.displayName = 'Textarea'
