import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-sm border px-1.5 py-0.5 font-mono text-[11px] leading-none tracking-wide uppercase",
  {
    variants: {
      variant: {
        default: "border-line bg-paper-2 text-ink-2",
        accent: "border-accent/40 bg-accent/10 text-accent",
        ok: "border-ok/40 bg-ok/10 text-ok",
        weak: "border-line text-ink-3",
      },
    },
    defaultVariants: { variant: "default" },
  },
);

export function Badge({
  className,
  variant,
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & VariantProps<typeof badgeVariants>) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}
