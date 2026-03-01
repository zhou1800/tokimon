import { ComponentRenderer } from "@tambo-ai/react";
import { z } from "zod";

import type { UIBlock } from "../../types";
import { TamboRenderContext, useTamboRenderContext } from "../renderContext";
import { toComponentContent } from "../toComponentBlock";

export const PanelPropsSchema = z.object({
  title: z.string().optional(),
  blocks: z.array(z.any()),
});

export type PanelProps = {
  title?: string;
  blocks: UIBlock[];
};

export function PanelBlock(props: PanelProps) {
  const ctx = useTamboRenderContext();
  const threadId = ctx.threadId;
  const messageId = ctx.messageId;
  const basePrefix = ctx.idPrefix;

  return (
    <div>
      {props.title ? <div className="tm-block-title">{props.title}</div> : null}
      <div style={{ display: "grid", gap: 10 }}>
        {props.blocks.map((block, idx) => {
          const id = `${basePrefix}.${idx}`;
          const content = toComponentContent(block, id);
          return (
            <div key={id} className="tm-block">
              <TamboRenderContext.Provider value={{ ...ctx, idPrefix: id }}>
                <ComponentRenderer content={content as any} threadId={threadId} messageId={messageId} />
              </TamboRenderContext.Provider>
            </div>
          );
        })}
      </div>
    </div>
  );
}
