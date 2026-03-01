import { createContext, useContext } from "react";

export type TamboRenderContextValue = {
  threadId: string;
  messageId: string;
  idPrefix: string;
};

export const TamboRenderContext = createContext<TamboRenderContextValue>({
  threadId: "local",
  messageId: "local",
  idPrefix: "block",
});

export function useTamboRenderContext(): TamboRenderContextValue {
  return useContext(TamboRenderContext);
}
