import { createContext, useContext } from "react";

export type ChatSendFn = (message: string) => void | Promise<void>;

export const ChatSendContext = createContext<ChatSendFn | null>(null);

export function useChatSend(): ChatSendFn | null {
  return useContext(ChatSendContext);
}
