import type { ChatConversation, ChatConversationSummary, ChatMessage, SendResponse } from "./types";

type ConversationsResponse = {
  ok?: boolean;
  conversations?: ChatConversationSummary[];
  error?: string;
};

type ConversationResponse = ChatConversation & {
  ok?: boolean;
  error?: string;
};

function readError(data: { error?: string } | null | undefined, res: Response): string {
  return data && typeof data.error === "string" ? data.error : `HTTP ${res.status}`;
}

export async function postSend(
  message: string,
  history: ChatMessage[],
  model?: string,
  threadId?: string,
): Promise<SendResponse> {
  const normalizedModel = typeof model === "string" ? model.trim() : "";
  const res = await fetch("/api/send", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      message,
      history,
      ...(normalizedModel ? { model: normalizedModel } : {}),
      ...(threadId ? { thread_id: threadId } : {}),
    }),
  });

  const data = (await res.json()) as SendResponse;
  if (!res.ok) {
    throw new Error(readError(data, res));
  }
  if (!data || data.ok !== true) {
    throw new Error(readError(data, res));
  }
  return data;
}

export async function listConversations(): Promise<ChatConversationSummary[]> {
  const res = await fetch("/api/conversations", { headers: { accept: "application/json" } });
  const data = (await res.json()) as ConversationsResponse;
  if (!res.ok) {
    throw new Error(readError(data, res));
  }
  if (!data || data.ok !== true || !Array.isArray(data.conversations)) {
    throw new Error(readError(data, res));
  }
  return data.conversations;
}

export async function getConversation(threadId: string): Promise<ChatConversation> {
  const res = await fetch(`/api/conversations/${encodeURIComponent(threadId)}`, { headers: { accept: "application/json" } });
  const data = (await res.json()) as ConversationResponse;
  if (!res.ok) {
    throw new Error(readError(data, res));
  }
  if (!data || data.ok !== true || !Array.isArray(data.messages)) {
    throw new Error(readError(data, res));
  }
  return data;
}
