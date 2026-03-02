import type { ChatMessage, SendResponse } from "./types";

export async function postSend(message: string, history: ChatMessage[], model?: string): Promise<SendResponse> {
  const normalizedModel = typeof model === "string" ? model.trim() : "";
  const res = await fetch("/api/send", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ message, history, ...(normalizedModel ? { model: normalizedModel } : {}) }),
  });

  const data = (await res.json()) as SendResponse;
  if (!res.ok) {
    const err = data && typeof data.error === "string" ? data.error : `HTTP ${res.status}`;
    throw new Error(err);
  }
  if (!data || data.ok !== true) {
    const err = data && typeof data.error === "string" ? data.error : "request failed";
    throw new Error(err);
  }
  return data;
}
