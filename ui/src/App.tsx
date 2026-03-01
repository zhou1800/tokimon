import { ComponentRenderer, TamboRegistryProvider } from "@tambo-ai/react";
import { useMemo, useState } from "react";

import { postSend } from "./api";
import type { ChatMessage, SendResponse, UIBlock } from "./types";
import { components } from "./tambo/registry";
import { TamboRenderContext } from "./tambo/renderContext";
import { ChatSendContext } from "./tambo/sendContext";
import { toComponentContent } from "./tambo/toComponentBlock";

type LogEntry = {
  role: "user" | "assistant";
  content: string;
  meta?: string;
  error?: boolean;
};

export function App() {
  const [input, setInput] = useState("");
  const [history, setHistory] = useState<ChatMessage[]>([]);
  const [log, setLog] = useState<LogEntry[]>([]);
  const [sending, setSending] = useState(false);
  const [lastResponse, setLastResponse] = useState<SendResponse | null>(null);

  const uiBlocks = useMemo(() => {
    const stepBlocks = lastResponse?.step_result?.ui_blocks;
    return (Array.isArray(stepBlocks) ? stepBlocks : lastResponse?.ui_blocks) ?? [];
  }, [lastResponse]);

  async function sendMessage(message: string) {
    setSending(true);
    setLog((l) => [...l, { role: "user", content: message }]);
    const nextHistory = [...history, { role: "user", content: message } satisfies ChatMessage];
    setHistory(nextHistory);

    try {
      const res = await postSend(message, nextHistory);
      setLastResponse(res);
      const reply = typeof res.reply === "string" ? res.reply : typeof res.summary === "string" ? res.summary : "";
      const meta = typeof res.status === "string" ? `status=${res.status}` : undefined;
      setLog((l) => [...l, { role: "assistant", content: reply || "(no reply)", meta }]);
      setHistory((h) => [...h, { role: "assistant", content: reply || "" }]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setLog((l) => [...l, { role: "assistant", content: `Error: ${msg}`, meta: "request failed", error: true }]);
    } finally {
      setSending(false);
    }
  }

  const onFormSend = async (message: string) => {
    await sendMessage(message);
  };

  return (
    <>
      <header className="tm-header">Tokimon Chat UI</header>
      <div className="tm-wrap">
        <div className="tm-log">
          {log.map((entry, i) => (
            <div key={i} className={`tm-msg ${entry.role === "user" ? "tm-user" : ""} ${entry.error ? "tm-error" : ""}`}>
              <div>
                {entry.role === "user" ? "You" : "Tokimon"}: {entry.content}
              </div>
              {entry.meta ? <div className="tm-meta">{entry.meta}</div> : null}
            </div>
          ))}

          {lastResponse ? (
            <div className="tm-pane">
              <details className="tm-structured">
                <summary>Structured result</summary>
                <pre className="tm-pre">{JSON.stringify(lastResponse.step_result ?? lastResponse, null, 2)}</pre>
              </details>

              {uiBlocks.length ? (
                <TamboRegistryProvider components={components as any}>
                  <ChatSendContext.Provider value={onFormSend}>
                    {uiBlocks.map((block: UIBlock, idx: number) => {
                      const threadId = "local";
                      const messageId = `${lastResponse.run_id ?? "local"}:${lastResponse.step_id ?? "ui"}`;
                      const id = `block-${idx}`;
                      const content = toComponentContent(block, id);
                      const label =
                        typeof (block as any)?.title === "string"
                          ? String((block as any).title)
                          : typeof (block as any)?.component === "string"
                            ? String((block as any).component)
                            : typeof (block as any)?.type === "string"
                              ? String((block as any).type)
                              : "block";
                      return (
                        <div key={id} className="tm-block">
                          <div className="tm-block-title">{label}</div>
                          <TamboRenderContext.Provider value={{ threadId, messageId, idPrefix: id }}>
                            <ComponentRenderer content={content as any} threadId={threadId} messageId={messageId} />
                          </TamboRenderContext.Provider>
                        </div>
                      );
                    })}
                  </ChatSendContext.Provider>
                </TamboRegistryProvider>
              ) : null}
            </div>
          ) : null}
        </div>

        <form
          className="tm-form"
          onSubmit={(e) => {
            e.preventDefault();
            const msg = input.trim();
            if (!msg || sending) return;
            setInput("");
            void sendMessage(msg);
          }}
        >
          <textarea
            className="tm-input"
            placeholder="Type a message…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={sending}
          />
          <button className="tm-button" type="submit" disabled={sending}>
            Send
          </button>
        </form>
      </div>
    </>
  );
}
