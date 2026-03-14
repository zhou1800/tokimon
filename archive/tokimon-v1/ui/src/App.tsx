import { ComponentRenderer, TamboRegistryProvider } from "@tambo-ai/react";
import { startTransition, useEffect, useMemo, useRef, useState } from "react";

import { getConversation, listConversations, postSend } from "./api";
import type { ChatConversationSummary, ChatMessage, ConversationLogEntry, StepResult, UIBlock } from "./types";
import { components } from "./tambo/registry";
import { TamboRenderContext } from "./tambo/renderContext";
import { ChatSendContext } from "./tambo/sendContext";
import { toComponentContent } from "./tambo/toComponentBlock";

const MODEL_PRESETS = [
  "gpt-5.4",
  "gpt-5.3-codex",
  "gpt-5.3-codex-spark",
  "gpt-5.2-codex",
  "gpt-5.2",
  "gpt-5.1-codex-max",
  "gpt-5.1-codex-mini",
] as const;

const DEFAULT_MODEL = "gpt-5.4";
const LEGACY_DEFAULT_MODELS = new Set(["gpt-5.3-codex", "gpt-5.2"]);
const THREAD_STORAGE_KEY = "tokimon.threadId";

function toHistory(messages: ConversationLogEntry[]): ChatMessage[] {
  return messages.map((message) => ({ role: message.role, content: message.content }));
}

function readStoredThreadId(): string | null {
  try {
    const stored = localStorage.getItem(THREAD_STORAGE_KEY);
    const normalized = (stored ?? "").trim();
    return normalized || null;
  } catch {
    return null;
  }
}

function formatTimestamp(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

export function App() {
  const [input, setInput] = useState("");
  const [history, setHistory] = useState<ChatMessage[]>([]);
  const [log, setLog] = useState<ConversationLogEntry[]>([]);
  const [conversations, setConversations] = useState<ChatConversationSummary[]>([]);
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null);
  const [loadingThreadId, setLoadingThreadId] = useState<string | null>(null);
  const [sidebarError, setSidebarError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [bootstrapping, setBootstrapping] = useState(true);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const [model, setModel] = useState(() => {
    try {
      const stored = localStorage.getItem("tokimon.model");
      const normalized = (stored ?? "").trim();
      if (!normalized || LEGACY_DEFAULT_MODELS.has(normalized)) {
        return DEFAULT_MODEL;
      }
      return normalized;
    } catch {
      return DEFAULT_MODEL;
    }
  });
  const [lastStepResult, setLastStepResult] = useState<StepResult | null>(null);
  const [lastStepId, setLastStepId] = useState<string | null>(null);
  const modelPreset = MODEL_PRESETS.includes(model as (typeof MODEL_PRESETS)[number]) ? model : "__custom__";
  const isCustomModel = modelPreset === "__custom__";
  const activeConversation = conversations.find((item) => item.thread_id === currentThreadId) ?? null;

  useEffect(() => {
    try {
      localStorage.setItem("tokimon.model", model);
    } catch {
      return;
    }
  }, [model]);

  useEffect(() => {
    try {
      if (currentThreadId) {
        localStorage.setItem(THREAD_STORAGE_KEY, currentThreadId);
      } else {
        localStorage.removeItem(THREAD_STORAGE_KEY);
      }
    } catch {
      return;
    }
  }, [currentThreadId]);

  useEffect(() => {
    if (!sending) {
      inputRef.current?.focus();
    }
  }, [sending, currentThreadId]);

  const uiBlocks = useMemo(() => {
    return Array.isArray(lastStepResult?.ui_blocks) ? lastStepResult.ui_blocks : [];
  }, [lastStepResult]);

  async function refreshConversations(): Promise<ChatConversationSummary[]> {
    const summaries = await listConversations();
    startTransition(() => {
      setConversations(summaries);
      setSidebarError(null);
    });
    return summaries;
  }

  async function loadConversation(threadId: string, options: { silent?: boolean } = {}): Promise<void> {
    const { silent = false } = options;
    if (!silent) {
      setLoadingThreadId(threadId);
    }
    try {
      const conversation = await getConversation(threadId);
      startTransition(() => {
        setCurrentThreadId(conversation.thread_id);
        setHistory(toHistory(conversation.messages));
        setLog(conversation.messages);
        setLastStepResult(conversation.last_step_result ?? null);
        setLastStepId(conversation.last_step_id ?? null);
        if (typeof conversation.model === "string" && conversation.model.trim()) {
          setModel(conversation.model.trim());
        }
        setSidebarError(null);
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setSidebarError(msg);
    } finally {
      setLoadingThreadId(null);
      setBootstrapping(false);
    }
  }

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      try {
        const summaries = await listConversations();
        if (cancelled) {
          return;
        }
        startTransition(() => {
          setConversations(summaries);
          setSidebarError(null);
        });
        let preferredThreadId = readStoredThreadId();
        if (preferredThreadId && !summaries.some((item) => item.thread_id === preferredThreadId)) {
          preferredThreadId = null;
        }
        if (!preferredThreadId && summaries[0]) {
          preferredThreadId = summaries[0].thread_id;
        }
        if (preferredThreadId) {
          await loadConversation(preferredThreadId, { silent: true });
          return;
        }
        startTransition(() => {
          setCurrentThreadId(null);
          setHistory([]);
          setLog([]);
          setLastStepResult(null);
          setLastStepId(null);
        });
      } catch (err) {
        if (!cancelled) {
          const msg = err instanceof Error ? err.message : String(err);
          setSidebarError(msg);
        }
      } finally {
        if (!cancelled) {
          setBootstrapping(false);
        }
      }
    }

    void bootstrap();
    return () => {
      cancelled = true;
    };
  }, []);

  async function sendMessage(message: string) {
    setSending(true);
    setSidebarError(null);
    setLog((items) => [...items, { role: "user", content: message }]);
    const nextHistory = [...history, { role: "user", content: message } satisfies ChatMessage];
    setHistory(nextHistory);

    try {
      const res = await postSend(message, nextHistory, model, currentThreadId ?? undefined);
      const reply = typeof res.reply === "string" ? res.reply : typeof res.summary === "string" ? res.summary : "";
      const meta = typeof res.status === "string" ? `status=${res.status}` : undefined;
      const nextThreadId = typeof res.thread_id === "string" && res.thread_id ? res.thread_id : currentThreadId;

      setLog((items) => [
        ...items,
        {
          role: "assistant",
          content: reply || "(no reply)",
          meta,
          ...(typeof res.status === "string" && res.status !== "SUCCESS" ? { error: true } : {}),
          ...(typeof res.step_id === "string" && res.step_id ? { step_id: res.step_id } : {}),
        },
      ]);
      setHistory((items) => [...items, { role: "assistant", content: reply || "" }]);
      startTransition(() => {
        setCurrentThreadId(nextThreadId ?? null);
        setLastStepResult(res.step_result ?? null);
        setLastStepId(typeof res.step_id === "string" ? res.step_id : null);
      });

      try {
        await refreshConversations();
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setSidebarError(msg);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setLog((items) => [...items, { role: "assistant", content: `Error: ${msg}`, meta: "request failed", error: true }]);
    } finally {
      setSending(false);
    }
  }

  const onFormSend = async (message: string) => {
    await sendMessage(message);
  };

  const submitInput = () => {
    const msg = input.trim();
    if (!msg || sending) return;
    setInput("");
    void sendMessage(msg);
  };

  const startNewChat = () => {
    startTransition(() => {
      setCurrentThreadId(null);
      setHistory([]);
      setLog([]);
      setLastStepResult(null);
      setLastStepId(null);
      setSidebarError(null);
    });
  };

  return (
    <div className="tm-shell">
      <aside className="tm-sidebar">
        <div className="tm-sidebar-head">
          <div>
            <div className="tm-eyebrow">Tokimon</div>
            <h1 className="tm-sidebar-title">Chat history</h1>
          </div>
          <button className="tm-new-chat" type="button" onClick={startNewChat} disabled={sending}>
            New chat
          </button>
        </div>

        {sidebarError ? <div className="tm-sidebar-error">{sidebarError}</div> : null}

        <div className="tm-history-list">
          {conversations.length ? (
            conversations.map((conversation) => {
              const isActive = conversation.thread_id === currentThreadId;
              const isLoading = loadingThreadId === conversation.thread_id;
              return (
                <button
                  key={conversation.thread_id}
                  type="button"
                  className={`tm-history-item ${isActive ? "tm-history-item-active" : ""}`}
                  onClick={() => void loadConversation(conversation.thread_id)}
                  disabled={sending || isLoading}
                >
                  <div className="tm-history-item-top">
                    <span className="tm-history-item-title">{conversation.title}</span>
                    <span className="tm-history-item-time">{formatTimestamp(conversation.updated_at)}</span>
                  </div>
                  <div className="tm-history-item-preview">{conversation.preview}</div>
                  <div className="tm-history-item-meta">
                    {conversation.message_count} messages
                    {conversation.model ? ` | ${conversation.model}` : ""}
                    {isLoading ? " | loading" : ""}
                  </div>
                </button>
              );
            })
          ) : (
            <div className="tm-history-empty">
              Saved conversations will appear here after you send a message.
            </div>
          )}
        </div>
      </aside>

      <main className="tm-main">
        <header className="tm-header">
          <div>
            <div className="tm-eyebrow">Local chat UI</div>
            <div className="tm-header-title">{activeConversation?.title ?? (log.length ? "Current chat" : "New chat")}</div>
          </div>
          <div className="tm-header-meta">
            {activeConversation ? `Updated ${formatTimestamp(activeConversation.updated_at)}` : "Stored under runs/chat-ui"}
          </div>
        </header>

        <div className="tm-log">
          {bootstrapping ? (
            <div className="tm-empty-state">Loading saved conversations...</div>
          ) : log.length ? (
            <>
              {log.map((entry, index) => (
                <div
                  key={`${entry.step_id ?? "msg"}-${index}`}
                  className={`tm-msg ${entry.role === "user" ? "tm-user" : "tm-assistant"} ${entry.error ? "tm-error" : ""}`}
                >
                  <div className="tm-msg-label">{entry.role === "user" ? "You" : "Tokimon"}</div>
                  <div className="tm-msg-body">{entry.content}</div>
                  {entry.meta ? <div className="tm-meta">{entry.meta}</div> : null}
                </div>
              ))}

              {lastStepResult ? (
                <div className="tm-pane">
                  <details className="tm-structured">
                    <summary>Latest structured result</summary>
                    <pre className="tm-pre">{JSON.stringify(lastStepResult, null, 2)}</pre>
                  </details>

                  {uiBlocks.length ? (
                    <TamboRegistryProvider components={components as any}>
                      <ChatSendContext.Provider value={onFormSend}>
                        {uiBlocks.map((block: UIBlock, index: number) => {
                          const threadId = currentThreadId ?? "local";
                          const messageId = `${threadId}:${lastStepId ?? "ui"}`;
                          const id = `block-${index}`;
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
            </>
          ) : (
            <div className="tm-empty-state">
              Start a conversation. Each thread is persisted on disk and restored into the sidebar on reload.
            </div>
          )}
        </div>

        <form
          className="tm-form"
          onSubmit={(e) => {
            e.preventDefault();
            submitInput();
          }}
        >
          <div className="tm-controls">
            <label className="tm-control">
              Model
              <select
                className="tm-model"
                value={modelPreset}
                onChange={(e) => {
                  const next = e.target.value;
                  if (next === "__custom__") {
                    setModel("");
                    return;
                  }
                  setModel(next);
                }}
                disabled={sending}
              >
                <option value="__custom__">Custom...</option>
                {MODEL_PRESETS.map((preset) => (
                  <option key={preset} value={preset}>
                    {preset}
                  </option>
                ))}
              </select>
            </label>
            {isCustomModel ? (
              <label className="tm-control">
                Custom model
                <input
                  className="tm-model"
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  placeholder="e.g. gpt-5.4"
                  disabled={sending}
                />
              </label>
            ) : null}
          </div>

          <textarea
            ref={inputRef}
            className="tm-input"
            placeholder="Message Tokimon..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key !== "Enter" || e.shiftKey || e.nativeEvent.isComposing) return;
              e.preventDefault();
              submitInput();
            }}
            disabled={sending}
            autoFocus
          />
          <button className="tm-button" type="submit" disabled={sending}>
            {sending ? "Sending..." : "Send"}
          </button>
        </form>
      </main>
    </div>
  );
}
