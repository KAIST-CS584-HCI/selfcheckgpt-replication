import { useEffect, useRef, useState } from "react";
import type { KeyboardEvent } from "react";

type Role = "user" | "assistant" | "error";
type Message = { role: Role; text: string };

// Bottom-docked chat that drives the local Claude Code CLI via /api/chat. The
// server streams the CLI's NDJSON over SSE; we render assistant text live and
// carry the session_id forward so the conversation is multi-turn.
export function ChatDock() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const sessionId = useRef<string | undefined>(undefined);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [messages, collapsed]);

  const send = async () => {
    const message = input.trim();
    if (!message || streaming) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text: message }, { role: "assistant", text: "" }]);
    setStreaming(true);
    try {
      await streamReply(message, sessionId, applyEvent);
    } catch (err) {
      applyEvent({ kind: "error", text: String(err) });
    } finally {
      setStreaming(false);
    }
  };

  // Folds a parsed SSE event into the message list. Assistant text deltas append
  // to the in-flight (last) message; errors are pushed as their own line.
  const applyEvent = (ev: ChatEvent) => {
    if (ev.kind === "session") sessionId.current = ev.id;
    else if (ev.kind === "delta") appendToLast(setMessages, ev.text);
    else if (ev.kind === "error") setMessages((m) => [...m, { role: "error", text: ev.text }]);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    e.stopPropagation(); // keep arrows from flipping slides while typing
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className={"chat-dock" + (collapsed ? " collapsed" : "")}>
      <button className="chat-head" onClick={() => setCollapsed((c) => !c)}>
        <span>Chat with Claude Code</span>
        <span className="chat-caret">{collapsed ? "▴" : "▾"}</span>
      </button>
      {!collapsed && (
        <>
          <div className="chat-log" ref={logRef}>
            {messages.map((m, i) => (
              <div key={i} className={"chat-msg chat-" + m.role}>
                {m.text || (streaming && i === messages.length - 1 ? "…" : "")}
              </div>
            ))}
          </div>
          <div className="chat-input-row">
            <textarea
              className="chat-input"
              placeholder="Ask Claude Code to edit slides, or anything…"
              value={input}
              disabled={streaming}
              rows={1}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
            />
            <button className="chat-send" onClick={send} disabled={streaming || !input.trim()}>
              {streaming ? "…" : "Send"}
            </button>
          </div>
        </>
      )}
    </div>
  );
}

type ChatEvent =
  | { kind: "session"; id: string }
  | { kind: "delta"; text: string }
  | { kind: "error"; text: string };

// POSTs the message and reads the SSE response, decoding each frame and handing
// the caller a normalized ChatEvent. Resolves when the stream ends.
async function streamReply(
  message: string,
  sessionId: React.MutableRefObject<string | undefined>,
  onEvent: (ev: ChatEvent) => void,
): Promise<void> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ message, sessionId: sessionId.current }),
  });
  if (!res.ok || !res.body) throw new Error(`Chat failed (${res.status}). Is the dev server running?`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let sep: number;
    while ((sep = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, sep);
      buf = buf.slice(sep + 2);
      const ev = parseFrame(frame);
      if (ev) onEvent(ev);
    }
  }
}

// Turns one SSE frame into a ChatEvent. `message`/`stderr` frames carry a line
// of the CLI's NDJSON; we pull session_id and assistant text deltas out of it.
function parseFrame(frame: string): ChatEvent | null {
  const event = matchField(frame, "event") ?? "message";
  const data = matchField(frame, "data") ?? "";
  if (event === "done") return null;
  if (event === "error") return { kind: "error", text: data || "Claude Code failed to start." };
  if (event === "stderr") return null; // diagnostics only; ignore in the UI
  return parseClaudeLine(data);
}

function parseClaudeLine(line: string): ChatEvent | null {
  let json: any;
  try {
    json = JSON.parse(line);
  } catch {
    return null;
  }
  if (json.type === "system" && json.subtype === "init" && json.session_id) {
    return { kind: "session", id: json.session_id };
  }
  if (json.type === "stream_event" && json.event?.type === "content_block_delta") {
    const delta = json.event.delta;
    if (delta?.type === "text_delta") return { kind: "delta", text: delta.text };
  }
  if (json.type === "result") {
    if (json.session_id) return { kind: "session", id: json.session_id };
  }
  return null;
}

function matchField(frame: string, field: string): string | undefined {
  const line = frame.split("\n").find((l) => l.startsWith(field + ": "));
  return line?.slice(field.length + 2);
}

function appendToLast(setMessages: React.Dispatch<React.SetStateAction<Message[]>>, text: string): void {
  setMessages((m) => {
    const last = m[m.length - 1];
    if (!last || last.role !== "assistant") return m;
    return [...m.slice(0, -1), { ...last, text: last.text + text }];
  });
}
