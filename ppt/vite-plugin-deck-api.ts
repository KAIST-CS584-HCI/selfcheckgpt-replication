// Dev-only API for mutating deck.json from the preview UI.
// Server is authoritative: it reads, edits, and rewrites the file by id, so the
// client never ships a whole deck back. The file write triggers Vite HMR, which
// reloads the preview with the new deck.

import type { Plugin, Connect } from "vite";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { readFile, writeFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import type { IncomingMessage, ServerResponse } from "node:http";

const HERE = dirname(fileURLToPath(import.meta.url));
const DECK_PATH = resolve(HERE, "deck.json");
const CLAUDE_BIN = process.env.CLAUDE_BIN ?? "claude";

export function deckApi(): Plugin {
  return {
    name: "deck-api",
    configureServer(server) {
      server.middlewares.use("/api/slides/delete", handleDeleteSlide);
      server.middlewares.use("/api/slides/rename", handleRenameSlide);
      server.middlewares.use("/api/chat", handleChat);
    },
  };
}

async function handleDeleteSlide(req: IncomingMessage, res: ServerResponse, next: Connect.NextFunction) {
  if (req.method !== "POST") return next();
  try {
    const { id } = await readJsonBody(req);
    await deleteSlideById(id);
    sendJson(res, 200, { ok: true });
  } catch (err) {
    sendJson(res, 500, { error: String(err) });
  }
}

async function handleRenameSlide(req: IncomingMessage, res: ServerResponse, next: Connect.NextFunction) {
  if (req.method !== "POST") return next();
  try {
    const { id, label } = await readJsonBody(req);
    await renameSlideById(id, label);
    sendJson(res, 200, { ok: true });
  } catch (err) {
    sendJson(res, 500, { error: String(err) });
  }
}

// Chats with the local Claude Code CLI. Spawns `claude` headless in this dir as
// a full agent and streams its NDJSON output to the browser over SSE. Multi-turn
// continuity is the client's job: it echoes back the session_id it last saw.
async function handleChat(req: IncomingMessage, res: ServerResponse, next: Connect.NextFunction) {
  if (req.method !== "POST") return next();
  try {
    const { message, sessionId } = await readJsonBody(req);
    openEventStream(res);
    streamClaude(message, sessionId, req, res);
  } catch (err) {
    sendJson(res, 500, { error: String(err) });
  }
}

function streamClaude(message: string, sessionId: string | undefined, req: IncomingMessage, res: ServerResponse) {
  // stdin 'ignore': the prompt rides in via -p, so the CLI must not wait ~3s for
  // piped stdin before starting.
  const child = spawn(CLAUDE_BIN, claudeArgs(message, sessionId), {
    cwd: HERE,
    stdio: ["ignore", "pipe", "pipe"],
  });

  pipeLines(child.stdout, (line) => sendEvent(res, "message", line));
  pipeLines(child.stderr, (line) => sendEvent(res, "stderr", line));

  child.on("error", (err) => {
    sendEvent(res, "error", String(err));
    res.end();
  });
  child.on("close", () => {
    sendEvent(res, "done", "");
    res.end();
  });
  req.on("close", () => child.kill());
}

function claudeArgs(message: string, sessionId?: string): string[] {
  return [
    "-p", message,
    "--output-format", "stream-json",
    "--verbose",
    "--include-partial-messages",
    "--dangerously-skip-permissions",
    ...(sessionId ? ["--resume", sessionId] : []),
  ];
}

async function deleteSlideById(id: string): Promise<void> {
  const deck = JSON.parse(await readFile(DECK_PATH, "utf8"));
  deck.slides = deck.slides.filter((s: { id: string }) => s.id !== id);
  await writeFile(DECK_PATH, JSON.stringify(deck, null, 2) + "\n", "utf8");
}

// Sets the sidebar-only navLabel. A blank label, or one equal to the title,
// drops the field so the label falls back to the title.
async function renameSlideById(id: string, label: string): Promise<void> {
  const deck = JSON.parse(await readFile(DECK_PATH, "utf8"));
  const slide = deck.slides.find((s: { id: string }) => s.id === id);
  if (!slide) return;
  const trimmed = (label ?? "").trim();
  if (!trimmed || trimmed === slide.title) delete slide.navLabel;
  else slide.navLabel = trimmed;
  await writeFile(DECK_PATH, JSON.stringify(deck, null, 2) + "\n", "utf8");
}

function readJsonBody(req: IncomingMessage): Promise<any> {
  return new Promise((resolve, reject) => {
    let raw = "";
    req.on("data", (chunk) => (raw += chunk));
    req.on("end", () => {
      try {
        resolve(raw ? JSON.parse(raw) : {});
      } catch (e) {
        reject(e);
      }
    });
    req.on("error", reject);
  });
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  res.statusCode = status;
  res.setHeader("content-type", "application/json");
  res.end(JSON.stringify(body));
}

function openEventStream(res: ServerResponse): void {
  res.statusCode = 200;
  res.setHeader("content-type", "text/event-stream");
  res.setHeader("cache-control", "no-cache");
  res.setHeader("connection", "keep-alive");
  res.flushHeaders?.();
}

// One SSE frame per event. `data` is a single line (no embedded newlines from
// the CLI's NDJSON), so a one-line data payload is sufficient.
function sendEvent(res: ServerResponse, event: string, data: string): void {
  res.write(`event: ${event}\ndata: ${data}\n\n`);
}

// Buffers a child stream and invokes `onLine` once per complete `\n`-delimited
// line, flushing any trailing partial on stream end.
function pipeLines(stream: NodeJS.ReadableStream, onLine: (line: string) => void): void {
  let buf = "";
  stream.on("data", (chunk) => {
    buf += chunk;
    let nl: number;
    while ((nl = buf.indexOf("\n")) !== -1) {
      const line = buf.slice(0, nl);
      buf = buf.slice(nl + 1);
      if (line) onLine(line);
    }
  });
  stream.on("end", () => {
    if (buf) onLine(buf);
  });
}
