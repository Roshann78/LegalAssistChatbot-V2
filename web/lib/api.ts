import type {
  AskBody,
  AskEvent,
  MetricsSnapshot,
  Stats,
  UploadInfo,
} from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
  }
}

async function failure(res: Response): Promise<ApiError> {
  let message = `Request failed (${res.status})`;
  try {
    const body = (await res.json()) as { message?: string };
    if (body.message) message = body.message;
  } catch {
    /* non-JSON error body */
  }
  return new ApiError(message, res.status);
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { cache: "no-store" });
  if (!res.ok) throw await failure(res);
  return (await res.json()) as T;
}

export const getStats = () => getJson<Stats>("/v1/stats");
export const getMetrics = () => getJson<MetricsSnapshot>("/v1/metrics");

export async function uploadDocument(file: File): Promise<UploadInfo> {
  const res = await fetch(`${API_URL}/v1/documents/upload?filename=${encodeURIComponent(file.name)}`, {
    method: "POST",
    headers: { "content-type": "application/octet-stream" },
    body: file,
  });
  if (!res.ok) throw await failure(res);
  return (await res.json()) as UploadInfo;
}

export async function deleteUpload(uploadId: string): Promise<void> {
  await fetch(`${API_URL}/v1/documents/session/${encodeURIComponent(uploadId)}`, { method: "DELETE" });
}

export async function sendFeedback(requestId: string, rating: "up" | "down"): Promise<void> {
  const res = await fetch(`${API_URL}/v1/feedback`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ request_id: requestId, rating }),
  });
  if (!res.ok) throw await failure(res);
}

/** Parse one SSE block ("event: x\ndata: {...}") into a typed event. */
export function parseSseBlock(block: string): AskEvent | null {
  let type = "";
  let data = "";
  for (const line of block.split("\n")) {
    if (line.startsWith("event: ")) type = line.slice(7).trim();
    else if (line.startsWith("data: ")) data += line.slice(6);
  }
  if (!type || !data) return null;
  return { type, ...(JSON.parse(data) as object) } as AskEvent;
}

/** POST /v1/ask and call onEvent for every streamed event. Resolves when the stream ends. */
export async function streamAsk(
  body: AskBody,
  onEvent: (e: AskEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_URL}/v1/ask`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) throw await failure(res);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut: number;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const event = parseSseBlock(buffer.slice(0, cut));
      buffer = buffer.slice(cut + 2);
      if (event) onEvent(event);
    }
  }
}
