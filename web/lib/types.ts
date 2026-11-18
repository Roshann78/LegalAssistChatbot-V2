export type Domain = "" | "legal" | "banking";

export interface Citation {
  index: number;
  doc_id: string;
  title: string;
  section_or_article: string | null;
  page: number | null;
  status: string;
  snippet: string;
}

export interface DoneInfo {
  conversation_id?: string;
  request_id?: string;
  finish_reason?: string;
  disclaimer?: string;
  latency_ms?: number;
}

export type AskEvent =
  | { type: "token"; content: string }
  | { type: "citations"; citations: Citation[]; invalid: number[]; answer: string; notes: string[] }
  | { type: "abstained"; message: string; conversation_id: string; request_id: string; disclaimer: string }
  | ({ type: "done" } & DoneInfo)
  | { type: "error"; code: string; message: string; request_id: string };

export interface AskBody {
  question: string;
  domain: Domain | null;
  conversation_id: string | null;
  upload_id: string | null;
}

export interface UploadInfo {
  upload_id: string;
  filename: string;
  pages: number;
  chunks: number;
  expires_in_seconds: number;
}

export interface Stats {
  documents: number;
  chunks: number;
}

export interface StageLatency {
  p50: number;
  p95: number;
  n: number;
}

export interface MetricsSnapshot {
  requests: number;
  answered: number;
  abstained: number;
  errors: number;
  error_rate: number;
  abstention_rate: number;
  approx_output_tokens_per_request: number;
  latency_ms: Record<string, StageLatency>;
}

/** One question/answer exchange, kept in the browser session only. */
export interface Turn {
  id: string;
  question: string;
  answer: string;
  citations: Citation[];
  notes: string[];
  abstained: boolean;
  error: string | null;
  requestId: string | null;
  latencyMs: number | null;
  status: "streaming" | "done";
  feedback: "up" | "down" | null;
}
