"use client";

import { useEffect, useRef, useState } from "react";
import { AnswerPreview } from "@/components/AnswerPreview";
import { useApp } from "@/components/AppContext";
import { Card } from "@/components/Card";
import { IconChat, IconChevronRight } from "@/components/Icons";
import { UploadCard } from "@/components/UploadCard";
import { API_URL, sendFeedback, streamAsk } from "@/lib/api";
import type { AskEvent, Domain, Turn } from "@/lib/types";

const MAX_QUESTION = 1000; // mirrors the API's default MAX_QUESTION_LENGTH
const DOMAIN_CHIPS: { value: Domain; label: string }[] = [
  { value: "", label: "Any" },
  { value: "legal", label: "Legal" },
  { value: "banking", label: "Banking" },
];

export default function AskPage() {
  const { domain, setDomain, upload, conversationId, setConversationId, turns, setTurns, selectedId, setSelectedId } =
    useApp();
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [previewOpen, setPreviewOpen] = useState(true);
  const abort = useRef<AbortController | null>(null);
  const box = useRef<HTMLTextAreaElement>(null);

  useEffect(() => () => abort.current?.abort(), []);

  const shown = turns.find((t) => t.id === selectedId) ?? turns[turns.length - 1] ?? null;

  const patch = (id: string, fn: (t: Turn) => Turn) =>
    setTurns((all) => all.map((t) => (t.id === id ? fn(t) : t)));

  function apply(id: string, e: AskEvent) {
    switch (e.type) {
      case "token":
        patch(id, (t) => ({ ...t, answer: t.answer + e.content }));
        break;
      case "citations":
        // The sanitized answer replaces the streamed text (invalid markers removed).
        patch(id, (t) => ({ ...t, answer: e.answer, citations: e.citations, notes: e.notes }));
        break;
      case "abstained":
        setConversationId(e.conversation_id);
        patch(id, (t) => ({ ...t, answer: e.message, abstained: true, requestId: e.request_id }));
        break;
      case "done":
        if (e.conversation_id) setConversationId(e.conversation_id);
        patch(id, (t) => ({
          ...t,
          requestId: e.request_id ?? t.requestId,
          latencyMs: e.latency_ms ?? t.latencyMs,
        }));
        break;
      case "error":
        patch(id, (t) => ({ ...t, error: `${e.message} (request ${e.request_id.slice(0, 8)})`, requestId: e.request_id }));
        break;
    }
  }

  async function ask() {
    const q = question.trim();
    if (!q || busy) return;
    const id = crypto.randomUUID();
    const turn: Turn = {
      id,
      question: q,
      answer: "",
      citations: [],
      notes: [],
      abstained: false,
      error: null,
      requestId: null,
      latencyMs: null,
      status: "streaming",
      feedback: null,
    };
    setTurns((all) => [...all, turn]);
    setSelectedId(id);
    setPreviewOpen(true);
    setQuestion("");
    setBusy(true);
    abort.current = new AbortController();
    try {
      await streamAsk(
        { question: q, domain: upload ? null : domain || null, conversation_id: conversationId, upload_id: upload?.upload_id ?? null },
        (e) => apply(id, e),
        abort.current.signal,
      );
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        const message =
          err instanceof TypeError ? `Could not reach the API at ${API_URL}. Is it running, and does CORS allow this site?` : (err as Error).message;
        patch(id, (t) => ({ ...t, error: message }));
      }
    } finally {
      patch(id, (t) => ({ ...t, status: "done" }));
      setBusy(false);
      box.current?.focus();
    }
  }

  async function rate(rating: "up" | "down") {
    if (!shown?.requestId) return;
    const id = shown.id;
    const previous = shown.feedback;
    patch(id, (t) => ({ ...t, feedback: rating }));
    try {
      await sendFeedback(shown.requestId, rating);
    } catch {
      patch(id, (t) => ({ ...t, feedback: previous }));
    }
  }

  return (
    <>
      <main className="main">
        <header className="page-head">
          <h1>Ask a question</h1>
          <p>Ask about Indian law or banking regulation. Every answer shows the sources it used.</p>
        </header>

        <Card
          icon={<IconChat />}
          tone="blue"
          title="Your question"
          subtitle={upload ? `Using the uploaded file “${upload.filename}”` : "Searching the main corpus"}
        >
          <div className="inset">
            <label className="field">
              <span>
                Question <em>*</em>
              </span>
              <textarea
                ref={box}
                value={question}
                maxLength={MAX_QUESTION}
                rows={4}
                placeholder="e.g. What is the punishment for theft?"
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) ask();
                }}
              />
            </label>
            <div className="row-between">
              <div className="chips" role="radiogroup" aria-label="Domain" aria-disabled={!!upload}>
                {DOMAIN_CHIPS.map((c) => (
                  <button
                    key={c.label}
                    role="radio"
                    aria-checked={domain === c.value}
                    className={`chip${domain === c.value ? " on" : ""}`}
                    disabled={!!upload}
                    onClick={() => setDomain(c.value)}
                  >
                    {c.label}
                  </button>
                ))}
              </div>
              <div className="row-gap">
                <small className="muted">{question.length}/{MAX_QUESTION} · Ctrl+Enter</small>
                <button className="btn btn-primary" onClick={ask} disabled={busy || !question.trim()}>
                  {busy ? "Answering…" : "Ask"}
                </button>
              </div>
            </div>
            {upload && <p className="muted small">Domain filters apply to the main corpus, not to uploaded files.</p>}
          </div>
        </Card>

        <UploadCard />

        <p className="muted small footnote">
          Informational only; not legal advice. Answers can be incomplete or wrong. Check the cited sources.
        </p>
      </main>

      <aside className={`preview${previewOpen ? "" : " closed"}`}>
        <button
          className="preview-toggle"
          onClick={() => setPreviewOpen((v) => !v)}
          aria-label={previewOpen ? "Hide answer panel" : "Show answer panel"}
          aria-expanded={previewOpen}
        >
          <IconChevronRight style={{ transform: previewOpen ? "none" : "rotate(180deg)" }} />
        </button>
        {previewOpen && (
          <div className="preview-scroll">
            <header className="page-head">
              <h1>Answer</h1>
              <p>See exactly what was returned, with its sources.</p>
            </header>
            <AnswerPreview turn={shown} onFeedback={rate} />
          </div>
        )}
      </aside>
    </>
  );
}
