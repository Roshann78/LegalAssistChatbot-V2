"use client";

import { Fragment, useState } from "react";
import type { Citation, Turn } from "@/lib/types";
import { IconAlert, IconSpark, IconThumbDown, IconThumbUp } from "./Icons";

const MARKER = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

/** Render answer text, turning "[1]" / "[1, 2]" markers into jump links to the sources. */
function AnswerText({ text, count }: { text: string; count: number }) {
  const parts: React.ReactNode[] = [];
  let last = 0;
  for (const m of text.matchAll(MARKER)) {
    const at = m.index ?? 0;
    parts.push(<Fragment key={`t${at}`}>{text.slice(last, at)}</Fragment>);
    const nums = m[1].split(",").map((n) => parseInt(n.trim(), 10));
    parts.push(
      <sup key={`m${at}`} className="cite-marks">
        {nums.map((n) =>
          n >= 1 && n <= count ? (
            <a key={n} href={`#source-${n}`} aria-label={`Source ${n}`}>
              {n}
            </a>
          ) : null,
        )}
      </sup>,
    );
    last = at + m[0].length;
  }
  parts.push(<Fragment key="end">{text.slice(last)}</Fragment>);
  return <p className="answer-text">{parts}</p>;
}

function Source({ c }: { c: Citation }) {
  const [open, setOpen] = useState(false);
  const where = [c.section_or_article, c.page ? `p. ${c.page}` : null].filter(Boolean).join(" · ");
  return (
    <li id={`source-${c.index}`} className="source">
      <span className="source-index">{c.index}</span>
      <div>
        <strong>{c.title}</strong>
        {c.status !== "in_force" && <span className="tag tag-warn">{c.status.replace("_", " ")}</span>}
        {where && <small>{where}</small>}
        <button className="link-btn" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
          {open ? "Hide passage" : "Show passage"}
        </button>
        {open && <blockquote>{c.snippet}</blockquote>}
      </div>
    </li>
  );
}

interface Props {
  turn: Turn | null;
  onFeedback: (rating: "up" | "down") => void;
}

export function AnswerPreview({ turn, onFeedback }: Props) {
  if (!turn) {
    return (
      <div className="preview-empty">
        <span className="tile tile-violet">
          <IconSpark />
        </span>
        <h3>No answer yet</h3>
        <p className="muted">Ask a question and the answer, with the sources it relied on, appears here.</p>
      </div>
    );
  }

  const streaming = turn.status === "streaming";
  return (
    <article className="paper" aria-live="polite" aria-busy={streaming}>
      <header className="paper-head">
        <span className="logo small">
          <IconSpark />
        </span>
        <div>
          <p className="muted small">Question</p>
          <h3>{turn.question}</h3>
        </div>
      </header>

      <h4 className="section-label">Answer</h4>
      {turn.error ? (
        <p className="error" role="alert">
          <IconAlert /> {turn.error}
        </p>
      ) : turn.answer ? (
        turn.abstained ? (
          <div className="callout callout-warn">
            <IconAlert />
            <p>{turn.answer}</p>
          </div>
        ) : (
          <AnswerText text={turn.answer} count={turn.citations.length} />
        )
      ) : (
        <p className="muted">{streaming ? "Searching the documents…" : "No answer returned."}</p>
      )}
      {streaming && <span className="caret" aria-hidden="true" />}

      {turn.notes.length > 0 && (
        <div className="callout callout-info">
          <IconAlert />
          <div>
            {turn.notes.map((n) => (
              <p key={n}>{n}</p>
            ))}
          </div>
        </div>
      )}

      {turn.citations.length > 0 && (
        <>
          <h4 className="section-label">Sources</h4>
          <ol className="sources">
            {turn.citations.map((c) => (
              <Source key={c.index} c={c} />
            ))}
          </ol>
        </>
      )}

      <footer className="paper-foot">
        <p className="muted small">
          Informational only; not legal advice.
          {turn.latencyMs !== null && ` · ${turn.latencyMs} ms`}
          {turn.requestId && <span className="mono"> · {turn.requestId.slice(0, 8)}</span>}
        </p>
        {turn.requestId && !streaming && (
          <div className="feedback" role="group" aria-label="Rate this answer">
            <button
              className={`icon-btn${turn.feedback === "up" ? " on" : ""}`}
              onClick={() => onFeedback("up")}
              aria-pressed={turn.feedback === "up"}
              aria-label="Helpful"
            >
              <IconThumbUp />
            </button>
            <button
              className={`icon-btn${turn.feedback === "down" ? " on" : ""}`}
              onClick={() => onFeedback("down")}
              aria-pressed={turn.feedback === "down"}
              aria-label="Not helpful"
            >
              <IconThumbDown />
            </button>
          </div>
        )}
      </footer>
    </article>
  );
}
