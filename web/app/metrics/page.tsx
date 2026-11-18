"use client";

import { useCallback, useEffect, useState } from "react";
import { Card } from "@/components/Card";
import { IconChart } from "@/components/Icons";
import { API_URL, getMetrics } from "@/lib/api";
import type { MetricsSnapshot } from "@/lib/types";

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
const STAGE_LABELS: Record<string, string> = {
  understand_ms: "Query understanding",
  retrieve_ms: "Retrieval",
  rerank_ms: "Reranking",
  generate_ms: "Generation",
  total_ms: "Total",
};

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="stat">
      <small>{label}</small>
      <strong>{value}</strong>
      {hint && <small className="muted">{hint}</small>}
    </div>
  );
}

export default function MetricsPage() {
  const [m, setM] = useState<MetricsSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setM(await getMetrics());
      setError(null);
    } catch (e) {
      setError(e instanceof TypeError ? `Could not reach the API at ${API_URL}.` : (e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 10_000);
    return () => clearInterval(t);
  }, [load]);

  const stages = m ? Object.entries(m.latency_ms) : [];

  return (
    <main className="main wide">
      <header className="page-head">
        <h1>Metrics</h1>
        <p>Live operational numbers for this API process. They reset when the server restarts.</p>
      </header>

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      <Card
        icon={<IconChart />}
        tone="violet"
        title="Requests"
        subtitle="Since the server started"
        action={
          <button className="btn btn-outline" onClick={load} disabled={loading}>
            {loading ? "Refreshing…" : "Refresh"}
          </button>
        }
      >
        <div className="stat-grid">
          <Stat label="Requests" value={m ? String(m.requests) : "—"} />
          <Stat label="Answered" value={m ? String(m.answered) : "—"} />
          <Stat label="Abstained" value={m ? String(m.abstained) : "—"} hint={m ? pct(m.abstention_rate) : undefined} />
          <Stat label="Errors" value={m ? String(m.errors) : "—"} hint={m ? pct(m.error_rate) : undefined} />
        </div>
      </Card>

      <Card icon={<IconChart />} tone="amber" title="Latency by stage" subtitle="Median and 95th percentile, most recent 1000 requests">
        {stages.length === 0 ? (
          <p className="muted">No requests recorded yet. Ask a question first.</p>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Stage</th>
                <th>p50 (ms)</th>
                <th>p95 (ms)</th>
                <th>Samples</th>
              </tr>
            </thead>
            <tbody>
              {stages.map(([k, v]) => (
                <tr key={k}>
                  <td>{STAGE_LABELS[k] ?? k}</td>
                  <td>{v.p50.toFixed(1)}</td>
                  <td>{v.p95.toFixed(1)}</td>
                  <td>{v.n}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </main>
  );
}
