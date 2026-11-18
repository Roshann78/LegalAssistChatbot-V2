"use client";

import { useEffect, useState } from "react";
import { Card } from "@/components/Card";
import { IconBook } from "@/components/Icons";
import { UploadCard } from "@/components/UploadCard";
import { API_URL, getStats } from "@/lib/api";
import type { Stats } from "@/lib/types";

export default function DocumentsPage() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getStats()
      .then(setStats)
      .catch((e) =>
        setError(e instanceof TypeError ? `Could not reach the API at ${API_URL}.` : (e as Error).message),
      );
  }, []);

  return (
    <main className="main wide">
      <header className="page-head">
        <h1>Documents</h1>
        <p>The corpus the assistant searches, and your own temporary uploads.</p>
      </header>

      <Card icon={<IconBook />} tone="blue" title="Main corpus" subtitle="Statutes, judgments, RBI directions and schemes">
        {error ? (
          <p className="error" role="alert">
            {error}
          </p>
        ) : (
          <div className="stat-grid two">
            <div className="stat">
              <small>Documents</small>
              <strong>{stats ? stats.documents : "—"}</strong>
            </div>
            <div className="stat">
              <small>Indexed passages</small>
              <strong>{stats ? stats.chunks : "—"}</strong>
            </div>
          </div>
        )}
        <p className="muted small spaced">
          The corpus is managed on the server with the ingestion CLI; it cannot be changed from this page.
        </p>
      </Card>

      <UploadCard />
    </main>
  );
}
