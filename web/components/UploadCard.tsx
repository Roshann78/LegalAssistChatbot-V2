"use client";

import { useRef, useState } from "react";
import { deleteUpload, uploadDocument } from "@/lib/api";
import { useApp } from "./AppContext";
import { Card } from "./Card";
import { IconFile, IconTrash, IconUpload } from "./Icons";

export function UploadCard() {
  const { upload, setUpload } = useApp();
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onPick(file: File | undefined) {
    if (!file) return;
    setError(null);
    setBusy(true);
    try {
      if (upload) await deleteUpload(upload.upload_id).catch(() => undefined);
      setUpload(await uploadDocument(file));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setBusy(false);
      if (input.current) input.current.value = "";
    }
  }

  async function remove() {
    if (!upload) return;
    await deleteUpload(upload.upload_id).catch(() => undefined);
    setUpload(null);
  }

  return (
    <Card
      icon={<IconFile />}
      tone="green"
      title="Chat with a document"
      subtitle="Upload a PDF or text file to ask questions about just that file."
      action={
        <>
          <input
            ref={input}
            type="file"
            accept=".pdf,.txt"
            hidden
            onChange={(e) => onPick(e.target.files?.[0])}
            aria-label="Choose a document to upload"
          />
          <button className="btn btn-outline" onClick={() => input.current?.click()} disabled={busy}>
            <IconUpload /> {busy ? "Uploading…" : upload ? "Replace" : "Import file"}
          </button>
        </>
      }
    >
      <div className="inset">
        {upload ? (
          <div className="file-chip">
            <IconFile />
            <div>
              <strong>{upload.filename}</strong>
              <small>
                {upload.pages} page{upload.pages === 1 ? "" : "s"} · {upload.chunks} passages · questions use only this
                file
              </small>
            </div>
            <button className="icon-btn" onClick={remove} aria-label="Remove uploaded document">
              <IconTrash />
            </button>
          </div>
        ) : (
          <p className="muted">
            No document attached; questions use the main corpus. Limits: .pdf or .txt, 5&nbsp;MB, 100 pages. Files are
            deleted after 30 minutes. Scanned PDFs are not supported.
          </p>
        )}
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
      </div>
    </Card>
  );
}
