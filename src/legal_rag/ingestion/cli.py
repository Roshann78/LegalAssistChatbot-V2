"""Ingestion CLI: validate, draft, add, update, remove, list, reindex, verify, stats."""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import yaml

from legal_rag.config.settings import Settings
from legal_rag.ingestion.chunking import chunk_document, strategy_for
from legal_rag.ingestion.indexer import Indexer
from legal_rag.ingestion.manifest import draft_entries, load_manifest, validate_manifest
from legal_rag.ingestion.parser import clean_pages, parse_file
from legal_rag.retrieval.embedder import build_embedder
from legal_rag.retrieval.memory_store import FingerprintMismatchError, InMemoryVectorStore


def _table(rows: list[tuple[str, ...]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) for r in rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="legal-rag-ingest")
    p.add_argument("--dry-run", action="store_true", help="report what would change; write nothing")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    sub.add_parser("draft")
    sub.add_parser("list")
    sub.add_parser("verify")
    sub.add_parser("stats")
    for name in ("add", "update", "remove"):
        sub.add_parser(name).add_argument("doc_id")
    r = sub.add_parser("reindex")
    r.add_argument("--all", action="store_true", required=True)
    return p


def run(argv: list[str], settings: Settings | None = None) -> int:
    args = build_parser().parse_args(argv)
    s = settings or Settings()
    records = load_manifest(s.manifest_path) if s.manifest_path.exists() else []
    by_id = {r.doc_id: r for r in records}
    embedder = build_embedder(s.embedding_model)
    store = InMemoryVectorStore.load(s.index_path)
    idx = Indexer(s, embedder, store, base_dir=s.data_dir)

    def save() -> None:
        if not args.dry_run:
            store.save(s.index_path)

    if args.cmd == "validate":
        errs = validate_manifest(records, s.data_dir)
        print("\n".join(errs) if errs else f"OK: {len(records)} documents valid")
        return 1 if errs else 0
    if args.cmd == "draft":
        known = set(by_id)
        print(yaml.safe_dump(draft_entries(s.data_dir / "raw", known), sort_keys=False))
        return 0
    if args.cmd == "list":
        rows = [("doc_id", "type", "status", "indexed")]
        indexed = set(store.list_docs())
        rows += [
            (r.doc_id, r.doc_type.value, r.status.value, str(r.doc_id in indexed)) for r in records
        ]
        print(_table(rows))
        return 0
    if args.cmd == "verify":
        rep = idx.verify(records)
        for k, v in rep.items():
            print(f"{k}: {v or 'none'}")
        return 1 if any(rep.values()) else 0
    if args.cmd == "stats":
        rows = [("doc_id", "chunks", "avg_chars", "max_chars")]
        for r in records:
            pages = clean_pages(parse_file(s.data_dir / r.path))
            ch = chunk_document(
                {"doc_id": r.doc_id, "title": r.title},
                pages,
                strategy_for(r.doc_type.value, s.chunk_strategy),
                size=s.chunk_size,
                overlap=s.chunk_overlap,
            )
            sizes = [len(c.text) for c in ch] or [0]
            rows.append((r.doc_id, str(len(ch)), str(sum(sizes) // len(sizes)), str(max(sizes))))
        print(_table(rows))
        return 0

    if (
        args.cmd in ("add", "update", "remove")
        and args.doc_id not in by_id
        and args.cmd != "remove"
    ):
        print(f"unknown doc_id {args.doc_id!r}: add it to the manifest first", file=sys.stderr)
        return 2
    try:
        if args.cmd == "remove":
            n = store.delete_by_doc_id(args.doc_id) if not args.dry_run else 0
            print(f"removed {n} chunks for {args.doc_id}")
        elif args.dry_run:
            stale = [
                r.doc_id
                for r in records
                if store.doc_hashes().get(r.doc_id) != (r.file_hash or "x")
            ]
            print(f"dry-run: would (re)process up to {len(stale)} docs: {stale}")
        elif args.cmd == "reindex":
            store.clear()
            store.fingerprint = None
            summary = idx.index_all(records, force=True)
            print(
                f"indexed={len(summary.indexed)} chunks={summary.chunks} seconds={summary.seconds}"
            )
        else:
            force = args.cmd == "update"
            outcome = idx.index_document(by_id[args.doc_id], force=force)
            print(f"{args.doc_id}: {outcome}")
    except FingerprintMismatchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    save()
    kinds = Counter(c.metadata.get("chunk_type") for c in store._chunks.values())
    print(f"index: {store.count()} chunks, {len(store.list_docs())} docs, types={dict(kinds)}")
    return 0


def main() -> None:
    sys.exit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
