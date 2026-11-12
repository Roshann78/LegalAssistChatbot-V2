"""Ingestion CLI (validate, draft, parse, add, update, remove, list, reindex, verify, ...)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from legal_rag.config.settings import Settings
from legal_rag.ingestion.cache import EmbeddingCache
from legal_rag.ingestion.chunking import chunk_document, strategy_for
from legal_rag.ingestion.indexer import Indexer
from legal_rag.ingestion.manifest import (
    draft_entries,
    load_manifest,
    sha256_file,
    validate_manifest,
)
from legal_rag.ingestion.normalize import ensure_normalized
from legal_rag.ingestion.report import lookup_section, snippet, write_report
from legal_rag.retrieval.embedder import build_embedder
from legal_rag.retrieval.factory import build_store
from legal_rag.retrieval.memory_store import FingerprintMismatchError


def _table(rows: list[tuple[str, ...]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) for r in rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="legal-rag-ingest")
    p.add_argument("--dry-run", action="store_true", help="report what would change; write nothing")
    p.add_argument("--yes", action="store_true", help="skip confirmation prompts")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("validate", "draft", "list", "verify", "status", "stats"):
        sub.add_parser(name)
    for name in ("add", "update", "remove"):
        sub.add_parser(name).add_argument("doc_id")
    sp = sub.add_parser("parse")
    sp.add_argument("doc_id", nargs="?")
    sp.add_argument("--all", action="store_true")
    sub.add_parser("reindex").add_argument("--all", action="store_true", required=True)
    rp = sub.add_parser("report", help="verify, then write the ingestion report")
    rp.add_argument("--out", default="docs/ingestion_report.md")
    lk = sub.add_parser("lookup", help="find the chunk for a section by metadata only")
    lk.add_argument("doc_id")
    lk.add_argument("section")
    return p


def _confirm(msg: str, yes: bool) -> bool:
    if yes:
        return True
    if not sys.stdin.isatty():
        print(f"{msg} -- refusing without --yes in a non-interactive session", file=sys.stderr)
        return False
    return input(f"{msg} [y/N] ").strip().lower() == "y"


def run(argv: list[str], settings: Settings | None = None) -> int:
    args = build_parser().parse_args(argv)
    s = settings or Settings()
    records = load_manifest(s.manifest_path) if s.manifest_path.exists() else []
    by_id = {r.doc_id: r for r in records}

    if args.cmd == "validate":
        errs = validate_manifest(records, s.data_dir)
        print("\n".join(errs) if errs else f"OK: {len(records)} documents valid")
        return 1 if errs else 0
    if args.cmd == "draft":
        print(yaml.safe_dump(draft_entries(s.data_dir / "raw", set(by_id)), sort_keys=False))
        return 0

    embedder = build_embedder(s.embedding_model)
    store = build_store(s, embedder)
    cache = EmbeddingCache(s.embedding_cache_path, embedder.name)
    idx = Indexer(s, embedder, store, base_dir=s.data_dir, cache=cache)

    def need(doc_id: str) -> bool:
        if doc_id not in by_id:
            print(f"unknown doc_id {doc_id!r}: add it to the manifest first", file=sys.stderr)
            return False
        return True

    try:
        return _dispatch(args, s, records, by_id, idx, store, need)
    except FingerprintMismatchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    except FileNotFoundError as exc:
        print(f"ERROR: source file missing: {exc.filename}", file=sys.stderr)
        return 2
    finally:
        cache.close()


def _dispatch(args, s, records, by_id, idx: Indexer, store, need) -> int:
    cmd = args.cmd
    if cmd == "list":
        indexed = set(store.list_docs())
        rows = [("doc_id", "type", "status", "indexed")]
        rows += [
            (r.doc_id, r.doc_type.value, r.status.value, str(r.doc_id in indexed)) for r in records
        ]
        print(_table(rows))
        return 0
    if cmd == "status":
        print(
            f"documents in manifest: {len(records)}\ndocuments indexed:     "
            f"{len(store.list_docs())}\nchunks:                {store.count()}\n"
            f"index fingerprint:     {store.fingerprint or '-'}\n"
            f"config fingerprint:    {idx.fingerprint}\nchunk types:           "
            f"{dict(idx.chunk_type_counts())}"
        )
        return 0
    if cmd == "verify":
        report = idx.verify(records)
        print(
            _table(
                [
                    ("check", "count", "doc_ids"),
                    *[(k, str(len(v)), ", ".join(v) or "-") for k, v in report.items()],
                ]
            )
        )
        return 1 if any(report.values()) else 0
    if cmd == "stats":
        rows = [("doc_id", "chunks", "avg_tokens", "max_tokens")]
        for r in records:
            norm, _ = ensure_normalized(
                r.doc_id, s.data_dir / r.path, s.processed_dir, s.low_text_chars
            )
            ch = chunk_document(
                {"doc_id": r.doc_id, "title": r.title},
                norm.to_pages(),
                strategy_for(r.doc_type.value, s.chunk_strategy),
                size=s.chunk_size,
                overlap=s.chunk_overlap,
                max_tokens=s.chunk_max_tokens,
            )
            tok = [c.metadata["token_count"] for c in ch] or [0]
            rows.append((r.doc_id, str(len(ch)), str(sum(tok) // len(tok)), str(max(tok))))
        print(_table(rows))
        return 0
    if cmd == "lookup":
        ids = lookup_section(idx, args.doc_id, args.section)
        for cid in ids:
            print(f"{cid}: {snippet(idx, cid, args.doc_id)}")
        if len(ids) != 1:
            print(f"expected exactly 1 chunk, found {len(ids)}", file=sys.stderr)
        return 0 if len(ids) == 1 else 1
    if cmd == "report":
        summary = idx.index_all(records)
        store.persist()
        text = write_report(Path(args.out), idx, records, summary)
        problems = idx.verify(records)
        print(text)
        return 1 if any(problems.values()) else 0
    if cmd == "parse":
        targets = (
            records
            if args.all
            else [by_id[args.doc_id]]
            if args.doc_id and need(args.doc_id)
            else []
        )
        if not targets:
            return 2
        for r in targets:
            if args.dry_run:
                print(f"{r.doc_id}: would parse")
            else:
                print(f"{r.doc_id}: {idx.parse(r)}")
        return 0
    if cmd == "remove":
        if args.dry_run:
            print(f"dry-run: would remove chunks of {args.doc_id}")
            return 0
        if not _confirm(f"Remove all chunks of {args.doc_id!r} (source files are kept)?", args.yes):
            return 1
        print(f"removed {idx.remove(args.doc_id)} chunks for {args.doc_id}")
    elif cmd in ("add", "update"):
        if not need(args.doc_id):
            return 2
        rec = by_id[args.doc_id]
        if args.dry_run:
            changed = store.doc_hashes().get(rec.doc_id) != sha256_file(s.data_dir / rec.path)
            verb = "indexed" if changed or cmd == "update" else "skipped"
            print(f"dry-run: {rec.doc_id} would be {verb}")
            return 0
        outcome = idx.index_document(rec, force=cmd == "update")
        print(f"{rec.doc_id}: {outcome}" + (" (nothing to do)" if outcome == "skipped" else ""))
    elif cmd == "reindex":
        if args.dry_run:
            print(
                f"dry-run: would rebuild {len(records)} documents under fingerprint "
                f"{idx.fingerprint}"
            )
            return 0
        if not _confirm(f"Rebuild the whole index ({len(records)} documents)?", args.yes):
            return 1
        store.fingerprint = None  # new data is written first; stale chunks are removed per doc
        summary = idx.index_all(records, force=True)
        for orphan in sorted(set(store.list_docs()) - set(by_id)):
            store.delete_by_doc_id(orphan)
        print(
            f"indexed={len(summary.indexed)} chunks={store.count()} "
            f"cache_hits={summary.cache_hits} seconds={summary.seconds}"
        )
    store.persist()
    print(
        f"index: {store.count()} chunks, {len(store.list_docs())} docs, "
        f"types={dict(idx.chunk_type_counts())}"
    )
    return 0


def main() -> None:
    sys.exit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
