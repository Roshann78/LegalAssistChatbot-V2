"""Heuristic classifier: infer manifest metadata for a dropped-in PDF/text file.

Everything here is a best guess from the folder, the file name, PDF metadata and the first pages.
The result always passes schema validation; the owner can correct any field in the manifest, and
`sync` never overwrites fields of entries that already exist.
"""

from __future__ import annotations

import re
from pathlib import Path

from legal_rag.ingestion.manifest import sha256_file

SUFFIXES = {".pdf", ".txt"}
DOMAINS = ("legal", "banking")

_BANKING = (
    "rbi", "reserve bank", "bank", "nbfc", "kyc", "loan", "credit", "deposit", "dicgc",
    "insolvency", "payment", "financial", "monetary", "lending", "fema", "upi",
)  # fmt: skip
_LEGAL = (
    "act", "court", "judgment", "constitution", "penal", "code", "supreme", "high court",
    "criminal", "civil", "petitioner", "sanhita", "adhiniyam", "tribunal",
)  # fmt: skip

# (doc_type, regex) tried in order against the normalised file name, then against the first page.
_NAME_RULES = (
    ("mapping_table", r"mapping|correspondence|comparison table"),
    ("faq", r"\bfaqs?\b|frequently asked"),
    ("master_direction", r"master direction"),
    ("circular", r"circular|notification"),
    ("constitution", r"constitution"),
    ("scheme", r"\bscheme\b|yojana"),
    ("report", r"annual report|\breport\b|financial stability|\bsurvey\b"),
    ("judgment", r"^(sc|hc) |\bvs?\b|judg(e)?ment"),
    ("rule", r"\brules?\b|\bregulations?\b"),
    ("statute", r"\bact\b|\bcode\b|sanhita|adhiniyam"),
)
_TEXT_RULES = (
    ("mapping_table", r"correspondence table|comparative table"),
    ("faq", r"frequently asked questions"),
    ("master_direction", r"master direction"),
    ("circular", r"\bcircular\b"),
    ("constitution", r"constitution of india"),
    ("judgment", r"(in the (supreme|high) court).{0,600}(petitioner|appellant|respondent)"),
    ("report", r"annual report|financial stability report"),
    ("statute", r"\ban act to\b|be it enacted"),
    ("rule", r"\brules?,? \d{4}"),
)
_JUNK_TITLE = re.compile(r"microsoft word|untitled|\.docx?\b|\.pdf\b|^\W*$", re.I)
_YEAR = re.compile(r"(?<!\d)(1[89]\d\d|20\d\d)(?!\d)")
_ACT_TITLE = re.compile(r"\bThe [A-Z][A-Za-z ,&()'-]{3,90}?(?:Act|Code|Sanhita|Adhiniyam),? ?\d{4}")


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9_-]+", "-", name.lower()).strip("-_")
    s = re.sub(r"-{2,}", "-", s)
    return s or "doc"


def unique_id(base: str, taken: set[str]) -> str:
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


def _read_head(path: Path, max_pages: int = 3) -> tuple[str, dict, int, float]:
    """(first pages text, pdf metadata, page_count, low-text fraction). Never raises."""
    try:
        if path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore")
            return text[:6000], {}, max(1, text.count("\f") + 1), 0.0 if text.strip() else 1.0
        import fitz  # PyMuPDF

        with fitz.open(path) as doc:
            n = doc.page_count
            texts = [(doc[i].get_text("text") or "") for i in range(min(max_pages, n))]
            low = sum(1 for t in texts if len(t.strip()) < 50) / max(1, len(texts))
            return "\n".join(texts), dict(doc.metadata or {}), n, low
    except Exception:  # unreadable/corrupt file: classify from the name alone
        return "", {}, 0, 1.0


def _norm_name(stem: str) -> str:
    return re.sub(r"[\s_\-.]+", " ", stem.lower()).strip()


def infer_domain(path: Path, raw_dir: Path, name: str, text: str) -> str:
    try:
        parts = path.relative_to(raw_dir).parts
    except ValueError:
        parts = ()
    if len(parts) >= 2 and parts[0].lower() in DOMAINS:
        return parts[0].lower()
    low = text[:3000].lower()
    bank = sum(3 * (k in name) + (k in low) for k in _BANKING)
    legal = sum(3 * (k in name) + (k in low) for k in _LEGAL)
    return "banking" if bank > legal else "legal"


def infer_doc_type(name: str, text: str, domain: str) -> str:
    for dtype, pat in _NAME_RULES:
        if re.search(pat, name):
            return dtype
    low = re.sub(r"\s+", " ", text[:2500].lower())
    for dtype, pat in _TEXT_RULES:
        if re.search(pat, low):
            return dtype
    return "circular" if domain == "banking" else "statute"


def infer_title(stem: str, meta: dict, text: str) -> str:
    t = (meta.get("title") or "").strip()
    if len(t) >= 6 and not _JUNK_TITLE.search(t):
        return t[:200]
    head = re.sub(r"\s+", " ", text[:2500])
    m = _ACT_TITLE.search(head)
    if m:
        return m.group(0).strip()
    for line in (ln.strip() for ln in text.splitlines()[:25]):
        letters = sum(ch.isalpha() for ch in line)
        if 8 <= len(line) <= 150 and len(line.split()) >= 2 and letters / len(line) > 0.6:
            return line
    return re.sub(r"[_\-]+", " ", stem).strip().title() or stem


def infer_year(stem: str, title: str, text: str) -> int | None:
    for src in (stem, title, text[:2000]):
        m = _YEAR.search(src)
        if m:
            return int(m.group(1))
    return None


def classify_file(path: Path, raw_dir: Path, taken_ids: set[str], rel_path: str) -> dict:
    """Return a manifest entry (plain dict, valid for DocumentRecord) for `path`."""
    text, meta, pages, low_frac = _read_head(path)
    name = _norm_name(path.stem)
    domain = infer_domain(path, raw_dir, name, text)
    doc_type = infer_doc_type(name, text, domain)
    title = infer_title(path.stem, meta, text)
    entry: dict = {
        "doc_id": unique_id(slugify(path.stem), taken_ids),
        "title": title,
        "doc_type": doc_type,
        "domain": domain,
        "path": rel_path,
        "jurisdiction": "India",
        "status": "in_force",
        "language": "en",
        "source_type": "text" if path.suffix.lower() == ".txt" else "pdf",
        "file_hash": sha256_file(path),
    }
    if pages:
        entry["page_count"] = pages
    entry["text_quality"] = (
        "unknown"
        if not pages
        else "good"
        if low_frac == 0
        else "fair"
        if low_frac < 0.2
        else "poor"
        if low_frac < 0.5
        else "needs_ocr"
    )
    year = infer_year(path.stem, title, text)
    if year:
        entry["year"] = year
    if doc_type == "judgment":  # schema requires a court for judgments
        blob = (name + " " + text[:1500]).lower()
        entry["authority_or_court"] = (
            "Supreme Court of India"
            if "supreme court" in blob or name.startswith("sc ")
            else "High Court"
            if "high court" in blob or name.startswith("hc ")
            else "Court (to be verified)"
        )
    return entry


def discover(raw_dir: Path) -> list[Path]:
    """All .pdf/.txt files under raw_dir (recursive), sorted; hidden files skipped."""
    if not raw_dir.exists():
        return []
    return sorted(
        f
        for f in raw_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in SUFFIXES and not f.name.startswith(".")
    )
