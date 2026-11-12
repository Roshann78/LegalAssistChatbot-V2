# Corpus plan (~100 documents) — checklist, nothing downloaded

Place PDFs at `data/raw/<domain>/<doc_type>/<doc_id>.pdf`, then run `python -m legal_rag.ingestion draft`
and complete `data/manifest.yaml`. Prefer official sources (India Code, e-Gazette, Supreme Court, RBI);
record `source_url` for each.

- [ ] Core criminal law: BNS, BNSS, BSA (3) — India Code
- [ ] IPC→BNS, CrPC→BNSS, IEA→BSA mapping tables (3–5); also fill `data/mappings/ipc_to_bns.json`
      (`old_act, old_section, new_act, new_section, notes, verified`; only `verified: true` is used)
- [ ] IPC, CrPC, IEA (3) — mark `status: superseded` + `superseded_by`
- [ ] Constitution of India (1)
- [ ] Key statutes: RTI, DV Act, Consumer Protection, IT Act, Contract Act, … (10–15)
- [ ] Landmark Supreme Court judgments, full text (30–40) — supremecourt.gov.in
- [ ] RBI master directions/circulars: KYC, fair practices, ombudsman, … (8–12) — rbi.org.in
- [ ] Banking statutes/schemes: BR Act, PMJDY, MUDRA, … (8–12)
- [ ] Official FAQs/explainers (5–8)

Suggestions to consider: Negotiable Instruments Act, Limitation Act, Digital Personal Data Protection Act,
RBI digital-lending directions.
