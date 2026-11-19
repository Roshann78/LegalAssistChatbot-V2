"""Title-derived acronyms so "dpdp act" finds a document that never spells the acronym."""

from __future__ import annotations

from legal_rag.retrieval.query import derive_aliases, understand

TITLES = [
    "The Digital Personal Data Protection Act, 2023 (Act 22 of 2023)",
    "THE INDIAN CONTRACT ACT, 1872",
    "The Information Technology Act, 2000",
    "The Constitution of India",
]


def test_derive_aliases_from_titles():
    a = derive_aliases(TITLES)
    assert a["dpdp"] == "Digital Personal Data Protection Act" and a["dpdpa"] == a["dpdp"]
    assert a["it act"] == "Information Technology Act"
    assert not any("constitution" in v.lower() for v in a.values())


def test_understand_expands_derived_aliases_but_not_plain_words():
    a = derive_aliases(TITLES)
    assert understand("what id dpdp act", aliases=a).expansion == a["dpdp"]
    assert understand("explain IT Act section 66", aliases=a).expansion == a["it act"]
    assert understand("what is it about", aliases=a).expansion == ""
    assert understand("what id dpdp act", aliases=a, abbrev=False).expansion == ""
