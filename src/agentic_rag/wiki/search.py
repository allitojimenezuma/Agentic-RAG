"""BM25 search over curated wiki page fields with bounded link expansion.

The per-page BM25 "document" is the page title + tags + all section headings
+ all section texts (the first/intro section text serves as the inferred
summary — frontmatter has no summary field). ``types``/``tags`` are predicate
filters applied to the candidate set *before* scoring. Result lists are
deterministic: direct hits sorted by score desc, then bounded ``expand-link``
hits (fixed low score, capped per page and in total — never unbounded).
"""

from __future__ import annotations

import logging
import re
import unicodedata

from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from agentic_rag.wiki.model import Page, Wiki

logger = logging.getLogger(__name__)

# Fixed score for link-expansion hits: deterministic and below typical BM25
# scores for genuine matches (BM25Okapi epsilon-smooths IDF, so small
# negative scores are possible on tiny corpora — that is fine; expansion
# hits are always appended after the direct hits).
EXPAND_SCORE = 0.1

# Max expansion hits added per source page (min with k).
_PER_PAGE_EXPAND_CAP = 2


class SearchHit(BaseModel):
    """A single search result: a wiki page slug plus match provenance."""

    slug: str
    score: float
    sections: list[str]
    matched_via: str


def search(
    wiki: Wiki,
    query: str,
    *,
    k: int = 8,
    types: list[str] | None = None,
    tags: list[str] | None = None,
    expand_links: bool = True,
    depth: int = 1,
) -> list[SearchHit]:
    """BM25 search over curated page fields with bounded link expansion.

    Args:
        wiki: in-memory ``Wiki`` model (see ``agentic_rag.wiki.model``).
        query: free-text query, tokenized like the page documents.
        k: maximum number of direct hits (and total expansion hits).
        types: if given, keep only pages whose ``fm.type`` is in this list.
        tags: if given, keep only pages whose ``fm.tags`` intersects this list.
        expand_links: if True, append bounded ``expand-link`` hits for pages
            linked (via resolved ``outbound_links``) from the direct hits.
        depth: BFS levels for link expansion (1 = one hop from direct hits).

    Returns:
        Direct hits sorted by score desc, followed by ``expand-link`` hits.
    """
    if k <= 0 or not wiki.pages:
        return []

    tokens = _tokenize(query)
    if not tokens:
        return []

    candidates = [p for p in wiki.pages if _matches_filters(p, types, tags)]
    if not candidates:
        return []

    corpus = [_document_tokens(p) for p in candidates]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokens)

    ranked = sorted(
        ((p, float(s)) for p, s in zip(candidates, scores)),
        key=lambda ps: ps[1],
        reverse=True,
    )

    # Direct hits = top-k scored candidates that actually contain at least one
    # query token. Candidates with no token overlap are noise (no meaningful
    # matched_via) and are skipped. Note BM25Okapi epsilon-smoothes IDF, so a
    # genuine match can carry a small negative score on a tiny corpus — never
    # filter on the sign of the score.
    query_tokens = set(tokens)
    direct: list[SearchHit] = []
    for page, score in ranked:
        if len(direct) >= k:
            break
        matched_via, sections = _match_location(page, query_tokens)
        if matched_via is None:
            continue
        direct.append(
            SearchHit(slug=page.slug, score=score, sections=sections, matched_via=matched_via)
        )

    hits = direct
    if expand_links and depth >= 1 and direct:
        hits = direct + _expand_linked(wiki, direct, k, depth)
    return hits


def _tokenize(text: str) -> list[str]:
    """NFKD-normalize to ASCII, lowercase, split on non-alphanumerics.

    Mirrors ``io.markdown_parser.slugify`` normalization so tokens match
    how page slugs/titles are canonicalized.
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    return [t for t in re.split(r"[^a-z0-9]+", text) if t]


def _document_tokens(page: Page) -> list[str]:
    """Per-page BM25 document: title + tags + headings + section texts."""
    parts = [page.fm.title]
    parts.extend(page.fm.tags)
    parts.extend(section.heading for section in page.sections)
    parts.extend(section.text for section in page.sections)
    return _tokenize(" ".join(parts))


def _matches_filters(
    page: Page, types: list[str] | None, tags: list[str] | None
) -> bool:
    """Predicate filters applied to the candidate set before BM25."""
    if types is not None and page.fm.type not in types:
        return False
    if tags is not None and not any(tag in page.fm.tags for tag in tags):
        return False
    return True


def _match_location(page: Page, tokens: set[str]) -> tuple[str | None, list[str]]:
    """Where the query matched: title / tags / section:<heading> / body.

    First match wins, checked in that order. ``sections`` = the headings of
    sections whose body text matched ([] if only title/tags matched). Returns
    ``(None, [])`` when no query token appears anywhere in the page document.
    """
    if tokens & set(_tokenize(page.fm.title)):
        return "title", []

    tag_tokens: set[str] = set()
    for tag in page.fm.tags:
        tag_tokens |= set(_tokenize(tag))
    if tokens & tag_tokens:
        return "tags", []

    matched_sections = [
        section.heading
        for section in page.sections
        if tokens & set(_tokenize(section.text))
    ]
    for section in page.sections:
        if tokens & set(_tokenize(section.heading)):
            return f"section:{section.heading}", matched_sections
    if matched_sections:
        return "body", matched_sections
    return None, []


def _expand_linked(
    wiki: Wiki, direct: list[SearchHit], k: int, depth: int
) -> list[SearchHit]:
    """Bounded BFS over ``outbound_links`` starting from the direct hits.

    Adds at most ``min(2, k)`` expansion hits per source page and at most ``k``
    in total; ``depth`` = number of BFS levels (1 = one hop). Every expansion
    hit is a ``SearchHit(slug, EXPAND_SCORE, [], "expand-link")``.
    """
    expansion: list[SearchHit] = []
    seen = {hit.slug for hit in direct}
    frontier = [hit.slug for hit in direct]
    per_page_cap = min(_PER_PAGE_EXPAND_CAP, k)

    for _ in range(depth):
        next_frontier: list[str] = []
        for slug in frontier:
            page = wiki.by_slug.get(slug)
            if page is None:
                continue
            added = 0
            for target in page.outbound_links:
                if target in seen:
                    continue
                seen.add(target)
                expansion.append(
                    SearchHit(slug=target, score=EXPAND_SCORE, sections=[], matched_via="expand-link")
                )
                next_frontier.append(target)
                added += 1
                if added >= per_page_cap or len(expansion) >= k:
                    break
            if len(expansion) >= k:
                break
        if not next_frontier or len(expansion) >= k:
            break
        frontier = next_frontier
    return expansion
