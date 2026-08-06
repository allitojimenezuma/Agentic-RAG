#!/usr/bin/env python3
"""Generate an interactive Obsidian-style wiki app from the wiki.

Produces a single self-contained HTML file with two views:

- **Graph view** (default): the vis.js knowledge graph, same style as before.
- **Standard view**: an Obsidian-like note reader. Click any node in the graph
  to open its note; [[wiki links]] inside notes are clickable and navigate
  in-app; each note shows its backlinks at the bottom.

All page content and links are pre-rendered at generation time with
markdown-it-py, so the app is fully self-contained (only vis-network is loaded
from a CDN, matching the original graph).

Usage:
    python scripts/knowledge_graph.py --wiki wiki --output knowledge_graph.html
"""

import html as html_mod
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from markdown_it import MarkdownIt

from agentic_rag.io.markdown_parser import _LINK_RE, extract_links, parse_frontmatter, slugify
from agentic_rag.io.wiki_io import list_pages

TYPE_COLORS = {
    "entity": "#4CAF50",
    "concept": "#2196F3",
    "source": "#FF9800",
    "comparison": "#9C27B0",
    "overview": "#607D8B",
    "unknown": "#9E9E9E",
}

# Fenced code blocks (backticks or tildes): protected from [[link]] rewriting.
# Placeholders use private-use-area chars because markdown-it replaces \x00 (NUL)
# with U+FFFD during its normalize pass but passes PUA chars through untouched.
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})([^\n]*)\n(.*?)^\1[ \t]*$", re.MULTILINE | re.DOTALL)
_CODE_TOKEN_RE = re.compile(r"^(\ue000CODE\d+\ue001)$", re.MULTILINE)

_md = MarkdownIt("commonmark", {"html": True}).enable("table").enable("strikethrough")


# --------------------------------------------------------------------------- #
# Data building
# --------------------------------------------------------------------------- #

def _build_lookup(page_slugs: set[str]) -> dict[str, str]:
    """Map short/slugified names to full slugs (e.g. 'ml' -> 'concepts/machine-learning')."""
    lookup = {}
    for ps in page_slugs:
        short = ps.rsplit("/", 1)[-1] if "/" in ps else ps
        lookup[short] = ps
        lookup[slugify(ps.split("/")[-1])] = ps
        lookup[slugify(ps)] = ps
    return lookup


def _resolve(target: str, page_slugs: set[str], lookup: dict[str, str]) -> str | None:
    """Resolve a [[link]] target to a page slug, or None if it points nowhere."""
    if target in page_slugs:
        return target
    s = slugify(target)
    if s in lookup:
        return lookup[s]
    for ps in page_slugs:
        if ps.endswith("/" + s) or ps.rsplit("/", 1)[-1] == s:
            return ps
    return None


def _strip_frontmatter(content: str) -> str:
    """Return the markdown body after the YAML frontmatter block."""
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) == 3:
            return parts[2].lstrip("\n")
    return content


def _render_body(md: MarkdownIt, body: str, resolver, title: str | None = None) -> str:
    """Render markdown to HTML, converting [[Target|alias]] into in-app links.

    Fenced code blocks are protected from link rewriting so code samples keep
    their literal ``[[...]]`` text (rendered as regular code blocks).
    """
    # Drop a leading "# Title" heading that duplicates the note title.
    if title:
        m = re.match(rf"^#\s+{re.escape(title)}\s*$", body, re.MULTILINE)
        if m:
            body = body[m.end():].lstrip("\n")

    # Protect fenced code blocks.
    protected = {}

    def _protect(m):
        token = f"\ue000CODE{len(protected)}\ue001"
        protected[token] = m
        return token

    body = _FENCE_RE.sub(_protect, body)
    body = _CODE_TOKEN_RE.sub(lambda m: f"\n\n{m.group(0)}\n\n", body)

    # Rewrite [[links]] into clickable anchors (or muted spans if unresolved).
    def _link_repl(m):
        target = m.group(1).strip()
        alias = (m.group(2) or "").strip() or target
        slug = resolver(target)
        alias_html = html_mod.escape(alias)
        if slug:
            return f'<a href="#" class="wiki-link" data-page="{html_mod.escape(slug)}">{alias_html}</a>'
        return f'[[{alias_html}]]'

    body = _LINK_RE.sub(_link_repl, body)

    out = md.render(body)
    # Unwrap the paragraph markdown-it wraps around each code placeholder.
    out = out.replace("<p>\ue000CODE", "\ue000CODE").replace("\ue001</p>", "\ue001")
    for token, m in protected.items():
        lang = m.group(2).strip()
        code = html_mod.escape(m.group(3).rstrip("\n"))
        cls = f' class="language-{html_mod.escape(lang)}"' if lang else ""
        out = out.replace(token, f"<pre><code{cls}>{code}</code></pre>")
    return out


def build_app_data(wiki_path: Path) -> dict:
    """Build everything the app needs: graph nodes/edges + per-page data."""
    page_paths = list_pages(wiki_path)

    raw: dict[str, str] = {}
    meta: dict[str, dict] = {}
    nodes: dict[str, dict] = {}
    page_slugs: set[str] = set()

    # First pass: read pages, parse frontmatter, build graph nodes.
    for page_path in page_paths:
        slug = str(page_path.relative_to(wiki_path)).removesuffix(".md")
        page_slugs.add(slug)
        try:
            content = page_path.read_text(encoding="utf-8")
        except Exception:
            content = ""
        raw[slug] = content
        try:
            fm = parse_frontmatter(content)
        except Exception:
            fm = None
        if fm is not None:
            meta[slug] = {
                "title": fm.title,
                "type": fm.type or "unknown",
                "tags": list(fm.tags),
                "sources": list(fm.sources),
                "updated": fm.updated.isoformat(),
            }
        else:
            meta[slug] = {"title": slug, "type": "unknown", "tags": [], "sources": [], "updated": ""}
        ptype = meta[slug]["type"]
        nodes[slug] = {
            "id": slug,
            "label": meta[slug]["title"],
            "color": TYPE_COLORS.get(ptype, "#9E9E9E"),
            "shape": "dot" if ptype != "source" else "diamond",
            "size": 20 if ptype != "source" else 15,
            "title": f"{meta[slug]['title']}\nType: {ptype}",
        }

    lookup = _build_lookup(page_slugs)

    # Second pass: resolve [[links]] into edges + backlinks.
    edges_raw = []
    backlinks: dict[str, set[str]] = defaultdict(set)
    for source, content in raw.items():
        try:
            for link in extract_links(content):
                target = _resolve(link.target, page_slugs, lookup)
                if target and target != source:
                    edges_raw.append((source, target))
                    backlinks[target].add(source)
        except Exception:
            continue

    # Deduplicate and detect bidirectional edges (same as the original graph).
    edge_set = set(edges_raw)
    bidirectional = set()
    for src, tgt in edges_raw:
        if (tgt, src) in edge_set:
            bidirectional.add((min(src, tgt), max(src, tgt)))

    seen = set()
    edges = []
    for src, tgt in edges_raw:
        pair = (min(src, tgt), max(src, tgt))
        if pair in bidirectional:
            if pair not in seen:
                seen.add(pair)
                edges.append({"from": src, "to": tgt, "arrows": "both"})
        else:
            if (src, tgt) not in seen:
                seen.add((src, tgt))
                edges.append({"from": src, "to": tgt, "arrows": "to"})

    # Third pass: render each page's markdown for the standard view.
    pages_data = {}
    for slug in page_slugs:
        body = _strip_frontmatter(raw.get(slug, ""))
        pages_data[slug] = {
            "slug": slug,
            "title": meta[slug]["title"],
            "type": meta[slug]["type"],
            "tags": meta[slug]["tags"],
            "sources": meta[slug]["sources"],
            "updated": meta[slug]["updated"],
            "content": _render_body(
                _md, body, lambda t: _resolve(t, page_slugs, lookup), meta[slug]["title"]
            ),
            "backlinks": sorted(
                (
                    {"slug": b, "title": meta[b]["title"], "type": meta[b]["type"]}
                    for b in backlinks.get(slug, set())
                ),
                key=lambda b: b["title"].lower(),
            ),
        }

    return {"nodes": list(nodes.values()), "edges": edges, "pages": pages_data}


# --------------------------------------------------------------------------- #
# HTML generation
# --------------------------------------------------------------------------- #

def _to_json(obj) -> str:
    """JSON safe to embed inside a <script> block."""
    return (
        json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wiki Vault</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
    [hidden] { display: none !important; }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
        margin: 0; display: flex; flex-direction: column; overflow: hidden;
        font-family: -apple-system, "Segoe UI", "Inter", Roboto, sans-serif;
        background: #1a1a2e; color: #eee;
    }
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 5px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.28); }
    ::-webkit-scrollbar-track { background: transparent; }

    /* ---------- toolbar ---------- */
    #toolbar {
        flex: 0 0 auto; display: flex; align-items: center; gap: 10px;
        padding: 8px 14px; background: rgba(0,0,0,0.45);
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    #brand { font-weight: 600; letter-spacing: 0.5px; color: #fff; font-size: 14px; white-space: nowrap; }
    #brand .glyph { color: #4FC3F7; }
    .icon-btn {
        background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
        color: #ddd; border-radius: 6px; width: 30px; height: 28px; cursor: pointer; font-size: 14px;
        line-height: 1;
    }
    .icon-btn:hover:not(:disabled) { background: rgba(255,255,255,0.14); color: #fff; }
    .icon-btn:disabled { opacity: 0.35; cursor: default; }
    #search-wrap { position: relative; flex: 1; max-width: 420px; margin: 0 auto; }
    #search {
        width: 100%; background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12); color: #eee; border-radius: 6px;
        padding: 6px 10px; font-size: 13px; outline: none;
    }
    #search::placeholder { color: #77778f; }
    #search:focus { border-color: #4FC3F7; }
    #search-results {
        position: absolute; top: 100%; left: 0; right: 0; margin-top: 4px;
        background: #16213e; border: 1px solid rgba(255,255,255,0.12); border-radius: 8px;
        max-height: 60vh; overflow-y: auto; z-index: 50;
        box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    }
    .result-item {
        display: flex; align-items: center; gap: 8px; padding: 7px 10px;
        font-size: 13px; cursor: pointer; color: #ddd; text-decoration: none;
    }
    .result-item:hover { background: rgba(255,255,255,0.07); }
    .result-type { margin-left: auto; font-size: 11px; color: #77778f; text-transform: capitalize; }
    .result-empty { padding: 10px; font-size: 12px; color: #77778f; }
    #view-toggle { display: flex; gap: 4px; background: rgba(255,255,255,0.06); border-radius: 6px; padding: 3px; }
    .toggle-btn {
        background: transparent; border: none; color: #aaa; padding: 4px 12px;
        border-radius: 4px; font-size: 12px; cursor: pointer;
    }
    .toggle-btn.active { background: #2196F3; color: #fff; }

    /* ---------- graph view ---------- */
    #graph-view { flex: 1; position: relative; }
    #graph { position: absolute; inset: 0; }
    #legend {
        position: absolute; top: 12px; left: 12px; background: rgba(0,0,0,0.7);
        padding: 12px 16px; border-radius: 8px; font-size: 13px; z-index: 5;
    }
    #legend div { margin: 4px 0; }
    .dot { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
    .dot-sm { display: inline-block; width: 8px; height: 8px; border-radius: 50%; flex: 0 0 auto; }
    #stats {
        position: absolute; bottom: 12px; left: 12px; background: rgba(0,0,0,0.7);
        padding: 8px 12px; border-radius: 8px; font-size: 12px; z-index: 5;
    }
    #graph-hint {
        position: absolute; bottom: 12px; right: 12px; background: rgba(0,0,0,0.7);
        padding: 8px 12px; border-radius: 8px; font-size: 12px; color: #aaa; z-index: 5;
    }

    /* ---------- standard (note) view ---------- */
    #standard-view { flex: 1; overflow-y: auto; padding: 26px 20px 60px; }
    #note { max-width: 820px; margin: 0 auto; }
    #note-title { margin: 0 0 4px; font-size: 26px; color: #fff; }
    #note-meta {
        display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
        font-size: 12px; color: #8b8ba3; padding-bottom: 14px;
        border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 18px;
    }
    .sep { color: #56567a; }
    .badge {
        display: inline-block; padding: 2px 8px; border-radius: 10px;
        font-size: 11px; color: #fff; text-transform: capitalize;
    }
    .tag {
        display: inline-block; background: rgba(255,255,255,0.08);
        padding: 2px 8px; border-radius: 8px; font-size: 11px; color: #b9b9cf;
    }
    #empty-state { max-width: 820px; margin: 80px auto; text-align: center; color: #8b8ba3; }

    #note-content { font-size: 15px; line-height: 1.65; color: #d6d6e6; }
    #note-content h1 { font-size: 24px; margin: 26px 0 10px; color: #fff; }
    #note-content h2 { font-size: 19px; margin: 24px 0 8px; color: #fff; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 6px; }
    #note-content h3 { font-size: 16px; margin: 20px 0 6px; color: #e8e8f2; }
    #note-content h4 { font-size: 14.5px; margin: 16px 0 4px; color: #e8e8f2; }
    #note-content p { margin: 10px 0; }
    #note-content ul, #note-content ol { padding-left: 22px; margin: 10px 0; }
    #note-content li { margin: 4px 0; }
    #note-content a { color: #4FC3F7; text-decoration: none; }
    #note-content a:hover { text-decoration: underline; }
    #note-content .wiki-link { cursor: pointer; }
    #note-content .wiki-dangling { color: #7a7a8f; text-decoration: underline dashed; cursor: not-allowed; }
    #note-content code {
        background: rgba(255,255,255,0.09); border-radius: 4px; padding: 1px 5px;
        font-family: "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace;
        font-size: 13px; color: #9bd7ff;
    }
    #note-content pre {
        background: #101024; border: 1px solid rgba(255,255,255,0.1); border-radius: 8px;
        padding: 12px 14px; overflow-x: auto; line-height: 1.5; margin: 12px 0;
    }
    #note-content pre code { background: none; padding: 0; color: #c9d6e4; }
    #note-content blockquote {
        border-left: 3px solid #4FC3F7; margin: 14px 0; padding: 2px 14px;
        color: #a9a9c2; background: rgba(79,195,247,0.05); border-radius: 0 6px 6px 0;
    }
    #note-content table { border-collapse: collapse; margin: 14px 0; width: 100%; font-size: 13.5px; }
    #note-content th, #note-content td { border: 1px solid rgba(255,255,255,0.14); padding: 6px 10px; text-align: left; }
    #note-content th { background: rgba(255,255,255,0.06); }
    #note-content hr { border: none; border-top: 1px solid rgba(255,255,255,0.1); margin: 24px 0; }
    #note-content img { max-width: 100%; border-radius: 6px; }
    #note-content strong { color: #f2f2fa; }

    #backlinks { margin-top: 40px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 14px; }
    #backlinks h2 { font-size: 13px; color: #8b8ba3; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 10px; }
    .backlink {
        display: inline-flex; align-items: center; gap: 7px; margin: 0 6px 6px 0;
        padding: 5px 12px; background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1); border-radius: 14px;
        font-size: 13px; color: #ddd; text-decoration: none;
    }
    .backlink:hover { background: rgba(79,195,247,0.15); border-color: #4FC3F7; }
    .muted { color: #8b8ba3; font-size: 13px; }

    /* ---------- quick switcher ---------- */
    #switcher {
        position: fixed; inset: 0; background: rgba(0,0,0,0.55); z-index: 100;
        display: flex; align-items: flex-start; justify-content: center; padding-top: 12vh;
    }
    #switcher-box {
        width: 520px; max-width: 90vw; background: #16213e;
        border: 1px solid rgba(255,255,255,0.14); border-radius: 10px;
        box-shadow: 0 16px 48px rgba(0,0,0,0.6); overflow: hidden;
    }
    #switcher-input {
        width: 100%; background: rgba(255,255,255,0.05); border: none;
        border-bottom: 1px solid rgba(255,255,255,0.12); color: #eee;
        padding: 12px 14px; font-size: 14px; outline: none;
    }
    #switcher-list { list-style: none; margin: 0; padding: 6px; max-height: 50vh; overflow-y: auto; }
    #switcher-list li {
        display: flex; align-items: center; gap: 8px; padding: 7px 10px;
        border-radius: 6px; font-size: 13px; color: #ddd; cursor: pointer;
    }
    #switcher-list li.sel { background: #2196F3; color: #fff; }
    #switcher-list li.sel .result-type { color: #cfe8ff; }
    #switcher-hint { padding: 8px 14px; font-size: 11px; color: #7a7a8f; border-top: 1px solid rgba(255,255,255,0.08); }
</style>
</head>
<body>

<header id="toolbar">
    <div id="brand"><span class="glyph">✦</span> Wiki Vault</div>
    <button id="btn-back" class="icon-btn" title="Back (Alt+Left)" disabled>&#8592;</button>
    <button id="btn-fwd" class="icon-btn" title="Forward (Alt+Right)" disabled>&#8594;</button>
    <div id="search-wrap">
        <input id="search" type="text" placeholder="Search notes&hellip; (Ctrl/Cmd+K)" autocomplete="off" spellcheck="false">
        <div id="search-results" hidden></div>
    </div>
    <div id="view-toggle">
        <button id="btn-standard" class="toggle-btn" title="Standard view (Ctrl/Cmd+B)">Standard</button>
        <button id="btn-graph" class="toggle-btn active" title="Graph view (Ctrl/Cmd+B)">Graph</button>
    </div>
</header>

<div id="graph-view">
    <div id="legend">
        <div><span class="dot" style="background:#4CAF50"></span>Entity</div>
        <div><span class="dot" style="background:#2196F3"></span>Concept</div>
        <div><span class="dot" style="background:#FF9800"></span>Source</div>
        <div><span class="dot" style="background:#9C27B0"></span>Comparison</div>
        <div><span class="dot" style="background:#607D8B"></span>Overview</div>
    </div>
    <div id="stats">__STATS__</div>
    <div id="graph-hint">Click a node to open its note</div>
    <div id="graph"></div>
</div>

<main id="standard-view" hidden>
    <div id="empty-state">
        <p>No note selected. Click a node in the <b>Graph</b> view, use search, or press <b>Ctrl/Cmd+P</b> to jump.</p>
    </div>
    <article id="note" hidden>
        <h1 id="note-title"></h1>
        <div id="note-meta"></div>
        <div id="note-content"></div>
        <section id="backlinks"></section>
    </article>
</main>

<div id="switcher" hidden>
    <div id="switcher-box">
        <input id="switcher-input" placeholder="Jump to note&hellip; (type to filter)" autocomplete="off" spellcheck="false">
        <ul id="switcher-list"></ul>
        <div id="switcher-hint">&#8593;&#8595; navigate &middot; Enter open &middot; Esc close</div>
    </div>
</div>

<script>
var WIKI = { pages: __WIKI_DATA__ };
var pageList = Object.keys(WIKI.pages).map(function (k) { return WIKI.pages[k]; })
    .sort(function (a, b) { return a.title.localeCompare(b.title); });

var TYPE_COLORS = {
    entity: '#4CAF50', concept: '#2196F3', source: '#FF9800',
    comparison: '#9C27B0', overview: '#607D8B', unknown: '#9E9E9E'
};
function colorFor(t) { return TYPE_COLORS[t] || TYPE_COLORS['unknown']; }
function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
        return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
}

/* ---------------- graph ---------------- */
var container = document.getElementById('graph');
var network = null;
if (typeof vis !== 'undefined' && typeof vis.Network === 'function') {
    try {
        var nodes = new vis.DataSet(__NODES__);
        var edges = new vis.DataSet(__EDGES__);
        var options = {
            physics: {
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {
                    gravitationalConstant: -200,
                    centralGravity: 0.01,
                    springLength: 200,
                    springConstant: 0.02,
                    damping: 0.4
                },
                stabilization: { iterations: 300 }
            },
            nodes: {
                font: { color: '#ccc', size: 12 },
                borderWidth: 2
            },
            interaction: { hover: true, zoomView: true, dragView: true, tooltipDelay: 100 },
            edges: {
                color: { color: 'rgba(150,150,150,0.5)', highlight: '#fff' },
                smooth: { type: 'continuous' },
                width: 1.5
            }
        };
        network = new vis.Network(container, { nodes: nodes, edges: edges }, options);
        network.on('click', function (params) {
            if (params.nodes && params.nodes.length) openPage(params.nodes[0]);
        });
    } catch (err) {
        network = null;
    }
}
if (!network) {
    container.innerHTML = '<div class="muted" style="padding:40px;text-align:center">' +
        'The graph library (vis-network) failed to load &mdash; check your connection. ' +
        'The standard view still works.</div>';
}

/* ---------------- history / navigation ---------------- */
var navHistory = [], navIdx = -1;

function openPage(slug) {
    if (!WIKI.pages[slug]) return;
    navHistory = navHistory.slice(0, navIdx + 1);
    navHistory.push(slug);
    navIdx++;
    closeOverlays();
    showStandard(slug);
}

function showStandard(slug) {
    document.getElementById('graph-view').hidden = true;
    document.getElementById('standard-view').hidden = false;
    setToggle('standard');
    if (slug && WIKI.pages[slug]) renderNote(slug);
    else showEmpty();
    document.getElementById('standard-view').scrollTop = 0;
    updateNav();
}

function showGraph() {
    document.getElementById('standard-view').hidden = true;
    document.getElementById('graph-view').hidden = false;
    setToggle('graph');
    updateNav();
    setTimeout(resizeGraph, 0);
}

function showEmpty() {
    document.getElementById('note').hidden = true;
    document.getElementById('empty-state').hidden = false;
    document.title = 'Wiki Vault';
}

function renderNote(slug) {
    var p = WIKI.pages[slug];
    document.getElementById('empty-state').hidden = true;
    document.getElementById('note').hidden = false;
    document.getElementById('note-title').textContent = p.title;
    document.title = p.title + ' \u00b7 Wiki Vault';

    var bits = [];
    bits.push('<span class="badge" style="background:' + colorFor(p.type) + '">' + esc(p.type) + '</span>');
    if (p.updated) bits.push('<span>updated ' + esc(p.updated) + '</span>');
    if (p.sources && p.sources.length) bits.push('<span>sources: ' + p.sources.map(esc).join(', ') + '</span>');
    if (p.tags && p.tags.length) {
        bits.push('<span>tags: ' + p.tags.map(function (t) {
            return '<span class="tag">' + esc(t) + '</span>';
        }).join(' ') + '</span>');
    }
    document.getElementById('note-meta').innerHTML = bits.join(' <span class="sep">&middot;</span> ');

    document.getElementById('note-content').innerHTML = p.content;

    var bl = document.getElementById('backlinks');
    if (p.backlinks && p.backlinks.length) {
        bl.innerHTML = '<h2>Backlinks</h2>' + p.backlinks.map(function (b) {
            return '<a href="#" class="backlink" data-page="' + esc(b.slug) + '">' +
                '<span class="dot-sm" style="background:' + colorFor(b.type) + '"></span>' +
                esc(b.title) + '</a>';
        }).join('');
    } else {
        bl.innerHTML = '<h2>Backlinks</h2><p class="muted">No other notes link to this note yet.</p>';
    }
}

function updateNav() {
    document.getElementById('btn-back').disabled = navIdx < 0;
    document.getElementById('btn-fwd').disabled = navIdx >= navHistory.length - 1;
}

function goBack() {
    if (navIdx > 0) { navIdx--; showStandard(navHistory[navIdx]); }
    else if (navIdx === 0) { navIdx = -1; showGraph(); }
}

function goForward() {
    if (navIdx < navHistory.length - 1) { navIdx++; showStandard(navHistory[navIdx]); }
}

/* ---------------- view toggle ---------------- */
function setToggle(mode) {
    document.getElementById('btn-standard').classList.toggle('active', mode === 'standard');
    document.getElementById('btn-graph').classList.toggle('active', mode === 'graph');
}

function showStandardFromToggle() {
    if (navIdx >= 0) showStandard(navHistory[navIdx]);
    else if (pageList.length) openPage(pageList[0].slug);
}

function toggleView() {
    if (document.getElementById('standard-view').hidden) showStandardFromToggle();
    else showGraph();
}

function resizeGraph() {
    if (!network) return;
    try {
        network.setSize(container.clientWidth, container.clientHeight);
        network.redraw();
    } catch (err) { /* ignore */ }
}

/* ---------------- search ---------------- */
var searchInput = document.getElementById('search');
var searchResults = document.getElementById('search-results');

function runSearch() {
    var q = searchInput.value.trim().toLowerCase();
    if (!q) { searchResults.hidden = true; return; }
    var hits = [];
    pageList.forEach(function (p) {
        var title = p.title.toLowerCase();
        var inTitle = title.indexOf(q) !== -1;
        var inMeta = (p.type + ' ' + (p.tags || []).join(' ')).toLowerCase().indexOf(q) !== -1;
        if (inTitle || inMeta) {
            hits.push({ p: p, score: title.indexOf(q) === 0 ? 0 : (inTitle ? 1 : 2) });
        }
    });
    hits.sort(function (a, b) { return a.score - b.score || a.p.title.localeCompare(b.p.title); });
    hits = hits.slice(0, 10);
    searchResults.innerHTML = hits.length
        ? hits.map(function (h) {
            return '<a href="#" class="result-item" data-page="' + esc(h.p.slug) + '">' +
                '<span class="dot-sm" style="background:' + colorFor(h.p.type) + '"></span>' +
                esc(h.p.title) + '<span class="result-type">' + esc(h.p.type) + '</span></a>';
        }).join('')
        : '<div class="result-empty">No matching notes.</div>';
    searchResults.hidden = false;
}

function clearSearch() {
    searchInput.value = '';
    searchResults.hidden = true;
    searchInput.blur();
}

searchInput.addEventListener('input', runSearch);
searchInput.addEventListener('focus', runSearch);
searchInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') {
        var first = searchResults.querySelector('[data-page]');
        if (first) openPage(first.getAttribute('data-page'));
        e.preventDefault();
    } else if (e.key === 'Escape') {
        clearSearch();
        e.preventDefault();
    }
});
document.addEventListener('click', function (e) {
    if (!e.target.closest('#search-wrap')) clearSearch();
});

/* ---------------- quick switcher ---------------- */
var switcher = document.getElementById('switcher');
var switcherInput = document.getElementById('switcher-input');
var switcherList = document.getElementById('switcher-list');
var swHits = [], swSel = 0;

function openSwitcher() {
    switcher.hidden = false;
    switcherInput.value = '';
    swSel = 0;
    swHits = pageList.slice();
    drawSwitcher();
    switcherInput.focus();
}

function closeSwitcher() { switcher.hidden = true; }

function drawSwitcher() {
    switcherList.innerHTML = swHits.map(function (p, i) {
        return '<li data-page="' + esc(p.slug) + '" class="' + (i === swSel ? 'sel' : '') + '">' +
            '<span class="dot-sm" style="background:' + colorFor(p.type) + '"></span>' +
            esc(p.title) + '<span class="result-type">' + esc(p.type) + '</span></li>';
    }).join('');
    var sel = switcherList.querySelector('.sel');
    if (sel) sel.scrollIntoView({ block: 'nearest' });
}

switcherInput.addEventListener('input', function () {
    var q = switcherInput.value.trim().toLowerCase();
    swHits = pageList.filter(function (p) {
        if (!q) return true;
        return (p.title + ' ' + p.type + ' ' + (p.tags || []).join(' ')).toLowerCase().indexOf(q) !== -1;
    });
    swSel = 0;
    drawSwitcher();
});

switcherInput.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowDown') { swSel = Math.min(swSel + 1, swHits.length - 1); drawSwitcher(); e.preventDefault(); }
    else if (e.key === 'ArrowUp') { swSel = Math.max(swSel - 1, 0); drawSwitcher(); e.preventDefault(); }
    else if (e.key === 'Enter') {
        if (swHits[swSel]) openPage(swHits[swSel].slug);
        closeSwitcher();
        e.preventDefault();
    }
    else if (e.key === 'Escape') { closeSwitcher(); e.preventDefault(); }
});

switcher.addEventListener('click', function (e) {
    if (e.target === switcher) closeSwitcher();
});

/* ---------------- overlays / shortcuts ---------------- */
function closeOverlays() {
    searchResults.hidden = true;
    searchInput.blur();
    closeSwitcher();
}

document.addEventListener('keydown', function (e) {
    var mod = e.ctrlKey || e.metaKey;
    var k = e.key.toLowerCase();
    if (mod && k === 'p') { e.preventDefault(); openSwitcher(); }
    else if (mod && k === 'k') { e.preventDefault(); searchInput.focus(); searchInput.select(); }
    else if (mod && k === 'b') { e.preventDefault(); toggleView(); }
    else if (e.altKey && e.key === 'ArrowLeft') { e.preventDefault(); goBack(); }
    else if (e.altKey && e.key === 'ArrowRight') { e.preventDefault(); goForward(); }
    else if (e.key === 'Escape') { closeOverlays(); }
});

/* ---------------- wiring ---------------- */
document.addEventListener('click', function (e) {
    var el = e.target.closest ? e.target.closest('[data-page]') : null;
    if (el) { e.preventDefault(); openPage(el.getAttribute('data-page')); }
});

document.getElementById('btn-back').addEventListener('click', goBack);
document.getElementById('btn-fwd').addEventListener('click', goForward);
document.getElementById('btn-graph').addEventListener('click', showGraph);
document.getElementById('btn-standard').addEventListener('click', showStandardFromToggle);

window.addEventListener('resize', function () {
    if (!document.getElementById('graph-view').hidden) resizeGraph();
});

updateNav();
</script>
</body>
</html>
"""


def generate_html(data: dict, output_path: Path) -> None:
    """Write the self-contained Obsidian-style wiki app to disk."""
    html = _TEMPLATE
    html = html.replace("__WIKI_DATA__", _to_json(data["pages"]))
    html = html.replace("__NODES__", _to_json(data["nodes"]))
    html = html.replace("__EDGES__", _to_json(data["edges"]))
    html = html.replace("__STATS__", f"{len(data['nodes'])} pages, {len(data['edges'])} links")
    output_path.write_text(html, encoding="utf-8")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate an Obsidian-style wiki app from the wiki")
    parser.add_argument("--wiki", default="wiki", help="Wiki directory path")
    parser.add_argument("--output", default="knowledge_graph.html", help="Output HTML file")
    args = parser.parse_args()

    wiki_path = Path(args.wiki)
    if not wiki_path.is_dir():
        print(f"Error: wiki directory not found: {wiki_path}")
        return

    data = build_app_data(wiki_path)
    output = Path(args.output)
    generate_html(data, output)
    print(
        f"Generated: {output} "
        f"({len(data['nodes'])} nodes, {len(data['edges'])} edges, {len(data['pages'])} pages)"
    )


if __name__ == "__main__":
    main()
