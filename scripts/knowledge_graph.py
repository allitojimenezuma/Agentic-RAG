#!/usr/bin/env python3
"""Generate an interactive knowledge graph visualization from the wiki."""

import json
from pathlib import Path
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentic_rag.io.wiki_io import list_pages
from agentic_rag.io.markdown_parser import extract_links, parse_frontmatter, slugify


def build_graph(wiki_path: Path) -> dict:
    """Build nodes and edges from wiki pages."""
    pages = list_pages(wiki_path)
    nodes = {}
    edges = []
    page_slugs = set()

    # First pass: collect all pages
    for page_path in pages:
        slug = str(page_path.relative_to(wiki_path)).removesuffix(".md")
        page_slugs.add(slug)
        try:
            content = page_path.read_text(encoding="utf-8")
            fm = parse_frontmatter(content)
            nodes[slug] = {
                "id": slug,
                "label": fm.title or slug,
                "type": fm.type or "unknown",
                "group": fm.type or "unknown",
            }
        except Exception:
            nodes[slug] = {"id": slug, "label": slug, "type": "unknown", "group": "unknown"}

    # Second pass: extract links
    # Build a lookup: slugified name -> full slug
    slug_lookup = {}
    for ps in page_slugs:
        # "concepts/ml" -> "ml", "entities/álvaro-jiménez-martínez" -> "álvaro-jiménez-martínez"
        short = ps.rsplit("/", 1)[-1] if "/" in ps else ps
        slug_lookup[short] = ps
        slug_lookup[slugify(ps.split("/")[-1])] = ps
        # Also map the full slugified title
        slug_lookup[slugify(ps)] = ps

    for page_path in pages:
        source_slug = str(page_path.relative_to(wiki_path)).removesuffix(".md")
        try:
            content = page_path.read_text(encoding="utf-8")
            links = extract_links(content)
            for link in links:
                target = link.target
                # Resolve: try exact match, then slugify match
                target_slug = None
                if target in page_slugs:
                    target_slug = target
                else:
                    s = slugify(target)
                    # Try matching the slugified target against lookup
                    if s in slug_lookup:
                        target_slug = slug_lookup[s]
                    else:
                        # Try partial match (slug ends with the target slugified)
                        for ps in page_slugs:
                            if ps.endswith("/" + s) or ps.rsplit("/", 1)[-1] == s:
                                target_slug = ps
                                break
                if target_slug and target_slug != source_slug:
                    edges.append((source_slug, target_slug))
        except Exception:
            continue

    # Deduplicate edges
    seen = set()
    unique_edges = []
    for src, tgt in edges:
        key = (src, tgt)
        if key not in seen:
            seen.add(key)
            unique_edges.append({"from": src, "to": tgt})
    edges = unique_edges

    return {"nodes": list(nodes.values()), "edges": edges}


def generate_html(graph: dict, output_path: Path) -> None:
    """Generate an interactive HTML graph using vis.js."""
    colors = {
        "entity": "#4CAF50",
        "concept": "#2196F3",
        "source": "#FF9800",
        "comparison": "#9C27B0",
        "overview": "#607D8B",
        "unknown": "#9E9E9E",
    }

    nodes_js = json.dumps([
        {
            "id": n["id"],
            "label": n["label"],
            "color": colors.get(n["type"], "#9E9E9E"),
            "shape": "dot" if n["type"] != "source" else "diamond",
            "size": 20 if n["type"] != "source" else 15,
            "title": f"{n['label']}\nType: {n['type']}",
        }
        for n in graph["nodes"]
    ])
    edges_js = json.dumps([
        {"from": e["from"], "to": e["to"], "arrows": "to"}
        for e in graph["edges"]
    ])

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Wiki Knowledge Graph</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ margin: 0; font-family: sans-serif; background: #1a1a2e; color: #eee; }}
        #graph {{ width: 100vw; height: 100vh; }}
        #legend {{
            position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.7);
            padding: 12px 16px; border-radius: 8px; font-size: 13px;
        }}
        #legend div {{ margin: 4px 0; }}
        .dot {{ display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }}
        #stats {{
            position: absolute; bottom: 10px; left: 10px; background: rgba(0,0,0,0.7);
            padding: 8px 12px; border-radius: 8px; font-size: 12px;
        }}
    </style>
</head>
<body>
    <div id="legend">
        <div><span class="dot" style="background:#4CAF50"></span>Entity</div>
        <div><span class="dot" style="background:#2196F3"></span>Concept</div>
        <div><span class="dot" style="background:#FF9800"></span>Source</div>
        <div><span class="dot" style="background:#9C27B0"></span>Comparison</div>
        <div><span class="dot" style="background:#607D8B"></span>Overview</div>
    </div>
    <div id="stats">{len(graph['nodes'])} pages, {len(graph['edges'])} links</div>
    <div id="graph"></div>
    <script>
        var nodes = new vis.DataSet({nodes_js});
        var edges = new vis.DataSet({edges_js});
        var container = document.getElementById('graph');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
            physics: {{
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {{
                    gravitationalConstant: -200,
                    centralGravity: 0.01,
                    springLength: 200,
                    springConstant: 0.02,
                    damping: 0.4,
                }},
                stabilization: {{ iterations: 300 }},
            }},
            nodes: {{
                font: {{ color: '#ccc', size: 12 }},
                borderWidth: 2,
            }},
            interaction: {{ hover: true, zoomView: true, dragView: true, tooltipDelay: 100 }},
            edges: {{
                color: {{ color: 'rgba(150,150,150,0.5)', highlight: '#fff' }},
                smooth: {{ type: 'continuous' }},
                width: 1.5,
            }},
        }};
        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>"""
    output_path.write_text(html)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate knowledge graph from wiki")
    parser.add_argument("--wiki", default="wiki", help="Wiki directory path")
    parser.add_argument("--output", default="knowledge_graph.html", help="Output HTML file")
    args = parser.parse_args()

    wiki_path = Path(args.wiki)
    if not wiki_path.is_dir():
        print(f"Error: wiki directory not found: {wiki_path}")
        return

    graph = build_graph(wiki_path)
    output = Path(args.output)
    generate_html(graph, output)
    print(f"Generated: {output} ({len(graph['nodes'])} nodes, {len(graph['edges'])} edges)")


if __name__ == "__main__":
    main()
