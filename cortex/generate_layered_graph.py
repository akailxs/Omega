#!/usr/bin/env python3
"""Generate layered graph data from Obsidian vault for 3D visualization.

Scans the vault, classifies each note into a layer (0-3),
extracts wikilinks, and outputs JSON for cortex_layers.html.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

VAULT_PATH = Path(__file__).resolve().parent.parent.parent
OUTPUT_PATH = Path(__file__).resolve().parent / "layered_graph.json"

EXCLUDED = {
    ".obsidian", ".git", ".trash", "__pycache__", "node_modules",
    "_DEV", "memory", "00. Agents", ".smart-env", ".claude",
    "my-agent", "00. Modèles", "00. Pièces jointes", "04. Journal",
}

WIKI_LINK = re.compile(r"\[\[([^\]|#\n]+?)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
HEADING = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)
FRONTMATTER = re.compile(r"^---\s*\n.*?\n---\s*\n?", re.DOTALL)
YAML_TAGS = re.compile(r"tags:\s*\n((?:\s+-\s+.+\n)*)", re.MULTILINE)
YAML_TAG_ITEM = re.compile(r"^\s+-\s+(.+)$", re.MULTILINE)


def extract_title(content: str, fallback: str) -> str:
    clean = FRONTMATTER.sub("", content, count=1)
    m = HEADING.search(clean)
    return m.group(1).strip() if m else fallback


def extract_links(content: str) -> list[str]:
    return [m.strip() for m in WIKI_LINK.findall(content)]


def extract_tags(content: str) -> list[str]:
    tags = set()
    yaml_match = YAML_TAGS.search(content)
    if yaml_match:
        for t in YAML_TAG_ITEM.findall(yaml_match.group(1)):
            tags.add(t.strip().strip("\"'"))
    return sorted(tags)


def classify_type(note_id: str, weight: int) -> str:
    """Classify note into type based on path and connectivity.

    Layer 3 (idea): MOCs, wiki-index, high-connectivity hubs (8+)
    Layer 2 (concept): Mental Models
    Layer 1 (technical): Bibliothèque, Projects, Areas
    Layer 0 (resource): Sources, Inbox, everything else
    """
    p = note_id.lower()

    if note_id.startswith("MOC") or note_id == "wiki-index.md":
        return "idea"
    if weight >= 20:
        return "idea"

    if "01. knowledge/mental models/" in p:
        return "concept"

    if any(x in p for x in [
        "01. knowledge/bibliothèque/",
        "02. projects/",
        "03. areas/",
    ]):
        return "technical"

    return "resource"


TYPE_TO_LAYER = {"resource": 0, "technical": 1, "concept": 2, "idea": 3}


def build_name_index(note_ids: list[str]) -> dict[str, str]:
    index: dict[str, str] = {}
    for nid in note_ids:
        stem = Path(nid).stem.strip().lower()
        index.setdefault(stem, nid)
        index.setdefault(nid.lower(), nid)
    return index


def resolve_link(link: str, name_index: dict[str, str]) -> str | None:
    norm = link.strip().lower()
    if not norm:
        return None

    hit = name_index.get(norm)
    if hit:
        return hit

    for k, v in name_index.items():
        if k.endswith("/" + norm) or k == norm:
            return v

    for k, v in name_index.items():
        if norm in k or k in norm:
            return v

    return None


def main() -> None:
    files = []
    for f in sorted(VAULT_PATH.rglob("*.md")):
        if any(part in EXCLUDED for part in f.relative_to(VAULT_PATH).parts):
            continue
        if f.name == "CLAUDE.md" or f.name == ".obsidianignore":
            continue
        files.append(f)

    notes: dict[str, dict] = {}
    for f in files:
        try:
            content = f.read_text("utf-8", errors="ignore")
        except OSError:
            continue

        rel = f.relative_to(VAULT_PATH)
        nid = rel.as_posix()
        notes[nid] = {
            "id": nid,
            "title": extract_title(content, rel.stem),
            "tags": extract_tags(content),
            "raw_links": extract_links(content),
        }

    name_index = build_name_index(list(notes.keys()))

    edges: list[dict] = []
    seen: set[tuple[str, str]] = set()
    conn: Counter[str] = Counter()

    for src_id, note in notes.items():
        for link in note["raw_links"]:
            tgt_id = resolve_link(link, name_index)
            if not tgt_id or tgt_id == src_id or tgt_id not in notes:
                continue
            key = tuple(sorted((src_id, tgt_id)))
            if key in seen:
                continue
            seen.add(key)
            edges.append({"source": src_id, "target": tgt_id})
            conn[src_id] += 1
            conn[tgt_id] += 1

    # Classify and build output
    nodes_out = []
    for nid, note in notes.items():
        w = conn[nid]
        ntype = classify_type(nid, w)
        nodes_out.append({
            "id": nid,
            "title": note["title"],
            "tags": note["tags"],
            "type": ntype,
            "layer_index": TYPE_TO_LAYER[ntype],
            "connections": w,
            "links": [
                e["target"] if e["source"] == nid else e["source"]
                for e in edges if e["source"] == nid or e["target"] == nid
            ],
        })

    node_layer = {n["id"]: n["layer_index"] for n in nodes_out}
    edges_out = [
        {
            "source": e["source"],
            "target": e["target"],
            "type": "vertical" if node_layer.get(e["source"], 0) != node_layer.get(e["target"], 0) else "horizontal",
        }
        for e in edges
    ]

    layers = [
        {"layer_index": 0, "label": "Resources"},
        {"layer_index": 1, "label": "Technical Logic"},
        {"layer_index": 2, "label": "Conceptual Layer"},
        {"layer_index": 3, "label": "Core Ideas"},
    ]

    output = {
        "nodes": sorted(nodes_out, key=lambda n: n["title"].lower()),
        "edges": edges_out,
        "layers": layers,
    }

    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), "utf-8")

    print(f"✓ {len(nodes_out)} nodes | {len(edges_out)} edges → {OUTPUT_PATH.name}")
    for la in layers:
        c = sum(1 for n in nodes_out if n["layer_index"] == la["layer_index"])
        v = sum(1 for e in edges_out if e["type"] == "vertical"
                and (node_layer.get(e["source"]) == la["layer_index"]
                     or node_layer.get(e["target"]) == la["layer_index"]))
        print(f"  L{la['layer_index']} {la['label']:20s}: {c:3d} nodes, {v:3d} vertical edges")


if __name__ == "__main__":
    main()
