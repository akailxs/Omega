from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


VAULT_PATH = Path(__file__).resolve().parent
OUTPUT_PATH = VAULT_PATH / "data.json"
EXCLUDED_DIRECTORIES = {
    ".obsidian",
    ".git",
    ".trash",
    "__pycache__",
    "node_modules",
    "memory",
}
WIKI_LINK_PATTERN = re.compile(r"\[\[([^\]|#\n]+?)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
TAG_PATTERN = re.compile(r"(#[a-zA-Z0-9_À-ÿ\-]+)")
HEADING_PATTERN = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n.*?\n---\s*\n?", re.DOTALL)


def strip_frontmatter(content: str) -> str:
    return FRONTMATTER_PATTERN.sub("", content, count=1)


def extract_title(content: str, fallback: str) -> str:
    clean_content = strip_frontmatter(content)
    match = HEADING_PATTERN.search(clean_content)
    if match:
        return match.group(1).strip()
    return fallback


def extract_tags(content: str) -> list[str]:
    return sorted(set(TAG_PATTERN.findall(content)))


def extract_links(content: str) -> list[str]:
    return [match.strip() for match in WIKI_LINK_PATTERN.findall(content)]


def iter_markdown_files(vault_path: Path) -> list[Path]:
    markdown_files: list[Path] = []
    for path in vault_path.rglob("*.md"):
        if any(part in EXCLUDED_DIRECTORIES for part in path.parts):
            continue
        markdown_files.append(path)
    return sorted(markdown_files)


def resolve_group(relative_path: Path) -> str:
    return relative_path.parts[0] if len(relative_path.parts) > 1 else "Root"


def build_name_index(relative_paths: list[Path]) -> dict[str, str]:
    index: dict[str, str] = {}
    for relative_path in relative_paths:
        node_id = relative_path.as_posix()
        stem = relative_path.stem.strip().lower()
        index.setdefault(stem, node_id)
        index.setdefault(node_id.lower(), node_id)
    return index


def resolve_link_target(link: str, name_index: dict[str, str]) -> str | None:
    normalized = link.strip().lower()
    if not normalized:
        return None

    direct = name_index.get(normalized)
    if direct is not None:
        return direct

    for candidate, node_id in name_index.items():
        if candidate.endswith(f"/{normalized}") or candidate == normalized:
            return node_id

    for candidate, node_id in name_index.items():
        if normalized in candidate or candidate in normalized:
            return node_id

    return None


def main() -> None:
    markdown_files = iter_markdown_files(VAULT_PATH)
    relative_paths = [path.relative_to(VAULT_PATH) for path in markdown_files]
    name_index = build_name_index(relative_paths)

    notes_by_id: dict[str, dict[str, object]] = {}
    for path, relative_path in zip(markdown_files, relative_paths):
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            print(f"Impossible de lire {relative_path.as_posix()}: {exc}")
            continue

        node_id = relative_path.as_posix()
        notes_by_id[node_id] = {
            "id": node_id,
            "label": extract_title(content, relative_path.stem),
            "group": resolve_group(relative_path),
            "path": node_id,
            "tags": extract_tags(content),
            "raw_links": extract_links(content),
        }

    edge_keys: set[tuple[str, str]] = set()
    links: list[dict[str, object]] = []
    connection_counts: Counter[str] = Counter()

    for source_id, note in notes_by_id.items():
        raw_links = note.get("raw_links", [])
        if not isinstance(raw_links, list):
            continue

        for raw_link in raw_links:
            if not isinstance(raw_link, str):
                continue
            target_id = resolve_link_target(raw_link, name_index)
            if target_id is None or target_id == source_id or target_id not in notes_by_id:
                continue

            edge_key = tuple(sorted((source_id, target_id)))
            if edge_key in edge_keys:
                continue

            edge_keys.add(edge_key)
            links.append(
                {
                    "source": source_id,
                    "target": target_id,
                }
            )
            connection_counts[source_id] += 1
            connection_counts[target_id] += 1

    nodes: list[dict[str, object]] = []
    for node_id, note in notes_by_id.items():
        weight = connection_counts[node_id]
        nodes.append(
            {
                "id": note["id"],
                "label": note["label"],
                "group": note["group"],
                "size": max(6, 6 + (weight * 2)),
                "weight": weight,
                "path": note["path"],
                "tags": note["tags"],
            }
        )

    nodes.sort(key=lambda node: str(node["label"]).lower())
    links.sort(key=lambda link: (str(link["source"]).lower(), str(link["target"]).lower()))

    payload = {
        "nodes": nodes,
        "links": links,
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Graph exporté vers {OUTPUT_PATH}")
    print(f"{len(nodes)} nœuds | {len(links)} liens")


if __name__ == "__main__":
    main()
