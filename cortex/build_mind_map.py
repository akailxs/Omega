#!/usr/bin/env python3
"""
OMEGA — Build 3D Mind Map Graph Data
Reads vault → extracts wiki links + TF-IDF semantic similarity → outputs JSON
Includes graph-UI compiler: layer classification, Jaccard clustering, zoom levels
"""
import re, json, math, yaml
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone

VAULT = Path("/Users/akai/Library/Mobile Documents/iCloud~md~obsidian/Documents/Omega")
EXCLUDE = {"node_modules", ".obsidian", "memory", ".git", ".venv", ".trash",
           "my-agent", "00. Agents", "00. Modèles"}
SIM_THRESHOLD = 0.12
MAX_SEMANTIC_PER_NODE = 4
TODAY = datetime.now(timezone.utc)
STOP_TAG_FREQ = 0.35  # Tags in >35% of tagged notes excluded from Jaccard

STOPWORDS = {
    # French
    'le','la','les','un','une','des','de','du','et','en','est','à','au','aux',
    'pour','par','sur','dans','avec','qui','que','qu','ce','se','sa','son','ses',
    'il','elle','ils','elles','je','tu','nous','vous','on','ne','pas','plus',
    'mais','ou','donc','or','ni','car','sont','avoir','être','fait','peut',
    'cette','cet','ces','leur','leurs','même','bien','tout','très','plus',
    'aussi','comme','si','car','donc','alors','ainsi','entre','vers','dont',
    # English
    'the','of','and','is','in','to','a','that','it','for','are','as','was',
    'with','this','be','by','from','or','an','at','not','but','have','has',
    'its','which','they','we','you','can','all','one','will','about',
    # Common note words
    'note','voir','aussi','cf','exemple','exemples','permet','permet'
}


def extract_tags(content: str):
    tags = re.findall(r'(#[a-zA-Z0-9_À-ÿ\-]+)', content)
    return list(set(tags))

def extract_yaml_tags(content: str):
    """Extract tags from YAML frontmatter (normalized lowercase list)."""
    m = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
    if not m:
        return []
    try:
        fm = yaml.safe_load(m.group(1))
        if fm and isinstance(fm, dict) and 'tags' in fm:
            tags = fm['tags']
            if isinstance(tags, list):
                return [str(t).strip().lower() for t in tags if t]
            elif isinstance(tags, str):
                return [t.strip().lower() for t in tags.split(',') if t.strip()]
    except Exception:
        pass
    return []

# ── Graph-UI compiler helpers ────────────────────────────────────────────────
def gui_clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

def gui_norm(x, mn, mx):
    return gui_clamp((x - mn) / (mx - mn)) if mx > mn else 0.0

def gui_recency(mtime_epoch, max_days=180):
    last_dt = datetime.fromtimestamp(mtime_epoch, tz=timezone.utc)
    days = (TODAY - last_dt).total_seconds() / 86400
    return gui_clamp(1.0 - days / max_days)

def jaccard(a, b):
    sa, sb = set(a), set(b)
    u = len(sa | sb)
    return len(sa & sb) / u if u else 0.0

def uf_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def uf_union(parent, rank, a, b):
    ra, rb = uf_find(parent, a), uf_find(parent, b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    if rank[ra] == rank[rb]:
        rank[ra] += 1

def extract_summary(text, max_chars=220):
    """Extract a short summary: first meaningful lines after frontmatter/title."""
    # Strip frontmatter
    text = re.sub(r'^---[\s\S]*?---\s*', '', text)
    # Strip tags, aliases blocks
    text = re.sub(r'^(tags|aliases|cssclasses)\s*:.*$', '', text, flags=re.MULTILINE)
    lines = text.strip().split('\n')
    out = []
    chars = 0
    for line in lines:
        l = line.strip()
        if not l:
            continue
        # Skip headings (# Title)
        if l.startswith('#'):
            continue
        # Skip pure wiki-link lines or image embeds
        if re.match(r'^!?\[\[', l):
            continue
        # Clean inline markup
        clean = re.sub(r'\[\[([^\]|]+?)(\|[^\]]+)?\]\]', r'\1', l)
        clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean)
        clean = re.sub(r'\*([^*]+)\*', r'\1', clean)
        clean = re.sub(r'`[^`]+`', '', clean)
        clean = clean.strip()
        if len(clean) < 8:
            continue
        out.append(clean)
        chars += len(clean)
        if chars >= max_chars:
            break
    summary = ' '.join(out)
    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(' ', 1)[0] + '...'
    return summary

def tokenize(text):
    # Remove frontmatter, code blocks, links
    text = re.sub(r'^---[\s\S]*?---', '', text)
    text = re.sub(r'```[\s\S]*?```', ' ', text)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = text.lower()
    text = re.sub(r'[^\w\sàâäéèêëîïôùûüçœæ]', ' ', text)
    words = text.split()
    return [w for w in words if len(w) > 2 and w not in STOPWORDS and not w.isdigit()]

def extract_links(text):
    return re.findall(r'\[\[([^\]|#\n]+?)(?:\|[^\]]+)?\]\]', text)

def get_top_folder(path_str):
    parts = Path(path_str).parts
    return parts[0] if len(parts) > 1 else "Root"

# ── Read notes ───────────────────────────────────────────────────────────────
print("Reading vault...")
notes = {}
for f in VAULT.rglob("*.md"):
    if any(ex in f.parts for ex in EXCLUDE):
        continue
    rel = str(f.relative_to(VAULT))
    try:
        text = f.read_text(encoding="utf-8", errors="ignore")
        notes[rel] = text
    except Exception as e:
        print(f"  skip {rel}: {e}")

paths = list(notes.keys())
n = len(paths)
print(f"  → {n} notes")

# ── Name index for wiki-link resolution ──────────────────────────────────────
stem_to_idx = {}
for i, path in enumerate(paths):
    stem = Path(path).stem.lower()
    stem_to_idx[stem] = i
    # Also index without leading numbers like "01 - "
    clean = re.sub(r'^(loi\s*\d+\s*[—-]\s*)', '', stem)
    if clean != stem:
        stem_to_idx[clean] = i

# ── Explicit edges from [[wiki links]] ───────────────────────────────────────
print("Extracting wiki links...")
edges = []
edge_set = set()

for i, path in enumerate(paths):
    for link in extract_links(notes[path]):
        link_clean = link.strip().lower()
        # Try exact stem match
        j = stem_to_idx.get(link_clean)
        if j is None:
            # Partial match: link is substring of stem or vice versa
            for stem, idx in stem_to_idx.items():
                if link_clean in stem or stem in link_clean:
                    j = idx
                    break
        if j is not None and j != i:
            a, b = min(i, j), max(i, j)
            if (a, b) not in edge_set:
                edge_set.add((a, b))
                edges.append({"source": a, "target": b, "type": "explicit"})

print(f"  → {len(edges)} explicit links")

# ── TF-IDF semantic similarity (stdlib only) ─────────────────────────────────
print("Computing TF-IDF similarity...")
token_lists = [tokenize(notes[p]) for p in paths]

# Document frequency
df = Counter()
for tl in token_lists:
    df.update(set(tl))

# IDF with smoothing
idf = {w: math.log((n + 1) / (df[w] + 1)) + 1 for w in df}

# TF-IDF vectors (sparse dicts)
def tfidf_vec(tokens):
    if not tokens:
        return {}
    tf = Counter(tokens)
    total = len(tokens)
    return {w: (c / total) * idf.get(w, 1) for w, c in tf.items()}

vecs = [tfidf_vec(tl) for tl in token_lists]

# L2 norms
norms = []
for v in vecs:
    norm = math.sqrt(sum(x * x for x in v.values()))
    norms.append(norm if norm > 0 else 1.0)

# Cosine similarity (sparse dot product)
def cosine(i, j):
    a, b = vecs[i], vecs[j]
    if not a or not b:
        return 0.0
    # Use smaller vec to iterate
    if len(a) > len(b):
        a, b = b, a
    dot = sum(a[w] * b[w] for w in a if w in b)
    return dot / (norms[i] * norms[j])

# Find top semantic neighbors per node
semantic_added = 0
for i in range(n):
    sims = []
    for j in range(n):
        if i != j:
            s = cosine(i, j)
            if s >= SIM_THRESHOLD:
                sims.append((j, s))
    sims.sort(key=lambda x: -x[1])
    for j, s in sims[:MAX_SEMANTIC_PER_NODE]:
        a, b = min(i, j), max(i, j)
        if (a, b) not in edge_set:
            edge_set.add((a, b))
            edges.append({"source": a, "target": b, "type": "semantic", "weight": round(s, 3)})
            semantic_added += 1

print(f"  → {semantic_added} semantic links")

# ── Build node list ───────────────────────────────────────────────────────────
conn_count = Counter()
explicit_count = Counter()
for e in edges:
    conn_count[e["source"]] += 1
    conn_count[e["target"]] += 1
    if e["type"] == "explicit":
        explicit_count[e["source"]] += 1
        explicit_count[e["target"]] += 1

nodes = []
for i, path in enumerate(paths):
    tokens = token_lists[i]
    f_path = VAULT / path
    try:
        mtime = f_path.stat().st_mtime
    except Exception:
        mtime = 0.0
    yaml_tags = extract_yaml_tags(notes[path])
    is_lit = path.startswith("00. Sources")
    nodes.append({
        "id": i,
        "name": Path(path).stem,
        "path": path,
        "folder": get_top_folder(path),
        "connections": conn_count[i],
        "explicitLinks": explicit_count[i],
        "wordCount": len(tokens),
        "topWords": [w for w, _ in Counter(tokens).most_common(5)],
        "summary": extract_summary(notes[path]),
        "tags": extract_tags(notes[path]),
        "yamlTags": yaml_tags,
        "mtime": mtime,
        "isLiterature": is_lit,
    })

# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH-UI COMPILER — Layer classification, Jaccard clustering, zoom levels
# ═══════════════════════════════════════════════════════════════════════════════
print("Running graph-UI compiler...")

# Maturity scores
TAG_BOOST_SET = {'source', 'ref', 'article', 'import', 'resource', 'project',
                 'concept', 'moc', 'modèle-mental', 'antifragilité'}

for nd in nodes:
    bl = explicit_count[nd["id"]]
    ac = min(50, bl * 3 + max(nd["connections"] - 1, 0) +
         (20 if 'moc' in nd["yamlTags"] else 0) +
         (5 if bl >= 5 else 0) + 1)
    nd["accessCount"] = ac

    s = (0.35 * gui_norm(ac, 0, 50) +
         0.30 * gui_norm(bl, 0, 10) +
         0.25 * gui_norm(nd["wordCount"], 0, 1000) +
         0.10 * gui_recency(nd["mtime"], 180))

    if any(t in TAG_BOOST_SET for t in nd["yamlTags"]):
        s = min(1.0, s + 0.15)
    nd["maturityScore"] = round(s, 4)

# Layer assignment
LIT_TAGS = {'source', 'ref', 'article', 'import', 'resource'}
CONCEPT_TAGS = {'modèle-mental', 'antifragilité'}
MOC_TAGS = {'moc'}

for nd in nodes:
    tl = set(nd["yamlTags"])
    score = nd["maturityScore"]
    bl = explicit_count[nd["id"]]
    days = (TODAY - datetime.fromtimestamp(nd["mtime"], tz=timezone.utc)).total_seconds() / 86400 if nd["mtime"] > 0 else 999

    if nd["isLiterature"] or (tl & LIT_TAGS):
        nd["layer"] = "literature"
    elif bl > 5 and score > 0.70:
        nd["layer"] = "evergreen"
    elif ('project' in tl and days < 14) or score > 0.65:
        nd["layer"] = "project"
    elif (tl & CONCEPT_TAGS) or (tl & MOC_TAGS) or (0.35 <= score <= 0.65):
        nd["layer"] = "concept"
    elif score < 0.20:
        nd["layer"] = "fleeting"
    else:
        nd["layer"] = "raw"

# IDF stop-tag filtering
tagged_nodes = [nd for nd in nodes if nd["yamlTags"]]
n_tagged = len(tagged_nodes)
tag_freq = Counter()
for nd in tagged_nodes:
    for t in set(nd["yamlTags"]):
        tag_freq[t] += 1
stop_tags = {t for t, c in tag_freq.items() if n_tagged > 0 and c / n_tagged > STOP_TAG_FREQ}
print(f"  Stop tags: {stop_tags}")

def filtered_tags(nd):
    return [t for t in nd["yamlTags"] if t not in stop_tags]

# Jaccard clustering with union-find
n_nodes = len(nodes)
parent = list(range(n_nodes))
rank_uf = [0] * n_nodes

for i in range(n_nodes):
    ft_i = filtered_tags(nodes[i])
    if not ft_i:
        continue
    for j in range(i + 1, n_nodes):
        ft_j = filtered_tags(nodes[j])
        if not ft_j:
            continue
        if jaccard(ft_i, ft_j) > 0.5:
            uf_union(parent, rank_uf, i, j)

groups = defaultdict(list)
for i in range(n_nodes):
    r = uf_find(parent, i)
    groups[r].append(i)

clusters = []
node_cluster_map = {}

for _, members in groups.items():
    if len(members) < 2:
        continue
    rep_idx = max(members, key=lambda m: nodes[m]["accessCount"])
    cluster_id = f"cl_{nodes[rep_idx]['name']}"
    clusters.append({
        "clusterId": cluster_id,
        "memberIds": [nodes[m]["id"] for m in sorted(members)],
        "representativeId": nodes[rep_idx]["id"],
        "representativeLabel": nodes[rep_idx]["name"],
        "size": len(members)
    })
    for m in members:
        if m != rep_idx:
            node_cluster_map[m] = cluster_id

# Assign cluster + zoom to each node
for nd in nodes:
    nd["compressedInto"] = node_cluster_map.get(nd["id"], None)
    if nd["compressedInto"]:
        nd["zoomLevel"] = "cluster"
    elif nd["layer"] in ("evergreen", "project"):
        nd["zoomLevel"] = "content"
    else:
        nd["zoomLevel"] = "node"

# Actions
actions = []
for cl in clusters:
    actions.append({"action": "compress", "targetId": cl["clusterId"]})
for nd_id, cl_id in sorted(node_cluster_map.items()):
    actions.append({"action": "expand", "targetId": nodes[nd_id]["name"]})

# Layer stats
layer_counts = Counter(nd["layer"] for nd in nodes)
print(f"  Layers: {dict(sorted(layer_counts.items(), key=lambda x: -x[1]))}")
print(f"  Clusters: {len(clusters)} ({sum(c['size'] for c in clusters)} nodes)")

# ── Network analysis ──────────────────────────────────────────────────────────
display_nodes = [nd for nd in nodes if nd["folder"] != "memory"]
display_ids = {nd["id"] for nd in display_nodes}
display_links = [e for e in edges if e["source"] in display_ids and e["target"] in display_ids]

n_d = len(display_nodes)
n_e = len(display_links)
density = round(2 * n_e / (n_d * (n_d - 1)), 4) if n_d > 1 else 0

conns = [nd["connections"] for nd in display_nodes]
avg_conn = sum(conns) / len(conns) if conns else 0
max_conn = max(conns) if conns else 0

# Gini coefficient (centralization)
sorted_c = sorted(conns)
gini = sum((2 * (i + 1) - n_d - 1) * c for i, c in enumerate(sorted_c))
gini = abs(gini / (n_d * n_d * avg_conn)) if avg_conn > 0 else 0

# Cross-folder ratio (distribution)
cross = sum(1 for e in display_links if
    nodes[e["source"]]["folder"] != nodes[e["target"]]["folder"])
cross_ratio = cross / n_e if n_e > 0 else 0

analysis = {
    "nodes": n_d,
    "edges": n_e,
    "density": density,
    "avgConnections": round(avg_conn, 2),
    "maxConnections": max_conn,
    "gini": round(gini, 3),
    "crossFolderRatio": round(cross_ratio, 3),
    "centralScore": min(100, round(gini * 120)),
    "decentralScore": min(100, round((1 - cross_ratio) * 90)),
    "distribScore": min(100, round(cross_ratio * 140)),
    "layerCounts": dict(layer_counts),
    "clusterCount": len(clusters),
}

graph = {
    "nodes": nodes,
    "links": edges,
    "clusters": clusters,
    "actions": actions,
    "analysis": analysis
}

out = VAULT / "mind_map.json"
out.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")

print(f"\n✓ DONE")
print(f"  {n_d} nodes | {n_e} edges | density {density}")
print(f"  Centralized: {analysis['centralScore']}% | Distributed: {analysis['distribScore']}%")
print(f"  → {out}")
print(f"\nOuvrir le cerveau:")
print(f"  cd '{VAULT}' && python3 -m http.server 8765")
print(f"  → http://localhost:8765/mind_map.html")
