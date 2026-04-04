# %% [markdown]
# # Graph Theory Analysis — Sacred Concept Network
#
# Loads the balanced concept graph saved by script 25 and runs:
#   1. Centrality (degree, eigenvector, betweenness, closeness)
#   2. Louvain community detection + tradition entropy per community
#   3. Cross-tradition edges ranked by weight
#   4. Tradition subgraph stats (density, avg degree, diameter)
#   5. Shortest paths between tradition hubs
#   6. Network diameter + narrated path
#
# Saves results as:
#   26_graph_analysis.md   — human-readable report
#   26_graph_analysis.json — structured data for future use

# %% Imports
import json
import math
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    print("Warning: python-louvain not installed. Community detection will be skipped.")
    print("  pip install python-louvain")

TRADITION_COLORS = {
    "Abrahamic": "#e05c5c",
    "Dharmic":   "#f0a500",
    "Buddhist":  "#44bb99",
    "Taoist":    "#9b59b6",
}
TRADITIONS = list(TRADITION_COLORS.keys())

# %% Load graph
with open("25_concept_graph.pkl", "rb") as f:
    G = pickle.load(f)

print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Work on largest connected component for path-based metrics
lcc_nodes = max(nx.connected_components(G), key=len)
LCC = G.subgraph(lcc_nodes).copy()
print(f"Largest connected component: {LCC.number_of_nodes()} nodes, {LCC.number_of_edges()} edges")

def node_label(G, n):
    return f"C{n} [{G.nodes[n]['label']}]"

def tradition_entropy(trad_counts: dict) -> float:
    total = sum(trad_counts.values())
    if total == 0:
        return 0.0
    probs = [v / total for v in trad_counts.values() if v > 0]
    return float(-sum(p * math.log(p) for p in probs))

def trad_counts_for_node(G, n) -> dict:
    return {t: G.nodes[n].get(f"n_{t.lower()}", 0) for t in TRADITIONS}

def trad_counts_for_nodes(G, nodes) -> dict:
    totals = {t: 0 for t in TRADITIONS}
    for n in nodes:
        for t in TRADITIONS:
            totals[t] += G.nodes[n].get(f"n_{t.lower()}", 0)
    return totals

# %% ── 1. Centrality ──────────────────────────────────────────────────────────
print("\nComputing centrality measures ...")

degree_cent    = nx.degree_centrality(G)
between_cent   = nx.betweenness_centrality(G, weight="weight")
closeness_cent = nx.closeness_centrality(G)
try:
    eigen_cent = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
except nx.PowerIterationFailedConvergence:
    eigen_cent = nx.degree_centrality(G)
    print("  Eigenvector centrality did not converge — using degree as fallback")

def top_n_centrality(cent_dict, n=10):
    return sorted(cent_dict.items(), key=lambda x: -x[1])[:n]

print("\nTop 10 — Betweenness:")
for node, score in top_n_centrality(between_cent):
    print(f"  {node_label(G, node):40s}  {score:.4f}  [{G.nodes[node]['dom_tradition']}]")

print("\nTop 10 — Eigenvector:")
for node, score in top_n_centrality(eigen_cent):
    print(f"  {node_label(G, node):40s}  {score:.4f}  [{G.nodes[node]['dom_tradition']}]")

# %% ── 2. Community detection ─────────────────────────────────────────────────
communities_data = []

if HAS_LOUVAIN:
    print("\nRunning Louvain community detection ...")
    partition   = community_louvain.best_partition(G, weight="weight", random_state=42)
    comm_groups = defaultdict(list)
    for node, comm in partition.items():
        comm_groups[comm].append(node)

    print(f"\n{len(comm_groups)} communities found")
    for comm_id, nodes in sorted(comm_groups.items(), key=lambda x: -len(x[1])):
        tc      = trad_counts_for_nodes(G, nodes)
        ent     = tradition_entropy(tc)
        dom     = max(tc, key=tc.get)
        dom_pct = tc[dom] / max(sum(tc.values()), 1)

        print(f"\n  Community {comm_id} ({len(nodes)} nodes | entropy={ent:.3f} | dom={dom} {dom_pct:.0%})")
        print(f"  Traditions: {' | '.join(f'{t}: {v}' for t, v in tc.items() if v > 0)}")
        for n in sorted(nodes):
            print(f"    {node_label(G, n):40s}  [{G.nodes[n]['dom_tradition']}]")

        communities_data.append({
            "community_id":    comm_id,
            "nodes":           nodes,
            "node_labels":     [G.nodes[n]["label"] for n in nodes],
            "tradition_counts": tc,
            "entropy":         ent,
            "dominant":        dom,
            "dominant_pct":    float(dom_pct),
        })
else:
    print("\nSkipping community detection (python-louvain not installed)")

# %% ── 3. Cross-tradition edges ───────────────────────────────────────────────
print("\n=== Cross-tradition edges (ranked by weight) ===")

cross_edges = []
for u, v, data in G.edges(data=True):
    t_u = G.nodes[u]["dom_tradition"]
    t_v = G.nodes[v]["dom_tradition"]
    if t_u != t_v:
        cross_edges.append({
            "node_a":   u,
            "label_a":  G.nodes[u]["label"],
            "trad_a":   t_u,
            "node_b":   v,
            "label_b":  G.nodes[v]["label"],
            "trad_b":   t_v,
            "weight":   data["weight"],
        })

cross_edges.sort(key=lambda x: -x["weight"])
print(f"\nTotal cross-tradition edges: {len(cross_edges)}")
print(f"Top 20:")
for e in cross_edges[:20]:
    print(f"  [{e['weight']:3d}]  {e['trad_a']:10s} {e['label_a']:30s}  ↔  {e['trad_b']:10s} {e['label_b']}")

# %% ── 4. Tradition subgraph stats ────────────────────────────────────────────
print("\n=== Tradition subgraph statistics ===")

subgraph_stats = {}
for t in TRADITIONS:
    t_nodes = [n for n in G.nodes if G.nodes[n]["dom_tradition"] == t]
    if len(t_nodes) < 2:
        continue
    SG    = G.subgraph(t_nodes).copy()
    lcc_t = max(nx.connected_components(SG), key=len) if nx.number_connected_components(SG) > 0 else set()
    SG_lcc = SG.subgraph(lcc_t).copy()

    stats = {
        "nodes":        len(t_nodes),
        "edges":        SG.number_of_edges(),
        "density":      float(nx.density(SG)),
        "avg_degree":   float(sum(dict(SG.degree()).values()) / max(len(t_nodes), 1)),
        "lcc_size":     len(lcc_t),
        "diameter":     nx.diameter(SG_lcc) if len(SG_lcc) > 1 else 0,
    }
    subgraph_stats[t] = stats
    print(f"\n  {t}: {stats['nodes']} nodes | {stats['edges']} edges | "
          f"density={stats['density']:.3f} | avg_degree={stats['avg_degree']:.1f} | "
          f"diameter={stats['diameter']}")

# %% ── 5. Shortest paths between tradition hubs ───────────────────────────────
# Hub = highest eigenvector centrality node per tradition (in LCC)

print("\n=== Shortest paths between tradition hubs ===")

trad_hubs = {}
for t in TRADITIONS:
    t_nodes_in_lcc = [n for n in LCC.nodes if LCC.nodes[n]["dom_tradition"] == t]
    if not t_nodes_in_lcc:
        continue
    hub = max(t_nodes_in_lcc, key=lambda n: eigen_cent.get(n, 0))
    trad_hubs[t] = hub
    print(f"  Hub for {t}: {node_label(LCC, hub)}")

shortest_paths = {}
trad_list = [t for t in TRADITIONS if t in trad_hubs]

for i, t_a in enumerate(trad_list):
    for t_b in trad_list[i+1:]:
        h_a, h_b = trad_hubs[t_a], trad_hubs[t_b]
        try:
            path = nx.shortest_path(LCC, h_a, h_b, weight=None)
            key  = f"{t_a} → {t_b}"
            shortest_paths[key] = {
                "from":        h_a,
                "to":          h_b,
                "path_nodes":  path,
                "path_labels": [LCC.nodes[n]["label"] for n in path],
                "length":      len(path) - 1,
            }
            print(f"\n  {key} (length {len(path)-1}):")
            for n in path:
                print(f"    {node_label(LCC, n):40s} [{LCC.nodes[n]['dom_tradition']}]")
        except nx.NetworkXNoPath:
            print(f"  No path: {t_a} → {t_b}")

# %% ── 6. Network diameter ────────────────────────────────────────────────────
print("\n=== Network diameter (largest connected component) ===")

diameter = nx.diameter(LCC)
periphery = nx.periphery(LCC)
center    = nx.center(LCC)

print(f"  Diameter: {diameter}")
print(f"  Center nodes (eccentricity = radius):")
for n in center[:5]:
    print(f"    {node_label(LCC, n)} [{LCC.nodes[n]['dom_tradition']}]")

# Find the diameter path — pick one periphery pair that achieves the diameter
diam_path = None
for n in periphery[:5]:
    lengths, paths = nx.single_source_shortest_path_length(LCC, n), nx.single_source_shortest_path(LCC, n)
    far_node = max(lengths, key=lengths.get)
    if lengths[far_node] == diameter:
        diam_path = paths[far_node]
        break

print(f"\n  Diameter path ({diameter} hops):")
if diam_path:
    for n in diam_path:
        print(f"    {node_label(LCC, n):40s} [{LCC.nodes[n]['dom_tradition']}]")

diameter_data = {
    "diameter":      diameter,
    "center_nodes":  [{"node": n, "label": LCC.nodes[n]["label"],
                       "tradition": LCC.nodes[n]["dom_tradition"]} for n in center],
    "diameter_path": [{"node": n, "label": LCC.nodes[n]["label"],
                       "tradition": LCC.nodes[n]["dom_tradition"]} for n in (diam_path or [])],
}

# %% ── Save markdown report ───────────────────────────────────────────────────
def md_table(headers, rows):
    sep  = "| " + " | ".join("---" for _ in headers) + " |"
    head = "| " + " | ".join(headers) + " |"
    body = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return "\n".join([head, sep, body])

lines = ["# Sacred Concept Network — Graph Theory Analysis\n"]

# 1. Centrality
lines.append("## 1. Centrality Rankings\n")
for name, cent in [("Betweenness", between_cent), ("Eigenvector", eigen_cent),
                   ("Degree", degree_cent), ("Closeness", closeness_cent)]:
    lines.append(f"### {name} (Top 10)\n")
    rows = [(rank+1, node_label(G, n), G.nodes[n]["dom_tradition"], f"{s:.4f}")
            for rank, (n, s) in enumerate(top_n_centrality(cent))]
    lines.append(md_table(["Rank", "Cluster", "Tradition", "Score"], rows))
    lines.append("")

# 2. Communities
if communities_data:
    lines.append("## 2. Communities (Louvain)\n")
    for cd in sorted(communities_data, key=lambda x: -x["entropy"]):
        tc_str = " | ".join(f"{t}: {v}" for t, v in cd["tradition_counts"].items() if v > 0)
        lines.append(f"### Community {cd['community_id']} — entropy {cd['entropy']:.3f} | {tc_str}\n")
        for n, lbl in zip(cd["nodes"], cd["node_labels"]):
            lines.append(f"- C{n} [{lbl}] — {G.nodes[n]['dom_tradition']}")
        lines.append("")

# 3. Cross-tradition edges
lines.append("## 3. Cross-Tradition Edges (Top 20)\n")
rows = [(e["weight"], e["trad_a"], e["label_a"], e["trad_b"], e["label_b"])
        for e in cross_edges[:20]]
lines.append(md_table(["Weight", "Tradition A", "Concept A", "Tradition B", "Concept B"], rows))
lines.append("")

# 4. Subgraph stats
lines.append("## 4. Tradition Subgraph Statistics\n")
rows = [(t, s["nodes"], s["edges"], f"{s['density']:.3f}",
         f"{s['avg_degree']:.1f}", s["diameter"])
        for t, s in subgraph_stats.items()]
lines.append(md_table(["Tradition", "Nodes", "Edges", "Density", "Avg Degree", "Diameter"], rows))
lines.append("")

# 5. Shortest paths
lines.append("## 5. Shortest Paths Between Tradition Hubs\n")
for key, sp in shortest_paths.items():
    lines.append(f"### {key} (length {sp['length']})\n")
    for n, lbl in zip(sp["path_nodes"], sp["path_labels"]):
        lines.append(f"- C{n} [{lbl}] — {LCC.nodes[n]['dom_tradition']}")
    lines.append("")

# 6. Diameter
lines.append("## 6. Network Diameter\n")
lines.append(f"**Diameter:** {diameter} hops\n")
if diam_path:
    lines.append("**Diameter path:**\n")
    for n in diam_path:
        lines.append(f"- C{n} [{LCC.nodes[n]['label']}] — {LCC.nodes[n]['dom_tradition']}")
lines.append("")

md_text = "\n".join(lines)
with open("26_graph_analysis.md", "w") as f:
    f.write(md_text)
print("\nSaved: 26_graph_analysis.md")

# %% ── Save JSON ──────────────────────────────────────────────────────────────
output = {
    "graph": {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "lcc_nodes": LCC.number_of_nodes(),
        "lcc_edges": LCC.number_of_edges(),
    },
    "centrality": {
        "betweenness": [{"node": n, "label": G.nodes[n]["label"],
                         "tradition": G.nodes[n]["dom_tradition"], "score": round(s, 5)}
                        for n, s in top_n_centrality(between_cent, 20)],
        "eigenvector": [{"node": n, "label": G.nodes[n]["label"],
                         "tradition": G.nodes[n]["dom_tradition"], "score": round(s, 5)}
                        for n, s in top_n_centrality(eigen_cent, 20)],
        "degree":      [{"node": n, "label": G.nodes[n]["label"],
                         "tradition": G.nodes[n]["dom_tradition"], "score": round(s, 5)}
                        for n, s in top_n_centrality(degree_cent, 20)],
        "closeness":   [{"node": n, "label": G.nodes[n]["label"],
                         "tradition": G.nodes[n]["dom_tradition"], "score": round(s, 5)}
                        for n, s in top_n_centrality(closeness_cent, 20)],
    },
    "communities":         communities_data,
    "cross_tradition_edges": cross_edges[:50],
    "tradition_subgraphs": subgraph_stats,
    "shortest_paths":      shortest_paths,
    "diameter":            diameter_data,
}

with open("26_graph_analysis.json", "w") as f:
    json.dump(output, f, indent=2)
print("Saved: 26_graph_analysis.json")

# %% ── Summary plot: centrality comparison ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (name, cent) in zip(axes, [("Betweenness", between_cent), ("Eigenvector", eigen_cent)]):
    top    = top_n_centrality(cent, 12)
    labels = [G.nodes[n]["label"][:25] for n, _ in top]
    scores = [s for _, s in top]
    colors = [TRADITION_COLORS.get(G.nodes[n]["dom_tradition"], "#aaa") for n, _ in top]
    ax.barh(range(len(top)), scores[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_xlabel("Score")
    ax.set_title(f"{name} Centrality — Top 12", fontweight="bold")

patches = [mpatches.Patch(color=c, label=t) for t, c in TRADITION_COLORS.items()]
fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.05))
plt.tight_layout()
plt.savefig("26_centrality.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 26_centrality.png")
