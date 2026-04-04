# %% [markdown]
# # Concept Network — Centroid Similarity Edges (Script 36b)
#
# Loads precomputed data from script 36 (no DB access, no ML inference):
#   - 36_concept_graph.pkl  — cluster nodes with all metadata
#   - 36_centroids.pkl      — L2-normalised cluster centroids
#
# Draws edges only between clusters whose centroids are semantically similar
# AND which are both multi-tradition. These edges answer:
# "Do these ideas mean similar things across traditions?"
# rather than the co-occurrence question in script 36.
#
# Nodes with many centroid edges spanning multiple traditions are the
# strongest perennial philosophy candidates.

# %% Imports
import pickle
import numpy as np
import networkx as nx
from pyvis.network import Network

# %% Parameters
CENTROID_SIM_THRESHOLD  = 0.7  # cosine similarity cutoff — tune up/down to taste
CENTROID_MIN_TRADITIONS = 2    # cluster must span this many traditions to qualify

TRADITION_COLORS = {
    "Abrahamic":  "#e63946",
    "Dharmic":    "#ffd166",
    "Buddhist":   "#06d6a0",
    "Taoist":     "#7209b7",
    "Norse":      "#118ab2",
    "Confucian":  "#f4845f",
    "Philosophy": "#f72585",
    "Scientific": "#b5e853",
    "Literature": "#48cae4",
    "Historical": "#cb997e",
}

# %% Load
with open("36_concept_graph.pkl", "rb") as f:
    G = pickle.load(f)

with open("36_centroids.pkl", "rb") as f:
    centroids = pickle.load(f)

cluster_ids = sorted(G.nodes())
print(f"Loaded {len(cluster_ids)} clusters")

# %% Multi-tradition clusters only
multi_trad = [
    c for c in cluster_ids
    if sum(1 for k, v in G.nodes[c].items() if k.startswith("n_") and v > 0) >= CENTROID_MIN_TRADITIONS
]
print(f"Multi-tradition clusters (>= {CENTROID_MIN_TRADITIONS}): {len(multi_trad)}")

# %% Centroid similarity edges
centroid_mat = np.stack([centroids[c] for c in multi_trad])  # (n, 768)
sim_matrix   = centroid_mat @ centroid_mat.T                  # (n, n) cosine similarities

edges = []
for i in range(len(multi_trad)):
    for j in range(i + 1, len(multi_trad)):
        sim = float(sim_matrix[i, j])
        if sim >= CENTROID_SIM_THRESHOLD:
            edges.append((multi_trad[i], multi_trad[j], sim))

edges.sort(key=lambda x: -x[2])
print(f"Centroid similarity edges (sim >= {CENTROID_SIM_THRESHOLD}): {len(edges)}")
print("\nTop 20 edges:")
for a, b, sim in edges[:20]:
    print(f"  {sim:.3f}  C{a} [{G.nodes[a]['label']}]  ↔  C{b} [{G.nodes[b]['label']}]")

# %% Build graph
G2 = nx.Graph()

connected = {c for a, b, _ in edges for c in (a, b)}
for c in connected:
    G2.add_node(c, **G.nodes[c])

for a, b, sim in edges:
    G2.add_edge(a, b, weight=sim)

print(f"\nGraph: {G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges")

between_cent = nx.betweenness_centrality(G2, weight="weight")
print("\nTop 10 nodes by betweenness centrality:")
for c, bc in sorted(between_cent.items(), key=lambda x: -x[1])[:10]:
    print(f"  C{c} [{G.nodes[c]['label']}]  betweenness={bc:.3f}  degree={G2.degree(c)}")

# %% Pyvis
def scale_size(n, lo=15, hi=60, min_n=None, max_n=None):
    if max_n == min_n:
        return (lo + hi) / 2
    return lo + (hi - lo) * ((n - min_n) / (max_n - min_n)) ** 0.5

all_n   = [G2.nodes[c]["n_passages"] for c in G2.nodes]
min_n   = min(all_n)
max_n   = max(all_n)

net = Network(
    height="900px",
    width="100%",
    bgcolor="#1a1a2e",
    font_color="white",
    notebook=False,
)

for c in G2.nodes:
    nd      = G2.nodes[c]
    color   = TRADITION_COLORS.get(nd["dom_tradition"], "#aaaaaa")
    tooltip = (
        f"Cluster {c}: {nd['label']}\n"
        f"Passages: {nd['n_passages']}\n"
        f"Traditions: {nd['trad_breakdown']}\n"
        f"Dominant: {nd['dom_tradition']} ({nd['dom_pct']:.0%})\n\n"
        f"Exemplar:\n{nd['exemplar']}"
    )
    net.add_node(
        c,
        label=nd["label"],
        title=tooltip,
        color=color,
        size=scale_size(nd["n_passages"], min_n=min_n, max_n=max_n),
        font={"size": 11, "color": "white"},
    )

for a, b, data in G2.edges(data=True):
    sim = data["weight"]
    net.add_edge(
        int(a), int(b),
        value=sim,
        title=f"Centroid similarity: {sim:.3f}",
        color={"color": "#ffd70066", "highlight": "#ffd700cc"},
    )

net.set_options("""
{
  "physics": {
    "solver": "forceAtlas2Based",
    "forceAtlas2Based": {
      "gravitationalConstant": -150,
      "centralGravity": 0.005,
      "springLength": 220,
      "springConstant": 0.03,
      "damping": 0.9
    },
    "maxVelocity": 15,
    "stabilization": {"iterations": 300, "updateInterval": 50}
  },
  "edges": {
    "smooth": {"type": "continuous"}
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 100
  }
}
""")

out_file = "36b_concept_network_centroid.html"
net.save_graph(out_file)
print(f"\nSaved: {out_file}")

with open("36b_concept_graph.pkl", "wb") as f:
    pickle.dump(G2, f)
print("Saved: 36b_concept_graph.pkl")
