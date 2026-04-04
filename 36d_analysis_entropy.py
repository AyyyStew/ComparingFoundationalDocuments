# %% [markdown]
# # Cross-Tradition Entropy Analysis (Script 36d)
#
# Loads any concept network graph pickle and computes Shannon entropy
# over tradition counts per node.
#
# High entropy = tradition counts are spread evenly = no tradition dominates.
# These are the strongest candidates for universal / perennial concepts.
#
# Works with: 36_concept_graph.pkl, 36b_concept_graph.pkl, 36c_concept_graph.pkl

# %% Imports
import pickle
import numpy as np
import pandas as pd

# %% Config — point at whichever graph you want
GRAPH_PKL = "36_concept_graph.pkl"
TOP_N     = 20   # how many nodes to show

TRADITION_KEYS = [
    "n_abrahamic", "n_dharmic", "n_buddhist", "n_taoist",
    "n_norse", "n_confucian", "n_philosophy", "n_scientific",
    "n_literature", "n_historical",
]

# %% Load
with open(GRAPH_PKL, "rb") as f:
    G = pickle.load(f)

print(f"Loaded {G.number_of_nodes()} nodes from {GRAPH_PKL}")

# %% Compute entropy per node
def shannon_entropy(counts):
    counts = np.array(counts, dtype=float)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return 0.0
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))

def max_entropy(n_traditions):
    """Entropy of a perfectly uniform distribution over n traditions."""
    return np.log2(n_traditions) if n_traditions > 1 else 0.0

rows = []
for c in G.nodes:
    nd     = G.nodes[c]
    counts = [nd.get(k, 0) for k in TRADITION_KEYS]
    n_present   = sum(1 for x in counts if x > 0)
    entropy     = shannon_entropy(counts)
    norm_entropy = entropy / max_entropy(n_present) if n_present > 1 else 0.0
    rows.append({
        "cluster":       c,
        "label":         nd.get("label", ""),
        "n_passages":    nd.get("n_passages", 0),
        "dom_tradition": nd.get("dom_tradition", ""),
        "dom_pct":       nd.get("dom_pct", 1.0),
        "n_traditions":  n_present,
        "entropy":       entropy,
        "norm_entropy":  norm_entropy,  # 0=one tradition dominates, 1=perfectly even
        **{k: nd.get(k, 0) for k in TRADITION_KEYS},
    })

df = pd.DataFrame(rows).sort_values("norm_entropy", ascending=False).reset_index(drop=True)

# %% Print results
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.width", 120)

print(f"\n=== Top {TOP_N} nodes by cross-tradition entropy ===")
print("(norm_entropy: 0.0 = one tradition dominates, 1.0 = perfectly even)\n")

display_cols = ["cluster", "label", "n_passages", "n_traditions", "norm_entropy", "dom_tradition", "dom_pct"]
print(df[display_cols].head(TOP_N).to_string(index=False))

print(f"\n\n=== Full breakdown for top {min(TOP_N, 10)} ===")
trad_cols = [k for k in TRADITION_KEYS if df[k].sum() > 0]  # skip empty traditions
for _, row in df.head(min(TOP_N, 10)).iterrows():
    print(f"\nC{int(row['cluster'])} [{row['label']}]  "
          f"entropy={row['norm_entropy']:.3f}  passages={int(row['n_passages'])}")
    for k in trad_cols:
        v = int(row[k])
        if v > 0:
            trad = k.replace("n_", "").capitalize()
            pct  = v / row["n_passages"] * 100
            bar  = "█" * int(pct / 5)
            print(f"  {trad:<12} {v:4d}  ({pct:5.1f}%)  {bar}")

# %% Save to CSV
out_csv = GRAPH_PKL.replace(".pkl", "_entropy.csv")
df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")
