# %% [markdown]
# # Outlier Detection — Sacred Texts
#
# Two types of outliers:
#
# 1. Within-tradition outliers
#    Passages with the lowest cosine similarity to their own tradition centroid.
#    These are the "weird" passages — content that doesn't fit their text's pattern.
#
# 2. Cross-tradition wanderers
#    Passages that are far from their own tradition's centroid but close to
#    another tradition's centroid. A Bible verse that semantically lives in
#    Buddhist space, a Gita verse that sounds Taoist, etc.
#    Scored by: sim_to_best_other - sim_to_own  (higher = further from home)

# %% Imports
import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from umap import UMAP
import plotly.graph_objects as go

from db.schema import get_conn
from analysis_utils import SKIP_CORPORA

MODEL_NAME = "all-mpnet-base-v2"
TOP_N      = 10   # outliers to show per tradition

TRADITION_COLORS = {
    "Abrahamic": "#e05c5c",
    "Dharmic":   "#f0a500",
    "Buddhist":  "#44bb99",
    "Taoist":    "#9b59b6",
}
TRADITIONS = list(TRADITION_COLORS.keys())

# %% Load
CORPORA_AT_28 = {
    "Bible — KJV (King James Version)",
    "Bhagavad Gita",
    "Dhammapada (Müller)",
    "Dao De Jing (Linnell)",
    "Yoga Sutras of Patanjali (Johnston)",
}

conn = get_conn()
placeholders = ", ".join("?" * len(CORPORA_AT_28))
rows = conn.execute(f"""
    SELECT t.name AS tradition, c.name AS corpus,
           p.id AS passage_id, p.unit_label, p.text, e.vector
    FROM embedding e
    JOIN passage p ON e.passage_id = p.id
    JOIN corpus c  ON p.corpus_id = c.id
    JOIN corpus_tradition t ON c.tradition_id = t.id
    WHERE e.model_name = ?
    AND c.name IN ({placeholders})
    ORDER BY c.name, p.id
""", [MODEL_NAME] + list(CORPORA_AT_28)).fetchall()
conn.close()

df = pd.DataFrame(rows, columns=["tradition", "corpus", "passage_id", "unit_label", "text", "vector"])
df = df[~df["corpus"].isin(SKIP_CORPORA)].reset_index(drop=True)
df["vector"] = df["vector"].apply(np.array)

vecs  = np.stack(df["vector"].values).astype("float32")
norms = np.linalg.norm(vecs, axis=1, keepdims=True)
vecs_normed = vecs / np.clip(norms, 1e-9, None)

print(f"Loaded {len(df):,} passages")
print(df.groupby(["tradition", "corpus"]).size().to_string())

# %% Compute tradition centroids (L2-normalised mean)
centroids = {}
for t in TRADITIONS:
    mask = (df["tradition"] == t).values
    c    = vecs_normed[mask].mean(axis=0)
    centroids[t] = c / (np.linalg.norm(c) + 1e-9)

# Similarity of every passage to every tradition centroid
# Shape: (n_passages, n_traditions)
centroid_mat = np.stack([centroids[t] for t in TRADITIONS])   # (4, 768)
sim_to_centroids = vecs_normed @ centroid_mat.T                 # (n_passages, 4)

trad_to_idx = {t: i for i, t in enumerate(TRADITIONS)}

df["sim_own"] = [
    float(sim_to_centroids[i, trad_to_idx[df.loc[i, "tradition"]]])
    for i in range(len(df))
]

# Best other-tradition similarity and which tradition it is
other_sims = []
other_trads = []
for i in range(len(df)):
    own_idx = trad_to_idx[df.loc[i, "tradition"]]
    sims    = sim_to_centroids[i].copy()
    sims[own_idx] = -np.inf
    best_idx = int(sims.argmax())
    other_sims.append(float(sim_to_centroids[i, best_idx]))
    other_trads.append(TRADITIONS[best_idx])

df["sim_best_other"]  = other_sims
df["closest_other"]   = other_trads
df["wander_score"]    = df["sim_best_other"] - df["sim_own"]

# %% ── Build report (written to file + printed) ──────────────────────────────
report_lines = []

def emit(line=""):
    """Print and accumulate for file output."""
    print(line)
    report_lines.append(line)

# ── Type 1: Within-tradition outliers
emit("\n" + "="*70)
emit(f"WITHIN-TRADITION OUTLIERS (lowest sim to own tradition centroid)")
emit("="*70)

for t in TRADITIONS:
    sub = df[df["tradition"] == t].nsmallest(TOP_N, "sim_own")
    emit(f"\n{'─'*60}")
    emit(f"  {t} — bottom {TOP_N} (most unlike their tradition)")
    for _, row in sub.iterrows():
        emit(f"\n  [{row['sim_own']:.3f}] {row['corpus']} | {row['unit_label']}")
        emit(f"    {row['text'][:200]}")

# ── Type 2: Cross-tradition wanderers
emit("\n" + "="*70)
emit(f"CROSS-TRADITION WANDERERS (far from own, close to another)")
emit("="*70)

for t in TRADITIONS:
    sub = df[df["tradition"] == t].nlargest(TOP_N, "wander_score")
    emit(f"\n{'─'*60}")
    emit(f"  {t} wanderers — most drawn toward other traditions")
    for _, row in sub.iterrows():
        emit(f"\n  [own={row['sim_own']:.3f}  other={row['sim_best_other']:.3f}  "
             f"Δ={row['wander_score']:.3f}] → closest to: {row['closest_other']}")
        emit(f"  {row['corpus']} | {row['unit_label']}")
        emit(f"    {row['text'][:200]}")

# %% ── Plot: UMAP scatter highlighting wanderers ─────────────────────────────
print("\nFitting 2D UMAP for visualisation ...")
reducer = UMAP(
    n_components=2,
    n_neighbors=30,
    min_dist=0.1,
    metric="cosine",
    random_state=42,
    low_memory=False,
)
X2 = reducer.fit_transform(vecs_normed).astype("float32")

# Top wanderers per tradition
wanderer_ids = set()
for t in TRADITIONS:
    ids = df[df["tradition"] == t].nlargest(TOP_N, "wander_score")["passage_id"].tolist()
    wanderer_ids.update(ids)

is_wanderer = df["passage_id"].isin(wanderer_ids).values

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor("#1a1a2e")

for ax, (title, highlight_mask, highlight_col) in zip(axes, [
    ("Within-tradition outliers",   df["sim_own"].rank() <= len(TRADITIONS) * TOP_N,   "sim_own"),
    ("Cross-tradition wanderers",    is_wanderer,                                       "wander_score"),
]):
    ax.set_facecolor("#1a1a2e")

    # Background — all passages faint
    ax.scatter(X2[:, 0], X2[:, 1], s=1, alpha=0.08,
               c=[TRADITION_COLORS[t] for t in df["tradition"]], linewidths=0)

    # Highlighted outliers
    hi_idx = np.where(highlight_mask)[0]
    for i in hi_idx:
        t     = df.iloc[i]["tradition"]
        color = TRADITION_COLORS[t]
        ax.scatter(X2[i, 0], X2[i, 1], s=60, color=color,
                   edgecolors="white", linewidths=0.8, zorder=5, alpha=0.9)

    patches = [mpatches.Patch(color=c, label=t) for t, c in TRADITION_COLORS.items()]
    ax.legend(handles=patches, fontsize=8, loc="lower right",
              facecolor="#1a1a2e", labelcolor="white", framealpha=0.5)
    ax.set_title(title, fontsize=11, fontweight="bold", color="white")
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle("Sacred Text Outliers — Highlighted in Semantic Space",
             fontsize=13, fontweight="bold", color="white")
plt.tight_layout()
plt.savefig("28_outliers.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print("Saved: 28_outliers.png")

# %% ── Summary stats ──────────────────────────────────────────────────────────
emit("\n=== Summary: mean similarity to own centroid per tradition ===")
emit(df.groupby("tradition")["sim_own"].agg(["mean", "min", "max"]).round(3).to_string())

emit("\n=== Where do wanderers end up? (top destination tradition) ===")
wanderer_df = pd.concat([
    df[df["tradition"] == t].nlargest(TOP_N, "wander_score")
    for t in TRADITIONS
])
emit(wanderer_df.groupby(["tradition", "closest_other"]).size().unstack(fill_value=0).to_string())

# %% ── Save report ────────────────────────────────────────────────────────────
with open("28_outliers_report.txt", "w") as f:
    f.write("\n".join(report_lines))
print("\nSaved: 28_outliers_report.txt")

# %% ── Interactive Plotly scatter ─────────────────────────────────────────────
import textwrap

df["ux"] = X2[:, 0]
df["uy"] = X2[:, 1]

# Wrap passage text for readable hover
def wrap_text(t, width=80):
    return "<br>".join(textwrap.wrap(t[:300], width))

df["hover"] = df.apply(
    lambda r: (
        f"<b>{r['corpus']}</b>  {r['unit_label']}<br>"
        f"Tradition: {r['tradition']}<br>"
        f"sim_own: {r['sim_own']:.3f}  wander: {r['wander_score']:.3f}<br>"
        f"Closest other: {r['closest_other']}<br><br>"
        f"{wrap_text(r['text'])}"
    ),
    axis=1,
)

wanderer_mask = df["passage_id"].isin(wanderer_ids).values
outlier_mask  = df["sim_own"].rank() <= len(TRADITIONS) * TOP_N

def _bg_to_base64(df_view, x_range, y_range, color_map, img_w=1200, img_h=900):
    dpi    = 150
    fw, fh = img_w / dpi, img_h / dpi
    fig_bg, ax_bg = plt.subplots(figsize=(fw, fh), dpi=dpi)
    for t, color in color_map.items():
        sub = df_view[df_view["tradition"] == t]
        ax_bg.scatter(sub["ux"], sub["uy"], s=2, color=color, alpha=0.25, linewidths=0)
    ax_bg.set_xlim(x_range); ax_bg.set_ylim(y_range)
    ax_bg.axis("off")
    fig_bg.patch.set_alpha(0)
    ax_bg.set_facecolor("none")
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig_bg.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                   transparent=True, pad_inches=0)
    plt.close(fig_bg)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def build_interactive(mask, title, filename):
    hi_df   = df[mask]
    pad_x   = (hi_df["ux"].max() - hi_df["ux"].min()) * 0.25 + 1.0
    pad_y   = (hi_df["uy"].max() - hi_df["uy"].min()) * 0.25 + 1.0
    x_range = [hi_df["ux"].min() - pad_x, hi_df["ux"].max() + pad_x]
    y_range = [hi_df["uy"].min() - pad_y, hi_df["uy"].max() + pad_y]
    in_view = (
        (df["ux"] >= x_range[0]) & (df["ux"] <= x_range[1]) &
        (df["uy"] >= y_range[0]) & (df["uy"] <= y_range[1])
    )

    bg_img = _bg_to_base64(df[in_view & ~mask], x_range, y_range, TRADITION_COLORS)

    fig = go.Figure()
    fig.add_layout_image(dict(
        source=bg_img,
        xref="x", yref="y",
        x=x_range[0], y=y_range[1],
        sizex=x_range[1] - x_range[0],
        sizey=y_range[1] - y_range[0],
        sizing="stretch",
        layer="below",
        opacity=1.0,
    ))

    for t, color in TRADITION_COLORS.items():
        t_mask = (df["tradition"] == t).values
        hi     = t_mask & mask
        fig.add_trace(go.Scatter(
            x=df.loc[hi, "ux"], y=df.loc[hi, "uy"],
            mode="markers",
            marker=dict(size=10, color=color, opacity=0.95,
                        line=dict(width=1.5, color="white")),
            text=df.loc[hi, "hover"],
            hovertemplate="%{text}<extra></extra>",
            legendgroup=t, showlegend=True, name=t,
        ))

    fig.update_layout(
        title=title, height=750, autosize=True,
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(title="Tradition"),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=x_range),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   range=y_range, scaleanchor="x"),
    )
    fig.write_html(filename)
    print(f"Saved: {filename}")

build_interactive(
    outlier_mask,
    "Within-Tradition Outliers — Passages Least Like Their Own Tradition",
    "28_outliers_within.html",
)
build_interactive(
    wanderer_mask,
    "Cross-Tradition Wanderers — Passages Drawn Toward Other Traditions",
    "28_outliers_cross.html",
)
