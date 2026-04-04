# %% [markdown]
# # Final Analysis — Full Corpus
#
# Runs four analyses on whatever traditions are currently in the database:
#
#   1. Top-k similarity heatmap  — how similar are all tradition pairs at their best?
#   2. Random-pair + lift        — baseline similarity and concentration scores
#   3. Similarity distributions  — overlapping histograms, within vs cross-tradition
#   4. Narrative flow            — how each corpus moves through semantic space
#
# No hardcoded tradition or corpus lists. Everything is derived from the DB at runtime.
# Skips translation variants (SKIP_CORPORA) and requires at least 2 passages per tradition.

# %% Imports
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from umap import UMAP
import plotly.graph_objects as go

from db.schema import get_conn
from analysis_utils import SKIP_CORPORA

MODEL_NAME  = "all-mpnet-base-v2"
TOP_K       = 5
N_SAMPLE    = 5000
RANDOM_SEED = 42
TOP_N       = 10   # outliers per tradition

# %% ── Load ───────────────────────────────────────────────────────────────────
conn = get_conn()
rows = conn.execute("""
    SELECT t.name AS tradition, c.name AS corpus,
           p.id AS passage_id, p.book, p.unit_number, p.unit_label, p.text, e.vector
    FROM embedding e
    JOIN passage p ON e.passage_id = p.id
    JOIN corpus c  ON p.corpus_id = c.id
    JOIN corpus_tradition t ON c.tradition_id = t.id
    WHERE e.model_name = ?
    ORDER BY t.name, c.name, p.id
""", [MODEL_NAME]).fetchall()
conn.close()

df = pd.DataFrame(rows, columns=[
    "tradition", "corpus", "passage_id", "book",
    "unit_number", "unit_label", "text", "vector"
])
df = df[~df["corpus"].isin(SKIP_CORPORA)].reset_index(drop=True)
df["vector"] = df["vector"].apply(np.array)

# Drop any tradition with fewer than 2 passages (can't do pairwise)
counts = df.groupby("tradition").size()
valid  = counts[counts >= 2].index.tolist()
df     = df[df["tradition"].isin(valid)].reset_index(drop=True)

traditions = sorted(df["tradition"].unique().tolist())

print(f"Traditions ({len(traditions)}): {traditions}")
print(f"\nPassages per tradition:")
print(df.groupby(["tradition", "corpus"]).size().to_string())

vecs   = np.stack(df["vector"].values).astype("float32")
norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
vecs_n = vecs / np.clip(norms, 1e-9, None)

trad_idx = {t: np.where(df["tradition"].values == t)[0] for t in traditions}

# %% ── 1. Top-k similarity heatmap ───────────────────────────────────────────
print("\n[1/4] Computing top-k similarity matrix ...")

topk_mat = pd.DataFrame(np.nan, index=traditions, columns=traditions)

for t_a in traditions:
    for t_b in traditions:
        idx_a = trad_idx[t_a]
        idx_b = trad_idx[t_b]
        V_a   = vecs_n[idx_a]
        V_b   = vecs_n[idx_b]
        sim   = V_a @ V_b.T

        if t_a == t_b:
            np.fill_diagonal(sim, -np.inf)
            k = min(TOP_K, sim.shape[1] - 1)
        else:
            k = min(TOP_K, sim.shape[1])

        topk_mat.loc[t_a, t_b] = float(np.partition(sim, -k, axis=1)[:, -k:].mean())

n = len(traditions)
fig_size = max(8, n * 0.9)
fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
vals = topk_mat.values.astype(float)
im   = ax.imshow(vals, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(n)); ax.set_xticklabels(traditions, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(n)); ax.set_yticklabels(traditions, fontsize=9)
for i in range(n):
    for j in range(n):
        if not np.isnan(vals[i, j]):
            ax.text(j, i, f"{vals[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    fontweight="bold" if i == j else "normal")
plt.colorbar(im, ax=ax, label="Mean top-k cosine similarity")
ax.set_title("Tradition Similarity — Top-k Nearest Neighbours\n(diagonal = within-tradition)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("final_topk_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: final_topk_heatmap.png")

# Also plot with high-volume low-similarity traditions excluded so the scale
# isn't crushed. Exclude any tradition whose max off-diagonal is an outlier.
HEATMAP_EXCLUDE = {"News"}   # extend if needed
t_filtered = [t for t in traditions if t not in HEATMAP_EXCLUDE]
if len(t_filtered) < len(traditions):
    nf = len(t_filtered)
    fig_size_f = max(8, nf * 0.9)
    fig2, ax2 = plt.subplots(figsize=(fig_size_f, fig_size_f * 0.85))
    vals_f = topk_mat.loc[t_filtered, t_filtered].values.astype(float)
    im2    = ax2.imshow(vals_f, cmap="YlOrRd", aspect="auto")
    ax2.set_xticks(range(nf)); ax2.set_xticklabels(t_filtered, rotation=45, ha="right", fontsize=9)
    ax2.set_yticks(range(nf)); ax2.set_yticklabels(t_filtered, fontsize=9)
    for i in range(nf):
        for j in range(nf):
            if not np.isnan(vals_f[i, j]):
                ax2.text(j, i, f"{vals_f[i, j]:.3f}", ha="center", va="center", fontsize=7,
                         fontweight="bold" if i == j else "normal")
    plt.colorbar(im2, ax=ax2, label="Mean top-k cosine similarity")
    excluded_str = ", ".join(HEATMAP_EXCLUDE)
    ax2.set_title(f"Tradition Similarity — Top-k (excluding: {excluded_str})\n(diagonal = within-tradition)",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("final_topk_heatmap_filtered.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: final_topk_heatmap_filtered.png")

# %% ── 2. Random-pair baseline + lift ────────────────────────────────────────
print("\n[2/4] Computing random-pair similarity + lift ...")

rng      = np.random.default_rng(RANDOM_SEED)
rand_mat = pd.DataFrame(np.nan, index=traditions, columns=traditions)

for t_a in traditions:
    for t_b in traditions:
        idx_a = trad_idx[t_a]
        idx_b = trad_idx[t_b]
        si    = rng.choice(idx_a, size=N_SAMPLE, replace=True)
        sj    = rng.choice(idx_b, size=N_SAMPLE, replace=True)
        if t_a == t_b:
            same = si == sj
            while same.any():
                sj[same] = rng.choice(idx_b, size=same.sum(), replace=True)
                same = si == sj
        rand_mat.loc[t_a, t_b] = float((vecs_n[si] * vecs_n[sj]).sum(axis=1).mean())

topk_vals = topk_mat.values.astype(float)
rand_vals = rand_mat.values.astype(float)
diff_vals  = topk_vals - rand_vals
ratio_vals = topk_vals / np.clip(rand_vals, 1e-9, None)

fig, axes = plt.subplots(1, 3, figsize=(max(18, n * 2.5), max(6, n * 0.85)))

for ax, (data, title, cmap, fmt) in zip(axes, [
    (rand_vals,  f"Random-pair baseline\n({N_SAMPLE:,} pairs each)", "YlOrRd", ".3f"),
    (diff_vals,  "Lift — difference\n(top-k minus random)",          "RdBu_r", ".3f"),
    (ratio_vals, "Lift — ratio\n(top-k ÷ random)",                   "YlOrRd", ".2f"),
]):
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(traditions, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(traditions, fontsize=8)
    for i in range(n):
        for j in range(n):
            if not np.isnan(data[i, j]):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        fontsize=6.5, fontweight="bold" if i == j else "normal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10, fontweight="bold")

plt.suptitle("Similarity Lift — Full Corpus", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("final_lift.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: final_lift.png")

# Filtered lift — same but excluding noisy traditions
if len(t_filtered) < len(traditions):
    nf         = len(t_filtered)
    topk_f     = topk_mat.loc[t_filtered, t_filtered].values.astype(float)
    rand_f     = rand_mat.loc[t_filtered, t_filtered].values.astype(float)
    diff_f     = topk_f - rand_f
    ratio_f    = topk_f / np.clip(rand_f, 1e-9, None)

    fig_f, axes_f = plt.subplots(1, 3, figsize=(max(18, nf * 2.5), max(6, nf * 0.85)))
    for ax_f, (data_f, title_f, cmap_f, fmt_f) in zip(axes_f, [
        (rand_f,  f"Random-pair baseline\n({N_SAMPLE:,} pairs each)", "YlOrRd", ".3f"),
        (diff_f,  "Lift — difference\n(top-k minus random)",          "RdBu_r", ".3f"),
        (ratio_f, "Lift — ratio\n(top-k ÷ random)",                   "YlOrRd", ".2f"),
    ]):
        im_f = ax_f.imshow(data_f, cmap=cmap_f, aspect="auto")
        ax_f.set_xticks(range(nf)); ax_f.set_xticklabels(t_filtered, rotation=45, ha="right", fontsize=8)
        ax_f.set_yticks(range(nf)); ax_f.set_yticklabels(t_filtered, fontsize=8)
        for i in range(nf):
            for j in range(nf):
                if not np.isnan(data_f[i, j]):
                    ax_f.text(j, i, format(data_f[i, j], fmt_f), ha="center", va="center",
                              fontsize=6.5, fontweight="bold" if i == j else "normal")
        plt.colorbar(im_f, ax=ax_f, fraction=0.046, pad=0.04)
        ax_f.set_title(title_f, fontsize=10, fontweight="bold")

    plt.suptitle(f"Similarity Lift — excluding: {excluded_str}", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("final_lift_filtered.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: final_lift_filtered.png")

# %% ── 3. Similarity distributions (interactive) ─────────────────────────────
print("\n[3/4] Sampling similarity distributions ...")


dist_data = {}
bins      = np.linspace(-0.2, 1.0, 80)
bin_mids  = (bins[:-1] + bins[1:]) / 2

for t in traditions:
    idx  = trad_idx[t]
    si   = rng.choice(idx, size=N_SAMPLE, replace=True)
    sj   = rng.choice(idx, size=N_SAMPLE, replace=True)
    same = si == sj
    while same.any():
        sj[same] = rng.choice(idx, size=same.sum(), replace=True)
        same = si == sj
    dist_data[f"Within: {t}"] = (vecs_n[si] * vecs_n[sj]).sum(axis=1)

cross_sims_all = []
for i, t_a in enumerate(traditions):
    for t_b in traditions[i+1:]:
        si = rng.choice(trad_idx[t_a], size=500, replace=True)
        sj = rng.choice(trad_idx[t_b], size=500, replace=True)
        cross_sims_all.append((vecs_n[si] * vecs_n[sj]).sum(axis=1))
dist_data["Cross-tradition (all pairs)"] = np.concatenate(cross_sims_all)

plotly_colors = [
    "#e05c5c","#f0a500","#44bb99","#9b59b6","#4a90d9","#e67e22",
    "#1abc9c","#e74c3c","#3498db","#2ecc71","#9b59b6","#aaaaaa",
]

def hex_to_rgba(hex_color, alpha=0.25):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# Pre-compute all histograms
hist_data = {}
for label, sims in dist_data.items():
    counts, _ = np.histogram(sims, bins=bins, density=True)
    hist_data[label] = (counts, float(np.median(sims)))

# Average within-tradition curve (mean across all per-tradition histograms)
within_labels  = [l for l in hist_data if l.startswith("Within:")]
within_counts  = np.stack([hist_data[l][0] for l in within_labels])
avg_counts     = within_counts.mean(axis=0)
avg_median     = float(np.median(np.concatenate([dist_data[l] for l in within_labels])))

fig_dist = go.Figure()

# 1. Average within-tradition — visible by default
fig_dist.add_trace(go.Scatter(
    x=bin_mids.tolist(), y=avg_counts.tolist(),
    mode="lines", fill="tozeroy",
    name="Within-tradition (average)",
    line=dict(color="#2c7bb6", width=2.5),
    fillcolor=hex_to_rgba("#2c7bb6", 0.2),
    legendgroup="avg", visible=True,
    hovertemplate="<b>Within avg</b><br>similarity: %{x:.3f}<br>density: %{y:.3f}<extra></extra>",
))
fig_dist.add_trace(go.Scatter(
    x=[avg_median, avg_median], y=[0, avg_counts.max()],
    mode="lines", line=dict(color="#2c7bb6", width=2, dash="dash"),
    showlegend=False, legendgroup="avg", visible=True,
    hovertemplate=f"Avg within median: {avg_median:.3f}<extra></extra>",
))

# 2. Cross-tradition — visible by default
cross_counts, cross_median = hist_data["Cross-tradition (all pairs)"]
fig_dist.add_trace(go.Scatter(
    x=bin_mids.tolist(), y=cross_counts.tolist(),
    mode="lines", fill="tozeroy",
    name="Cross-tradition (all pairs)",
    line=dict(color="#888888", width=2.5),
    fillcolor=hex_to_rgba("#888888", 0.2),
    legendgroup="cross", visible=True,
    hovertemplate="<b>Cross-tradition</b><br>similarity: %{x:.3f}<br>density: %{y:.3f}<extra></extra>",
))
fig_dist.add_trace(go.Scatter(
    x=[cross_median, cross_median], y=[0, cross_counts.max()],
    mode="lines", line=dict(color="#888888", width=2, dash="dash"),
    showlegend=False, legendgroup="cross", visible=True,
    hovertemplate=f"Cross median: {cross_median:.3f}<extra></extra>",
))

# 3. Individual traditions — hidden by default, toggleable
for idx_l, label in enumerate(within_labels):
    counts_i, median_i = hist_data[label]
    color = plotly_colors[idx_l % len(plotly_colors)]
    fig_dist.add_trace(go.Scatter(
        x=bin_mids.tolist(), y=counts_i.tolist(),
        mode="lines", fill="tozeroy",
        name=label, line=dict(color=color, width=1.2),
        fillcolor=hex_to_rgba(color, 0.15),
        legendgroup=label, visible="legendonly",
        hovertemplate=f"<b>{label}</b><br>similarity: %{{x:.3f}}<br>density: %{{y:.3f}}<extra></extra>",
    ))
    fig_dist.add_trace(go.Scatter(
        x=[median_i, median_i], y=[0, counts_i.max()],
        mode="lines", line=dict(color=color, width=1.2, dash="dash"),
        showlegend=False, legendgroup=label, visible="legendonly",
        hovertemplate=f"Median: {median_i:.3f}<extra></extra>",
    ))

fig_dist.update_layout(
    title="Similarity Distributions — Within vs Cross Tradition<br>"
          "<sup>Average within-tradition and cross-tradition shown by default. Click to toggle individual traditions.</sup>",
    xaxis_title="Cosine similarity",
    yaxis_title="Density",
    width=1000, height=550,
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend=dict(groupclick="togglegroup"),
    hovermode="x",
)
fig_dist.update_xaxes(showgrid=True, gridcolor="#eeeeee")
fig_dist.update_yaxes(showgrid=True, gridcolor="#eeeeee")
fig_dist.write_html("final_distributions.html")
print("Saved: final_distributions.html")

print("\nMedian similarities:")
for label, sims in dist_data.items():
    print(f"  {label:45s}: {np.median(sims):.4f}")

# %% ── 4. Narrative flow ──────────────────────────────────────────────────────
print("\n[4/4] Computing narrative flow ...")

BIBLE_ABBREV = [
    "Gen","Exo","Lev","Num","Deu","Jos","Jdg","Rut","1Sa","2Sa",
    "1Ki","2Ki","1Ch","2Ch","Ezr","Neh","Est","Job","Psa","Pro",
    "Ecc","Son","Isa","Jer","Lam","Eze","Dan","Hos","Joe","Amo",
    "Oba","Jon","Mic","Nah","Hab","Zep","Hag","Zec","Mal",
    "Mat","Mar","Luk","Joh","Act","Rom","1Co","2Co","Gal","Eph",
    "Php","Col","1Th","2Th","1Ti","2Ti","Tit","Phm","Heb","Jam",
    "1Pe","2Pe","1Jo","2Jo","3Jo","Jud","Rev",
]

print("  Fitting 2D UMAP on full corpus ...")
reducer = UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
               metric="cosine", random_state=42, low_memory=False)
X2 = reducer.fit_transform(vecs_n).astype("float32")
df["ux"] = X2[:, 0]
df["uy"] = X2[:, 1]

def get_flow_sequence(corpus_name):
    """Return (coords, labels) for one corpus, aggregated to natural units."""
    sub = df[df["corpus"] == corpus_name].copy()

    if corpus_name == "Bible — KJV (King James Version)":
        sub["_order"] = sub["book"].apply(
            lambda b: BIBLE_ABBREV.index(b[:3]) if b[:3] in BIBLE_ABBREV else 9999
        )
        grp = (sub.groupby(["_order", "book"])
               .agg(ux=("ux", "mean"), uy=("uy", "mean"))
               .reset_index().sort_values("_order"))
        labels = [BIBLE_ABBREV[r["_order"]] if r["_order"] < len(BIBLE_ABBREV) else r["book"][:3]
                  for _, r in grp.iterrows()]
        return grp[["ux", "uy"]].values, labels

    # Generic: group by book (= chapter for most texts), order by min passage_id
    import re
    book_order = sub.groupby("book")["passage_id"].min().sort_values().index.tolist()
    grp = (sub.groupby("book").agg(ux=("ux", "mean"), uy=("uy", "mean"))
           .reindex(book_order).reset_index())
    def short_label(name):
        m = re.search(r"(\d+)", name)
        return m.group(1) if m else name[:6]
    labels = [short_label(b) for b in grp["book"]]
    return grp[["ux", "uy"]].values, labels

# One subplot per corpus — auto grid
corpora  = sorted(df["corpus"].unique().tolist())
n_corpus = len(corpora)
ncols    = min(4, n_corpus)
nrows    = (n_corpus + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 5))
axes_flat = np.array(axes).flatten()

for idx, corpus in enumerate(corpora):
    ax = axes_flat[idx]
    try:
        coords, labels = get_flow_sequence(corpus)
    except Exception as e:
        ax.set_visible(False)
        continue

    n_pts = len(coords)
    pad   = max(1.0, ((coords[:, 0].max() - coords[:, 0].min()) + (coords[:, 1].max() - coords[:, 1].min())) * 0.15)
    ax.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
    ax.set_ylim(coords[:, 1].min() - pad, coords[:, 1].max() + pad)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    in_view = ((X2[:, 0] >= xlim[0]) & (X2[:, 0] <= xlim[1]) &
               (X2[:, 1] >= ylim[0]) & (X2[:, 1] <= ylim[1]))
    ax.scatter(X2[in_view, 0], X2[in_view, 1], s=1, alpha=0.15, color="#aaaaaa", linewidths=0)

    cmap_f = cm.get_cmap("coolwarm")
    colors = [cmap_f(i / max(n_pts - 1, 1)) for i in range(n_pts)]

    for i in range(n_pts - 1):
        ax.annotate("", xy=coords[i+1], xytext=coords[i],
                    arrowprops=dict(arrowstyle="->", color=colors[i], lw=1.0, alpha=0.7))

    for i, (coord, color) in enumerate(zip(coords, colors)):
        ax.scatter(*coord, color=color, s=35, zorder=5, linewidths=0)

    fontsize = 5.5 if n_pts > 50 else 7 if n_pts > 20 else 8
    for coord, label in zip(coords, labels):
        ax.annotate(label, xy=coord, fontsize=fontsize, ha="center", va="bottom",
                    xytext=(0, 4), textcoords="offset points", color="#333333", zorder=6)

    ax.scatter(*coords[0],  marker="^", s=60, color="steelblue", zorder=7, label="Start")
    ax.scatter(*coords[-1], marker="s", s=60, color="tomato",    zorder=7, label="End")

    # Short title: strip author parentheticals for readability
    short = corpus.split("(")[0].strip()
    ax.set_title(short, fontsize=9, fontweight="bold")
    ax.set_facecolor("white")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")

    if idx == 0:
        ax.legend(fontsize=7, loc="lower right", framealpha=0.7)

# Hide unused subplots
for idx in range(n_corpus, len(axes_flat)):
    axes_flat[idx].set_visible(False)

sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
sm.set_array([])
# Add colorbar to last visible axes slot if spare, else skip
spare = [axes_flat[i] for i in range(n_corpus, len(axes_flat))]
if spare:
    cbar = fig.colorbar(sm, ax=spare[0], fraction=0.5, pad=0.05)
    cbar.set_ticks([0, 1]); cbar.set_ticklabels(["Start", "End"])
    spare[0].set_visible(False)

fig.patch.set_facecolor("white")
fig.suptitle("Narrative Flow — Full Corpus in Semantic Space",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("final_narrative_flow.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
print("Saved: final_narrative_flow.png")

# %% ── 5. Outliers ────────────────────────────────────────────────────────────
print("\n[5/5] Computing outliers ...")

# Tradition centroids
centroids = {}
for t in traditions:
    mask = (df["tradition"] == t).values
    c    = vecs_n[mask].mean(axis=0)
    centroids[t] = c / (np.linalg.norm(c) + 1e-9)

centroid_mat     = np.stack([centroids[t] for t in traditions])
sim_to_centroids = vecs_n @ centroid_mat.T
trad_to_col      = {t: i for i, t in enumerate(traditions)}

df["sim_own"] = [
    float(sim_to_centroids[i, trad_to_col[df.loc[i, "tradition"]]])
    for i in range(len(df))
]

other_sims, other_trads = [], []
for i in range(len(df)):
    own_col  = trad_to_col[df.loc[i, "tradition"]]
    sims_row = sim_to_centroids[i].copy()
    sims_row[own_col] = -np.inf
    best = int(sims_row.argmax())
    other_sims.append(float(sim_to_centroids[i, best]))
    other_trads.append(traditions[best])

df["sim_best_other"] = other_sims
df["closest_other"]  = other_trads
df["wander_score"]   = df["sim_best_other"] - df["sim_own"]

# Fit UMAP if not already done by narrative flow section
if "ux" not in df.columns:
    print("  Fitting 2D UMAP ...")
    reducer_o = UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                     metric="cosine", random_state=42, low_memory=False)
    X2_o = reducer_o.fit_transform(vecs_n).astype("float32")
    df["ux"] = X2_o[:, 0]
    df["uy"] = X2_o[:, 1]

wanderer_ids  = set()
for t in traditions:
    wanderer_ids.update(
        df[df["tradition"] == t].nlargest(TOP_N, "wander_score")["passage_id"].tolist()
    )

outlier_mask  = df["sim_own"].rank() <= len(traditions) * TOP_N
wanderer_mask = df["passage_id"].isin(wanderer_ids).values

# ── Text report ───────────────────────────────────────────────────────────────
report_lines = []

def emit(line=""):
    print(line)
    report_lines.append(line)

emit("=" * 70)
emit("WITHIN-TRADITION OUTLIERS (lowest sim to own tradition centroid)")
emit("=" * 70)
for t in traditions:
    sub = df[df["tradition"] == t].nsmallest(TOP_N, "sim_own")
    emit(f"\n{'─'*60}")
    emit(f"  {t} — bottom {TOP_N}")
    for _, row in sub.iterrows():
        emit(f"\n  [{row['sim_own']:.3f}] {row['corpus']} | {row['unit_label']}")
        emit(f"    {row['text'][:200]}")

emit("\n" + "=" * 70)
emit("CROSS-TRADITION WANDERERS (far from own, close to another)")
emit("=" * 70)
for t in traditions:
    sub = df[df["tradition"] == t].nlargest(TOP_N, "wander_score")
    emit(f"\n{'─'*60}")
    emit(f"  {t} wanderers")
    for _, row in sub.iterrows():
        emit(f"\n  [own={row['sim_own']:.3f}  other={row['sim_best_other']:.3f}  "
             f"Δ={row['wander_score']:.3f}] → {row['closest_other']}")
        emit(f"  {row['corpus']} | {row['unit_label']}")
        emit(f"    {row['text'][:200]}")

emit("\n=== Where do wanderers end up? ===")
wanderer_df = pd.concat([
    df[df["tradition"] == t].nlargest(TOP_N, "wander_score") for t in traditions
])
emit(wanderer_df.groupby(["tradition", "closest_other"]).size().unstack(fill_value=0).to_string())

with open("final_outliers_report.txt", "w") as f:
    f.write("\n".join(report_lines))
print("Saved: final_outliers_report.txt")

# ── Interactive plots ─────────────────────────────────────────────────────────
def wrap_text(t, width=80):
    return "<br>".join(textwrap.wrap(t[:300], width))

df["hover"] = df.apply(
    lambda r: (
        f"<b>{r['corpus']}</b>  {r['unit_label']}<br>"
        f"Tradition: {r['tradition']}<br>"
        f"sim_own: {r['sim_own']:.3f}  wander: {r['wander_score']:.3f}<br>"
        f"Closest other: {r['closest_other']}<br><br>"
        f"{wrap_text(r['text'])}"
    ), axis=1,
)

trad_color_out = dict(zip(traditions, plotly_colors[:len(traditions)]))

def _bg_to_base64(df_view, x_range, y_range, trad_color_map, img_w=1200, img_h=900):
    """Render background scatter as a PNG and return base64-encoded data URI."""
    import io, base64
    dpi    = 150
    fw, fh = img_w / dpi, img_h / dpi
    fig_bg, ax_bg = plt.subplots(figsize=(fw, fh), dpi=dpi)
    for t, color in trad_color_map.items():
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


def build_outlier_plot(mask, title, filename):
    hi_df   = df[mask]
    pad_x   = (hi_df["ux"].max() - hi_df["ux"].min()) * 0.25 + 1.0
    pad_y   = (hi_df["uy"].max() - hi_df["uy"].min()) * 0.25 + 1.0
    x_range = [hi_df["ux"].min() - pad_x, hi_df["ux"].max() + pad_x]
    y_range = [hi_df["uy"].min() - pad_y, hi_df["uy"].max() + pad_y]
    in_view = (
        (df["ux"] >= x_range[0]) & (df["ux"] <= x_range[1]) &
        (df["uy"] >= y_range[0]) & (df["uy"] <= y_range[1])
    )

    # Render background as a static image — no interactive overhead
    bg_img = _bg_to_base64(
        df[in_view & ~mask], x_range, y_range, trad_color_out
    )

    fig_o = go.Figure()

    # Background image pinned to data coordinates
    fig_o.add_layout_image(dict(
        source=bg_img,
        xref="x", yref="y",
        x=x_range[0], y=y_range[1],
        sizex=x_range[1] - x_range[0],
        sizey=y_range[1] - y_range[0],
        sizing="stretch",
        layer="below",
        opacity=1.0,
    ))

    # Interactive highlighted points only
    for t in traditions:
        color  = trad_color_out[t]
        t_mask = (df["tradition"] == t).values
        hi     = t_mask & mask
        fig_o.add_trace(go.Scatter(
            x=df.loc[hi, "ux"], y=df.loc[hi, "uy"],
            mode="markers",
            marker=dict(size=10, color=color, opacity=0.95,
                        line=dict(width=1.5, color="white")),
            text=df.loc[hi, "hover"],
            hovertemplate="%{text}<extra></extra>",
            legendgroup=t, showlegend=True, name=t,
        ))

    fig_o.update_layout(
        title=title, height=750,
        autosize=True,
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(title="Tradition"),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=x_range),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   range=y_range, scaleanchor="x"),
    )
    fig_o.write_html(filename)
    print(f"Saved: {filename}")

build_outlier_plot(
    outlier_mask,
    "Within-Tradition Outliers — Passages Least Like Their Own Tradition",
    "final_outliers_within.html",
)
build_outlier_plot(
    wanderer_mask,
    "Cross-Tradition Wanderers — Passages Drawn Toward Other Traditions",
    "final_outliers_cross.html",
)

print("\nDone. Outputs:")
print("  final_topk_heatmap.png")
print("  final_topk_heatmap_filtered.png  (if excluded traditions present)")
print("  final_lift.png")
print("  final_lift_filtered.png          (if excluded traditions present)")
print("  final_distributions.html")
print("  final_narrative_flow.png")
print("  final_outliers_report.txt")
print("  final_outliers_within.html")
print("  final_outliers_cross.html")
