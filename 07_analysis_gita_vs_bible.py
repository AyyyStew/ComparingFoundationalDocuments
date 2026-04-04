# %% [markdown]
# # Bhagavad Gita vs Bible — Cross-tradition Analysis
#
# 1. Joint chapter UMAP — colored by tradition
# 2. Gita chapter × Bible book heatmap
# 3. Most similar chapters (cross-tradition pairs)
# 4. Most similar verses
# 5. Most dissimilar verses
# 6. Nearest Bible neighbor per Gita verse — which Bible books claim the most
# 7. Similarity distribution — cross-tradition vs within-tradition baselines

# %% Imports
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics.pairwise import cosine_similarity

from db.schema import get_conn

MODEL_NAME   = "all-mpnet-base-v2"
BIBLE_CORPUS = "Bible — KJV (King James Version)"
GITA_CORPUS  = "Bhagavad Gita"
RANDOM_STATE = 42
TOP_N        = 20   # for ranked tables

# %% Load embeddings from DuckDB
conn = get_conn()

def load_corpus(conn, corpus_name: str) -> pd.DataFrame:
    rows = conn.execute(
        """
        SELECT p.id, p.book, p.section, p.unit_number, p.unit_label, p.text, e.vector
        FROM passage p
        JOIN embedding e ON e.passage_id = p.id
        JOIN corpus c    ON c.id = p.corpus_id
        WHERE c.name       = ?
          AND e.model_name = ?
        ORDER BY p.id
        """,
        [corpus_name, MODEL_NAME],
    ).fetchall()
    return pd.DataFrame(rows, columns=["id", "book", "section", "unit_number", "unit_label", "text", "vector"])

bible = load_corpus(conn, BIBLE_CORPUS)
gita  = load_corpus(conn, GITA_CORPUS)
conn.close()

bible_emb = np.vstack(bible["vector"].values).astype(np.float32)
gita_emb  = np.vstack(gita["vector"].values).astype(np.float32)

print(f"Bible: {len(bible):,} verses | Gita: {len(gita):,} verses")

# %% Aggregate to chapter level (mean pooling)
def chapter_embeddings(df: pd.DataFrame, emb: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df["chapter_key"] = df["book"] + "|" + df["section"].astype(str)
    groups = df.groupby("chapter_key", sort=False)
    records = []
    for ck, grp in groups:
        vecs = emb[grp.index]
        records.append({
            "chapter_key": ck,
            "book":        grp["book"].iloc[0],
            "section":     grp["section"].iloc[0],
            "embedding":   vecs.mean(axis=0),
            "n_verses":    len(grp),
        })
    # preserve canonical order
    order = df["chapter_key"].drop_duplicates().tolist()
    result = pd.DataFrame(records).set_index("chapter_key").loc[order].reset_index()
    return result

bible_chaps = chapter_embeddings(bible, bible_emb)
gita_chaps  = chapter_embeddings(gita,  gita_emb)

bible_chap_emb = np.vstack(bible_chaps["embedding"].values).astype(np.float32)
gita_chap_emb  = np.vstack(gita_chaps["embedding"].values).astype(np.float32)

print(f"Bible chapters: {len(bible_chaps):,} | Gita chapters: {len(gita_chaps):,}")

# ── Plot 1: Joint chapter UMAP ────────────────────────────────────────────────
# %% Run joint UMAP on all chapters from both traditions
print("Running joint chapter UMAP...")

all_chap_emb = np.vstack([bible_chap_emb, gita_chap_emb])
all_labels   = (["Bible"] * len(bible_chaps)) + (["Gita"] * len(gita_chaps))

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                    metric="cosine", random_state=RANDOM_STATE)
all_2d = reducer.fit_transform(all_chap_emb)

bible_2d = all_2d[:len(bible_chaps)]
gita_2d  = all_2d[len(bible_chaps):]

# Color Bible chapters by book
bible_book_order = bible["book"].drop_duplicates().tolist()
n_bible_books    = len(bible_book_order)
bible_book_cmap  = {b: cm.nipy_spectral(i / n_bible_books) for i, b in enumerate(bible_book_order)}
bible_point_cols = bible_chaps["book"].map(bible_book_cmap).tolist()

fig, ax = plt.subplots(figsize=(16, 12))

# Bible chapters — small, colored by book, low alpha
ax.scatter(bible_2d[:, 0], bible_2d[:, 1],
           c=bible_point_cols, s=6, alpha=0.35, linewidths=0, label="_nolegend_")

# Gita chapters — larger, black outline, labeled
gita_colors = cm.plasma(np.linspace(0.1, 0.9, len(gita_chaps)))
for i, row in gita_chaps.iterrows():
    ax.scatter(gita_2d[i, 0], gita_2d[i, 1],
               color=gita_colors[i], s=80, zorder=5,
               edgecolors="black", linewidths=0.8)
    ax.annotate(f"BG {row['section']}\n{row['book'][:18]}",
                (gita_2d[i, 0], gita_2d[i, 1]),
                fontsize=5.5, ha="center", va="bottom",
                xytext=(0, 5), textcoords="offset points", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6))

ax.set_title("Joint Chapter UMAP — Bhagavad Gita vs Bible (KJV)\n"
             "Bible chapters colored by book · Gita chapters in black-outlined circles",
             fontsize=12)
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig("07_joint_chapter_umap.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Plot 2: Gita chapter × Bible book heatmap ─────────────────────────────────
# %% Mean cosine similarity: each Gita chapter vs each Bible book
print("Computing Gita chapter × Bible book similarities...")

bible_book_order_unique = bible_chaps["book"].drop_duplicates().tolist()
gita_chapter_labels     = [f"Ch{row['section']}: {row['book'][:22]}"
                            for _, row in gita_chaps.iterrows()]

heatmap = np.zeros((len(gita_chaps), len(bible_book_order_unique)), dtype=np.float32)

for j, book in enumerate(bible_book_order_unique):
    book_mask   = bible_chaps["book"].values == book
    book_emb    = bible_chap_emb[book_mask]
    sims        = cosine_similarity(gita_chap_emb, book_emb)  # (18, n_book_chaps)
    heatmap[:, j] = sims.mean(axis=1)

vmin, vmax = heatmap.min(), heatmap.max()
vmid = (vmin + vmax) / 2

fig, ax = plt.subplots(figsize=(22, 8))
im = ax.imshow(heatmap, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
ax.set_yticks(range(len(gita_chapter_labels)))
ax.set_yticklabels(gita_chapter_labels, fontsize=7)
ax.set_xticks(range(len(bible_book_order_unique)))
ax.set_xticklabels(bible_book_order_unique, rotation=90, fontsize=6)
cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01, label="Mean cosine similarity")
cbar.set_ticks([vmin, vmid, vmax])
cbar.set_ticklabels([f"{vmin:.3f}", f"{vmid:.3f} (mid)", f"{vmax:.3f}"])
ax.axvline(38.5, color="white", lw=1.2, alpha=0.7)
ax.text(19, -1.2, "Old Testament", ha="center", fontsize=7, color="grey")
ax.text(52, -1.2, "New Testament", ha="center", fontsize=7, color="grey")
ax.set_title("Bhagavad Gita Chapters × Bible Books — Mean Cosine Similarity", fontsize=13)
plt.tight_layout()
plt.savefig("07_gita_bible_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Plot 3: Most similar chapters ────────────────────────────────────────────
# %% All Gita-chapter × Bible-chapter pairs, ranked by cosine similarity
print("Ranking cross-tradition chapter pairs...")

chap_sim_matrix = cosine_similarity(gita_chap_emb, bible_chap_emb)  # (18, n_bible_chaps)

pair_rows = []
for gi in range(len(gita_chaps)):
    for bi in range(len(bible_chaps)):
        pair_rows.append({
            "gita_chapter":  f"Ch{gita_chaps.iloc[gi]['section']}: {gita_chaps.iloc[gi]['book']}",
            "bible_chapter": f"{bible_chaps.iloc[bi]['book']} {bible_chaps.iloc[bi]['section']}",
            "similarity":    chap_sim_matrix[gi, bi],
        })

pairs_df = pd.DataFrame(pair_rows).sort_values("similarity", ascending=False)

print(f"\nTop {TOP_N} most similar chapter pairs (Gita ↔ Bible):")
print(pairs_df.head(TOP_N).to_string(index=False))

print(f"\nTop {TOP_N} most dissimilar chapter pairs (Gita ↔ Bible):")
print(pairs_df.tail(TOP_N).to_string(index=False))

# ── Plot 4 & 5: Most similar / dissimilar verses ─────────────────────────────
# %% Full verse × verse similarity matrix (Gita rows, Bible cols)
print("\nComputing full verse similarity matrix (640 × 31,102)...")
verse_sim_matrix = cosine_similarity(gita_emb, bible_emb)  # (640, 31102)

# Per Gita verse: max similarity to any Bible verse
max_sim_per_gita = verse_sim_matrix.max(axis=1)
max_idx_per_gita = verse_sim_matrix.argmax(axis=1)

# Most similar pairs
top_gita_idx  = np.argsort(max_sim_per_gita)[::-1][:TOP_N]

print(f"\nTop {TOP_N} most similar verse pairs (Gita ↔ Bible):")
sim_rows = []
for gi in top_gita_idx:
    bi = max_idx_per_gita[gi]
    sim_rows.append({
        "similarity":   max_sim_per_gita[gi],
        "gita_ref":     f"Ch{gita.iloc[gi]['section']} v{gita.iloc[gi]['unit_label']}",
        "gita_text":    gita.iloc[gi]["text"][:90],
        "bible_ref":    f"{bible.iloc[bi]['book']} {bible.iloc[bi]['section']}:{bible.iloc[bi]['unit_number']}",
        "bible_text":   bible.iloc[bi]["text"][:90],
    })
sim_df = pd.DataFrame(sim_rows)
pd.set_option("display.max_colwidth", 92)
print(sim_df.to_string(index=False))

# Most dissimilar pairs (lowest max similarity)
bot_gita_idx = np.argsort(max_sim_per_gita)[:TOP_N]

print(f"\nTop {TOP_N} most dissimilar Gita verses (lowest best-match similarity to Bible):")
dis_rows = []
for gi in bot_gita_idx:
    bi = max_idx_per_gita[gi]
    dis_rows.append({
        "best_sim":   max_sim_per_gita[gi],
        "gita_ref":   f"Ch{gita.iloc[gi]['section']} v{gita.iloc[gi]['unit_label']}",
        "gita_text":  gita.iloc[gi]["text"][:90],
        "bible_ref":  f"{bible.iloc[bi]['book']} {bible.iloc[bi]['section']}:{bible.iloc[bi]['unit_number']}",
        "bible_text": bible.iloc[bi]["text"][:90],
    })
print(pd.DataFrame(dis_rows).to_string(index=False))

# ── Plot 6: Nearest Bible neighbor — which books claim the most Gita verses ──
# %% For each Gita verse, find its nearest Bible verse and record the book
nearest_bible_book = bible["book"].values[max_idx_per_gita]
book_claim_counts  = pd.Series(nearest_bible_book).value_counts()

fig, ax = plt.subplots(figsize=(14, 6))
colors  = [cm.nipy_spectral(i / n_bible_books)
           for i, b in enumerate(bible_book_order)
           if b in book_claim_counts.index]
book_claim_counts.plot(kind="bar", ax=ax, color=colors[:len(book_claim_counts)], edgecolor="none")
ax.set_title("Which Bible Books Claim the Most Gita Verses?\n"
             "(nearest Bible verse per Gita verse)", fontsize=12)
ax.set_xlabel("Bible Book")
ax.set_ylabel("Number of Gita verses claimed")
ax.tick_params(axis="x", labelsize=7, rotation=90)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("07_nearest_neighbor_claims.png", dpi=150)
plt.show()

# ── Plot 7: Similarity distribution ──────────────────────────────────────────
# %% Cross-tradition vs within-tradition similarity distributions
print("\nComputing within-tradition similarity samples...")

# Cross-tradition: all Gita→Bible verse sims (sample for speed)
rng = np.random.default_rng(RANDOM_STATE)
sample_gi = rng.integers(0, len(gita),   size=2000)
sample_bi = rng.integers(0, len(bible),  size=2000)
cross_sims = (gita_emb[sample_gi] * bible_emb[sample_bi]).sum(axis=1) / (
    np.linalg.norm(gita_emb[sample_gi],  axis=1) *
    np.linalg.norm(bible_emb[sample_bi], axis=1) + 1e-9
)

# Within-Bible: random pairs
b1 = rng.integers(0, len(bible), size=2000)
b2 = rng.integers(0, len(bible), size=2000)
bible_sims = (bible_emb[b1] * bible_emb[b2]).sum(axis=1) / (
    np.linalg.norm(bible_emb[b1], axis=1) *
    np.linalg.norm(bible_emb[b2], axis=1) + 1e-9
)

# Within-Gita: all pairs (only 640 verses so manageable)
gita_sim_matrix = cosine_similarity(gita_emb)
mask = np.triu(np.ones_like(gita_sim_matrix, dtype=bool), k=1)
gita_sims = gita_sim_matrix[mask]

fig, ax = plt.subplots(figsize=(11, 5))
bins = np.linspace(-0.2, 1.0, 80)
ax.hist(bible_sims, bins=bins, alpha=0.5, color="steelblue",   label="Within Bible (random pairs)", density=True)
ax.hist(gita_sims,  bins=bins, alpha=0.5, color="darkorange",  label="Within Gita (all pairs)",     density=True)
ax.hist(cross_sims, bins=bins, alpha=0.5, color="mediumpurple",label="Gita ↔ Bible (random pairs)", density=True)

ax.axvline(np.median(bible_sims), color="steelblue",    lw=1.5, ls="--", alpha=0.8)
ax.axvline(np.median(gita_sims),  color="darkorange",   lw=1.5, ls="--", alpha=0.8)
ax.axvline(np.median(cross_sims), color="mediumpurple", lw=1.5, ls="--", alpha=0.8)

ax.set_xlabel("Cosine similarity")
ax.set_ylabel("Density")
ax.set_title("Cosine Similarity Distributions — Cross-tradition vs Within-tradition", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("07_similarity_distributions.png", dpi=150)
plt.show()

print("\nMedian similarities:")
print(f"  Within Bible:  {np.median(bible_sims):.4f}")
print(f"  Within Gita:   {np.median(gita_sims):.4f}")
print(f"  Gita ↔ Bible:  {np.median(cross_sims):.4f}")
