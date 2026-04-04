# %% [markdown]
# # Bhagavad Gita vs Wisdom Books & Epistles — Chapter-level Analysis
#
# Zooms in on the parts of the Bible most philosophically adjacent to the Gita:
#   - Wisdom books: Job, Psalms, Proverbs, Ecclesiastes
#   - Epistles: Romans → Jude
#
# Heatmap: each Gita chapter vs each Bible chapter in these sections.

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics.pairwise import cosine_similarity

from db.schema import get_conn

MODEL_NAME   = "all-mpnet-base-v2"
BIBLE_CORPUS = "Bible — KJV (King James Version)"
GITA_CORPUS  = "Bhagavad Gita"

WISDOM_BOOKS = ["Job", "Psalms", "Proverbs", "Ecclesiastes"]

EPISTLES = [
    "Romans", "I Corinthians", "II Corinthians", "Galatians",
    "Ephesians", "Philippians", "Colossians", "I Thessalonians",
    "II Thessalonians", "I Timothy", "II Timothy", "Titus",
    "Philemon", "Hebrews", "James", "I Peter", "II Peter",
    "I John", "II John", "III John", "Jude",
]

FOCUS_BOOKS = WISDOM_BOOKS + EPISTLES

# %% Load from DuckDB
conn = get_conn()

def load_corpus(conn, corpus_name):
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

bible_full = load_corpus(conn, BIBLE_CORPUS)
gita       = load_corpus(conn, GITA_CORPUS)
conn.close()

# Filter Bible to focus books only
bible = bible_full[bible_full["book"].isin(FOCUS_BOOKS)].reset_index(drop=True)
bible_emb = np.vstack(bible["vector"].values).astype(np.float32)
gita_emb  = np.vstack(gita["vector"].values).astype(np.float32)

print(f"Gita: {len(gita):,} verses")
print(f"Bible focus ({len(FOCUS_BOOKS)} books): {len(bible):,} verses")

# %% Aggregate to chapter level
def to_chapters(df, emb):
    df = df.copy()
    df["chapter_key"] = df["book"] + "|" + df["section"].astype(str)
    order = df["chapter_key"].drop_duplicates().tolist()
    records = []
    for ck, grp in df.groupby("chapter_key", sort=False):
        records.append({
            "chapter_key": ck,
            "book":        grp["book"].iloc[0],
            "section":     grp["section"].iloc[0],
            "label":       f"{grp['book'].iloc[0]} {grp['section'].iloc[0]}",
            "embedding":   emb[grp.index].mean(axis=0),
        })
    return pd.DataFrame(records).set_index("chapter_key").loc[order].reset_index()

gita_chaps  = to_chapters(gita,  gita_emb)
bible_chaps = to_chapters(bible, bible_emb)

gita_chap_emb  = np.vstack(gita_chaps["embedding"].values).astype(np.float32)
bible_chap_emb = np.vstack(bible_chaps["embedding"].values).astype(np.float32)

print(f"Gita chapters: {len(gita_chaps)} | Bible focus chapters: {len(bible_chaps)}")

# %% Cosine similarity matrix: Gita chapters (rows) × Bible chapters (cols)
sim_matrix = cosine_similarity(gita_chap_emb, bible_chap_emb)  # (18, n_bible_chaps)

# Row labels: Gita chapters
gita_labels = [f"Ch{row['section']}: {row['book'][:24]}" for _, row in gita_chaps.iterrows()]

# Col labels: Bible chapters
bible_labels = bible_chaps["label"].tolist()

# %% Plot — one heatmap per row, one row per Bible book, one figure per section
# Shared color scale across all panels so cells are directly comparable
vmin, vmax = sim_matrix.min(), sim_matrix.max()
vmid = (vmin + vmax) / 2

def plot_section(section_name, section_books, fname):
    books_present = [b for b in section_books if b in bible_chaps["book"].values]
    n_books = len(books_present)

    # Each row is one heatmap. Width is fixed; height scales with number of books.
    row_height = 5
    fig, axes = plt.subplots(
        n_books, 1,
        figsize=(18, n_books * row_height),
        squeeze=False,
    )

    for idx, book in enumerate(books_present):
        ax = axes[idx][0]
        book_mask  = bible_chaps["book"].values == book
        book_slice = sim_matrix[:, book_mask]   # (18, n_chapters_in_book)
        chap_nums  = bible_chaps[book_mask]["section"].tolist()

        im = ax.imshow(book_slice, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)

        # Y axis — Gita chapters
        ax.set_yticks(range(len(gita_labels)))
        ax.set_yticklabels(gita_labels, fontsize=7)

        # X axis — Bible chapters
        ax.set_xticks(range(len(chap_nums)))
        ax.set_xticklabels([f"Ch {c}" for c in chap_nums], fontsize=7, rotation=90)

        ax.set_ylabel("Gita chapter", fontsize=7)
        ax.set_title(f"Gita vs {book}", fontsize=10, fontweight="bold", loc="left")

        fig.colorbar(im, ax=ax, fraction=0.01, pad=0.01, label="Cosine sim")

    fig.suptitle(
        f"Bhagavad Gita vs {section_name} — Chapter-level Cosine Similarity\n"
        f"(shared scale: {vmin:.3f} – {vmax:.3f})",
        fontsize=13, y=1.005,
    )
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {fname}")

plot_section("Wisdom Books", WISDOM_BOOKS, "08_gita_vs_wisdom_epistles__wisdom_heatmap.png")
plot_section("Epistles",     EPISTLES,     "08_gita_vs_widsom_epistles__epistles_heatmap.png")

# %% Top similar pairs — printed table
print(f"\nTop 20 most similar Gita ↔ Bible chapter pairs (Wisdom + Epistles):")
pairs = []
for gi in range(len(gita_chaps)):
    for bi in range(len(bible_chaps)):
        pairs.append({
            "similarity":    sim_matrix[gi, bi],
            "gita_chapter":  gita_labels[gi],
            "bible_chapter": bible_labels[bi],
        })

pairs_df = pd.DataFrame(pairs).sort_values("similarity", ascending=False)
pd.set_option("display.max_colwidth", 40)
print(pairs_df.head(20).to_string(index=False))
