# %% [markdown]
# # Easter Bonus — The Words of Jesus in Semantic Space
#
# Four takes on the same question: do the Gospels trace the same path?
#
# Data: red-letter verses from kjv.json (*r markup = Jesus speaking)
# Embeddings already in DuckDB — we just filter to red-letter verses.
#
# Four outputs:
#   bonus_easter_1_annotated.png  — all 4 paths, key story beats labelled
#   bonus_easter_2_redletter.png  — Jesus's words only (red-letter verses)
#   bonus_easter_3_passion.png    — final chapters only (Passion week)
#   bonus_easter_4_parallel.png   — synoptic parallels highlighted

# %% Imports
import json, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from collections import defaultdict
from umap import UMAP

from db.schema import get_conn

MODEL_NAME = "all-mpnet-base-v2"
CORPUS     = "Bible — KJV (King James Version)"
GOSPELS    = ["Matthew", "Mark", "Luke", "John"]

GOSPEL_COLORS = {
    "Matthew": "#ff6b6b",
    "Mark":    "#ffd93d",
    "Luke":    "#6bcb77",
    "John":    "#4d96ff",
}
BG_COLOR = "#0d1117"

# ── Key story beats per Gospel (chapter → label) ─────────────────────────────
STORY_BEATS = {
    "Matthew": {
        1:  "Birth", 3: "Baptism", 5: "Sermon\non the Mount",
        14: "Walking\non Water", 21: "Triumphal\nEntry",
        26: "Last Supper", 27: "Crucifixion", 28: "Resurrection",
    },
    "Mark": {
        1:  "Baptism", 6: "Walking\non Water", 11: "Triumphal\nEntry",
        14: "Last Supper", 15: "Crucifixion", 16: "Resurrection",
    },
    "Luke": {
        2:  "Birth", 4: "Temptation", 6: "Sermon\non the Plain",
        15: "Prodigal Son", 19: "Triumphal\nEntry",
        22: "Last Supper", 23: "Crucifixion", 24: "Resurrection",
    },
    "John": {
        1:  "In the Beginning", 3: "Born Again", 6: "Bread of Life",
        11: "Lazarus", 13: "Last Supper",
        14: "I Am the Way", 19: "Crucifixion", 20: "Resurrection",
    },
}

# ── Passion week chapters (last stretch of each Gospel) ──────────────────────
PASSION_CHAPTERS = {
    "Matthew": range(21, 29),
    "Mark":    range(11, 17),
    "Luke":    range(19, 25),
    "John":    range(12, 22),
}

# ── Synoptic parallel events: (Matthew ch, Mark ch, Luke ch) ─────────────────
SYNOPTIC_PARALLELS = [
    ("Baptism",         3,  1,  3),
    ("Temptation",      4,  1,  4),
    ("Sermon",          5,  None, 6),
    ("Feeding 5000",   14,  6,  9),
    ("Transfiguration",17,  9,  9),
    ("Triumphal Entry",21, 11, 19),
    ("Last Supper",    26, 14, 22),
    ("Crucifixion",    27, 15, 23),
    ("Resurrection",   28, 16, 24),
]

# %% ── Parse red-letter verses from kjv.json ──────────────────────────────────
print("Parsing red-letter verses from kjv.json ...")
with open("data/bibles/kjv.json") as f:
    kjv_data = json.load(f)

# Map (book, chapter, verse_str) → has_jesus_speech
red_letter = set()   # set of "Book:chapter:verse" strings
for item in kjv_data:
    r = item.get("r", "")
    t = item.get("t", "")
    if not r or "*r" not in str(t):
        continue
    parts = r.split(":")
    if len(parts) < 4:
        continue
    _, book, chapter, verse = parts[0], parts[1], parts[2], parts[3]
    if book in set(GOSPELS):
        red_letter.add(f"{book}:{chapter}:{verse}")

print(f"  Red-letter verses: {len(red_letter):,}")

# %% ── Load Gospel embeddings from DuckDB ─────────────────────────────────────
conn = get_conn()

# Load ALL KJV verses (for background) + gospel verses
rows = conn.execute("""
    SELECT p.book, p.unit_label, p.id AS passage_id, e.vector
    FROM embedding e
    JOIN passage p ON e.passage_id = p.id
    JOIN corpus c  ON p.corpus_id = c.id
    WHERE e.model_name = ?
    AND c.name = ?
    ORDER BY p.id
""", [MODEL_NAME, CORPUS]).fetchall()
conn.close()

df = pd.DataFrame(rows, columns=["book", "unit_label", "passage_id", "vector"])
df["vector"]  = df["vector"].apply(np.array)
df["chapter"] = df["unit_label"].str.split(":").str[0].astype(int)
df["verse"]   = df["unit_label"].str.split(":").str[1]

# Tag red-letter verses
df["red_letter"] = df.apply(
    lambda r: f"{r['book']}:{r['chapter']}:{r['verse']}" in red_letter, axis=1
)

gospel_df = df[df["book"].isin(GOSPELS)].copy()
bg_df     = df[~df["book"].isin(GOSPELS)].copy()

print(f"Gospel verses: {len(gospel_df):,}  "
      f"({gospel_df['red_letter'].sum():,} red-letter)")
print(f"Background (rest of Bible): {len(bg_df):,} verses")

# %% ── Compute chapter-mean vectors ──────────────────────────────────────────
def chapter_means(source_df):
    rows = []
    for (book, chapter), grp in source_df.groupby(["book", "chapter"]):
        vecs = np.stack(grp["vector"].values)
        mean_vec = vecs.mean(axis=0)
        mean_vec /= (np.linalg.norm(mean_vec) + 1e-9)
        rows.append({"book": book, "chapter": int(chapter), "vector": mean_vec})
    return pd.DataFrame(rows)

# All-verse chapter means (for plots 1, 3, 4)
all_ch   = chapter_means(gospel_df)
# Red-letter chapter means (for plot 2) — only chapters where Jesus speaks
rl_ch    = chapter_means(gospel_df[gospel_df["red_letter"]])
# Background: book-level means for all non-gospel Bible books
bg_ch    = chapter_means(bg_df)

# %% ── UMAP fits ──────────────────────────────────────────────────────────────
def fit_umap(ch_df, bg=None, n_neighbors=15, min_dist=0.08):
    """Fit UMAP on ch_df vectors (+ optional bg), return coords for ch_df only."""
    if bg is not None:
        all_vecs = np.vstack([
            np.stack(ch_df["vector"].values),
            np.stack(bg["vector"].values),
        ]).astype("float32")
        n_ch = len(ch_df)
    else:
        all_vecs = np.stack(ch_df["vector"].values).astype("float32")
        n_ch = len(ch_df)

    reducer = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                   metric="cosine", random_state=42, low_memory=False)
    X2 = reducer.fit_transform(all_vecs).astype("float32")
    return X2[:n_ch], X2[n_ch:] if bg is not None else None

print("\nFitting UMAP (all-verse chapters + Bible background) ...")
X_all, X_bg = fit_umap(all_ch, bg=bg_ch)
all_ch["ux"] = X_all[:, 0]; all_ch["uy"] = X_all[:, 1]
bg_ch["ux"]  = X_bg[:, 0];  bg_ch["uy"]  = X_bg[:, 1]

print("Fitting UMAP (red-letter chapters only) ...")
X_rl, _ = fit_umap(rl_ch, n_neighbors=12, min_dist=0.06)
rl_ch["ux"] = X_rl[:, 0]; rl_ch["uy"] = X_rl[:, 1]

# %% ── Plotting helpers ───────────────────────────────────────────────────────
def style_ax(ax, title=None, title_color="white", fontsize=12):
    ax.set_facecolor(BG_COLOR)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#222233")
    if title:
        ax.set_title(title, fontsize=fontsize, fontweight="bold",
                     color=title_color, pad=8)

def draw_bg_dots(ax, bg_source, alpha=0.18, xlim=None, ylim=None):
    bx, by = bg_source["ux"].values, bg_source["uy"].values
    if xlim is not None:
        m = (bx >= xlim[0]) & (bx <= xlim[1]) & (by >= ylim[0]) & (by <= ylim[1])
        bx, by = bx[m], by[m]
    ax.scatter(bx, by, s=5, color="#ffffff", alpha=alpha, linewidths=0, zorder=1)

def draw_path(ax, ch_df, gospel, color, beat_labels=None,
              fontsize=7, zoom=False, alpha_scale=True):
    sub = ch_df[ch_df["book"] == gospel].sort_values("chapter")
    coords = sub[["ux", "uy"]].values
    chapters = sub["chapter"].tolist()
    n = len(coords)
    if n == 0:
        return

    if zoom:
        pad = max(0.6, (coords[:, 0].max() - coords[:, 0].min() + coords[:, 1].max() - coords[:, 1].min()) * 0.20)
        ax.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
        ax.set_ylim(coords[:, 1].min() - pad, coords[:, 1].max() + pad)

    for i in range(n - 1):
        alpha = (0.4 + 0.6 * (i / max(n - 2, 1))) if alpha_scale else 0.7
        ax.annotate("", xy=coords[i + 1], xytext=coords[i],
                    arrowprops=dict(
                        arrowstyle="->, head_width=0.22, head_length=0.22",
                        color=color, lw=1.5, alpha=alpha,
                    ), zorder=5)

    # Glow + solid dot
    ax.scatter(coords[:, 0], coords[:, 1], s=90, color=color,
               alpha=0.20, linewidths=0, zorder=6)
    ax.scatter(coords[:, 0], coords[:, 1], s=22, color=color,
               alpha=1.0, linewidths=0, zorder=7)

    # Start / end
    ax.scatter(*coords[0],  marker="^", s=110, color=color,
               edgecolors="white", linewidths=0.8, zorder=8)
    ax.scatter(*coords[-1], marker="s", s=90,  color=color,
               edgecolors="white", linewidths=0.8, zorder=8)

    # Labels
    if beat_labels:
        for i, ch in enumerate(chapters):
            if ch in beat_labels:
                txt = beat_labels[ch]
                ax.annotate(txt, xy=coords[i],
                            fontsize=fontsize, ha="center", va="bottom",
                            xytext=(0, 7), textcoords="offset points",
                            color="white", fontweight="bold", zorder=9,
                            path_effects=[pe.withStroke(linewidth=2.5,
                                                        foreground=BG_COLOR)])
    else:
        step = 2 if n > 20 else 1
        for i, (coord, ch) in enumerate(zip(coords, chapters)):
            if i % step == 0 or i == n - 1:
                ax.annotate(str(ch), xy=coord,
                            fontsize=fontsize, ha="center", va="bottom",
                            xytext=(0, 5), textcoords="offset points",
                            color="white", fontweight="bold", zorder=9,
                            path_effects=[pe.withStroke(linewidth=2,
                                                        foreground=BG_COLOR)])

def gospel_legend(ax, extra=None):
    patches = [mpatches.Patch(color=GOSPEL_COLORS[g], label=g) for g in GOSPELS]
    if extra:
        patches += extra
    ax.legend(handles=patches, fontsize=9, loc="lower right",
              facecolor="#1a1f2e", edgecolor="#333344",
              labelcolor="white", framealpha=0.85)

def save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# Plot 1 — All paths, key story beats annotated
# ════════════════════════════════════════════════════════════════════════════
print("\nPlot 1: Annotated story beats ...")
fig1 = plt.figure(figsize=(20, 24))
fig1.patch.set_facecolor(BG_COLOR)
gs1 = GridSpec(2, 4, figure=fig1, height_ratios=[1.2, 1],
               hspace=0.28, wspace=0.12,
               left=0.03, right=0.97, top=0.92, bottom=0.03)

ax1_ov = fig1.add_subplot(gs1[0, :])
ax1_gs = [fig1.add_subplot(gs1[1, i]) for i in range(4)]

draw_bg_dots(ax1_ov, bg_ch, alpha=0.12)
for g in GOSPELS:
    draw_path(ax1_ov, all_ch, g, GOSPEL_COLORS[g],
              beat_labels=STORY_BEATS[g], fontsize=7)
gospel_legend(ax1_ov)
style_ax(ax1_ov, "The Four Gospels — Key Story Beats", fontsize=14)
ax1_ov.text(0.01, 0.02, "Background = rest of Bible",
            transform=ax1_ov.transAxes, fontsize=8,
            color="#555566", ha="left", va="bottom")

for ax, gospel in zip(ax1_gs, GOSPELS):
    draw_path(ax, all_ch, gospel, GOSPEL_COLORS[gospel],
              beat_labels=STORY_BEATS[gospel], fontsize=8, zoom=True)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    draw_bg_dots(ax, bg_ch, alpha=0.15, xlim=xlim, ylim=ylim)
    style_ax(ax, gospel, title_color=GOSPEL_COLORS[gospel])

fig1.text(0.5, 0.975, "The Four Gospels in Semantic Space",
          ha="center", va="top", fontsize=20, fontweight="bold", color="white")
fig1.text(0.5, 0.958, "Chapter-level embeddings  •  Key story beats labelled  •  KJV",
          ha="center", va="top", fontsize=10, color="#8888aa")
save(fig1, "bonus_easter_1_annotated.png")


# ════════════════════════════════════════════════════════════════════════════
# Plot 2 — Jesus's words only (red-letter chapters)
# ════════════════════════════════════════════════════════════════════════════
print("Plot 2: Red-letter (Jesus's words only) ...")
fig2 = plt.figure(figsize=(20, 24))
fig2.patch.set_facecolor(BG_COLOR)
gs2 = GridSpec(2, 4, figure=fig2, height_ratios=[1.2, 1],
               hspace=0.28, wspace=0.12,
               left=0.03, right=0.97, top=0.92, bottom=0.03)

ax2_ov = fig2.add_subplot(gs2[0, :])
ax2_gs = [fig2.add_subplot(gs2[1, i]) for i in range(4)]

draw_bg_dots(ax2_ov, rl_ch, alpha=0.08)   # other-gospel chapters as faint bg
for g in GOSPELS:
    draw_path(ax2_ov, rl_ch, g, GOSPEL_COLORS[g],
              beat_labels=STORY_BEATS[g], fontsize=7)
gospel_legend(ax2_ov)
style_ax(ax2_ov, "Where Jesus Speaks — Red-Letter Chapters", fontsize=14)
ax2_ov.text(0.01, 0.02, "Only chapters containing Jesus's direct speech (red-letter KJV)",
            transform=ax2_ov.transAxes, fontsize=8,
            color="#555566", ha="left", va="bottom")

for ax, gospel in zip(ax2_gs, GOSPELS):
    draw_path(ax, rl_ch, gospel, GOSPEL_COLORS[gospel],
              beat_labels=STORY_BEATS[gospel], fontsize=8, zoom=True)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    # background = other gospels' red-letter chapters
    other_rl = rl_ch[rl_ch["book"] != gospel]
    draw_bg_dots(ax, other_rl, alpha=0.20, xlim=xlim, ylim=ylim)
    style_ax(ax, gospel, title_color=GOSPEL_COLORS[gospel])
    n = len(rl_ch[rl_ch["book"] == gospel])
    ax.text(0.03, 0.03, f"{n} chapters w/ red-letter",
            transform=ax.transAxes, fontsize=7.5,
            color="#888899", ha="left", va="bottom")

fig2.text(0.5, 0.975, "The Words of Jesus — Semantic Space",
          ha="center", va="top", fontsize=20, fontweight="bold", color="#ff4444")
fig2.text(0.5, 0.958, "Chapter means of red-letter verses only  •  KJV  •  all-mpnet-base-v2",
          ha="center", va="top", fontsize=10, color="#8888aa")
save(fig2, "bonus_easter_2_redletter.png")


# ════════════════════════════════════════════════════════════════════════════
# Plot 3 — Passion week only
# ════════════════════════════════════════════════════════════════════════════
print("Plot 3: Passion week ...")
fig3 = plt.figure(figsize=(20, 10))
fig3.patch.set_facecolor(BG_COLOR)
gs3 = GridSpec(1, 4, figure=fig3, wspace=0.12,
               left=0.03, right=0.97, top=0.85, bottom=0.05)
axes3 = [fig3.add_subplot(gs3[0, i]) for i in range(4)]

for ax, gospel in zip(axes3, GOSPELS):
    passion_chs = set(PASSION_CHAPTERS[gospel])
    passion_df  = all_ch[(all_ch["book"] == gospel) &
                         (all_ch["chapter"].isin(passion_chs))]
    # Use passion chapters' bounding box but show all-gospel background
    coords = passion_df[["ux", "uy"]].values
    if len(coords) == 0:
        continue
    pad = max(0.8, (coords[:, 0].max() - coords[:, 0].min() + coords[:, 1].max() - coords[:, 1].min()) * 0.22)
    ax.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
    ax.set_ylim(coords[:, 1].min() - pad, coords[:, 1].max() + pad)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    # Background: full gospel (faint) + rest of Bible (very faint)
    draw_bg_dots(ax, all_ch[all_ch["book"] == gospel],
                 alpha=0.15, xlim=xlim, ylim=ylim)
    draw_bg_dots(ax, bg_ch, alpha=0.08, xlim=xlim, ylim=ylim)

    draw_path(ax, passion_df.rename(columns={"book": "book"}),
              gospel, GOSPEL_COLORS[gospel],
              beat_labels=STORY_BEATS[gospel], fontsize=8, zoom=False)

    style_ax(ax, gospel, title_color=GOSPEL_COLORS[gospel])
    ch_range = f"Ch. {min(passion_chs)}–{max(passion_chs)}"
    ax.text(0.03, 0.03, ch_range, transform=ax.transAxes,
            fontsize=8, color="#888899", ha="left", va="bottom")

fig3.text(0.5, 0.97, "Passion Week — The Final Chapters in Semantic Space",
          ha="center", va="top", fontsize=18, fontweight="bold", color="white")
fig3.text(0.5, 0.94, "Triumphal Entry → Crucifixion → Resurrection  •  KJV",
          ha="center", va="top", fontsize=10, color="#8888aa")
save(fig3, "bonus_easter_3_passion.png")


# ════════════════════════════════════════════════════════════════════════════
# Plot 4 — Synoptic parallels highlighted
# ════════════════════════════════════════════════════════════════════════════
print("Plot 4: Synoptic parallels ...")
PARALLEL_COLORS = plt.cm.Set2(np.linspace(0, 1, len(SYNOPTIC_PARALLELS)))

fig4, ax4 = plt.subplots(figsize=(16, 12))
fig4.patch.set_facecolor(BG_COLOR)

draw_bg_dots(ax4, bg_ch, alpha=0.10)

# Draw all four paths faintly first
for gospel in GOSPELS:
    sub = all_ch[all_ch["book"] == gospel].sort_values("chapter")
    coords = sub[["ux", "uy"]].values
    color  = GOSPEL_COLORS[gospel]
    for i in range(len(coords) - 1):
        ax4.annotate("", xy=coords[i + 1], xytext=coords[i],
                     arrowprops=dict(arrowstyle="->", color=color,
                                     lw=1.0, alpha=0.25), zorder=3)
    ax4.scatter(coords[:, 0], coords[:, 1], s=12, color=color,
                alpha=0.4, linewidths=0, zorder=4)

# Highlight parallel chapters with bright rings + event label
for idx, (event, mat_ch, mar_ch, luk_ch) in enumerate(SYNOPTIC_PARALLELS):
    pcolor = PARALLEL_COLORS[idx]
    parallel_map = {"Matthew": mat_ch, "Mark": mar_ch, "Luke": luk_ch}
    coords_event = []
    for gospel, ch in parallel_map.items():
        if ch is None:
            continue
        row = all_ch[(all_ch["book"] == gospel) & (all_ch["chapter"] == ch)]
        if row.empty:
            continue
        ux, uy = row.iloc[0]["ux"], row.iloc[0]["uy"]
        coords_event.append((ux, uy))
        ax4.scatter(ux, uy, s=200, color=pcolor, alpha=0.35,
                    linewidths=0, zorder=8)
        ax4.scatter(ux, uy, s=60, color=pcolor, alpha=1.0,
                    edgecolors="white", linewidths=0.8, zorder=9)

    # Draw lines connecting parallel chapters
    if len(coords_event) > 1:
        xs = [c[0] for c in coords_event]
        ys = [c[1] for c in coords_event]
        ax4.plot(xs, ys, color=pcolor, lw=1.2, alpha=0.5,
                 linestyle="--", zorder=7)

    # Label near centroid of the parallel group
    if coords_event:
        cx = np.mean([c[0] for c in coords_event])
        cy = np.mean([c[1] for c in coords_event])
        ax4.annotate(event, xy=(cx, cy),
                     fontsize=8, ha="center", va="bottom",
                     xytext=(0, 10), textcoords="offset points",
                     color="white", fontweight="bold", zorder=10,
                     path_effects=[pe.withStroke(linewidth=2.5,
                                                 foreground=BG_COLOR)])

# Gospel path legend
gospel_patches = [mpatches.Patch(color=GOSPEL_COLORS[g], label=g, alpha=0.6)
                  for g in ["Matthew", "Mark", "Luke"]]
ring_patch = mpatches.Patch(color="#aaaaaa", label="● Synoptic parallels")
ax4.legend(handles=gospel_patches + [ring_patch],
           fontsize=9, loc="lower right",
           facecolor="#1a1f2e", edgecolor="#333344",
           labelcolor="white", framealpha=0.85)

style_ax(ax4)
fig4.text(0.5, 0.97, "Where the Synoptic Gospels Tell the Same Story",
          ha="center", va="top", fontsize=18, fontweight="bold", color="white")
fig4.text(0.5, 0.945, "Dashed lines connect parallel events in Matthew, Mark & Luke  •  KJV",
          ha="center", va="top", fontsize=10, color="#8888aa")
ax4.text(0.01, 0.01, "John excluded (not a Synoptic Gospel)",
         transform=ax4.transAxes, fontsize=8,
         color="#555566", ha="left", va="bottom")

save(fig4, "bonus_easter_4_parallel.png")

print("\nAll done.")
