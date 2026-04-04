# %% [markdown]
# # Easter Bonus — The Words of Jesus (Red-Letter, Clean 2×2)
#
# 2×2 grid — one panel per Gospel.
# Data: chapter means of red-letter verses only (Jesus speaking, *r markup).
# Each panel shows one Gospel in colour; the other three paths in gray.
# Shared axes across all panels so position is directly comparable.

# %% Imports
import json, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
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

# Key beats — keep to 5 per gospel, no overlapping events
STORY_BEATS = {
    "Matthew": {
        5:  "Sermon on\nthe Mount",
        16: "Peter's\nConfession",
        21: "Triumphal\nEntry",
        26: "Last Supper",
        28: "Resurrection",
    },
    "Mark": {
        1:  "Baptism",
        9:  "Transfiguration",
        11: "Triumphal\nEntry",
        14: "Last Supper",
        16: "Resurrection",
    },
    "Luke": {
        6:  "Sermon on\nthe Plain",
        15: "Prodigal Son",
        19: "Triumphal\nEntry",
        22: "Last Supper",
        24: "Resurrection",
    },
    "John": {
        3:  "Born Again",
        6:  "Bread of Life",
        11: "Lazarus",
        14: "I Am the Way",
        20: "Resurrection",
    },
}

# %% Parse red-letter verses
print("Parsing red-letter verses ...")
with open("data/bibles/kjv.json") as f:
    kjv_data = json.load(f)

red_letter = set()
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

print(f"  {len(red_letter):,} red-letter verses found")

# %% Load Gospel embeddings
conn = get_conn()
rows = conn.execute("""
    SELECT p.book, p.unit_label, e.vector
    FROM embedding e
    JOIN passage p ON e.passage_id = p.id
    JOIN corpus c  ON p.corpus_id = c.id
    WHERE e.model_name = ?
    AND c.name = ?
    AND p.book IN ('Matthew','Mark','Luke','John')
    ORDER BY p.id
""", [MODEL_NAME, CORPUS]).fetchall()
conn.close()

df = pd.DataFrame(rows, columns=["book", "unit_label", "vector"])
df["vector"]  = df["vector"].apply(np.array)
df["chapter"] = df["unit_label"].str.split(":").str[0].astype(int)
df["verse"]   = df["unit_label"].str.split(":").str[1]
df["red_letter"] = df.apply(
    lambda r: f"{r['book']}:{r['chapter']}:{r['verse']}" in red_letter, axis=1
)

print(f"Loaded {len(df):,} Gospel verses  ({df['red_letter'].sum():,} red-letter)")

# %% Chapter means — red-letter verses only
rl_df = df[df["red_letter"]].copy()

ch_rows = []
for (book, chapter), grp in rl_df.groupby(["book", "chapter"]):
    vecs = np.stack(grp["vector"].values)
    mean_vec = vecs.mean(axis=0)
    mean_vec /= (np.linalg.norm(mean_vec) + 1e-9)
    ch_rows.append({"book": book, "chapter": int(chapter), "vector": mean_vec})

ch_df = pd.DataFrame(ch_rows)
print(f"Chapter means: {len(ch_df)} chapters across all Gospels")

# %% Fit UMAP on all red-letter chapter means together
vecs_all = np.stack(ch_df["vector"].values).astype("float32")

print("Fitting UMAP ...")
reducer = UMAP(
    n_components=2,
    n_neighbors=12,
    min_dist=0.06,
    metric="cosine",
    random_state=42,
    low_memory=False,
)
X2 = reducer.fit_transform(vecs_all).astype("float32")
ch_df["ux"] = X2[:, 0]
ch_df["uy"] = X2[:, 1]

# Shared axis limits across all panels — full extent with padding
pad_x = (X2[:, 0].max() - X2[:, 0].min()) * 0.06
pad_y = (X2[:, 1].max() - X2[:, 1].min()) * 0.06
XLIM = (X2[:, 0].min() - pad_x, X2[:, 0].max() + pad_x)
YLIM = (X2[:, 1].min() - pad_y, Y2max := X2[:, 1].max() + pad_y)
YLIM = (X2[:, 1].min() - pad_y, X2[:, 1].max() + pad_y)

# %% Drawing helpers
def labeled_box(ax, txt, xy, color, fontsize=8):
    """White text in a semi-transparent colored pill box."""
    ax.annotate(
        txt, xy=xy,
        xytext=(0, 9), textcoords="offset points",
        ha="center", va="bottom",
        fontsize=fontsize, fontweight="bold", color="white",
        zorder=12,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor=color, alpha=0.75,
            edgecolor="white", linewidth=0.6,
        ),
    )

def draw_gray_path(ax, gospel):
    sub = ch_df[ch_df["book"] == gospel].sort_values("chapter")
    coords = sub[["ux", "uy"]].values
    if len(coords) < 2:
        return
    for i in range(len(coords) - 1):
        ax.annotate("", xy=coords[i + 1], xytext=coords[i],
                    arrowprops=dict(
                        arrowstyle="->, head_width=0.15, head_length=0.15",
                        color="#444455", lw=1.0, alpha=0.6,
                    ), zorder=2)
    ax.scatter(coords[:, 0], coords[:, 1], s=14, color="#444455",
               alpha=0.7, linewidths=0, zorder=3)

def gospel_gradient(base_hex, n):
    """n colors from a pale tint → full saturated color."""
    rgb  = np.array(mcolors.to_rgb(base_hex))
    pale = rgb * 0.30 + np.array([1.0, 1.0, 1.0]) * 0.70
    return [tuple(pale + (rgb - pale) * t) for t in np.linspace(0, 1, n)]

def draw_main_path(ax, gospel, beats):
    sub    = ch_df[ch_df["book"] == gospel].sort_values("chapter")
    coords = sub[["ux", "uy"]].values
    chaps  = sub["chapter"].tolist()
    color  = GOSPEL_COLORS[gospel]
    n      = len(coords)
    if n < 2:
        return

    seg_colors = gospel_gradient(color, n)   # one color per point → n-1 segments

    # Arrows — gradient from pale → saturated
    for i in range(n - 1):
        ax.annotate("", xy=coords[i + 1], xytext=coords[i],
                    arrowprops=dict(
                        arrowstyle="->, head_width=0.28, head_length=0.28",
                        color=seg_colors[i], lw=2.2, alpha=0.90,
                    ), zorder=6)

    # Glow + solid dots — also gradient
    for i, (coord, c) in enumerate(zip(coords, seg_colors)):
        ax.scatter(*coord, s=120, color=c, alpha=0.18, linewidths=0, zorder=7)
        ax.scatter(*coord, s=30,  color=c, alpha=1.0,  linewidths=0, zorder=8)

    # Start / end markers
    ax.scatter(*coords[0],  marker="^", s=130, color=color,
               edgecolors="white", linewidths=1.0, zorder=9)
    ax.scatter(*coords[-1], marker="s", s=110, color=color,
               edgecolors="white", linewidths=1.0, zorder=9)

    # Beat labels — pill boxes
    for i, ch in enumerate(chaps):
        if ch in beats:
            labeled_box(ax, beats[ch], coords[i], color, fontsize=8)

def style_ax(ax, gospel):
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#222233")
    ax.set_title(gospel, fontsize=14, fontweight="bold",
                 color=GOSPEL_COLORS[gospel], pad=10)

# %% Build figure
fig = plt.figure(figsize=(18, 18))
fig.patch.set_facecolor(BG_COLOR)
gs = GridSpec(2, 2, figure=fig,
              hspace=0.10, wspace=0.06,
              left=0.03, right=0.97, top=0.91, bottom=0.04)

axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

for ax, gospel in zip(axes, GOSPELS):
    # Gray background paths (other three gospels)
    for other in GOSPELS:
        if other != gospel:
            draw_gray_path(ax, other)

    # Main coloured path
    draw_main_path(ax, gospel, STORY_BEATS[gospel])
    style_ax(ax, gospel)

    n_ch = len(ch_df[ch_df["book"] == gospel])
    ax.text(0.02, 0.02, f"{n_ch} chapters w/ red-letter speech",
            transform=ax.transAxes, fontsize=8,
            color="#555566", ha="left", va="bottom")

# Shared legend — start / end markers
start_p = ax.scatter([], [], marker="^", s=80, color="white", label="Chapter 1")
end_p   = ax.scatter([], [], marker="s", s=80, color="white", label="Final chapter")
gray_p  = mpatches.Patch(color="#444455", label="Other Gospels")
fig.legend(handles=[start_p, end_p, gray_p],
           fontsize=10, loc="lower center", ncol=3,
           facecolor="#1a1f2e", edgecolor="#333344",
           labelcolor="white", framealpha=0.9,
           bbox_to_anchor=(0.5, 0.01))

# Title block
fig.text(0.5, 0.975, "The Words of Jesus — Semantic Space",
         ha="center", va="top", fontsize=22, fontweight="bold", color="#ff4444")
fig.text(0.5, 0.953,
         "Chapter means of red-letter verses only  •  KJV  •  Shared axes — position is directly comparable",
         ha="center", va="top", fontsize=10, color="#8888aa")

plt.savefig("bonus_easter_2.png", dpi=180, bbox_inches="tight", facecolor=BG_COLOR)
plt.show()
print("Saved: bonus_easter_2.png")
