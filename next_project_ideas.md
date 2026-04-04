# Next Project: Archetypal Characters in Sacred Texts

## Core Question

Are the *relationships* between characters in sacred texts geometrically similar across traditions — even when the characters themselves live in different regions of embedding space?

Not "do Teacher figures cluster together?" but "does the vector from Teacher → Student point in the same direction across traditions?"

## The Jungian Frame

Jung proposed that myths across cultures share archetypal roles. This project tests that idea empirically in embedding space.

**Four archetypal pairs to test first:**

| Archetype | Character A | Character B | Text |
|-----------|-------------|-------------|------|
| Teacher → Questioner | Krishna | Arjuna | Bhagavad Gita |
| Divine → Sufferer | God (Yahweh) | Job | Book of Job |
| Teacher → Disciples | Jesus | The Twelve | Gospels (KJV, red-letter) |
| Buddha → Seeker | The Buddha | Ananda / Sariputta | Pali Canon |

**The vector analogy test:**
```
Krishna_centroid - Arjuna_centroid  =  "Teacher→Questioner direction"
God_centroid     - Job_centroid     =  ?

cosine_similarity(Krishna-Arjuna, God-Job) — are these the same relationship?
```

If the displacement vectors align, that's a real finding: the *semantic geometry of the relationship* is preserved across traditions even when the content and characters are completely different.

The analogy framing (like word2vec's king - man + woman = queen) gives you a clean, explainable result for a non-technical audience.

## What's Already Feasible

### Bhagavad Gita — Krishna vs Arjuna
The Gita CSV (already in DB) likely has a speaker column — check before writing any code.
If not, the structure is well-known: Arjuna speaks in early chapters, Krishna dominates from Ch 2 onward. Could hardcode speaker ranges.

### Book of Job — Job vs God (Yahweh)
Speaker attribution is clean and well-documented:
- Job's speeches: Ch 3, 6–7, 9–10, 12–14, 16–17, 19, 21, 23–24, 26–31
- God's speeches: Ch 38–41
- Elihu, Eliphaz, Bildad, Zophar also separable (the "bad advisors" — their own archetype?)
Already in DuckDB. Could hardcode verse ranges.

### Jesus vs Disciples
Red-letter markup done (`*r` in kjv.json). Disciple speech is harder but Gospels have clear attribution in many verses ("Peter said...", "Thomas answered...").

## Data Gaps / New Ingestion Needed

| Text | Characters | Source | Notes |
|------|-----------|--------|-------|
| Mahabharata | Karna, Duryodhana, Vyasa | Gutenberg #15474 | Speaker tags inconsistent across translations — find a clean one |
| Ramayana | Rama, Sita, Ravana, Hanuman | Gutenberg #24869 | Valmiki translation has reasonable speaker markup |
| Pali Canon | Buddha, Ananda, Sariputta, Mara | Access to Insight / SuttaCentral | SuttaCentral has JSON with speaker tags — best source |

## The Shadow / Adversary Cluster

A particularly interesting sub-question: do Adversary figures (Satan, Mara, Ravana, Duryodhana) occupy a geometrically consistent position relative to their Hero?

```
Satan_centroid - Job_centroid
Mara_centroid  - Buddha_centroid
Ravana_centroid - Rama_centroid

→ Do these vectors align?
```

If yes: the Shadow is always "over there" relative to the Hero, regardless of tradition.

## DB Schema Changes for New Repo

Add a `speaker` column to the passage table from the start:

```sql
CREATE TABLE passage (
    id          INTEGER PRIMARY KEY,
    corpus_id   INTEGER REFERENCES corpus(id),
    book        TEXT,
    chapter     INTEGER,
    verse       INTEGER,
    unit_label  TEXT,
    speaker     TEXT,    -- NEW: "Krishna", "Arjuna", "Job", "God", NULL if narrator
    text        TEXT NOT NULL,
    metadata    JSON
)
```

## Model Thoughts

`all-mpnet-base-v2` is solid but worth trying:
- `intfloat/e5-large-v2` — better at semantic similarity, same size
- `text-embedding-3-large` (OpenAI API) — best geometry for analogy tasks but costs money per token

Run both on the Job/Gita data first and see if the analogy vectors are cleaner before committing to a full re-embed.

## Minimum Viable First Script

1. Extract Job's speech and God's speech from DB using hardcoded verse ranges
2. Extract Krishna's speech and Arjuna's speech from Gita (check CSV for speaker col first)
3. Compute character centroids
4. Compute displacement vectors: `God - Job`, `Krishna - Arjuna`
5. Measure cosine similarity between the two vectors
6. UMAP plot: show all four character clouds + draw the displacement vectors as arrows

That's a self-contained, publishable finding with just what's already in the DB.
