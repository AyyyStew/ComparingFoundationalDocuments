# Are World Religions Saying the Same Thing?

An NLP exploration of sacred texts, literature, philosophy, and historical documents using sentence embeddings. Every passage from 37 corpora is embedded with `all-mpnet-base-v2` and stored in DuckDB, then analyzed for cross-tradition similarity, clustering, narrative flow, and outliers.

The full write-up lives in `project_thoughts.qmd`.

---

## Dataset

Passages and embeddings are hosted on Kaggle:
**[Sacred Texts Corpus — Embeddings Across Traditions](https://www.kaggle.com/datasets/williamsteward/sacred-texts-corpus-embeddings-across-traditions)**

To use it locally:

```bash
# Download parquet files from Kaggle into data/parquet/, then:
python import_parquet.py
```

To regenerate the export:

```bash
python export_parquet.py
```

---

## Scripts

Scripts are numbered in the order they were written. Ingest scripts (`*_ingest_*.py`) load source texts into the DuckDB database and compute embeddings. Analysis scripts produce figures and reports.

### Ingest

| Script                               | What it ingests                                                                                                    |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| `01_compute_bible_embeddings.py`     | KJV Bible (verse-level)                                                                                            |
| `06_ingest_bhagavad_gita.py`         | Bhagavad Gita                                                                                                      |
| `09_ingest_dhammapada.py`            | Dhammapada (Müller translation)                                                                                    |
| `10_ingest_dao_de_jing.py`           | Dao De Jing (Linnell translation)                                                                                  |
| `11_ingest_yoga_sutras.py`           | Yoga Sutras of Patanjali (Johnston translation)                                                                    |
| `12_ingest_literature.py`            | Frankenstein, Pride and Prejudice, Don Quixote, Romeo and Juliet                                                   |
| `13_ingest_historical.py`            | US Constitution, Federalist Papers, Magna Carta, Hammurabi, Communist Manifesto, Luther's 95 Theses                |
| `15_ingest_news.py`                  | News articles (sports + business) as a secular baseline                                                            |
| `29_ingest_sacred.py`                | Quran, Upanishads, Bhagavata Purana, Diamond Sutra, Jataka Tales, Varieties of Religious Experience                |
| `30_ingest_norse_confucian.py`       | Poetic Edda, Analects of Confucius                                                                                 |
| `31_ingest_philosophy.py`            | Plato's Republic, Aristotle's Ethics, Descartes, Kant, Nietzsche (Zarathustra + Beyond Good and Evil), Freud, Jung |
| `32_ingest_scientific_literature.py` | Darwin's Origin of Species, Newton's Opticks, Siddhartha, Crime and Punishment, Beowulf                            |

### Data Cleaning

| Script                   | What it does                                                                                    |
| ------------------------ | ----------------------------------------------------------------------------------------------- |
| `filter_gutenberg.py`    | Removes Project Gutenberg boilerplate (license headers, footers) that slipped through ingestion |
| `clean_diamond_sutra.py` | Removes translator name fragments and footnote leakage from the Diamond Sutra                   |

### Analysis

| Script                                         | Description                                                                                                     |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `02_analysis_kjv.py`                           | KJV UMAP: book-level embeddings with HDBSCAN clustering, verse-level UMAP, OT vs NT density                     |
| `03_analysis_kjv_pca.py`                       | PCA on KJV book embeddings — what are the main axes of semantic variation in the Bible?                         |
| `04_analysis_translation.py`                   | Compares KJV, ACV, YLT, BBE translations at the verse level                                                     |
| `05_analysis_chapter_aggregation.py`           | Compares 6 methods for aggregating verse embeddings to chapter level                                            |
| `07_analysis_gita_vs_bible.py`                 | First cross-tradition comparison: Gita vs Bible similarity distributions and nearest neighbours                 |
| `08_analysis_gita_vs_wisdom_epistles.py`       | Gita vs Wisdom books (Job, Psalms, Proverbs, Ecclesiastes) and New Testament Epistles                           |
| `14_analysis_tradition_landscape.py`           | UMAP of all traditions (pre-news): corpus-level and book-level views                                            |
| `16_analysis_tradition_landscape_with_news.py` | Same UMAP with news articles added as a secular baseline                                                        |
| `17a–17d_analysis_bertopic_*.py`               | BERTopic topic modelling: full corpus, sacred only, sacred vs historical, sacred vs sports news                 |
| `18_analysis_cross_tradition_nn.py`            | Cross-tradition nearest neighbours across all groups; BERTopic on "bridge" passages                             |
| `19_analysis_sacred_vs_literature.py`          | Per-pair heatmaps: each sacred corpus vs each literature corpus                                                 |
| `20_analysis_sacred_vs_sacred.py`              | Book-level heatmaps for all sacred corpus pairs                                                                 |
| `21a–21b_analysis_gmm_*.py`                    | GMM soft clustering on full corpus: BIC/AIC sweep to choose K, then K=20 topic analysis                         |
| `22a–22b_analysis_gmm_sacred_*.py`             | Same GMM pipeline restricted to sacred traditions only                                                          |
| `23_analysis_sacred_cross_tradition_nn.py`     | Top-k nearest-neighbour similarity heatmap for the 5 core sacred texts                                          |
| `23b_analysis_sacred_random_similarity.py`     | Random-pair baseline heatmap (same scale as 23, for direct comparison)                                          |
| `23c_analysis_sacred_similarity_lift.py`       | Lift analysis: how much does top-k exceed random? Difference and ratio heatmaps                                 |
| `24_analysis_concept_network.py`               | HDBSCAN concept clusters → co-occurrence network for sacred texts                                               |
| `25_analysis_concept_network_balanced.py`      | Rebalanced concept network (caps Bible passages to reduce dominance)                                            |
| `26_analysis_graph_theory.py`                  | Graph-theoretic analysis of the balanced concept network (centrality, bridges)                                  |
| `27_analysis_narrative_flow.py`                | Does each text travel through semantic space book-by-book, or circle the same ground?                           |
| `28_analysis_outliers.py`                      | Outlier detection: passages least like their own tradition, and passages drawn toward others                    |
| `33_analysis_full_corpus_umap.py`              | UMAP of the full 37-corpus dataset with all traditions                                                          |
| `34_analysis_universal_distinct_passages.py`   | Finds the most cross-tradition universal passages and the most tradition-distinct ones                          |
| `36_analysis_concept_network_full.py`          | Concept co-occurrence network extended to the full corpus                                                       |
| `final_analysis.py`                            | Omnibus script: top-k heatmap, lift, similarity distributions, narrative flow, and outliers for the full corpus |

### Utilities

| Script                | Description                                                                                   |
| --------------------- | --------------------------------------------------------------------------------------------- |
| `inspect_corpus.py`   | CLI tool to preview any corpus in the DB (`python inspect_corpus.py "Corpus Name" --limit 5`) |
| `analysis_utils.py`   | Shared constants (corpora to skip, colour maps)                                               |
| `postprocess_html.py` | Injects responsive CSS/JS into all Plotly HTML outputs                                        |
| `export_parquet.py`   | Exports DuckDB tables to Parquet for Kaggle upload                                            |
| `import_parquet.py`   | Rebuilds DuckDB from downloaded Parquet files                                                 |

---

## Passage Granularity

Different source types use different atomic units for embedding, chosen to keep
semantic density roughly comparable to a scripture verse (~20–80 words):

| Type                            | Unit              | Notes                                              |
| ------------------------------- | ----------------- | -------------------------------------------------- |
| Scripture (Bible, Gita, Sutras) | Verse / sutra     | Native unit of the text                            |
| Dhammapada                      | Verse             | Native numbered verses                             |
| Dao De Jing                     | Chapter           | Each chapter is already short (~60–120 words)      |
| Novels                          | 4-sentence chunk  | Split within chapters; stage directions stripped   |
| Shakespeare                     | Scene             | Full scene dialogue, stage directions stripped     |
| Legal (Hammurabi, Magna Carta)  | Numbered clause   | Each law/clause is one passage                     |
| Political (Constitution)        | Article + Section | Preamble + each Article/Section is one passage     |
| Theses (Luther)                 | Numbered thesis   | Each thesis is one passage                         |
| Essay (Manifesto)               | Paragraph         | Each paragraph is one passage; short lines skipped |
| Long essays (Federalist Papers) | 4-sentence chunk  | Same density target as novels                      |

Cross-tradition similarity scores are **not** perfectly length-normalized. A novel chunk and a Bible verse may differ in density. This is a known limitation and should be considered when interpreting cosine similarity results.

---

## Sources

Bible verses from [scrollmapper/bible_databases](https://github.com/scrollmapper/bible_databases)

kjv.json from https://github.com/jburson/bible-data/blob/main/data/kjv/kjv.json
for red letter annotations

Bhagavad Gita from https://www.kaggle.com/datasets/madhurpant/bhagavad-gita-verses-dataset

Bhagavata Purana from https://www.kaggle.com/datasets/madhurpant/srimad-bhagawatam-bhagavata-purana-dataset

News articles from https://www.kaggle.com/datasets/asad1m9a9h6mood/news-articles

Quran from https://www.kaggle.com/datasets/imrankhan197/the-quran-dataset

From Project Gutenberg:

- Dhammapada: https://www.gutenberg.org/ebooks/2017
- Dao De Jing: https://www.gutenberg.org/ebooks/49965
- Yoga Sutras: https://www.gutenberg.org/ebooks/2526
- Don Quixote: https://www.gutenberg.org/ebooks/996
- Frankenstein: https://www.gutenberg.org/ebooks/84
- Pride and Prejudice: https://www.gutenberg.org/ebooks/1342
- Romeo and Juliet: https://www.gutenberg.org/ebooks/1513
- US Constitution: https://www.gutenberg.org/ebooks/5
- Federalist Papers: https://www.gutenberg.org/ebooks/1404
- Magna Carta: https://www.gutenberg.org/ebooks/10000
- Hammurabi: https://www.gutenberg.org/ebooks/17150
- Communist Manifesto: https://www.gutenberg.org/ebooks/61
- Civilization and its Discontents: https://www.gutenberg.org/ebooks/78221
- Psychology of the Unconscious: https://www.gutenberg.org/ebooks/65903
- Critique of Pure Reason: https://www.gutenberg.org/ebooks/4280
- Discourse on Method: https://www.gutenberg.org/ebooks/59
- Analects of Confucius: https://www.gutenberg.org/ebooks/3330
- Thus Spake Zarathustra: https://www.gutenberg.org/ebooks/1998
- Beyond Good and Evil: https://www.gutenberg.org/ebooks/4363
- The Republic: https://www.gutenberg.org/ebooks/1497
- Ethics of Aristotle: https://www.gutenberg.org/ebooks/8438
- Sacred Books of the East: https://www.gutenberg.org/ebooks/12894
- Poetic Edda: https://www.gutenberg.org/ebooks/73533
- Upanishads: https://www.gutenberg.org/ebooks/3283
- Opticks: https://www.gutenberg.org/ebooks/33504
- Varieties of Religious Experience: https://www.gutenberg.org/ebooks/621
- On the Origin of Species: https://www.gutenberg.org/ebooks/1228
- Siddhartha: https://www.gutenberg.org/ebooks/2500
- Buddhist Birth Stories (Jataka Tales): https://www.gutenberg.org/ebooks/51880
- Diamond Sutra: https://www.gutenberg.org/ebooks/64623
- Crime and Punishment: https://www.gutenberg.org/ebooks/2554
- Luther's 95 Theses: https://www.gutenberg.org/ebooks/274




Sources added after the projects thoughts write up:
- Sri Guru Granth Sahib: https://sikher.com/ | https://sourceforge.net/projects/sikher/
- avesta yasna : https://www.avesta.org/yasna/index.html
- avesta vendidad : https://www.avesta.org/vendidad/index.html
- Chuang Tzu: https://www.gutenberg.org/ebooks/59709
- Kojiki: https://sacred-texts.com/shi/kj/index.htm