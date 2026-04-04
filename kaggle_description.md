## About this Dataset

166,000+ passages from 37 texts — sacred scriptures, philosophy, literature, historical documents, and news — encoded with `all-mpnet-base-v2` (768-dimensional sentence embeddings) and stored as Parquet.

The dataset was built to explore a simple question: **are world religions saying the same thing?** Do the Bhagavad Gita and the Bible occupy the same corner of semantic space, or are they genuinely distinct? What about the Dao De Jing vs the Upanishads? And how far does sacred language sit from Aristotle, Darwin, or the Communist Manifesto?

### What's included

**Four Parquet files:**

- `corpus_tradition.parquet` — 11 tradition labels (Abrahamic, Dharmic, Buddhist, Taoist, Philosophy, Literature, Historical, Scientific, Norse, Confucian, News)
- `corpus.parquet` — 37 corpora with metadata (type, language, era)
- `passage.parquet` — every passage with its book/section label and text
- `embedding.parquet` — one 768-float vector per passage, model `all-mpnet-base-v2`

**Traditions and texts:**

| Tradition | Texts |
|-----------|-------|
| Abrahamic | Bible (KJV, ACV, YLT, BBE), Quran |
| Dharmic | Bhagavad Gita, Yoga Sutras, Upanishads, Bhagavata Purana |
| Buddhist | Dhammapada, Diamond Sutra, Jataka Tales |
| Taoist | Dao De Jing |
| Philosophy | Plato, Aristotle, Descartes, Kant, Nietzsche, Freud, Jung |
| Scientific | Darwin, Newton, Varieties of Religious Experience |
| Literature | Frankenstein, Pride and Prejudice, Don Quixote, Romeo and Juliet, Siddhartha, Crime and Punishment, Beowulf |
| Historical | US Constitution, Federalist Papers, Magna Carta, Hammurabi, Communist Manifesto, Luther's 95 Theses |
| Norse | Poetic Edda |
| Confucian | Analects of Confucius |
| News | Sports and business news articles (secular baseline) |

### Key findings from the analysis

- Within-tradition cosine similarity (~0.31) is meaningfully higher than cross-tradition (~0.22), confirming traditions are semantically distinct — but not completely separate
- Dharmic texts (Gita, Yoga Sutras) and Buddhist texts cluster closer to each other than either does to Abrahamic texts
- Philosophy clusters near sacred texts in embedding space; hard science (Darwin, Newton) forms its own isolated region
- The Dao De Jing, with only 81 short chapters, is the most internally coherent tradition — its passages are the least like any other tradition's centroid

### How to use

```python
import pandas as pd
import numpy as np

passages = pd.read_parquet("passage.parquet")
embeddings = pd.read_parquet("embedding.parquet")
embeddings["vector"] = embeddings["vector"].apply(np.array)

df = passages.merge(embeddings, left_on="id", right_on="passage_id")
```

Alternatively, use the companion GitHub repository to load everything into DuckDB and reproduce the full analysis: [github link]

### Notes

- Passage granularity varies by text type (Bible = verse, novels = 4-sentence chunk, Dao De Jing = chapter). Cosine similarity scores are not perfectly length-normalized — see the repo README for details.
- Three non-KJV Bible translations (ACV, YLT, BBE) are included for translation comparison but are excluded from cross-tradition analyses.
