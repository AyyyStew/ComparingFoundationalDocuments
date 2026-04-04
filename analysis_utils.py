"""
analysis_utils.py
Shared constants and helpers for analysis scripts.
"""
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

# Archaic / KJV English words not in sklearn's stoplist
ARCHAIC_STOP_WORDS = {
    # Archaic pronouns and verb forms
    "shall", "thee", "thou", "thy", "thine", "ye", "hath", "doth",
    "unto", "hast", "canst", "wouldst", "shouldst", "art", "wilt",
    "saith", "sayeth",
    # Archaic connectives and prepositions
    "thereof", "therein", "thereto", "thereby", "therefor",
    "whereby", "whereof", "wherefore", "wherein",
    "hereof", "herein", "heretofore", "hereunto",
    "aforementioned", "aforesaid", "notwithstanding",
    # Exclamations
    "yea", "nay", "lo", "behold", "verily", "amen",
    # High-frequency generic verbs
    "said", "say", "says", "came", "come", "go", "went",
    "let", "make", "made",
    # High-frequency generic nouns / adjectives
    "like", "also", "one", "two", "three",
    "way", "day", "days", "years", "year",
    "new", "old", "thing", "things",
}

STOP_WORDS = list(ENGLISH_STOP_WORDS | ARCHAIC_STOP_WORDS)


def make_vectorizer(**kwargs) -> CountVectorizer:
    """Return a CountVectorizer with the shared stop word list.
    kwargs are passed through to CountVectorizer (e.g. min_df, ngram_range).
    """
    defaults = dict(min_df=2, ngram_range=(1, 2))
    defaults.update(kwargs)
    return CountVectorizer(stop_words=STOP_WORDS, **defaults)


# Corpus / tradition constants shared across scripts
SKIP_CORPORA = {
    "Bible — ACV (A Conservative Version)",
    "Bible — BBE (Bible in Basic English)",
    "Bible — YLT (Young's Literal Translation)",
}

SACRED_TRADITIONS = {"Abrahamic", "Dharmic", "Buddhist", "Taoist"}

TRADITION_GROUP = {
    "Abrahamic":  "Sacred Texts",
    "Dharmic":    "Sacred Texts",
    "Buddhist":   "Sacred Texts",
    "Taoist":     "Sacred Texts",
    "Literature": "Literature",
    "Historical": "Historical",
    "News":       "News",
}

GROUP_COLORS = {
    "Sacred Texts": "#e05c5c",
    "Literature":   "#4a90d9",
    "Historical":   "#9b59b6",
    "News":         "#aaaaaa",
}

NEWS_CATEGORY_COLORS = {
    "sports":   "#44bb99",
    "business": "#bbaa44",
}
