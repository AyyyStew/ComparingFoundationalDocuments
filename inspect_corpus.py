#!/usr/bin/env python
"""
inspect_corpus.py — browse what's in corpus.duckdb

Usage:
    python inspect_corpus.py                        # list all corpora
    python inspect_corpus.py "Dao De Jing (Linnell)"           # show passages
    python inspect_corpus.py "Dao De Jing (Linnell)" --limit 5 # first 5
    python inspect_corpus.py "Dao De Jing (Linnell)" --chapter 3
"""

import argparse
from db.schema import get_conn

parser = argparse.ArgumentParser()
parser.add_argument("corpus",   nargs="?", help="Corpus name (partial match ok)")
parser.add_argument("--limit",  type=int, default=20)
parser.add_argument("--chapter", type=str, help="Filter by section/chapter")
parser.add_argument("--full",   action="store_true", help="Print full text (no truncation)")
args = parser.parse_args()

conn = get_conn()

# No corpus given — list everything
if not args.corpus:
    rows = conn.execute("""
        SELECT c.name, t.name AS tradition, c.type, COUNT(p.id) AS passages,
               SUM(CASE WHEN e.passage_id IS NOT NULL THEN 1 ELSE 0 END) AS embedded
        FROM corpus c
        JOIN corpus_tradition t ON t.id = c.tradition_id
        LEFT JOIN passage p ON p.corpus_id = c.id
        LEFT JOIN embedding e ON e.passage_id = p.id
        GROUP BY c.name, t.name, c.type
        ORDER BY t.name, c.name
    """).fetchall()
    print(f"\n{'Corpus':<45} {'Tradition':<15} {'Type':<12} {'Passages':>9} {'Embedded':>9}")
    print("-" * 95)
    for name, trad, typ, passages, embedded in rows:
        print(f"{name:<45} {trad:<15} {typ:<12} {passages:>9,} {embedded:>9,}")
    conn.close()
    exit()

# Find corpus by partial name match
matches = conn.execute(
    "SELECT id, name FROM corpus WHERE name ILIKE ?", [f"%{args.corpus}%"]
).fetchall()

if not matches:
    print(f"No corpus matching '{args.corpus}'")
    conn.close()
    exit()

if len(matches) > 1:
    print("Multiple matches:")
    for cid, cname in matches:
        print(f"  [{cid}] {cname}")
    conn.close()
    exit()

corpus_id, corpus_name = matches[0]
print(f"\nCorpus: {corpus_name}  (id={corpus_id})")

# Build query
where = "WHERE p.corpus_id = ?"
params = [corpus_id]
if args.chapter:
    where += " AND p.section = ?"
    params.append(args.chapter)

rows = conn.execute(
    f"""
    SELECT p.book, p.section, p.unit_number, p.unit_label, p.text
    FROM passage p
    {where}
    ORDER BY p.id
    LIMIT {args.limit}
    """,
    params,
).fetchall()

total = conn.execute(
    f"SELECT COUNT(*) FROM passage p {where}", params
).fetchone()[0]

print(f"Showing {len(rows)} of {total:,} passages\n")
print("-" * 80)
for book, section, unit_num, label, text in rows:
    ref = f"[{book} | sec={section} unit={unit_num} label={label}]"
    body = text if args.full else (text[:200] + "..." if len(text) > 200 else text)
    print(f"{ref}\n{body}\n")

conn.close()
