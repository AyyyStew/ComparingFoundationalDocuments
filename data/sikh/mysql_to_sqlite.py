#!/usr/bin/env python3
"""Convert MySQL dump to SQLite database."""

import re
import sqlite3
import sys

INPUT = "data/sikh/gurbanidb_v2.mysql.sql"
OUTPUT = "data/sikh/gurbanidb.sqlite"

def convert_line(line):
    # Skip MySQL-specific directives
    if line.startswith("/*!") or line.startswith("LOCK TABLES") or line.startswith("UNLOCK TABLES"):
        return None
    if line.startswith("SET ") or line.startswith("-- MySQL") or line.startswith("-- Server"):
        return None

    # Remove backtick quoting
    line = line.replace("`", "")

    # Remove ENGINE=... and AUTO_INCREMENT=... at end of CREATE TABLE
    line = re.sub(r"\) ENGINE=\S+.*?;", ");", line)

    # Remove column-level AUTO_INCREMENT (keep as plain INTEGER)
    line = line.replace(" AUTO_INCREMENT", "")

    # Remove int(11) size specifiers
    line = re.sub(r"\bint\(\d+\)", "INTEGER", line)

    # Remove COMMENT '...'
    line = re.sub(r" COMMENT '[^']*'", "", line)

    # Remove KEY ... lines inside CREATE TABLE (SQLite doesn't support inline index defs)
    if re.match(r"^\s*(KEY|UNIQUE KEY|FULLTEXT KEY)\s", line):
        # Strip trailing comma from previous line handled below
        return ""

    return line


def process():
    with open(INPUT, "r", encoding="utf-8") as f:
        lines = f.readlines()

    converted = []
    for i, line in enumerate(lines):
        line = line.rstrip("\n")
        result = convert_line(line)
        if result is None:
            continue
        converted.append(result)

    # Re-join and fix trailing commas before closing paren (from removed KEY lines)
    sql = "\n".join(converted)
    # Remove trailing comma before closing paren of CREATE TABLE
    sql = re.sub(r",\s*\n\s*\)", "\n)", sql)
    # Convert MySQL escape sequences to SQLite (\'  -> '')
    sql = sql.replace("\\'", "''")

    print("Writing SQLite database...", file=sys.stderr)
    con = sqlite3.connect(OUTPUT)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=OFF")

    statements = sql.split(";\n")
    errors = 0
    for i, stmt in enumerate(statements):
        stmt = stmt.strip()
        if not stmt:
            continue
        try:
            con.executescript(stmt + ";")
        except sqlite3.Error as e:
            print(f"  Error on statement {i}: {e}", file=sys.stderr)
            print(f"  >> {stmt[:120]}", file=sys.stderr)
            errors += 1
            if errors > 10:
                print("Too many errors, aborting.", file=sys.stderr)
                break

    con.commit()
    con.close()
    print(f"Done. Errors: {errors}", file=sys.stderr)


if __name__ == "__main__":
    process()
