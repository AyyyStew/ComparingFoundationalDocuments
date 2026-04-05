"""
Download all Kojiki pages from sacred-texts.com into this directory.
Run once — skips pages already downloaded.
"""
import random
import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup

BASE    = "https://sacred-texts.com/shi/kj/"
OUT_DIR = Path(__file__).parent
UA      = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"}
DELAY_MIN = 2.0
DELAY_MAX = 10.0

session = requests.Session()
session.headers.update(UA)

# Fetch index to discover linked pages
idx = session.get(BASE + "index.htm", timeout=15)
idx.raise_for_status()
(OUT_DIR / "index.htm").write_text(idx.text, encoding="utf-8")
print("Saved index.htm")
time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

pages = set()
soup = BeautifulSoup(idx.text, "html.parser")
for a in soup.find_all("a", href=True):
    href = a["href"]
    if href.startswith("kj") and href.endswith(".htm"):
        pages.add(href)

# Also probe sequentially to catch Vol III pages not in index
for n in range(0, 150):
    pages.add(f"kj{n:03d}.htm")

pages = sorted(pages)
print(f"{len(pages)} candidate pages")

consecutive_404s = 0
for fname in pages:
    out = OUT_DIR / fname
    if out.exists():
        print(f"  skip {fname} (exists)")
        consecutive_404s = 0
        continue
    try:
        r = session.get(BASE + fname, timeout=15)
        if r.status_code == 404:
            print(f"  404  {fname}")
            consecutive_404s += 1
            if consecutive_404s >= 5:
                print("5 consecutive 404s — stopping probe")
                break
            continue
        r.raise_for_status()
        out.write_text(r.text, encoding="utf-8")
        print(f"  ok   {fname} ({len(r.text):,} chars)")
        consecutive_404s = 0
    except Exception as e:
        print(f"  ERR  {fname}: {e}")

    delay = random.uniform(DELAY_MIN, DELAY_MAX)
    time.sleep(delay)

print("Done.")
