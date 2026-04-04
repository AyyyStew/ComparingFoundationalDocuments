# %% [markdown]
# # Post-process Interactive HTML Files
#
# Plotly embeds fixed pixel widths in generated HTML. This script injects a
# small CSS + JS snippet into each HTML file that makes plots responsive.
# Idempotent — skips files already processed.

import glob
import os

INJECT = """
<style>
  body, .plotly-graph-div { width: 100% !important; max-width: 100% !important; }
</style>
<script>
  window.addEventListener('load', function() {
    var divs = document.querySelectorAll('.plotly-graph-div');
    divs.forEach(function(div) {
      Plotly.relayout(div, { width: div.parentElement.clientWidth, autosize: true });
    });
    window.addEventListener('resize', function() {
      divs.forEach(function(div) {
        Plotly.relayout(div, { width: div.parentElement.clientWidth });
      });
    });
  });
</script>
"""

MARKER = "<!-- postprocessed -->"

HTML_GLOB = "*.html"

files = sorted(glob.glob(HTML_GLOB))
print(f"Found {len(files)} HTML files.")

for path in files:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Strip any previous injection before re-processing
    if MARKER in content:
        start = content.index(MARKER)
        end   = content.index("</script>", start) + len("</script>")
        content = content[:start] + content[end:]

    if "<body" not in content:
        print(f"  SKIP (no <body>): {os.path.basename(path)}")
        continue

    # Inject right after the opening <body> tag
    insert_at = content.index("<body") + content[content.index("<body"):].index(">") + 1
    new_content = content[:insert_at] + "\n" + MARKER + INJECT + content[insert_at:]

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"  OK: {os.path.basename(path)}")

print("\nDone.")
