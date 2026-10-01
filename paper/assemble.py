"""Insert the generated table blocks of tables/blocks.md at the {{tab:label}} markers of paper.md and write paper_full.md."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
blocks = {}
for block in (HERE / "tables" / "blocks.md").read_text(encoding="utf8").split("\n\n"):
    m = re.match(r"::table (tab:[a-z_]+) \|", block.strip())
    if m:
        blocks[m.group(1)] = block.strip()
text = (HERE / "paper.md").read_text(encoding="utf8")
missing = []
def repl(m):
    key = m.group(1)
    if key not in blocks:
        missing.append(key)
        return m.group(0)
    return blocks[key]
out = re.sub(r"\{\{(tab:[a-z_]+)\}\}", repl, text)
(HERE / "paper_full.md").write_text(out, encoding="utf8")
print("assembled paper_full.md; missing blocks:", missing)
