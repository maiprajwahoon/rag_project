import re
from pathlib import Path

root = Path("data/raw/fastapi/docs/en/docs")
pat = re.compile(r"\{\*\s*(\S+)(.*?)\*\}")
total = found = 0
shown = 0
for md in sorted(root.rglob("*.md")):
    for m in pat.finditer(md.read_text(encoding="utf-8")):
        total += 1
        target = (md.parent / m.group(1)).resolve()
        if target.exists():
            found += 1
        if shown < 3:
            print(md.relative_to(root), "->", m.group(0)[:120])
            shown += 1
print("include lines:", total, "| target file exists:", found)
