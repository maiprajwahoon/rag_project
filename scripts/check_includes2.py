import re
from pathlib import Path

repo = Path("data/raw/fastapi")
root = repo / "docs/en/docs"
print("docs_src folders found:", [str(p) for p in repo.rglob("docs_src") if p.is_dir()][:5])

pat = re.compile(r"\{\*\s*(\S+)")
total = found = 0
missing = []
for md in sorted(root.rglob("*.md")):
    for m in pat.finditer(md.read_text(encoding="utf-8")):
        total += 1
        rel = m.group(1).split("docs_src/", 1)[-1]
        if (repo / "docs_src" / rel).exists():
            found += 1
        elif len(missing) < 3:
            missing.append(m.group(1))
print("include lines:", total, "| resolved against repo/docs_src:", found)
print("sample unresolved:", missing)
