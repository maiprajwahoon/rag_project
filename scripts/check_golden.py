import json
from pathlib import Path

root = Path("data/raw/fastapi/docs/en/docs")
bad = 0
rows = 0
for line in Path("eval/golden_draft.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    rows += 1
    for page in row["pages"]:
        if not (root / page).exists():
            print("MISSING PAGE:", row["id"], page)
            bad += 1
print(rows, "rows,", bad, "missing pages")