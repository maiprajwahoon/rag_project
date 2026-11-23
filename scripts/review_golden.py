import json
import re
from pathlib import Path

root = Path("data/raw/fastapi/docs/en/docs")
out: list[str] = []
for line in Path("eval/golden_draft.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    out.append(f"[{row['id']}] {row['question']}")
    if not row["answerable"]:
        out.append("   (should NOT be answerable from the docs)")
    for page in row["pages"]:
        text = (root / page).read_text(encoding="utf-8")
        heads = re.findall(r"^#{1,3} (.+)$", text, flags=re.MULTILINE)
        heads = [re.sub(r"\s*\{ #.*\}$", "", h) for h in heads][:10]
        out.append(f"   page: {page}")
        out.append("   headings: " + " | ".join(heads))
    out.append("")
Path("eval/review.txt").write_text("\n".join(out), encoding="utf-8")
print("wrote eval/review.txt")