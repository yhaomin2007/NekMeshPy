"""Reassemble repository transport chunks into their original binary files."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
PART = re.compile(r"^(.*)\.part-\d\d$")

while True:
    groups = {}
    for path in ROOT.rglob("*.part-*"):
        match = PART.match(str(path))
        if match:
            groups.setdefault(Path(match.group(1)), []).append(path)
    if not groups:
        break
    # Rebuild the most deeply nested chunk names first.
    target = max(groups, key=lambda p: str(p).count(".part-"))
    parts = sorted(groups[target])
    with target.open("wb") as stream:
        for part in parts:
            stream.write(part.read_bytes())
    for part in parts:
        part.unlink()
    print(f"restored {target.relative_to(ROOT)}")
