from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


def write_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


def write_jsonl(path, rows: Iterable[Dict], append: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if bool(append) else "w"
    with open(path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ordered_fieldnames(rows: Iterable[Dict], preferred: Optional[Sequence[str]] = None) -> List[str]:
    rows = list(rows)
    preferred = list(preferred or [])
    keys = {key for row in rows for key in row}
    return [key for key in preferred if key in keys] + sorted(key for key in keys if key not in preferred)


def write_csv(path, rows: Iterable[Dict], fieldnames: Optional[Sequence[str]] = None) -> None:
    rows = list(rows)
    if not rows:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or ordered_fieldnames(rows))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
