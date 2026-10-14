import json
from pathlib import Path
from typing import Any
from collections.abc import Iterator, Iterable


def _iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line_nmb, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} - line: {line_nmb} - invalid JSON: {e.msg}") from e
            yield row


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


def save_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
