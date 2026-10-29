import pytest
from pathlib import Path
from llm_product_cards.data import load_jsonl, save_jsonl



def test_roundtrip_keeps_rows_and_cyrillic(tmp_path: Path):
    rows = [
        {"product_id": "P-1", "title": "Наушники"},
        {"product_id": "P-2", "reviews": ["отлично", "так себе"]},
    ]                                                        # 2 dicts, one with Russian text and a nested list
    path = tmp_path / "out.jsonl"
    save_jsonl(path, rows)
    assert load_jsonl(path) == rows
    assert "\\u04" not in path.read_text(encoding="utf-8")   # ensure_ascii=False worked


def test_save_creates_missing_directories(tmp_path: Path):
    path = tmp_path / "nested" / "deeper" / "out.jsonl"
    save_jsonl(path, [{"product_id": "P-1"}])
    assert path.exists()                                     # save one row, assert path.exists()


def test_invalid_line_reports_line_number(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"a": 1}\n{not json}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line: 2"):
        load_jsonl(path)