import json

import pytest

from metis.evidence import EvidenceRecord, append_evidence, load_evidence


def test_append_preserves_records_order_and_content(tmp_path):
    path = tmp_path / "nested" / "evidence.jsonl"
    first = EvidenceRecord("run", 1, "left", "one", 1.5)
    second = EvidenceRecord("run", 2, "right", "two", 2.5)
    append_evidence(first, path)
    first_content = path.read_bytes()
    append_evidence(second, path)
    assert path.read_bytes().startswith(first_content)
    assert load_evidence(path) == [first, second]
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {
        "run_id": "run",
        "step": 1,
        "action": "left",
        "observation": "one",
        "score": 1.5,
    }


def test_malformed_evidence_fails_loudly(tmp_path):
    path = tmp_path / "evidence.jsonl"
    append_evidence(EvidenceRecord("run", 1, "left", "one", 1.5), path)
    with path.open("a", encoding="utf-8") as file:
        file.write('{"run_id": "truncated"\n')
    with pytest.raises(json.JSONDecodeError):
        load_evidence(path)
