import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvidenceRecord:
    run_id: str
    step: int
    action: str
    observation: str
    score: float


def append_evidence(record: EvidenceRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as file:
        json.dump(asdict(record), file)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def load_evidence(path: Path) -> list[EvidenceRecord]:
    records: list[EvidenceRecord] = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(EvidenceRecord(**json.loads(line)))

    return records
