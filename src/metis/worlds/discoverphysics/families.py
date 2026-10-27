from dataclasses import dataclass
from enum import Enum


class FamilyStatus(Enum):
    ACTIVE = "ACTIVE"
    WATCH = "WATCH"


@dataclass(frozen=True)
class FamilyAudit:
    name: str
    status: FamilyStatus
    counts_as_evidence: bool


FAMILY_REGISTRY = {
    "gravity": FamilyStatus.ACTIVE,
    "yukawa": FamilyStatus.ACTIVE,
    "fractional": FamilyStatus.WATCH,
}


def audit_family(name: str) -> FamilyAudit:
    if not isinstance(name, str):
        raise TypeError(f"family name must be a string, got {type(name).__name__}")
    if name not in FAMILY_REGISTRY:
        raise ValueError(f"unknown family: {name}")

    status = FAMILY_REGISTRY[name]
    return FamilyAudit(
        name=name,
        status=status,
        counts_as_evidence=status is FamilyStatus.ACTIVE,
    )
