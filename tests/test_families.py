import pytest

from metis.worlds.discoverphysics.families import (
    FAMILY_REGISTRY,
    FamilyAudit,
    FamilyStatus,
    audit_family,
)


def test_registry_matches_s03_sheet():
    assert FAMILY_REGISTRY == {
        "gravity": FamilyStatus.ACTIVE,
        "yukawa": FamilyStatus.ACTIVE,
        "fractional": FamilyStatus.WATCH,
    }


@pytest.mark.parametrize(
    "name, status, counts",
    [
        ("gravity", FamilyStatus.ACTIVE, True),
        ("yukawa", FamilyStatus.ACTIVE, True),
        ("fractional", FamilyStatus.WATCH, False),
    ],
)
def test_audit_family(name, status, counts):
    audit = audit_family(name)
    assert audit == FamilyAudit(name=name, status=status, counts_as_evidence=counts)


def test_watch_family_never_counts_as_scientific_evidence():
    for name, status in FAMILY_REGISTRY.items():
        if status is FamilyStatus.WATCH:
            assert audit_family(name).counts_as_evidence is False


@pytest.mark.parametrize("name", ["oscillator", "", "Gravity"])
def test_unknown_family_is_rejected(name):
    with pytest.raises(ValueError):
        audit_family(name)


@pytest.mark.parametrize("name", [None, 1, b"gravity"])
def test_non_string_name_is_rejected(name):
    with pytest.raises(TypeError):
        audit_family(name)


def test_audit_is_frozen():
    audit = audit_family("gravity")
    with pytest.raises(AttributeError):
        audit.status = FamilyStatus.WATCH
