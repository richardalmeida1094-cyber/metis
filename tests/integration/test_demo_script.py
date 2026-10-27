"""The demo script: a stopped and resumed run ends with the same evidence as an uninterrupted one."""

import sys
from pathlib import Path

import pytest

from .conftest import RecordingFactory, SimulatedCrash

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import run_discoverphysics as demo  # noqa: E402

ARGS = ["--budget", "6", "--regimes", "0:gravity,3:yukawa"]


def _die(code):
    raise SimulatedCrash(f"exit {code}")


def test_stop_and_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    reference = demo.main(["--out", str(tmp_path / "ref"), *ARGS], factory=RecordingFactory())

    monkeypatch.setattr(demo.os, "_exit", _die)
    with pytest.raises(SimulatedCrash):
        demo.main(["--out", str(tmp_path / "run"), *ARGS, "--stop-after", "2"], factory=RecordingFactory())
    resumed = demo.main(["--out", str(tmp_path / "run"), *ARGS, "--resume"], factory=RecordingFactory())

    assert resumed == reference


def test_refuses_to_overwrite_an_existing_run(tmp_path):
    demo.main(["--out", str(tmp_path), *ARGS], factory=RecordingFactory())
    with pytest.raises(SystemExit, match="already holds a run"):
        demo.main(["--out", str(tmp_path), *ARGS], factory=RecordingFactory())
