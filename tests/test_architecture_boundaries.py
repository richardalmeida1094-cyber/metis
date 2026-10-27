import ast
from pathlib import Path

import pytest


def simulator_import_lines(source, package):
    violations = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - node.level + 1]
                if node.module:
                    base += node.module.split(".")
                module = ".".join(base)
            else:
                module = node.module or ""
            modules = [module, *[f"{module}.{alias.name}" for alias in node.names]]
        if any(module == "metis.worlds" or module.startswith("metis.worlds.") for module in modules):
            violations.append(node.lineno)
    return violations


@pytest.mark.parametrize(
    "source",
    [
        "import metis.worlds",
        "import metis.worlds.family_a",
        "from metis.worlds import family_a",
        "from metis import worlds",
        "from . import worlds",
        "from .worlds import family_a",
    ],
)
def test_detector_rejects_alternate_import_forms(source):
    assert simulator_import_lines(source, ["metis"]) == [1]


@pytest.mark.parametrize(
    "source",
    [
        "from metis.experiment import Experiment",
        "from pathlib import Path",
    ],
)
def test_detector_allows_generic_imports(source):
    assert simulator_import_lines(source, ["metis"]) == []


def test_core_has_no_simulator_specific_imports():
    root = Path(__file__).resolve().parents[1] / "src" / "metis"
    violations = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if relative.parts[0] == "worlds":
            continue
        package = ["metis", *relative.parts[:-1]]
        for line in simulator_import_lines(path.read_text(encoding="utf-8"), package):
            violations.append(f"{relative}:{line}")
    assert not violations, "Simulator-specific imports: " + ", ".join(violations)
