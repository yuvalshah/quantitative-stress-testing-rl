"""Architecture tests: module layout, layered dependencies and lazy heavy imports.

These tests parse source files with ``ast`` so they run without any third-party
dependency installed. Import smoke tests skip when a dependency is missing.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / "rlstress"

# A module may import only from modules in strictly lower layers.
# Subpackages (e.g. ``data/``) count as one module; imports inside a subpackage are allowed.
LAYERS: dict[str, int] = {
    "config": 0,
    "exceptions": 0,
    "features": 1,
    "portfolio": 1,
    "data": 2,
    "risk": 2,
    "shocks": 3,
    "stress": 3,
    "env": 4,
    "risk_report": 4,
    "scenarios": 4,
    "shock_report": 5,
    "rl": 5,
    "search": 5,
    "mitigation": 5,
    "recommendation": 6,
    "scenario_library": 6,
    "dashboard": 7,
    "scenario_recommender": 7,
    "final_evaluation": 8,
    "unseen_evaluation": 9,
}

# Heavy or network-bound libraries must be imported inside functions, except where listed.
LAZY_ONLY = {
    "yfinance": set(),
    "stable_baselines3": {"rl"},
    "torch": {"rl"},
    "streamlit": set(),
    "sklearn": set(),
    "plotly": set(),
    "matplotlib": set(),
    "gymnasium": {"env"},
}

# Third-party packages each module's dependencies (by layer) are expected to need.
MODULE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "config": (),
    "exceptions": (),
    "stress": ("numpy", "pandas"),
    "shocks": ("numpy", "pandas", "scipy"),
    "shock_report": ("numpy", "pandas", "gymnasium", "scipy", "requests"),
    "risk_report": ("numpy", "pandas", "requests"),
    "data": ("numpy", "pandas", "requests"),
    "features": ("numpy", "pandas"),
    "portfolio": ("numpy", "pandas"),
    "risk": ("numpy",),
    "env": ("numpy", "pandas", "gymnasium", "scipy"),
    "scenarios": ("numpy", "pandas", "scipy"),
    "rl": ("numpy", "pandas", "gymnasium", "scipy", "requests", "stable_baselines3", "torch"),
    "search": ("numpy", "pandas", "gymnasium", "scipy"),
    "mitigation": ("numpy", "pandas", "scipy"),
    "recommendation": ("numpy", "pandas", "scipy"),
    "scenario_library": ("numpy", "pandas", "scipy"),
    "scenario_recommender": ("numpy", "pandas", "gymnasium", "scipy"),
    "final_evaluation": ("numpy", "scipy"),
    "unseen_evaluation": ("numpy", "pandas", "scipy"),
    "dashboard": (),
}


def _module_files() -> dict[str, Path]:
    """Top-level modules and subpackages, keyed by layer name."""
    files = {p.stem: p for p in PACKAGE_DIR.glob("*.py") if p.stem != "__init__"}
    files.update({p.name: p for p in PACKAGE_DIR.iterdir() if (p / "__init__.py").exists()})
    return files


def _source_files(module: str) -> list[Path]:
    """All source files of a module or subpackage."""
    path = _module_files()[module]
    return sorted(path.rglob("*.py")) if path.is_dir() else [path]


def _dotted_names(module: str) -> list[str]:
    """Importable dotted names of every file in a module or subpackage."""
    names = []
    for f in _source_files(module):
        rel = f.relative_to(PACKAGE_DIR).with_suffix("")
        parts = [p for p in rel.parts if p != "__init__"]
        names.append(".".join(["rlstress", *parts]))
    return names


def _internal_imports(tree: ast.Module) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if parts[0] == "rlstress" and len(parts) > 1:
                found.add(parts[1])
            elif parts[0] == "rlstress":
                found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "rlstress" and len(parts) > 1:
                    found.add(parts[1])
    return found


def _top_level_third_party(tree: ast.Module) -> set[str]:
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


def test_every_module_has_a_layer() -> None:
    assert set(_module_files()) == set(LAYERS)


@pytest.mark.parametrize("module", sorted(LAYERS))
def test_module_has_docstring(module: str) -> None:
    for f in _source_files(module):
        tree = ast.parse(f.read_text())
        assert ast.get_docstring(tree), f"{f} is missing a module docstring"


@pytest.mark.parametrize("module", sorted(LAYERS))
def test_imports_respect_layers(module: str) -> None:
    for f in _source_files(module):
        for dep in _internal_imports(ast.parse(f.read_text())):
            if dep == module:
                continue  # import within the same subpackage
            assert dep in LAYERS, f"{f.name} imports unknown module rlstress.{dep}"
            assert LAYERS[dep] < LAYERS[module], (
                f"{module} (layer {LAYERS[module]}) must not import "
                f"{dep} (layer {LAYERS[dep]})"
            )


@pytest.mark.parametrize("module", sorted(LAYERS))
def test_heavy_dependencies_are_lazy(module: str) -> None:
    for f in _source_files(module):
        top_level = _top_level_third_party(ast.parse(f.read_text()))
        for lib, allowed in LAZY_ONLY.items():
            if lib in top_level:
                assert module in allowed, f"{f.name} imports {lib} at top level"


@pytest.mark.parametrize("module", sorted(LAYERS))
def test_module_imports(module: str) -> None:
    for dep in MODULE_DEPENDENCIES[module]:
        pytest.importorskip(dep)
    for name in _dotted_names(module):
        importlib.import_module(name)
