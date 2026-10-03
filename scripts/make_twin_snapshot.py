"""Write the CPython reference results the browser (Pyodide) must reproduce.

    python scripts/make_twin_snapshot.py

Run after any change to src/forecasting/twin.py. tests/test_twin.py fails if the committed snapshot is stale,
and web/src/twin/parity.test.ts fails if Pyodide disagrees with it.
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIXTURE = ROOT / "web" / "src" / "twin" / "fixtures"


def load_fixture():
    spec = importlib.util.spec_from_file_location("twin_fixture", FIXTURE / "fixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute() -> dict:
    from forecasting import twin

    fixture = load_fixture()
    return {name: twin.simulate_bundle(fixture.bundle_for(request), request) for name, request in fixture.REQUESTS.items()}


if __name__ == "__main__":
    (FIXTURE / "snapshot.json").write_text(json.dumps(compute(), indent=1, sort_keys=True) + "\n")
    print("wrote", FIXTURE / "snapshot.json")
