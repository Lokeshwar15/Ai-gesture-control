import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.model_eval
def test_evaluate_model_script_runs_no_gate():
    script = ROOT / "ml" / "scripts" / "evaluate_model.py"
    r = subprocess.run(
        [sys.executable, str(script), "--no-gate"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if r.returncode != 0 and "tensorflow" in (r.stderr + r.stdout).lower():
        pytest.skip("tensorflow not installed")
    assert r.returncode == 0, r.stderr + r.stdout
