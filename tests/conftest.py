import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Ensure Tk/Tcl libraries are found in Windows virtual environment using POSIX slashes
if os.name == "nt":
    base_tcl = Path(sys.base_prefix) / "tcl"
    if (base_tcl / "tcl8.6").exists():
        os.environ["TCL_LIBRARY"] = (base_tcl / "tcl8.6").as_posix()
    if (base_tcl / "tk8.6").exists():
        os.environ["TK_LIBRARY"] = (base_tcl / "tk8.6").as_posix()


@pytest.fixture(scope="session")
def cascade():
    from ecg_edge.models import Cascade
    return Cascade()
