"""Full truth table of the HMI decision logic (Policy B)."""
import itertools

import pytest

from ecg_edge.decision import DASH, decide
from ecg_edge.models import ModelOutput


def mo(label, c=0.9): return ModelOutput(label, c, {label: c})


EXPECTED_TABLE = {
    ("SHOCKABLE", "NSR"): ("CONFLICT", f"CONFLICT {DASH} NOT CLASSIFIED"),
    ("SHOCKABLE", "TACHY"): ("CLASSIFIED", "VF/VT SHOCKABLE"),
    ("SHOCKABLE", "BRADY_ASY"): ("CONFLICT", f"CONFLICT {DASH} NOT CLASSIFIED"),
    ("SHOCKABLE", "VENTRICULAR"): ("CLASSIFIED", "VF/VT SHOCKABLE"),
    ("NON_SHOCKABLE", "NSR"): ("CLASSIFIED", "NORMAL SINUS RHYTHM"),
    ("NON_SHOCKABLE", "TACHY"): ("CLASSIFIED", "TACHYCARDIA"),
    ("NON_SHOCKABLE", "BRADY_ASY"): ("CLASSIFIED", f"OTHER {DASH} NOT CLASSIFIED"),
    ("NON_SHOCKABLE", "VENTRICULAR"): ("CLASSIFIED", "VENTRICULAR (NON-SHOCKABLE)"),
}


@pytest.mark.parametrize("shock,rhythm", list(EXPECTED_TABLE))
def test_exhaustive_eight_rows(shock, rhythm):
    expected_status, expected_headline = EXPECTED_TABLE[(shock, rhythm)]
    d = decide(mo(shock), mo(rhythm))
    assert d.status == expected_status
    assert d.headline == expected_headline


def test_exhaustive_combinations_count():
    assert len(EXPECTED_TABLE) == 8
    combinations = list(itertools.product(["NON_SHOCKABLE", "SHOCKABLE"],
                                          ["NSR", "TACHY", "BRADY_ASY", "VENTRICULAR"]))
    assert len(combinations) == 8
    for shock, rhythm in combinations:
        assert (shock, rhythm) in EXPECTED_TABLE


@pytest.mark.parametrize("shock,rhythm", [("SHOCKABLE", "NSR"), ("SHOCKABLE", "BRADY_ASY")])
def test_conflict_notes_and_voice(shock, rhythm):
    d = decide(mo(shock), mo(rhythm))
    assert d.status == "CONFLICT"
    assert d.severity == "WARNING"
    assert "Model disagreement" in d.speech
    assert len(d.notes) == 1
    assert shock in d.notes[0] and rhythm in d.notes[0]


def test_ventricular_non_shockable_properties():
    d = decide(mo("NON_SHOCKABLE"), mo("VENTRICULAR"))
    assert d.status == "CLASSIFIED"
    assert d.headline == "VENTRICULAR (NON-SHOCKABLE)"
    assert d.severity == "WARNING"
    assert "non shockable" in d.speech.lower()
