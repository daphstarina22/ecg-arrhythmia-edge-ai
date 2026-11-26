"""Full truth table of the HMI decision logic."""
import itertools

import pytest

from ecg_edge.decision import decide
from ecg_edge.models import ModelOutput


def mo(label, c=0.9): return ModelOutput(label, c, {label: c})


EXPECTED = {
    ("NSR", "NON_SHOCKABLE", None): "NSR — NON-SHOCKABLE",
    ("TACHY", "NON_SHOCKABLE", None): "TACHY — NON-SHOCKABLE",
    ("BRADY_ASY", "NON_SHOCKABLE", None): "BRADY_ASY — NON-SHOCKABLE",
    ("VENTRICULAR", "SHOCKABLE", "VF"): "VF — SHOCKABLE",
    ("VENTRICULAR", "SHOCKABLE", "VT"): "VT — SHOCKABLE",
}


@pytest.mark.parametrize("rhythm,shock,sub", list(EXPECTED))
def test_required_rows(rhythm, shock, sub):
    d = decide(mo(shock), mo(rhythm), mo(sub) if sub else None)
    assert d.status == "CLASSIFIED"
    assert d.headline == EXPECTED[(rhythm, shock, sub)]


@pytest.mark.parametrize("sub", ["VF", "VT"])
def test_ventricular_non_shockable_is_never_classified(sub):
    """The contradictory row must NOT yield a shockability statement."""
    d = decide(mo("NON_SHOCKABLE"), mo("VENTRICULAR"), mo(sub))
    assert d.status == "CONFLICT"
    assert "SHOCKABLE" not in d.headline
    assert "shockable" not in d.speech.lower()
    assert d.notes and "NON_SHOCKABLE" in d.notes[0] and "VENTRICULAR" in d.notes[0]


@pytest.mark.parametrize("rhythm", ["NSR", "TACHY", "BRADY_ASY"])
def test_shockable_with_nonventricular_rhythm_is_conflict(rhythm):
    d = decide(mo("SHOCKABLE"), mo(rhythm), mo("VT"))
    assert d.status == "CONFLICT" and d.severity != "CRITICAL"


def test_exhaustive_only_five_classified_rows():
    classified = set()
    for r, s, sub in itertools.product(["NSR", "TACHY", "BRADY_ASY", "VENTRICULAR"],
                                       ["NON_SHOCKABLE", "SHOCKABLE"], [None, "VT", "VF"]):
        d = decide(mo(s), mo(r), mo(sub) if sub else None)
        if d.status == "CLASSIFIED":
            classified.add(d.headline)
    assert classified == set(EXPECTED.values())


def test_ventricular_shockable_without_subtype_is_conflict():
    assert decide(mo("SHOCKABLE"), mo("VENTRICULAR"), None).status == "CONFLICT"
