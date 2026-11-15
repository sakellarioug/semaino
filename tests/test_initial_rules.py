"""Unit tests for initial and name joining rules (no trained models needed).

Tests the joining logic for Greek person initials, abbreviations, and toponyms
(e.g. 'Γ. Καναρίου', 'Κων. Λαχανᾶ', 'Ν. Καρλόβασι') while safeguarding against
ledger abbreviations (e.g. 'Γρ.' and 'Παρ.') and list markers.
Run with `python tests/test_initial_rules.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from semaino.ner_engine import _merge_name_initials  # noqa: E402


class DummyToken:
    def __init__(self, text, ner="O", upos=""):
        self.text = text
        self.ner = ner
        self.upos = upos


def test_single_letter_initial_with_person_name():
    tokens = [DummyToken("Γ", "S-PER"), DummyToken("."), DummyToken("Καναρίου", "S-PER")]
    _merge_name_initials(tokens)
    assert [t.ner for t in tokens] == ["B-PER", "I-PER", "E-PER"]


def test_abbreviated_first_name_with_person_name():
    tokens = [DummyToken("Κων", "S-PER"), DummyToken("."), DummyToken("Λαχανᾶ", "S-PER")]
    _merge_name_initials(tokens)
    assert [t.ner for t in tokens] == ["B-PER", "I-PER", "E-PER"]


def test_initial_with_preceding_article():
    tokens = [DummyToken("τοῦ"), DummyToken("Γ", "O"), DummyToken("."), DummyToken("Κατερίνη", "S-PER")]
    _merge_name_initials(tokens)
    assert [t.ner for t in tokens] == ["O", "B-PER", "I-PER", "E-PER"]


def test_chained_initials():
    tokens = [
        DummyToken("τοῦ"),
        DummyToken("Κ", "O"),
        DummyToken("."),
        DummyToken("Γ", "S-PER"),
        DummyToken("."),
        DummyToken("Κατζούλη", "S-PER"),
    ]
    _merge_name_initials(tokens)
    assert [t.ner for t in tokens] == ["O", "B-PER", "I-PER", "I-PER", "I-PER", "E-PER"]


def test_multi_word_surname():
    tokens = [
        DummyToken("Δημ", "S-PER"),
        DummyToken("."),
        DummyToken("Βαραγιάννη", "B-PER"),
        DummyToken("Ψαριανοῦ", "E-PER"),
    ]
    _merge_name_initials(tokens)
    assert [t.ner for t in tokens] == ["B-PER", "I-PER", "I-PER", "E-PER"]


def test_toponym_initial():
    tokens = [DummyToken("Ν", "S-GPE"), DummyToken("."), DummyToken("Καρλόβασι", "S-GPE")]
    _merge_name_initials(tokens)
    assert [t.ner for t in tokens] == ["B-GPE", "I-GPE", "E-GPE"]


def test_ledger_abbreviations_excluded():
    # Γρ. and Παρ. are currency units (Γρόσια / Παράδες) and must not be merged as persons
    tokens = [DummyToken("Γρ", "S-PER"), DummyToken("."), DummyToken("Παρ", "S-PER")]
    _merge_name_initials(tokens)
    assert [t.ner for t in tokens] == ["S-PER", "O", "S-PER"]


def test_list_marker_excluded():
    # "Α. Ορισμοί" without article should not be merged
    tokens = [DummyToken("Α", "O"), DummyToken("."), DummyToken("Ορισμοί", "O")]
    _merge_name_initials(tokens)
    assert [t.ner for t in tokens] == ["O", "O", "O"]


def test_raw_toolkit_tokens_lowercase_and_unaccented():
    # gr-nlp-toolkit gives lower-case, accent-free tokens; the article rule must still fire
    tokens = [DummyToken("του"), DummyToken("γ", "O"), DummyToken("."), DummyToken("κατερινη", "S-PERSON")]
    _merge_name_initials(tokens)
    assert [t.ner for t in tokens] == ["O", "B-PER", "I-PER", "E-PER"]


def test_raw_toolkit_tokens_ledger_abbreviation_still_excluded():
    tokens = [DummyToken("γρ", "S-PERSON"), DummyToken("."), DummyToken("παρ", "S-PERSON")]
    _merge_name_initials(tokens)
    assert [t.ner for t in tokens] == ["S-PERSON", "O", "S-PERSON"]


def test_raw_toolkit_list_marker_without_article_not_merged():
    tokens = [DummyToken("α", "O"), DummyToken("."), DummyToken("ορισμοι", "S-PERSON")]
    _merge_name_initials(tokens)
    assert [t.ner for t in tokens] == ["O", "O", "S-PERSON"]


if __name__ == "__main__":
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    bad = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            bad += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - bad}/{len(tests)} passed")
    sys.exit(1 if bad else 0)
