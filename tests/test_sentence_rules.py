"""Unit tests for the custom sentence-boundary rules (no trained spaCy model needed).

Uses a blank spaCy Greek pipeline, so it runs in seconds. Importing ner_engine also imports
gr-nlp-toolkit, which loads its BERT tokenizer at import time: once the models are cached, set
HF_HUB_OFFLINE=1 to avoid network checks.
Run with `pytest tests/test_sentence_rules.py` or `python tests/test_sentence_rules.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import spacy  # noqa: E402
from semaino.ner_engine import _install_sentence_rules  # noqa: E402


def _nlp():
    return _install_sentence_rules(spacy.blank("grc"))


def _starts(doc):
    """Texts of the tokens explicitly marked as sentence starts (token 0 always counts)."""
    return [t.text for t in doc if t.i == 0 or t.is_sent_start]


def test_blank_line_starts_new_sentence():
    doc = _nlp()("Σύντομη ιστορία\n\nΗ επέκταση της χώρας\n\nΟ Ιωάννης Καποδίστριας")
    assert _starts(doc) == ["Σύντομη", "Η", "Ο"]


def test_extra_blank_lines_and_spaces():
    doc = _nlp()("Τίτλος\n \n\nΚείμενο")
    assert _starts(doc) == ["Τίτλος", "Κείμενο"]


def test_single_newline_does_not_break():
    doc = _nlp()("πρώτη γραμμή\nδεύτερη γραμμή")
    assert _starts(doc) == ["πρώτη"]


def test_whitespace_never_starts_a_sentence():
    doc = _nlp()("α\n\nβ\n\n\nγ")
    assert all(not t.is_sent_start for t in doc if t.is_space)


def test_paragraph_break_wins_over_abbreviation_rule():
    # prevent_abbrev_split clears sentence starts after "ν." – a blank line must still win.
    doc = _nlp()("σύμφωνα με τον ν.\n\nΆρθρο πρώτο")
    assert "Άρθρο" in _starts(doc)


def test_install_is_idempotent():
    nlp = _nlp()
    _install_sentence_rules(nlp)
    assert nlp.pipe_names.count("paragraph_sentence_breaks") == 1
    assert nlp.pipe_names.count("prevent_abbrev_split") == 1
    assert nlp.pipe_names.index("paragraph_sentence_breaks") > nlp.pipe_names.index("prevent_abbrev_split")


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
