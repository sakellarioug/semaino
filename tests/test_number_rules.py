"""Unit tests for dotted number merging rules (no trained models needed).

Tests the numeric joining logic for Greek historical and accounting texts where
dots are used as thousand separators (e.g. 152.984 or 5.418).
Run with `python tests/test_number_rules.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from semaino.ner_engine import _merge_dotted_numbers  # noqa: E402


class DummyToken:
    def __init__(self, text, ner="O", upos=""):
        self.text = text
        self.ner = ner
        self.upos = upos


def test_standard_three_digit_thousands():
    tokens = [DummyToken("5", "S-CARDINAL"), DummyToken("."), DummyToken("418", "S-CARDINAL")]
    _merge_dotted_numbers(tokens)
    assert [t.ner for t in tokens] == ["B-CARDINAL", "I-CARDINAL", "E-CARDINAL"]


def test_unrecognised_tail_is_captured():
    # gr-nlp-toolkit often marks only the first digits as CARDINAL and tail as O
    tokens = [DummyToken("152", "S-CARDINAL"), DummyToken("."), DummyToken("984", "O")]
    _merge_dotted_numbers(tokens)
    assert [t.ner for t in tokens] == ["B-CARDINAL", "I-CARDINAL", "E-CARDINAL"]


def test_completely_unrecognised_number_is_captured():
    # Like 8.022 which gr-nlp-toolkit missed entirely
    tokens = [DummyToken("8", "O"), DummyToken("."), DummyToken("022", "O")]
    _merge_dotted_numbers(tokens)
    assert [t.ner for t in tokens] == ["B-CARDINAL", "I-CARDINAL", "E-CARDINAL"]


def test_multi_group_millions():
    tokens = [
        DummyToken("1"),
        DummyToken("."),
        DummyToken("234"),
        DummyToken("."),
        DummyToken("567"),
    ]
    _merge_dotted_numbers(tokens)
    assert [t.ner for t in tokens] == [
        "B-CARDINAL",
        "I-CARDINAL",
        "I-CARDINAL",
        "I-CARDINAL",
        "E-CARDINAL",
    ]


def test_sentence_period_not_merged():
    # 'Σελίδα 5. Οι άγραφες' - 'Οι' is not 3 digits
    tokens = [DummyToken("5", "O"), DummyToken("."), DummyToken("Οι", "O")]
    _merge_dotted_numbers(tokens)
    assert [t.ner for t in tokens] == ["O", "O", "O"]


def test_money_and_quantity_tags_preserved():
    tokens = [DummyToken("152", "B-MONEY"), DummyToken("."), DummyToken("984", "I-MONEY")]
    _merge_dotted_numbers(tokens)
    assert [t.ner for t in tokens] == ["B-MONEY", "I-MONEY", "E-MONEY"]


def test_comma_thousands_separator():
    tokens = [DummyToken("152"), DummyToken(","), DummyToken("984")]
    _merge_dotted_numbers(tokens)
    assert [t.ner for t in tokens] == ["B-CARDINAL", "I-CARDINAL", "E-CARDINAL"]


def _toks(text):
    return [DummyToken(w) for w in text.split()]


def test_spaced_list_number_not_merged_when_source_given():
    # '2. 100 people': the digits are not written together, so this is not 2,100
    tokens = _toks("2 . 100")
    _merge_dotted_numbers(tokens, "2. 100 people")
    assert [t.ner for t in tokens] == ["O", "O", "O"]


def test_unspaced_number_merged_when_source_given():
    tokens = _toks("5 . 418")
    _merge_dotted_numbers(tokens, "Σελίδα 5.418 εδώ")
    assert [t.ner for t in tokens] == ["B-CARDINAL", "I-CARDINAL", "E-CARDINAL"]


def test_partial_merge_stops_at_spaced_group():
    # 1.234 is written together, but ' 567' is spaced: only the first group is merged
    tokens = _toks("1 . 234 . 567")
    _merge_dotted_numbers(tokens, "1.234. 567")
    assert [t.ner for t in tokens] == ["B-CARDINAL", "I-CARDINAL", "E-CARDINAL", "O", "O"]


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
