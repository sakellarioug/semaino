"""Unit tests for the OCR-markup cleaner (no NLP models needed).

Cases are taken from real Marker output of a scanned Greek history text, plus edge cases that must NOT change.
Run with `pytest tests/test_text_cleaning.py` or `python tests/test_text_cleaning.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from semaino.text_cleaning import clean_ocr_markdown as clean  # noqa: E402

CASES = [
    # --- markup that must be removed (real examples from Marker output) ---
    ("heading", "#### Σύντομη ιστορία του κράτους", "Σύντομη ιστορία του κράτους"),
    ("heading becomes own paragraph", "κείμενο\n# Ο Γεώργιος Λογοθέτης\nσυνέχεια",
     "κείμενο\n\nΟ Γεώργιος Λογοθέτης\n\nσυνέχεια"),
    ("footnote ref", "καί θαλάσσης»<sup>1</sup>.", "καί θαλάσσης»."),
    ("footnote ref list", "εκδόσεις<sup>8,9,10</sup> με", "εκδόσεις με"),
    ("footnote ref with spaces", "μελέτες<sup>22, 23</sup>. Όμως", "μελέτες. Όμως"),
    ("footnote body label", "<sup>1.</sup> Λογοθέτη Λυκούργου", "Λογοθέτη Λυκούργου"),
    ("ordinal suffix kept", "την 6<sup>ος</sup> ημέρα, 28<sup>η</sup>", "την 6ος ημέρα, 28η"),
    ("image line", "πριν\n\n![](_page_18_Picture_1.jpeg)\n\nμετά", "πριν\n\nμετά"),
    ("image alt text kept", "![Χάρτης Σάμου](map.png)", "Χάρτης Σάμου"),
    ("link label kept", "δες [Σαμιακά](http://x.gr/a_b) εδώ", "δες Σαμιακά εδώ"),
    ("bullet list", "- 3. Στο ίδιο.", "3. Στο ίδιο."),
    ("blockquote", "> «ἐξοχώτατε!...»", "«ἐξοχώτατε!...»"),
    ("italic", " *ὅσαι, μετάσγουσαι καί αὐταί*", "ὅσαι, μετάσγουσαι καί αὐταί"),
    ("bold", "η **Σάμος** ήταν", "η Σάμος ήταν"),
    ("unpaired italic marker", "*ὅσαι, μετάσγουσαι", "ὅσαι, μετάσγουσαι"),
    ("latex greek accent", "$T\\tilde{\\omega}$ν Σαμίων", "Tῶν Σαμίων"),
    ("latex guillemet + footnote", "λόγια$\\gg^{64}$ και", "λόγια» και"),
    ("latex abbreviation", "$(\\Sigma \\pi.)$", "(Σπ.)"),
    ("latex mis-OCR word", "$\\Sigma_{\\tau 0}$ ίδιο", "Στ0 ίδιο"),
    ("html tags and entities", "<span id=\"page-3-0\"></span>Α &amp; Β<br/>Γ", "Α & Β Γ"),
    ("horizontal rule", "α\n\n---\n\nβ", "α\n\nβ"),
    ("table", "| Όνομα | Τόπος |\n|---|---|\n| Λυκούργος | Σάμος |", "Όνομα; Τόπος\n\nΛυκούργος; Σάμος"),
    ("backslash escapes", "ν\\. 4548/2018 \\*", "ν. 4548/2018 *"),
    ("blank lines collapsed", "α\n\n\n\nβ", "α\n\nβ"),
    # --- text that must NOT change ---
    ("prices untouched", "κόστος $5 και $10", "κόστος $5 και $10"),
    ("law reference untouched", "σύμφωνα με το άρθ. 4 του ν. 4548/2018", "σύμφωνα με το άρθ. 4 του ν. 4548/2018"),
    ("snake_case untouched", "αρχείο file_name_v2 εδώ", "αρχείο file_name_v2 εδώ"),
    ("multiplication untouched", "2 * 3 = 6", "2 * 3 = 6"),
    ("less-than untouched", "αν x < 5 και y > 2", "αν x < 5 και y > 2"),
    ("numbered line untouched", "35. Φωτεινός Δ., πιο πάνω.", "35. Φωτεινός Δ., πιο πάνω."),
    ("polytonic untouched", "ἄς ἀποθάνωμεν ὑπέρ πατρίδος", "ἄς ἀποθάνωμεν ὑπέρ πατρίδος"),
    ("paragraphs preserved", "Πρώτη.\n\nΔεύτερη.", "Πρώτη.\n\nΔεύτερη."),
]


def test_clean_ocr_markdown():
    failures = [(name, clean(src), want) for name, src, want in CASES if clean(src) != want]
    assert not failures, "\n".join(f"{n}: got {g!r}, want {w!r}" for n, g, w in failures)


def test_idempotent():
    for _, src, _ in CASES:
        once = clean(src)
        assert clean(once) == once, f"not idempotent for {src!r}"


if __name__ == "__main__":
    bad = 0
    for name, src, want in CASES:
        got = clean(src)
        ok = got == want
        bad += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n        got:  {got!r}\n        want: {want!r}"))
    idem = all(clean(clean(s)) == clean(s) for _, s, _ in CASES)
    print(f"  {'PASS' if idem else 'FAIL'}  idempotent")
    print(f"\n{len(CASES) - bad}/{len(CASES)} cases passed")
    sys.exit(1 if bad or not idem else 0)
