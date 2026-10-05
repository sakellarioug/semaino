"""Tests for the .docx text extractor (no NLP models needed).

Run with `pytest tests/test_docx_import.py` or `python tests/test_docx_import.py`.
"""
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from semaino.docx_import import DocxError, docx_to_text  # noqa: E402

W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def make_docx(body_xml, document_xml=None):
    """Build a minimal .docx in memory around the given <w:body> content."""
    if document_xml is None:
        document_xml = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                        f'<w:document {W}><w:body>{body_xml}<w:sectPr/></w:body></w:document>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types/>')
        zf.writestr("word/document.xml", document_xml)
    return buf.getvalue()


def para(*runs, style=None):
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return "<w:p>" + ppr + "".join(f"<w:r>{r}</w:r>" for r in runs) + "</w:p>"


def t(text):
    return f'<w:t xml:space="preserve">{text}</w:t>'


def test_paragraphs_become_blank_line_separated_blocks():
    body = (para(t("Η Αθήνα είναι η πρωτεύουσα."), style="Heading1")
            + para(t("Ο Περικλής "), t("έζησε εκεί."))
            + para()  # empty paragraph is skipped
            + para(t("Τέλος.")))
    assert docx_to_text(make_docx(body)) == (
        "Η Αθήνα είναι η πρωτεύουσα.\n\nΟ Περικλής έζησε εκεί.\n\nΤέλος.")


def test_runs_split_mid_word_are_joined():
    body = para(t("Θεσσα"), t("λονίκη"))
    assert docx_to_text(make_docx(body)) == "Θεσσαλονίκη"


def test_tabs_breaks_and_hyphens():
    body = para(t("Α"), "<w:tab/>", t("Β"), "<w:br/>", t("Γ"), "<w:noBreakHyphen/>",
                t("Δ"), "<w:softHyphen/>", t("Ε"))
    assert docx_to_text(make_docx(body)) == "Α Β\nΓ-ΔΕ"


def test_spaces_are_collapsed():
    body = para(t("  Η   Σάμος  "))
    assert docx_to_text(make_docx(body)) == "Η Σάμος"


def test_polytonic_text_is_preserved():
    text = "Ἐν ἀρχῇ ἦν ὁ λόγος."
    assert docx_to_text(make_docx(para(t(text)))) == text


def test_table_rows_join_cells():
    cell = lambda x: f"<w:tc>{para(t(x))}</w:tc>"
    body = (para(t("Πίνακας:"))
            + "<w:tbl>"
            + f"<w:tr>{cell('Πόλη')}{cell('Περιοχή')}</w:tr>"
            + f"<w:tr>{cell('Πάτρα')}{cell('Αχαΐα')}</w:tr>"
            + f"<w:tr>{cell('')}{cell('')}</w:tr>"
            + "</w:tbl>"
            + para(t("Μετά τον πίνακα.")))
    assert docx_to_text(make_docx(body)) == (
        "Πίνακας:\n\nΠόλη; Περιοχή\n\nΠάτρα; Αχαΐα\n\nΜετά τον πίνακα.")


def test_content_controls_are_read():
    body = f"<w:sdt><w:sdtContent>{para(t('Μέσα σε πεδίο.'))}</w:sdtContent></w:sdt>"
    assert docx_to_text(make_docx(body)) == "Μέσα σε πεδίο."


def test_accepts_path_and_file_object():
    data = make_docx(para(t("Κείμενο.")))
    assert docx_to_text(io.BytesIO(data)) == "Κείμενο."
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
        f.write(data)
    try:
        assert docx_to_text(f.name) == "Κείμενο."
    finally:
        Path(f.name).unlink()


def _raises_docx_error(data):
    try:
        docx_to_text(data)
    except DocxError:
        return True
    return False


def test_not_a_zip_is_rejected():
    assert _raises_docx_error(b"\xd0\xcf\x11\xe0 old binary .doc header")


def test_zip_without_document_is_rejected():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("hello.txt", "hi")
    assert _raises_docx_error(buf.getvalue())


def test_malformed_xml_is_rejected():
    assert _raises_docx_error(make_docx("", document_xml="<w:document><unclosed>"))


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
