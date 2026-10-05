"""Extract plain text from Word .docx files using only the standard library.

A .docx file is a zip archive; the body text lives in ``word/document.xml``. Paragraphs
become blocks separated by blank lines (so the paragraph detection used by the analysis
keeps working), and table rows become single lines with their cells joined by "; ".

Not kept: automatic list numbering/bullets, headers, footers, footnotes, comments and
text boxes. The old binary .doc format is not supported.
"""
import io
import re
import zipfile
import xml.etree.ElementTree as ET

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_W = "{%s}" % W_NS

# Refuse documents whose main XML part is implausibly large (zip-bomb guard).
MAX_DOCUMENT_XML_BYTES = 100 * 1024 * 1024

_SPACES = re.compile(r"[ \t\u00a0]+")


class DocxError(ValueError):
    """The file is not a readable .docx document."""


def _paragraph_text(p):
    parts = []
    for el in p.iter():
        tag = el.tag
        if tag == _W + "t":
            parts.append(el.text or "")
        elif tag == _W + "tab":
            parts.append(" ")
        elif tag in (_W + "br", _W + "cr"):
            parts.append("\n")
        elif tag == _W + "noBreakHyphen":
            parts.append("-")
        # w:softHyphen is an optional hyphenation point: contributes nothing.
    lines = [_SPACES.sub(" ", line).strip() for line in "".join(parts).split("\n")]
    return "\n".join(line for line in lines if line)


def _table_lines(tbl):
    lines = []
    for tr in tbl.findall(_W + "tr"):
        cells = []
        for tc in tr.findall(_W + "tc"):
            text = " ".join(t for t in (_paragraph_text(p) for p in tc.iter(_W + "p")) if t)
            text = text.replace("\n", " ").strip()
            if text:
                cells.append(text)
        if cells:
            lines.append("; ".join(cells))
    return lines


def _blocks(container):
    for child in container:
        if child.tag == _W + "p":
            text = _paragraph_text(child)
            if text:
                yield text
        elif child.tag == _W + "tbl":
            yield from _table_lines(child)
        elif child.tag == _W + "sectPr":
            continue
        else:
            # Content controls (w:sdt), custom XML, etc. wrap ordinary paragraphs/tables.
            yield from _blocks(child)


def docx_to_text(source):
    """Return the text of a .docx file.

    ``source`` may be bytes, a path, or a binary file-like object.
    Raises DocxError if the file is not a valid .docx document.
    """
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    try:
        with zipfile.ZipFile(source) as zf:
            try:
                info = zf.getinfo("word/document.xml")
            except KeyError:
                raise DocxError("This file is not a Word .docx document "
                                "(word/document.xml is missing).") from None
            if info.file_size > MAX_DOCUMENT_XML_BYTES:
                raise DocxError("This .docx document is too large to import.")
            xml_bytes = zf.read(info)
    except zipfile.BadZipFile:
        raise DocxError("This file is not a valid .docx document. If it is an old "
                        ".doc file, open it in Word and save it as .docx.") from None

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        raise DocxError(f"The .docx document could not be read: {e}") from None

    body = root.find(_W + "body")
    if body is None:
        return ""
    return "\n\n".join(_blocks(body))
