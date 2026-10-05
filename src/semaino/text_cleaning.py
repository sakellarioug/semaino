"""Turn Marker's OCR Markdown into plain prose before NLP analysis.

Marker (the PDF OCR step) emits Markdown and some HTML/LaTeX. Left in place, the tokenizer
treats that markup as words: `# # # #` headings, `< sup > 1 < / sup >` footnote references,
image links, list bullets, and LaTeX that Marker produces when it misreads a glyph
(e.g. `$T\\tilde{\\omega}$` for "Tῶ"). This module removes the markup while keeping the text
and the paragraph structure (blank lines) that the analysis relies on.

Pure standard library, so it can be tested without loading any NLP model.
"""
import html
import re
import unicodedata

__all__ = ["clean_ocr_markdown"]

# --- LaTeX --------------------------------------------------------------------------------
_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "θ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "omicron": "ο", "pi": "π", "varpi": "π",
    "rho": "ρ", "varrho": "ρ", "sigma": "σ", "varsigma": "ς", "tau": "τ", "upsilon": "υ",
    "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Alpha": "Α", "Beta": "Β", "Gamma": "Γ", "Delta": "Δ", "Epsilon": "Ε", "Zeta": "Ζ",
    "Eta": "Η", "Theta": "Θ", "Iota": "Ι", "Kappa": "Κ", "Lambda": "Λ", "Mu": "Μ", "Nu": "Ν",
    "Xi": "Ξ", "Omicron": "Ο", "Pi": "Π", "Rho": "Ρ", "Sigma": "Σ", "Tau": "Τ",
    "Upsilon": "Υ", "Phi": "Φ", "Chi": "Χ", "Psi": "Ψ", "Omega": "Ω",
}
_SYMBOLS = {"gg": "»", "ll": "«", "cdot": "·", "ldots": "…", "dots": "…", "prime": "΄",
            "S": "§", "%": "%", "&": "&", "$": "$", "#": "#", "_": "_", ",": " ", ";": " ", " ": " "}
# Accent commands -> combining marks (Greek: \tilde is the perispomeni/circumflex)
_ACCENTS = {"tilde": "\u0342", "acute": "\u0301", "grave": "\u0300", "ddot": "\u0308",
            "bar": "\u0304", "breve": "\u0306"}

_MATH = re.compile(r"\$\$(.+?)\$\$|(?<![\\$])\$([^$\n]+?)\$", re.S)


def _latex_to_text(expr):
    s = expr
    # superscripts/subscripts that are only numbers are footnote marks: drop them
    s = re.sub(r"\^\{\s*[\d,.\s]+\}|\^\d+", "", s)
    # accents: \tilde{\omega} or \tilde{x} or \tilde x -> base + combining mark
    def accent(m):
        base = m.group(2) or m.group(3) or ""
        return _latex_to_text(base) + _ACCENTS[m.group(1)]
    s = re.sub(r"\\(" + "|".join(_ACCENTS) + r")\s*(?:\{([^{}]*)\}|(\\?[A-Za-z]+|.))", accent, s)
    # \text{...}, \mathrm{...} etc.: keep the content
    s = re.sub(r"\\(?:text|mathrm|mathit|mathbf|textit|textbf|operatorname)\s*\{([^{}]*)\}", r"\1", s)
    # named commands; LaTeX ignores spaces after a command word
    def command(m):
        name = m.group(1)
        return _GREEK.get(name, _SYMBOLS.get(name, ""))
    s = re.sub(r"\\([A-Za-z]+)\s*", command, s)
    s = re.sub(r"\\(.)", lambda m: _SYMBOLS.get(m.group(1), m.group(1)), s)
    s = re.sub(r"[{}_^]", "", s)
    return unicodedata.normalize("NFC", s)


def _replace_math(m):
    expr = m.group(1) if m.group(1) is not None else m.group(2)
    # Only treat it as LaTeX if it looks like LaTeX; leave real prices like "$5 and $10" alone
    if not re.search(r"[\\^_{}]", expr):
        return m.group(0)
    return _latex_to_text(expr).strip()


# --- HTML ---------------------------------------------------------------------------------
_FOOTNOTE_REF = re.compile(r"^[\s\d,.;:\-–—*†‡]*$")
_BLOCK_TAGS = r"p|div|li|ul|ol|tr|table|h[1-6]|blockquote|section|article|header|footer"


def _replace_sup(m):
    inner = m.group(1)
    # <sup>1</sup>, <sup>8,9,10</sup>, <sup>82.</sup> are footnote references: remove.
    # <sup>ος</sup>, <sup>η</sup> are ordinal suffixes (6ος, 28η): keep, joined to the number.
    return "" if _FOOTNOTE_REF.match(inner) else inner


def _strip_html(text):
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<sup\b[^>]*>(.*?)</sup>", _replace_sup, text, flags=re.S | re.I)
    text = re.sub(r"<sub\b[^>]*>(.*?)</sub>", r"\1", text, flags=re.S | re.I)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(rf"</?(?:{_BLOCK_TAGS})\b[^>]*>", "\n", text, flags=re.I)
    text = re.sub(r"</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>", "", text)   # any other tag
    return html.unescape(text)


# --- Markdown -----------------------------------------------------------------------------
def _clean_line(line):
    if re.match(r"^\s*(```|~~~)", line):                       # code fences
        return ""
    if re.match(r"^\s*([-*_])(\s*\1){2,}\s*$", line):          # horizontal rules
        return ""
    if re.match(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$", line):   # table divider
        return ""
    heading = re.match(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$", line)
    if heading:                                                 # keep headings as own paragraph
        return "\n" + heading.group(1) + "\n"
    line = re.sub(r"^(\s*>\s?)+", "", line)                     # blockquotes
    line = re.sub(r"^\s*[-*+]\s+", "", line)                    # bullet lists
    if line.count("|") >= 2:                                    # table rows -> cells as phrases
        line = "; ".join(c.strip() for c in line.strip().strip("|").split("|") if c.strip())
    return line


def _strip_inline_markdown(text):
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)      # images: keep alt text only
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)       # links: keep the label
    text = re.sub(r"\[\^[^\]]+\]", "", text)                    # footnote refs [^1]
    text = re.sub(r"`([^`]*)`", r"\1", text)                    # inline code
    text = re.sub(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1", r"\2", text)              # bold
    text = re.sub(r"(?<![\w*\\])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])", r"\1", text)  # italic *x*
    text = re.sub(r"(?<![\w\\])_(?=\S)(.+?)(?<=\S)_(?!\w)", r"\1", text)      # italic _x_
    text = re.sub(r"(?<!\S)\*+(?=\S)|(?<=\S)(?<!\\)\*+(?!\S)", "", text)      # unpaired * markers
    text = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|>~])", r"\1", text)               # backslash escapes
    return text


def clean_ocr_markdown(text):
    """Return `text` with Markdown/HTML/LaTeX markup removed and paragraphs preserved."""
    if not text:
        return text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MATH.sub(_replace_math, text)
    text = _strip_html(text)
    text = "\n".join(_clean_line(line) for line in text.split("\n"))
    text = _strip_inline_markdown(text)
    text = re.sub(r"[ \t\u00a0]+", " ", text)                   # collapse spaces, keep newlines
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)                      # at most one blank line
    return text.strip()
