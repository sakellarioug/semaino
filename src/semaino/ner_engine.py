import spacy
import torch
import re
from gr_nlp_toolkit import Pipeline
from collections import defaultdict
import unicodedata
from spacy import displacy

from .text_cleaning import clean_ocr_markdown

# --- Device setup ---
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

MODERN_GREEK_MODELS = None
MAX_MODEL_TOKENS = 510
RESERVED_TOKENS = 2
MAX_SUBWORDS = MAX_MODEL_TOKENS - RESERVED_TOKENS

def _install_sentence_rules(nlp_spacy):
    """Add the custom sentence-boundary components to a spaCy pipeline (idempotent).

    The OdyCy pipeline object is cached and shared by the Modern and Ancient Greek engines,
    so both get the same rules whichever engine is loaded first.
    """
    from spacy.language import Language
    from spacy.tokens import Doc

    # --- Robust sentence-boundary fix for ν., τ., ρ. ---
    if not Language.has_factory("prevent_abbrev_split"):
        @Language.component("prevent_abbrev_split")
        def prevent_abbrev_split(doc: Doc):
            abbrevs = {",", "υπ'", "υπ", "γ.ε.μη", "αφμ", "α.φ.μ.", "λ", "γ", "Κ", "Τ,Κ", "α.φ.μ", "κιν", "τηλ", "ρ", "ν", "τ", "αρ", "παρ", "παρ.", "αυξ", "αρθρ", "αρθ", "άρθ", "αριθ", "αριθ.",  "πρωτ", "α.ε", "αύξ", "αύξ.", "αυξ." "αρίθ", "αριθμ", "αριθμ.", "κα", "Μ", "Η"}
            for i in range(len(doc) - 2):
                token = doc[i]
                if token.text.strip() == "." or token.text.strip() == ",":
                    doc[i].is_sent_start = False
                if any(token.text.lower().endswith(abbr) for abbr in abbrevs) and (doc[i + 1].text == "." or doc[i + 1].text == " "):
                    doc[i].is_sent_start = False
                    doc[i + 1].is_sent_start = False
                    doc[i + 2].is_sent_start = False
                if (doc[i + 1].text == "/" or doc[i + 1].text == "-"):
                    if i + 2 < len(doc):
                        after_next = doc[i + 2]
                        doc[i].is_sent_start = False
                        doc[i + 1].is_sent_start = False
                        after_next.is_sent_start = False
            return doc

    if "prevent_abbrev_split" not in nlp_spacy.pipe_names:
        if "parser" in nlp_spacy.pipe_names:
            nlp_spacy.add_pipe("prevent_abbrev_split", before="parser")
        elif "senter" in nlp_spacy.pipe_names:
            nlp_spacy.add_pipe("prevent_abbrev_split", after="senter")
        else:
            nlp_spacy.add_pipe("prevent_abbrev_split")

    # --- A blank line always ends a sentence ---
    # The statistical sentence splitter ignores line breaks, so a heading would otherwise be
    # merged with the paragraph after it (and a title split across two "paragraphs"). This
    # runs after prevent_abbrev_split so that a paragraph break always wins.
    if not Language.has_factory("paragraph_sentence_breaks"):
        @Language.component("paragraph_sentence_breaks")
        def paragraph_sentence_breaks(doc: Doc):
            newlines = 0
            for i, token in enumerate(doc):
                if token.is_space:
                    newlines += token.text.count("\n")
                    if i > 0:
                        token.is_sent_start = False   # never start a sentence with whitespace
                else:
                    if newlines >= 2:
                        token.is_sent_start = True
                    newlines = 0
            return doc

    if "paragraph_sentence_breaks" not in nlp_spacy.pipe_names:
        nlp_spacy.add_pipe("paragraph_sentence_breaks", after="prevent_abbrev_split")
    return nlp_spacy


def get_ancient_greek_model():
    """OdyCy (grc_odycy_joint_trf) with the custom sentence-boundary rules installed."""
    return _install_sentence_rules(get_spacy_model("grc_odycy_joint_trf"))


def get_modern_greek_models():
    global MODERN_GREEK_MODELS
    if MODERN_GREEK_MODELS is None:
        print("Lazy loading Modern Greek models...")
        from gr_nlp_toolkit import Pipeline
        from transformers import AutoTokenizer

        # We use OdyCy for sentence parsing as requested, since it performs better
        nlp_spacy = get_ancient_greek_model()
        nlp_gr = Pipeline("pos,ner,dp")
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
        MODERN_GREEK_MODELS = (nlp_spacy, nlp_gr, tokenizer)

    return MODERN_GREEK_MODELS

def is_date_separator(tok):
    return tok.text in {"/", "-", "."}
def is_money_connector(tok):
    return tok.text.isdigit() or tok.text in {",", ".", "€", "$", "£", "¥"}
def is_law_connector(tok):
    return tok.text.isdigit() or tok.text.lower() in {"/", "-", ".", ",", "'", "ν", "ν.", "αρθρο", "αρθρ.", "§", "παρ", "παρ."}
def strip_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))


def _merge_dotted_numbers(tokens):
    """Merge numeric tokens split by thousands dots (e.g. 5 . 418 or 152 . 984).

    In Greek accounting and historical texts, numbers often use '.' as a thousands separator.
    When gr-nlp-toolkit tokenises '152.984', it produces ['152', '.', '984'], tagging either
    only the prefix as CARDINAL or leaving the tail as O. This function unifies them into
    a single CARDINAL (or MONEY/QUANTITY if already marked) entity spanning the full number.
    """
    n = len(tokens)
    i = 0
    while i < n:
        t_i = tokens[i].text
        if re.match(r"^\d{1,3}$", t_i):
            j = i
            while j + 2 < n and tokens[j + 1].text in (".", ",") and re.match(r"^\d{3}$", tokens[j + 2].text):
                j += 2
            if j > i:
                label = "CARDINAL"
                for k in range(i, j + 1):
                    ner = getattr(tokens[k], "ner", "O") or "O"
                    if "MONEY" in ner:
                        label = "MONEY"
                        break
                    elif "QUANTITY" in ner:
                        label = "QUANTITY"

                tokens[i].ner = f"B-{label}"
                for k in range(i + 1, j):
                    tokens[k].ner = f"I-{label}"
                tokens[j].ner = f"E-{label}"
                i = j + 1
                continue
        i += 1


NON_PERSON_INITIALS = {"γρ", "γρ.", "παρ", "παρ.", "σουμα", "σοῦμα"}

GREEK_ARTICLES = {
    "ο", "ὁ", "του", "τοῦ", "τον", "τὸν", "τω", "τῷ",
    "η", "ἡ", "της", "τῆς", "την", "τὴν",
    "οι", "οἱ", "των", "τῶν", "τους", "τοὺς",
    "δια", "διὰ", "απο", "ἀπὸ", "από", "υπο", "ὑπὸ", "υπό",
    "κ", "κ.", "κα", "κα.", "κος", "κύριος", "κύριον", "κυρίου",
}


def _merge_name_initials(tokens):
    """Join single-letter or abbreviated name initials with adjacent person/place entities.

    In Greek texts, initials like 'Γ.' in 'Γ. Καναρίου' or 'Κων.' in 'Κων. Λαχανᾶ' are split
    by the period into disconnected tokens. This rule safely joins them under the same entity
    within the sentence, while excluding ledger abbreviations (e.g. 'Γρ.' and 'Παρ.').
    """
    n = len(tokens)
    i = 0
    while i < n - 2:
        t0 = tokens[i]
        t1 = tokens[i + 1]
        t2 = tokens[i + 2]

        t0_clean = t0.text.strip(" .")
        t2_clean = t2.text.strip(" .")

        if t1.text == "." and t0_clean.lower() not in NON_PERSON_INITIALS and t2_clean.lower() not in NON_PERSON_INITIALS:
            is_t0_abbr = bool(re.match(r"^[Α-Ωα-ωA-Za-zΆ-Ώά-ώἀ-ῼἈ-΅]{1,4}$", t0_clean))
            t0_ner = getattr(t0, "ner", "O") or "O"
            t2_ner = getattr(t2, "ner", "O") or "O"

            t0_is_per = "PER" in t0_ner
            t2_is_per = "PER" in t2_ner
            is_single_cap = bool(re.match(r"^[Α-ΩA-ZΆ-ΏἈ-΅]$", t0_clean))
            has_preceding_article = (i > 0 and tokens[i - 1].text.strip(" .").lower() in GREEK_ARTICLES)

            if (is_t0_abbr and t0_is_per and t2_is_per) or (is_single_cap and t2_is_per and has_preceding_article):
                t0.ner = "I-PER" if t0.ner.startswith(("I-", "E-")) else "B-PER"
                t1.ner = "I-PER"
                if t2.ner.startswith("S-"):
                    t2.ner = "E-PER"
                elif t2.ner.startswith("B-"):
                    t2.ner = "I-PER"
                i += 1
                continue

            t0_is_gpe = "GPE" in t0_ner
            t2_is_gpe = "GPE" in t2_ner
            if is_t0_abbr and t0_is_gpe and t2_is_gpe:
                t0.ner = "I-GPE" if t0.ner.startswith(("I-", "E-")) else "B-GPE"
                t1.ner = "I-GPE"
                if t2.ner.startswith("S-"):
                    t2.ner = "E-GPE"
                elif t2.ner.startswith("B-"):
                    t2.ner = "I-GPE"
                i += 1
                continue

        i += 1


# --- Restoring original case and diacritics -------------------------------------------
# gr-nlp-toolkit lowercases and strips all diacritics. To show the user's original text we
# locate each token in the source sentence using a "folded" copy of both: decomposed (NFD),
# combining marks removed, lowercased, final sigma unified. This covers monotonic, polytonic
# (breathings, circumflex, iota subscript) and diaeresis forms with one rule.
def _fold_char(c):
    base = ''.join(ch for ch in unicodedata.normalize('NFD', c) if not unicodedata.combining(ch))
    return base.lower().replace('ς', 'σ')


def build_folded_index(text):
    """Return (folded_text, index) where index[i] is the position in `text` of folded_text[i]."""
    folded, index = [], []
    for i, c in enumerate(text):
        for fc in _fold_char(c):
            folded.append(fc)
            index.append(i)
    return ''.join(folded), index


def restore_original_form(token_text, source, folded, index, cursor, pending_letters=0):
    """Find `token_text` in `source` at or after `cursor` (a position in `folded`).

    Only punctuation/whitespace may be skipped between consecutive tokens, plus the letters of
    any immediately preceding tokens that could not be matched (`pending_letters`), so a single
    miss cannot make the cursor jump ahead and break every following token.
    Returns (text, new_cursor, matched).
    """
    needle = ''.join(_fold_char(c) for c in token_text)
    if not needle:
        return token_text, cursor, False
    pos = folded.find(needle, cursor)
    if pos < 0 or sum(ch.isalnum() for ch in folded[cursor:pos]) > pending_letters + 2:
        return token_text, cursor, False
    start = index[pos]
    end = index[pos + len(needle) - 1] + 1
    while end < len(source) and unicodedata.combining(source[end]):
        end += 1   # keep trailing combining marks when the source itself is decomposed
    return source[start:end], pos + len(needle), True

@torch.no_grad()
def analyse_with_gr_toolkit(text, progress_callback=None):
    nlp_spacy, nlp_gr, tokenizer = get_modern_greek_models()
    
    tokens_info = []
    entities = []
    
    doc_spacy = nlp_spacy(text)
    sentences = [s.text for s in doc_spacy.sents]
    total_sents = len(sentences)
    
    paragraph_offsets = [m.start() for m in re.finditer(r'\n\s*\n', text)]
    
    for sent_idx, sent in enumerate(doc_spacy.sents):
        if progress_callback:
            progress_callback(sent_idx + 1, total_sents)
            
        para_idx = sum(1 for offset in paragraph_offsets if sent.start_char >= offset)
        
        encoding = tokenizer(sent.text, add_special_tokens=False)
        if len(encoding["input_ids"]) > MAX_SUBWORDS:
            # Smart chunking using SpaCy tokens to preserve exact whitespace and avoid splitting legal entities
            subsents = []
            current_chunk = ""
            current_len = 0
            
            for token in sent:
                w_len = len(tokenizer(token.text, add_special_tokens=False)["input_ids"])
                
                # Seek a safe syntactic split point (comma, semicolon, colon) after 350 tokens
                if current_len > 350 and token.text in {",", ";", ":"}:
                    current_chunk += token.text_with_ws
                    subsents.append(current_chunk)
                    current_chunk = ""
                    current_len = 0
                # Hard fallback split if no safe point is found and we hit the absolute ceiling
                elif current_len + w_len > MAX_SUBWORDS - 15:
                    subsents.append(current_chunk)
                    current_chunk = token.text_with_ws
                    current_len = w_len
                else:
                    current_chunk += token.text_with_ws
                    current_len += w_len
                    
            if current_chunk:
                subsents.append(current_chunk)
        else:
            subsents = [sent.text]
            
        for subsent in subsents:
            try:
                doc_gr = nlp_gr(subsent)
            except:
                doc_gr = Pipeline("pos,ner")(subsent)

            _merge_dotted_numbers(doc_gr.tokens)
            _merge_name_initials(doc_gr.tokens)

            current = None
            folded_subsent, folded_index = build_folded_index(subsent)
            original_cursor = 0
            pending_letters = 0   # letters of unmatched tokens since the last match
            for tok in doc_gr.tokens:
                head_text = "ROOT"
                if hasattr(tok, 'head') and tok.head is not None and tok.head > 0 and tok.head <= len(doc_gr.tokens):
                    head_text = doc_gr.tokens[tok.head - 1].text

                # Restore the original case and diacritics from the source text
                original_cased_text, original_cursor, matched = restore_original_form(
                    tok.text, subsent, folded_subsent, folded_index, original_cursor, pending_letters)
                pending_letters = 0 if matched else pending_letters + sum(ch.isalnum() for ch in tok.text)

                tokens_info.append({
                    "text": original_cased_text,
                    "pos": getattr(tok, "upos", ""),
                    "tag": getattr(tok, "feats", ""),
                    "dep": getattr(tok, "deprel", ""),
                    "head": head_text,
                    "head_id": getattr(tok, "head", 0),
                    "sentence_index": sent_idx,
                    "paragraph_index": para_idx,
                    "ner": getattr(tok, "ner", "O") or "O"
                })
                
                ner = getattr(tok, "ner", "O") or "O"
                text_analyse = tok.text
                
                if not ner or ner == "O":
                    if current:
                        if current["label"] == "DATE" and is_date_separator(tok):
                            current["text"] += text_analyse
                            continue
                        if current["label"] == "MONEY" and is_money_connector(tok):
                            current["text"] += text_analyse
                            continue
                        if current["label"] == "LAW" and is_law_connector(tok):
                            if tok.upos in ("PUNCT", "SYM") or current["text"].endswith((".", ",", "/", "-", "'")) or text_analyse.startswith((".", ",", "/", "-", "'")):
                                current["text"] += text_analyse
                            else:
                                current["text"] += " " + text_analyse
                            continue
                        entities.append(current)
                        current = None
                    continue
                
                parts = ner.split("-")
                prefix = parts[0] if len(parts) == 2 else "S"
                base = parts[-1].upper()
                if base == "PERSON": base = "PER"
                
                if base in {"DATE", "LAW", "MONEY", "CARDINAL"}:
                    if current and (
                        current["label"] == base or 
                        (current["label"] == "CARDINAL" and base == "MONEY") or 
                        (current["label"] == "MONEY" and base == "CARDINAL") or
                        (current["label"] == "LAW" and base in {"DATE", "CARDINAL"}) or
                        (current["label"] in {"DATE", "CARDINAL"} and base == "LAW")
                    ):
                        if current["label"] in {"CARDINAL", "DATE"} and base in {"MONEY", "LAW"}:
                            current["label"] = base
                        
                        if tok.upos in ("PUNCT", "SYM") or current["text"].endswith((".", ",", "/", "-", "'")) or text_analyse.startswith((".", ",", "/", "-", "'")):
                            current["text"] += text_analyse
                        else:
                            current["text"] += " " + text_analyse
                        continue
                    if current: entities.append(current)
                    current = {"text": text_analyse, "label": base, "sentence_index": sent_idx, "dep": getattr(tok, "deprel", ""), "head": head_text}
                    continue
                
                if prefix in ("B", "S"):
                    if current: 
                        entities.append(current)
                    current = {"text": text_analyse, "label": base, "sentence_index": sent_idx, "dep": getattr(tok, "deprel", ""), "head": head_text}
                    if prefix == "S":
                        entities.append(current)
                        current = None
                elif prefix in ("I", "E") and current:
                    if text_analyse in (".", ",", "/", "-", "'"):
                        current["text"] += text_analyse
                    elif current["label"] in ("CARDINAL", "MONEY", "QUANTITY") and current["text"].endswith((".", ",", "/", "-", "'")):
                        current["text"] += text_analyse
                    else:
                        current["text"] += " " + text_analyse
                    if prefix == "E":
                        entities.append(current)
                        current = None

            if current:
                entities.append(current)
                current = None

    for e in entities:
        if e["label"] == "MONEY":
            e["text"] = e["text"].strip()
            e["text"] = e["text"].rstrip(" .,;/-")
        elif e["label"] == "DATE":
            e["text"] = e["text"].strip()
            e["text"] = e["text"].rstrip(" .,;/-")
            e["text"] = re.sub(r'\s+(από|έως|μέχρι|στις|στα|στον|στην|στο)$', '', e["text"], flags=re.IGNORECASE)
            e["text"] = re.sub(r'^(από|έως|μέχρι|στις|στα|στον|στην|στο)\s+', '', e["text"], flags=re.IGNORECASE)
            if re.match(r'^[/\-\.]', e["text"]):
                sent_text = sentences[e["sentence_index"]]
                escaped_ent = re.escape(e["text"])
                pattern = r'(?:\d+\s*)' + escaped_ent
                match = re.search(pattern, sent_text, flags=re.IGNORECASE)
                if match:
                    e["text"] = match.group(0)
        elif e["label"] == "LAW":
            e["text"] = e["text"].strip()
            e["text"] = e["text"].rstrip(" .,;/-)")
            sent_text = sentences[e["sentence_index"]]
            escaped_ent = re.escape(e["text"])
            pattern = r'(?:(?:άρθ\.|άρθρο|αρθ\.|αρθ|παρ\.|παράγραφος)\s*\d+[α-ωΑ-Ω]?\s*(?:του\s*|της\s*)?)+' + escaped_ent
            match = re.search(pattern, sent_text, flags=re.IGNORECASE)
            if match:
                e["text"] = match.group(0)
            
    return entities, sentences, doc_spacy, tokens_info

def fuzzy_find_entity(needle, search_region, start_cursor=0):
    import re
    needle_clean = needle.replace(" ", "")
    if not needle_clean:
        return -1, -1
    escaped_chars = [re.escape(c) for c in needle_clean]
    pattern = r'\s*'.join(escaped_chars)
    match = re.search(pattern, search_region[start_cursor:], flags=re.IGNORECASE)
    if match:
        return start_cursor + match.start(), start_cursor + match.end()
    return -1, -1

def merged_entities_to_displacy(text, merged_entities, sentences):
    ents = []
    sentence_offsets = []
    search_pos = 0
    for sent in sentences:
        norm_sent = sent.strip()
        start = text.find(norm_sent, search_pos)
        if start == -1: continue
        end = start + len(norm_sent)
        sentence_offsets.append((start, end))
        search_pos = end

    norm_text = strip_accents(text).lower()
    sentence_cursors = {i: 0 for i in range(len(sentence_offsets))}

    for ent in merged_entities:
        ent_text = ent["text"]
        ent_label = ent.get("label")
        sent_idx = ent.get("sentence_index")
        
        if sent_idx is not None and sent_idx < len(sentence_offsets):
            sent_start, sent_end = sentence_offsets[sent_idx]
            search_region = norm_text[sent_start:sent_end]
            needle = strip_accents(ent_text).lower()
            cursor = sentence_cursors.get(sent_idx, 0)
            
            local_start, local_end = fuzzy_find_entity(needle, search_region, cursor)
            
            if local_start >= 0:
                start = sent_start + local_start
                end = sent_start + local_end
                original_text = text[start:end]
                ents.append({"start": start, "end": end, "label": ent_label, "text": original_text, "dep": ent.get("dep", ""), "head": ent.get("head", "")})
                sentence_cursors[sent_idx] = local_end
                continue
            
            # fallback without cursor
            local_start, local_end = fuzzy_find_entity(needle, search_region, 0)
            if local_start >= 0:
                start = sent_start + local_start
                end = sent_start + local_end
                original_text = text[start:end]
                ents.append({"start": start, "end": end, "label": ent_label, "text": original_text, "dep": ent.get("dep", ""), "head": ent.get("head", "")})
                continue
            
            # fallback to global text search
            global_start, global_end = fuzzy_find_entity(needle, norm_text, 0)
            if global_start >= 0:
                original_text = text[global_start:global_end]
                ents.append({"start": global_start, "end": global_end, "label": ent_label, "text": original_text, "dep": ent.get("dep", ""), "head": ent.get("head", "")})
        else:
            global_start, global_end = fuzzy_find_entity(strip_accents(ent_text).lower(), norm_text, 0)
            if global_start >= 0:
                original_text = text[global_start:global_end]
                ents.append({"start": global_start, "end": global_end, "label": ent_label, "text": original_text, "dep": ent.get("dep", ""), "head": ent.get("head", "")})
    
    return {"text": text, "ents": ents}

def format_displacy_paragraphs(html):
    import re
    match = re.search(r'(<div class="entities"[^>]*>)(.*?)(</div>\s*)$', html, flags=re.DOTALL)
    if match:
        prefix = match.group(1)
        content = match.group(2)
        suffix = match.group(3)
        
        paragraphs = re.split(r'(?:<br\s*/?>\s*){2,}', content)
        wrapped_paras = []
        for i, p in enumerate(paragraphs):
            if not p.strip(): continue
            p_idx = i + 1
            label = f'<div style="font-size: 0.7rem; font-weight: bold; color: #5a9df8; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;">Paragraph {p_idx}</div>'
            wrapped_paras.append(f'<div class="displacy-para" style="border-left: 4px solid #5a9df8; padding-left: 16px; margin-bottom: 24px; background: rgba(90, 157, 248, 0.03); border-radius: 0 8px 8px 0; padding-top: 12px; padding-bottom: 12px;">{label}<p style="margin: 0;">{p}</p></div>')
        
        return prefix + '\n'.join(wrapped_paras) + suffix
    return html.replace('\n\n', '<br><br>')

def add_entity_ids(html, clean_ents):
    counter = 0
    def repl(m):
        nonlocal counter
        try:
            label = clean_ents[counter]["label"]
            dep = clean_ents[counter].get("dep", "")
            head = clean_ents[counter].get("head", "")
        except IndexError:
            label = "UNKNOWN"
            dep = ""
            head = ""
        counter += 1
        return f'<mark{m.group(1)}class="entity" id="ent-{counter}" data-entity="{label}" data-dep="{dep}" data-head="{head}"{m.group(2)} tabindex="-1">'
    return re.sub(r'<mark([^>]*)class="entity"([^>]*)>', repl, html)

def process_markdown(text, progress_callback=None):
    # Remove OCR markup (headings, footnote <sup>, images, LaTeX...) so it is not analysed as words
    text = clean_ocr_markdown(text)
    # Preprocess text to ensure currency symbols are separated from numbers to prevent tokenisation drops
    text = re.sub(r'(\d)([€$£])', r'\1 \2', text)
    
    # 1. Run inference
    entities, sentences, doc_spacy, tokens_info = analyse_with_gr_toolkit(text, progress_callback)
    
    # 2. NER Displacy setup
    ner_data = merged_entities_to_displacy(text, entities, sentences)
    
    # Sort and filter overlaps for displacy manual rendering
    raw_ents = sorted(ner_data["ents"], key=lambda x: x["start"])
    clean_ents = []
    last_end = -1
    for ent in raw_ents:
        if ent["start"] >= last_end:
            clean_ents.append(ent)
            last_end = ent["end"]
            
    ner_data["ents"] = clean_ents
    
    ner_html = displacy.render(ner_data, style="ent", manual=True)
    ner_html = format_displacy_paragraphs(ner_html)
    ner_html = add_entity_ids(ner_html, clean_ents)
    
    # 3. Structural Grouping
    grouped_tokens = [[] for _ in sentences]
    for tok in tokens_info:
        grouped_tokens[tok["sentence_index"]].append(tok)
        
    structured_data = defaultdict(lambda: defaultdict(list))
    for tok in tokens_info:
        p_idx = tok["paragraph_index"]
        s_idx = tok["sentence_index"]
        structured_data[p_idx][s_idx].append(tok)
        
    grouped_paragraphs = []
    for p_idx in sorted(structured_data.keys()):
        paragraph = []
        for s_idx in sorted(structured_data[p_idx].keys()):
            paragraph.append(structured_data[p_idx][s_idx])
        grouped_paragraphs.append(paragraph)
    

    
    # Consolidate entities into blocks and remap head_id dependencies
    consolidated_paragraphs = []
    global_ent_id = 1
    for para in grouped_paragraphs:
        new_para = []
        for sent in para:
            new_sent = []
            current_ent = None
            token_remap = {}
            for tok_idx, tok in enumerate(sent, 1):
                if tok["ner"] == "O":
                    if current_ent:
                        new_sent.append(current_ent)
                        current_ent = None
                    new_sent.append({"is_ent": False, "id": tok_idx, **tok})
                else:
                    prefix = tok["ner"][0]
                    label = tok["ner"][2:] if len(tok["ner"]) > 2 else tok["ner"]
                    if label == "PERSON": label = "PER"   # one label for persons across sidebar, text and maps
                    
                    if prefix in ("B", "S") or not current_ent:
                        if current_ent:
                            new_sent.append(current_ent)
                        current_ent = {
                            "is_ent": True,
                            "text": tok["text"],
                            "ner": label,
                            "dep": tok["dep"],
                            "head": tok["head"],
                            "head_id": tok["head_id"],
                            "id": tok_idx,
                            "ent_id": global_ent_id,
                            "pos": tok.get("pos", "")
                        }
                        global_ent_id += 1
                    elif prefix in ("I", "E") and current_ent:
                        if tok["text"] in (".", ",", "'"):
                            current_ent["text"] += tok["text"]
                        elif current_ent.get("ner") in ("CARDINAL", "MONEY", "QUANTITY") and current_ent["text"].endswith((".", ",", "'")):
                            current_ent["text"] += tok["text"]
                        else:
                            current_ent["text"] += " " + tok["text"]
                        token_remap[tok_idx] = current_ent["id"]
            if current_ent:
                new_sent.append(current_ent)
                
            for item in new_sent:
                if item["head_id"] in token_remap:
                    item["head_id"] = token_remap[item["head_id"]]
            new_para.append(new_sent)
        consolidated_paragraphs.append(new_para)

    grouped_entities = defaultdict(list)
    for para in consolidated_paragraphs:
        for sent in para:
            for item in sent:
                if item.get("is_ent"):
                    grouped_entities[item["ner"]].append((item["text"], item["ent_id"]))
                    
    entity_order = []
    return ner_html, grouped_tokens, grouped_paragraphs, consolidated_paragraphs, grouped_entities, entity_order


SPACY_MODELS = {}
UGARIT_NER = None

def get_spacy_model(model_name):
    import spacy
    if model_name not in SPACY_MODELS:
        SPACY_MODELS[model_name] = spacy.load(model_name)
    return SPACY_MODELS[model_name]

def get_ugarit_ner():
    global UGARIT_NER
    if UGARIT_NER is None:
        from transformers import pipeline
        dev_id = 0 if device.type in ["cuda", "mps"] else -1
        UGARIT_NER = pipeline("token-classification", model="UGARIT/grc-ner-bert", aggregation_strategy="simple", device=dev_id)
    return UGARIT_NER

def process_markdown_spacy(text, model_choice, progress_callback=None):
    text = clean_ocr_markdown(text)
    text = re.sub(r'(\d)([€$£])', r'\1 \2', text)
    # analyse_with_spacy_only runs the whole pipeline, including rendering, and returns the same
    # 6-tuple as process_markdown.
    return analyse_with_spacy_only(text, model_choice, progress_callback)

def analyse_with_spacy_only(text, model_choice, progress_callback=None):
    if model_choice == "grc_odycy_joint_trf":
        nlp_spacy = get_ancient_greek_model()
    else:
        nlp_spacy, _, _ = get_modern_greek_models()
    ugarit = get_ugarit_ner()   # UGARIT/grc-ner-bert provides the entities for this engine
        
    doc_spacy = nlp_spacy(text)
    
    tokens_info = []
    entities = []
    
    paragraph_offsets = [m.start() for m in re.finditer(r'\n\s*\n', text)]
    
    sentences = [s.text for s in doc_spacy.sents]
    total_sents = len(sentences)
    
    for sent_idx, sent in enumerate(doc_spacy.sents):
        if progress_callback:
            progress_callback(sent_idx + 1, total_sents)
            
        para_idx = sum(1 for offset in paragraph_offsets if sent.start_char >= offset)
        
        current_ent = None
        sent_start_i = sent[0].i
        
        sent_ner_results = ugarit(sent.text)
        token_tags = {tok.i: "O" for tok in sent}
        
        for ent in sent_ner_results:
            ent_start = ent["start"]
            ent_end = ent["end"]
            label = ent.get("entity_group", ent.get("entity", "MISC"))
            if label == "LABEL_0": label = "O"
            if label == "O": continue
            
            overlapping_tokens = []
            for tok in sent:
                tok_start = tok.idx - sent.start_char
                tok_end = tok_start + len(tok.text)
                if max(ent_start, tok_start) < min(ent_end, tok_end):
                    overlapping_tokens.append(tok)
            
            for i, tok in enumerate(overlapping_tokens):
                prefix = "B" if i == 0 else "I"
                token_tags[tok.i] = f"{prefix}-{label}"

        sent_tokens = list(sent)
        n_sent = len(sent_tokens)
        s_i = 0
        while s_i < n_sent:
            t_text = sent_tokens[s_i].text
            if re.match(r"^\d{1,3}$", t_text):
                s_j = s_i
                while s_j + 2 < n_sent and sent_tokens[s_j + 1].text in (".", ",") and re.match(r"^\d{3}$", sent_tokens[s_j + 2].text):
                    s_j += 2
                if s_j > s_i:
                    label = "CARDINAL"
                    token_tags[sent_tokens[s_i].i] = f"B-{label}"
                    for s_k in range(s_i + 1, s_j):
                        token_tags[sent_tokens[s_k].i] = f"I-{label}"
                    token_tags[sent_tokens[s_j].i] = f"E-{label}"
                    s_i = s_j + 1
                    continue
            s_i += 1

        s_i = 0
        while s_i < n_sent - 2:
            t0 = sent_tokens[s_i]
            t1 = sent_tokens[s_i + 1]
            t2 = sent_tokens[s_i + 2]
            t0_clean = t0.text.strip(" .")
            t2_clean = t2.text.strip(" .")
            if t1.text == "." and t0_clean.lower() not in NON_PERSON_INITIALS and t2_clean.lower() not in NON_PERSON_INITIALS:
                t0_ner = token_tags[t0.i]
                t2_ner = token_tags[t2.i]
                is_t0_abbr = bool(re.match(r"^[Α-Ωα-ωA-Za-zΆ-Ώά-ώἀ-ῼἈ-΅]{1,4}$", t0_clean))
                t0_is_per = "PER" in t0_ner
                t2_is_per = "PER" in t2_ner
                is_single_cap = bool(re.match(r"^[Α-ΩA-ZΆ-ΏἈ-΅]$", t0_clean))
                has_preceding_article = (s_i > 0 and sent_tokens[s_i - 1].text.strip(" .").lower() in GREEK_ARTICLES)
                if (is_t0_abbr and t0_is_per and t2_is_per) or (is_single_cap and t2_is_per and has_preceding_article):
                    token_tags[t0.i] = "I-PER" if token_tags[t0.i].startswith(("I-", "E-")) else "B-PER"
                    token_tags[t1.i] = "I-PER"
                    if token_tags[t2.i].startswith("S-"):
                        token_tags[t2.i] = "E-PER"
                    elif token_tags[t2.i].startswith("B-"):
                        token_tags[t2.i] = "I-PER"
                    s_i += 1
                    continue
                t0_is_gpe = "GPE" in t0_ner
                t2_is_gpe = "GPE" in t2_ner
                if is_t0_abbr and t0_is_gpe and t2_is_gpe:
                    token_tags[t0.i] = "I-GPE" if token_tags[t0.i].startswith(("I-", "E-")) else "B-GPE"
                    token_tags[t1.i] = "I-GPE"
                    if token_tags[t2.i].startswith("S-"):
                        token_tags[t2.i] = "E-GPE"
                    elif token_tags[t2.i].startswith("B-"):
                        token_tags[t2.i] = "I-GPE"
                    s_i += 1
                    continue
            s_i += 1
        
        for tok in sent:
            local_idx = tok.i - sent_start_i + 1
            local_head = tok.head.i - sent_start_i + 1
            if tok.head.i == tok.i:
                local_head = 0  
                
            head_text = tok.head.text
            
            ner_tag = token_tags[tok.i]
                
            tokens_info.append({
                "text": tok.text,
                "pos": tok.pos_,
                "tag": tok.tag_,
                "dep": tok.dep_,
                "head": head_text,
                "head_id": local_head,
                "sentence_index": sent_idx,
                "paragraph_index": para_idx,
                "ner": ner_tag
            })
            
            text_analyse = tok.text
            
            if ner_tag == "O":
                if current_ent:
                    entities.append(current_ent)
                    current_ent = None
                continue
                
            parts = ner_tag.split("-")
            prefix = parts[0]
            base = parts[-1].upper()
            if base == "PERSON": base = "PER"
            
            if prefix in ("B", "S"):
                if current_ent: 
                    entities.append(current_ent)
                current_ent = {"text": text_analyse, "label": base, "sentence_index": sent_idx, "dep": tok.dep_, "head": head_text}
                if prefix == "S":
                    entities.append(current_ent)
                    current_ent = None
            elif prefix in ("I", "E") and current_ent:
                if text_analyse in (".", ",", "/", "-", "'") or current_ent["text"].endswith((".", ",", "/", "-", "'")):
                    current_ent["text"] += text_analyse
                else:
                    current_ent["text"] += " " + text_analyse
                if prefix == "E":
                    entities.append(current_ent)
                    current_ent = None

        if current_ent:
            entities.append(current_ent)
            current_ent = None

    for e in entities:
        if e["label"] == "MONEY":
            e["text"] = e["text"].strip()
            e["text"] = e["text"].rstrip(" .,;/-")
        elif e["label"] == "DATE":
            e["text"] = e["text"].strip()
            e["text"] = e["text"].rstrip(" .,;/-")
            e["text"] = re.sub(r'\s+(από|έως|μέχρι|στις|στα|στον|στην|στο)$', '', e["text"], flags=re.IGNORECASE)
            e["text"] = re.sub(r'^(από|έως|μέχρι|στις|στα|στον|στην|στο)\s+', '', e["text"], flags=re.IGNORECASE)
            if re.match(r'^[/\-\.]', e["text"]):
                sent_text = sentences[e["sentence_index"]]
                escaped_ent = re.escape(e["text"])
                pattern = r'(?:\d+\s*)' + escaped_ent
                match = re.search(pattern, sent_text, flags=re.IGNORECASE)
                if match:
                    e["text"] = match.group(0)
        elif e["label"] == "LAW":
            e["text"] = e["text"].strip()
            e["text"] = e["text"].rstrip(" .,;/-)")

    ner_data = merged_entities_to_displacy(text, entities, sentences)
    raw_ents = sorted(ner_data["ents"], key=lambda x: x["start"])
    clean_ents = []
    last_end = -1
    for ent in raw_ents:
        if ent["start"] >= last_end:
            clean_ents.append(ent)
            last_end = ent["end"]
            
    ner_data["ents"] = clean_ents
    ner_html = displacy.render(ner_data, style="ent", manual=True)
    ner_html = format_displacy_paragraphs(ner_html)
    ner_html = add_entity_ids(ner_html, clean_ents)
    
    grouped_tokens = [[] for _ in sentences]
    for tok in tokens_info:
        grouped_tokens[tok["sentence_index"]].append(tok)
        
    structured_data = defaultdict(lambda: defaultdict(list))
    for tok in tokens_info:
        p_idx = tok["paragraph_index"]
        s_idx = tok["sentence_index"]
        structured_data[p_idx][s_idx].append(tok)
        
    grouped_paragraphs = []
    for p_idx in sorted(structured_data.keys()):
        paragraph = []
        for s_idx in sorted(structured_data[p_idx].keys()):
            paragraph.append(structured_data[p_idx][s_idx])
        grouped_paragraphs.append(paragraph)
    

    
    consolidated_paragraphs = []
    global_ent_id = 1
    for para in grouped_paragraphs:
        new_para = []
        for sent in para:
            new_sent = []
            current_ent = None
            token_remap = {}
            for tok_idx, tok in enumerate(sent, 1):
                if tok["ner"] == "O":
                    if current_ent:
                        new_sent.append(current_ent)
                        current_ent = None
                    new_sent.append({"is_ent": False, "id": tok_idx, **tok})
                else:
                    prefix = tok["ner"][0]
                    label = tok["ner"][2:] if len(tok["ner"]) > 2 else tok["ner"]
                    if label == "PERSON": label = "PER"   # one label for persons across sidebar, text and maps
                    
                    if prefix in ("B", "S") or not current_ent:
                        if current_ent:
                            new_sent.append(current_ent)
                        current_ent = {
                            "is_ent": True,
                            "text": tok["text"],
                            "ner": label,
                            "dep": tok["dep"],
                            "head": tok["head"],
                            "head_id": tok["head_id"],
                            "id": tok_idx,
                            "ent_id": global_ent_id,
                            "pos": tok.get("pos", "")
                        }
                        global_ent_id += 1
                    elif prefix in ("I", "E") and current_ent:
                        if tok["text"] in (".", ",", "'"):
                            current_ent["text"] += tok["text"]
                        elif current_ent.get("ner") in ("CARDINAL", "MONEY", "QUANTITY") and current_ent["text"].endswith((".", ",", "'")):
                            current_ent["text"] += tok["text"]
                        else:
                            current_ent["text"] += " " + tok["text"]
                        token_remap[tok_idx] = current_ent["id"]
            if current_ent:
                new_sent.append(current_ent)
                
            for item in new_sent:
                if item["head_id"] in token_remap:
                    item["head_id"] = token_remap[item["head_id"]]
            new_para.append(new_sent)
        consolidated_paragraphs.append(new_para)

    grouped_entities = defaultdict(list)
    for para in consolidated_paragraphs:
        for sent in para:
            for item in sent:
                if item.get("is_ent"):
                    grouped_entities[item["ner"]].append((item["text"], item["ent_id"]))
                    
    entity_order = []
    return ner_html, grouped_tokens, grouped_paragraphs, consolidated_paragraphs, grouped_entities, entity_order
