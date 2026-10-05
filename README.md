# Semaino Hybrid Greek Language Analyser

> **Local and private by design**
>
> Semaino runs entirely on your own computer. Text extraction from PDFs (OCR), the language models and the linguistic analysis all run locally: no documents or text are sent to cloud services or external AI APIs, and once the models are installed no internet connection is required. This makes Semaino well suited to legal, financial and archival material that must remain under your control.
>
> *The only exception is the optional Geographic Map, which asks for your permission before sending the place names found in the text to OpenStreetMap and loading its map tiles. Start Semaino with `semaino serve --offline-map` to remove the map entirely, so that no external requests are made.*

**Semaino Hybrid Greek Language Analyser** is a powerful web application designed for Named Entity Recognition (NER), Part-of-Speech (POS) tagging, dependency parsing, and morphological analysis of Greek texts. It provides robust support for both **Modern Greek** and **Ancient/Polytonic Greek**.

It is built for linguists, historians, legal professionals, and researchers who need deep, interactive visualisations of complex Greek texts without compromising data privacy. Whether you are extracting entities from historical archives, parsing the grammar of a polytonic manuscript, or analysing modern legal documents, Semaino provides a secure, locally-hosted workspace to examine the linguistic architecture of your documents.

![Named Entities view: highlighted entities, entity sidebar and document minimap](docs/images/named_entities.png)

## Features

- **PDF OCR Pipeline**: Seamlessly upload PDFs which are converted to highly accurate markdown via **Marker OCR** (utilizing Surya vision models and accelerated by `llama.cpp` on Apple Metal/NVIDIA CUDA). Users can inspect the OCR'd text side-by-side with the original PDF before proceeding to NLP analysis.
- **Word Document Import**: `.docx` files are read directly (no OCR, no extra dependencies), keeping their paragraphs; table rows are imported as lines of text. Plain `.md` and `.txt` files are also accepted.
- **Automatic OCR Clean-up**: Markdown and HTML left over from OCR (heading marks, footnote numbers in `<sup>`, image links, emphasis markers, tables, LaTeX-encoded Greek letters) is stripped before analysis, so it is never tagged or displayed as words. Ordinals such as `6ος` are kept.
- **Paragraph-Aware Analysis**: A blank line always ends a sentence, so headings stay separate from body text and the results keep the paragraph structure of your document.
- **Multi-Engine NLP Analysis**:
  - **Modern Greek (Innovative Custom Pipeline)**: Recommended for most Greek texts from the 19th century to the present. While many widely used NLP toolkits (including spaCy) currently lack native transformer-level models for Modern Greek, this custom pipeline bridges that industry-wide gap. It combines a state-of-the-art open-source BERT transformer (via `gr-nlp-toolkit`) for deep-learning accuracy in NER, POS, and dependency parsing, seamlessly integrated with the structural integrity and production-grade architecture of the **spaCy** ecosystem. The original capitalisation and accents (including polytonic) are restored in the results.
  - **Ancient Greek**: Uses the `grc_odycy_joint_trf` spaCy model with `UGARIT/grc-ner-bert` for specialized Ancient Greek named entity recognition.

### Why Deterministic NLP over LLMs?
While Generative AI and LLMs (like ChatGPT) are incredibly powerful for text generation, they suffer from architectural limitations when it comes to **precise spatial extraction**:
1. **Positional Hallucination**: Because LLMs process text as subword tokens rather than raw characters, they routinely hallucinate character offsets and fail at deterministic positional mapping (e.g., trying to find the exact start/end character index of a word in a 100-page document).
2. **Disconnected Output**: LLMs output disconnected JSON objects. Without strict indices, it is nearly impossible to cleanly map the extracted entities back to their original physical coordinates in the text.
3. **Reproducibility**: Running the analysis engine twice on the exact same text will yield the exact same results. LLMs introduce statistical randomness, meaning your outputs can change unpredictably between runs.

**Semaino's Advantage**: Semaino relies on mathematically strict, deterministic NLP architectures (via **spaCy**, **gr-nlp-toolkit**, and **OdyCy**). When Semaino extracts an entity, it guarantees the absolute token index, start/end character offset, sentence number, and paragraph number. 
This strict positional mapping is the *only* reason Semaino's rich UI is possible—powering the clickable sidebar navigation, the spatial document minimap, and the physical SVG dependency arrows that draw exactly from word to word!

- **Rich Results UI**: A Material Design 3 dark-themed workspace offering:
  - **Named Entities**: interactive entity highlighting with a sidebar for quick navigation (click an entity to jump to it), category filters, an adjustable **text size**, and **interactive dependency arrows** drawn between words on hover.
  - **Document minimap**: a scaled copy of the text (or a block view of every word) that follows your scrolling; click it to jump to that part of the document.
  - **Dependency & POS**: a word-by-word grammatical reference in which every word carries its **paragraph.sentence.word** index, part of speech and dependency relation.
  - **Entity Mindmap**: an interactive network graph of how entities and key words relate, with several filters.
  - **Geographic Map**: places, facilities and geopolitical entities plotted on an interactive map. This is the only feature that goes online: it asks before contacting OpenStreetMap, and `semaino serve --offline-map` removes it entirely.
- **Save/Load Sessions**: Process long texts once, save the analyses as JSON files, and reload them instantly without re-running the models.

## Screenshots

*All screenshots use the generic sample text [`examples/sample_history.md`](examples/sample_history.md).*

<table width="100%">
  <tr>
    <td width="50%" valign="top">
      <img src="docs/images/landing.png" alt="Landing page with upload area, engine choice and saved analyses"><br>
      <b>Upload</b> a PDF, Word or text file and choose the engine
    </td>
    <td width="50%" valign="top">
      <img src="docs/images/ocr_inspect.png" alt="OCR inspection: original PDF beside the extracted text"><br>
      <b>Inspect OCR</b> output beside the original PDF
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <img src="docs/images/named_entities.png" alt="Named Entities view"><br>
      <b>Named Entities</b> with sidebar navigation and minimap
    </td>
    <td width="50%" valign="top">
      <img src="docs/images/dependency_arrows.png" alt="Dependency arrows drawn on hover, with larger text and the block minimap"><br>
      <b>Dependency arrows</b>, adjustable text size and block minimap
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <img src="docs/images/pos_dependency.png" alt="POS and dependency reference with paragraph.sentence.word indices"><br>
      <b>Dependency & POS</b> reference with word indices
    </td>
    <td width="50%" valign="top">
      <img src="docs/images/mindmap.png" alt="Entity mindmap network graph"><br>
      <b>Entity Mindmap</b>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <img src="docs/images/geographic_map.png" alt="Geographic map of the places mentioned in the text"><br>
      <b>Geographic Map</b> of places in the text
    </td>
    <td width="50%" valign="top">
    </td>
  </tr>
</table>

## User Guide
For a detailed walkthrough on how to use the interface, interpret the NLP results, and choose the correct language models, please read the [User Guide](docs/USER_GUIDE.md).

## Requirements
- **Python 3.10+**

## Installation

Semaino is packaged as a standard Python application. Hardware acceleration (Apple Metal or NVIDIA CUDA) is highly recommended for OCR and NLP inference.

### 1. Install Semaino
It's recommended to install Semaino in an isolated virtual environment (e.g. using `venv`, `uv`, or `pipx`).
```bash
# Clone the repository
git clone https://github.com/sakellarioug/Semaino.git
cd Semaino

# Create virtual environment and install Semaino
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

#### Setting up OCR (Optional but Recommended for PDFs)
Due to conflicting deep learning dependencies between modern OCR vision models and established NLP frameworks, **Marker OCR must be installed in a separate environment**. The easiest way to do this globally is using `pipx`:
```bash
pipx install marker-pdf
```
Semaino will automatically detect the `marker_single` binary from your system path and use it for PDF processing (or set `SEMAINO_MARKER_BIN` to its location).

### 2. Download Models
Semaino uses several AI models. Download them once using the built-in CLI:
```bash
semaino download-models
```

*(Note for Linux/Windows users with NVIDIA GPUs: To ensure models run on your GPU rather than CPU, install the CUDA-enabled version of PyTorch before downloading models: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`)*

You can check your setup at any time with:
```bash
semaino doctor
```

### 3. Usage
Start the web server:
```bash
semaino serve
```
Open your browser and navigate to `http://127.0.0.1:5001`. Try it with one of the files in `examples/`, e.g. `examples/sample_history.md`.

| Option | Purpose |
|---|---|
| `--port 5001` | Port to listen on |
| `--host 127.0.0.1` | Interface to bind to |
| `--data-dir PATH` | Where uploads and saved analyses are kept (default `~/.semaino`) |
| `--offline-map` | Remove the Geographic Map so no external requests are ever made |

**Offline use:**
Once the models have been downloaded, Semaino needs no internet connection. To make sure no external requests are ever made, start it without the Geographic Map:
```bash
semaino serve --offline-map
```

> After upgrading Semaino, restart `semaino serve` and re-analyse your documents to benefit from pipeline improvements. Saved analyses still open, but keep the results they were saved with.

## Running the Tests
```bash
python tests/test_preflight.py         # the web app starts, its basic routes answer and uploads (incl. .docx) are accepted
python tests/test_docx_import.py      # Word .docx text extraction
python tests/test_text_cleaning.py     # OCR clean-up rules
python tests/test_sentence_rules.py    # paragraph and sentence boundaries
python tests/test_ui_smoke.py          # results page in headless Chrome (needs Google Chrome or Chromium)
```
All five also run under `pytest`. Once the models are downloaded, set `HF_HUB_OFFLINE=1` to avoid network checks.

## Project Structure
- `src/semaino/`: Python package containing the web app, CLI, NLP engine, OCR clean-up, Word (`.docx`) import and templates.
- `tests/`: Unit tests and the end-to-end UI smoke test.
- `docs/`: User Guide and screenshots.
- `examples/`: Sample texts and PDFs.

## Author
**Georgios Sakellariou**
- GitHub: [@sakellarioug](https://github.com/sakellarioug)

*Author's Note: This project was developed with the assistance of AI coding tools to accelerate prototyping, testing, and UI implementation. The underlying NLP logic relies strictly on the deterministic open-source models listed below.*

## License
This project is licensed under the **Apache-2.0 License**. See the `LICENSE` file for details.

## Acknowledgments
Please see the `ACKNOWLEDGMENTS.md` file for a detailed list of the incredible open-source projects, models, and research groups that made this tool possible.
