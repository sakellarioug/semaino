"""Preflight check: the web app starts and its basic routes answer (no NLP models needed).

Run with `pytest tests/test_preflight.py` or `python tests/test_preflight.py`.
"""
import io
import sys
import tempfile
import time
import types
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from semaino.app import create_app  # noqa: E402


def _client():
    app = create_app(data_dir=tempfile.mkdtemp(prefix="semaino_preflight_"))
    app.config.update({"TESTING": True})
    return app.test_client()


def test_index_route():
    response = _client().get('/')
    assert response.status_code == 200
    assert b"Semaino" in response.data
    assert b'accept=".pdf,.docx,.md,.txt"' in response.data


def test_status_not_found():
    response = _client().get('/status/invalid_id')
    assert response.status_code == 404


def _docx_bytes(*paragraphs):
    ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", f"<w:document {ns}><w:body>{body}</w:body></w:document>")
    return buf.getvalue()


def test_docx_upload_reaches_analysis():
    """A .docx upload is converted to text and handed to the engine (a stand-in here)."""
    received = []
    fake = types.ModuleType("semaino.ner_engine")

    def fake_process(text, *args, progress_callback=None, **kwargs):
        received.append(text)
        return ["", [], [], [], {}, []]

    fake.process_markdown = fake_process
    fake.process_markdown_spacy = fake_process
    saved = sys.modules.get("semaino.ner_engine")
    sys.modules["semaino.ner_engine"] = fake
    try:
        client = _client()
        response = client.post('/analyse', content_type='multipart/form-data', data={
            'model_choice': 'custom_el',
            'file': (io.BytesIO(_docx_bytes("Η Αθήνα.", "Ο Περικλής.")), 'Κείμενο.DOCX'),
        })
        assert response.status_code == 200, response.get_json()
        task_id = response.get_json()['task_id']
        for _ in range(100):
            if client.get(f'/status/{task_id}').get_json()['status'] != 'processing':
                break
            time.sleep(0.05)
        assert client.get(f'/status/{task_id}').get_json()['status'] == 'complete'
        assert received == ["Η Αθήνα.\n\nΟ Περικλής."], received
    finally:
        if saved is not None:
            sys.modules["semaino.ner_engine"] = saved
        else:
            sys.modules.pop("semaino.ner_engine", None)


def test_old_doc_and_broken_docx_are_rejected():
    client = _client()
    for name, payload in (('old.doc', b'\xd0\xcf\x11\xe0'), ('broken.docx', b'not a zip')):
        response = client.post('/analyse', content_type='multipart/form-data',
                               data={'file': (io.BytesIO(payload), name)})
        assert response.status_code == 400, name
        assert '.docx' in response.get_json()['error'], response.get_json()


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
