"""End-to-end UI smoke test for the results page, driven through headless Chrome.

Why this exists: the results page broke completely without any server-side error (an unclosed
<template> silently disabled every <script>). Static checks cannot catch that kind of failure,
so this test loads the real page from a real Flask server and exercises it like a user:
sidebar navigation, highlighting, category filters, both minimaps, dependency arrows and tabs.

Usage
    pytest tests/test_ui_smoke.py               # synthetic document, no NLP models needed
    python tests/test_ui_smoke.py               # same, with a readable report
    python tests/test_ui_smoke.py --md doc.md   # run the real NLP pipeline on a document first

Requirements: Google Chrome or Chromium (set SEMAINO_CHROME to override the path).
All external hosts are blocked inside the browser, so the test never contacts OpenStreetMap.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semaino import app as app_module  # noqa: E402

# --------------------------------------------------------------------------------------
# Browser-side probe. Runs in <head> (always executed), then simulates the user's actions
# and POSTs a JSON report back to the test server.
# --------------------------------------------------------------------------------------
PROBE = r"""
<script>
window.__uiErrors = [];
window.addEventListener('error', e => window.__uiErrors.push(String(e.message)));
window.addEventListener('unhandledrejection', e => window.__uiErrors.push('promise: ' + String(e.reason)));
window.addEventListener('load', async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const checks = {}, details = {};
  const check = (name, ok, detail) => { checks[name] = !!ok; if (detail !== undefined) details[name] = detail; };
  // Wait until smooth scrolling has finished (scrollTop unchanged for 3 samples), max 4 s
  const settle = async el => { let last = -1, same = 0;
    for (let i = 0; i < 40 && same < 3; i++) { await sleep(100); same = el.scrollTop === last ? same + 1 : 0; last = el.scrollTop; } };
  const isClear = el => ['rgba(0, 0, 0, 0)', 'transparent'].includes(getComputedStyle(el).backgroundColor);
  try {
    await sleep(500);
    const view = document.getElementById('view-ner');
    const canvas = view.querySelector('.custom-ner-canvas');
    const miniText = view.querySelector('.minimap-content-text');
    const vp = view.querySelector('.minimap-viewport');

    check('page_functions_defined',
      ['switchTab', 'scrollToEntity', 'toggleCategory', 'toggleMinimapView', 'toggleMindmap']
        .every(f => typeof window[f] === 'function'));
    check('no_scripts_trapped_in_template',
      [...document.querySelectorAll('template')].every(t => !t.content.querySelector('script')));
    const seen = new Set(), dups = new Set();
    document.querySelectorAll('[id]').forEach(e => (seen.has(e.id) ? dups.add(e.id) : seen.add(e.id)));
    check('no_duplicate_ids', dups.size === 0, [...dups].slice(0, 5));

    // Sidebar: every entity link has a target in the main text
    const links = [...document.querySelectorAll('.sidebar a[onclick^="scrollToEntity"]')];
    const idOf = a => a.getAttribute('onclick').match(/\d+/)[0];
    const target = id => canvas.querySelector(`[data-ent-id="${id}"]`);
    const missing = links.filter(a => !target(idOf(a))).length;
    check('every_sidebar_entity_has_target', links.length > 0 && missing === 0, { links: links.length, missing });

    // Click an entity far down the document
    const link = links[Math.floor(links.length * 0.7)];
    const eid = idOf(link);
    link.click();
    await settle(view);
    const t = target(eid);
    const centre = t.getBoundingClientRect().top + t.offsetHeight / 2 - view.getBoundingClientRect().top;
    const atBottom = view.scrollTop >= view.scrollHeight - view.clientHeight - 2;
    check('sidebar_click_scrolls_to_entity',
      view.scrollTop > 0 && (Math.abs(centre - view.clientHeight / 2) < view.clientHeight * 0.15 || atBottom),
      { centre: Math.round(centre), viewHalf: view.clientHeight / 2, scrollTop: Math.round(view.scrollTop) });
    check('clicked_entity_highlighted_in_text_and_minimap',
      t.classList.contains('entity-selected') &&
      !!miniText.querySelector(`.entity-selected[data-ent-id="${eid}"]`) &&
      view.querySelectorAll('.entity-selected').length === view.querySelectorAll(`[data-ent-id="${eid}"]`).length);
    check('page_layout_not_shifted',
      ['.container', '.main'].every(s => document.querySelector(s).scrollTop === 0) &&
      document.scrollingElement.scrollTop === 0);

    // The minimap's viewport box frames the part of the text that is on screen
    const mm = miniText.querySelector(`[data-ent-id="${eid}"]`).getBoundingClientRect();
    const box = vp.getBoundingClientRect();
    const mid = mm.top + mm.height / 2;
    check('minimap_viewport_frames_visible_text', mid >= box.top - 2 && mid <= box.bottom + 2,
      { entityMid: Math.round(mid), boxTop: Math.round(box.top), boxBottom: Math.round(box.bottom) });

    // The minimap follows scrolling
    const snap = () => vp.style.top + '|' + miniText.style.transform;
    view.scrollTo({ top: 0 }); await settle(view); const s0 = snap();
    view.scrollTo({ top: view.scrollHeight }); await settle(view); const s1 = snap();
    check('text_minimap_follows_scroll', s0 !== s1, { top: s0, bottom: s1 });

    // Category filters update the main text, the text minimap and the block minimap
    const filterFailures = [];
    for (const cb of document.querySelectorAll('.category-toggle')) {
      const cat = cb.dataset.category;
      const main = [...canvas.querySelectorAll(`mark[data-entity="${cat}"]`)];
      const mini = [...miniText.querySelectorAll(`mark[data-entity="${cat}"]`)];
      const blocks = [...view.querySelectorAll(`.minimap-content-block [data-entity="${cat}"]`)];
      cb.click(); await sleep(350);   // marks have a 0.2 s colour transition
      if (!main.length || mini.length !== main.length || blocks.length !== main.length ||
          !main.every(isClear) || !mini.every(isClear) ||
          !blocks.every(b => b.classList.contains('filter-hidden'))) filterFailures.push(cat + ':hide');
      cb.click(); await sleep(350);   // marks have a 0.2 s colour transition
      if (main.some(isClear) || mini.some(isClear) || blocks.some(b => b.classList.contains('filter-hidden')))
        filterFailures.push(cat + ':restore');
    }
    check('category_filters_update_text_and_both_minimaps', filterFailures.length === 0, filterFailures);

    // Block view
    const toggleBtn = view.querySelector('.minimap-container button');
    const block = view.querySelector('.minimap-content-block');
    toggleBtn.click(); await sleep(120);
    check('block_minimap_shows', getComputedStyle(block).display !== 'none' && block.querySelectorAll('.mm-word').length > 0);
    view.scrollTo({ top: 0 }); await settle(view); const b0 = vp.style.top + block.style.transform;
    view.scrollTo({ top: view.scrollHeight }); await settle(view); const b1 = vp.style.top + block.style.transform;
    check('block_minimap_follows_scroll', b0 !== b1, { top: b0, bottom: b1 });
    toggleBtn.click(); await sleep(120);
    check('text_minimap_restored_after_toggle',
      getComputedStyle(miniText).display !== 'none' && getComputedStyle(block).display === 'none');

    // Interactive dependency arrows
    view.scrollTo({ top: 0 }); await settle(view);
    document.getElementById('toggle-interactive-arrows').click();
    const word = [...canvas.querySelectorAll('.custom-ner-word')].find(w => {
      const head = document.getElementById(w.dataset.headId); return head && head !== w;
    });
    word.dispatchEvent(new MouseEvent('mouseenter')); await sleep(60);
    const arrow = document.getElementById('custom-arrow-path');
    check('dependency_arrow_drawn_on_hover', arrow.style.opacity === '1' && !!arrow.getAttribute('d'));
    word.dispatchEvent(new MouseEvent('mouseleave'));
    document.getElementById('toggle-interactive-arrows').click();

    // Text size: the main text grows, the minimap clone follows, and the minimap stays in sync
    const fontPx = el => parseFloat(getComputedStyle(el).fontSize);
    const mirrorCanvas = () => miniText.querySelector('.minimap-mirror-canvas');
    const px0 = fontPx(canvas);
    document.getElementById('text-size-up').click();
    document.getElementById('text-size-up').click();
    await sleep(100);
    const px1 = fontPx(canvas);
    link.click(); await settle(view);
    const mm2 = miniText.querySelector(`[data-ent-id="${eid}"]`).getBoundingClientRect();
    const box2 = vp.getBoundingClientRect();
    const mid2 = mm2.top + mm2.height / 2;
    check('text_size_control_resizes_text_and_minimap',
      px1 > px0 && mirrorCanvas() && mirrorCanvas().style.fontSize === canvas.style.fontSize &&
      document.getElementById('text-size-value').textContent !== '100%' &&
      mid2 >= box2.top - 2 && mid2 <= box2.bottom + 2 &&
      !!miniText.querySelector(`.entity-selected[data-ent-id="${eid}"]`),
      { px0, px1, label: document.getElementById('text-size-value').textContent,
        entityMid: Math.round(mid2), boxTop: Math.round(box2.top), boxBottom: Math.round(box2.bottom) });
    setTextSize(1.05);   // back to the default (also resets the remembered value)
    check('text_size_reset_to_default', Math.abs(fontPx(canvas) - px0) < 0.5 &&
      document.getElementById('text-size-value').textContent === '100%');

    // Tabs
    window.confirm = () => true;
    const shown = id => getComputedStyle(document.getElementById('view-' + id)).display !== 'none';
    document.getElementById('tab-dep').click(); await sleep(100);
    check('tab_dependency_pos_shows', shown('dep') && !shown('ner') &&
      document.querySelectorAll('#view-dep .pos-mark').length > 0);

    // Every word in the dependency view carries paragraph.sentence.word, matching the NER numbering
    const posMarks = [...document.querySelectorAll('#view-dep .pos-mark')];
    const badLoc = posMarks.filter(m => !/^\d+\.\d+\.\d+$/.test(m.dataset.loc || '') ||
      m.querySelector('.pos-index').textContent.trim() !== m.dataset.loc);
    let compared = 0, mismatched = [];
    for (const m of posMarks) {
      const el = document.getElementById('cner-w-' + m.dataset.loc.replaceAll('.', '-'));
      if (!el || el.classList.contains('tp-ent')) continue;   // entities are merged into one mark
      compared++;
      const want = m.querySelector('span:not(.pos-index)').textContent.trim();
      if (el.textContent.trim() !== want) mismatched.push([m.dataset.loc, want, el.textContent.trim()]);
    }
    check('dependency_view_has_matching_indices',
      posMarks.length > 0 && badLoc.length === 0 && compared > 0 && mismatched.length === 0 &&
      document.querySelectorAll('#view-dep .pos-paragraph-group').length ===
        document.querySelectorAll('.custom-ner-canvas .displacy-para').length,
      { marks: posMarks.length, badLoc: badLoc.length, compared, mismatched: mismatched.slice(0, 3) });

    document.getElementById('dep-show-arrows').click(); await sleep(100);
    check('dependency_view_shortcut_enables_arrows', shown('ner') &&
      document.getElementById('toggle-interactive-arrows').checked);
    document.getElementById('toggle-interactive-arrows').click();
    document.getElementById('tab-mindmap').click(); await sleep(2000);
    check('tab_mindmap_draws_graph', shown('mindmap') && !!document.querySelector('#mindmap-network canvas'));
    const geoTab = document.getElementById('tab-geomap');
    details.geomap_tab_present = !!geoTab;
    if (geoTab) {
      geoTab.click(); await sleep(600);
      const gm = document.getElementById('geographic-map');
      check('tab_geomap_initialises_visible_map', shown('geomap') && gm.classList.contains('leaflet-container') &&
        gm.clientWidth > 0 && gm.clientHeight > 0);
    }
    links[0].click(); await sleep(300);
    check('sidebar_click_from_another_tab_returns_to_text', shown('ner') &&
      document.getElementById('tab-ner').classList.contains('active'));

    check('no_javascript_errors', window.__uiErrors.length === 0, window.__uiErrors.slice(0, 5));
  } catch (e) {
    check('probe_completed', false, String(e && e.stack || e));
  }
  fetch('/__ui_report', { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ checks, details }) });
});
</script>
"""


# --------------------------------------------------------------------------------------
# Test data
# --------------------------------------------------------------------------------------
def synthetic_analysis(paragraphs=40):
    """A pipeline-shaped result with enough text to scroll, every entity type the UI colours,
    multi-word entities, and an entity tagged PUNCT (previously never rendered)."""
    words = ["Ἡ", "ἐπιτροπή", "συνεδρίασε", "χθες", "καί", "ἀπεφάσισε", "τήν", "ἀποστολή",
             "ἐγγράφων", "πρός", "τό", "Ὑπουργεῖο", "μετά", "ἀπό", "μακρά", "συζήτηση"]
    ents = [("PER", "Ἰωάννης Καποδίστριας"), ("ORG", "Ἐθνική Τράπεζα"), ("GPE", "Σάμος"),
            ("DATE", "28 Ἰουλίου 1824"), ("LOC", "Αἰγαῖο"), ("LAW", "ν. 4548/2018"), ("NORP", "Ἕλληνες")]
    consolidated, grouped, sentences, ent_id = [], {}, [], 1
    for p in range(paragraphs):
        para = []
        for s in range(3):
            sent = []
            for i in range(1, 26):
                if i % 6 == 0:
                    label, text = ents[(p + s + i) % len(ents)]
                    pos = "PUNCT" if (p, s, i) == (5, 1, 6) else "PROPN"
                    sent.append({"is_ent": True, "id": i, "ent_id": ent_id, "text": text, "ner": label,
                                 "pos": pos, "dep": "nmod", "head": words[2], "head_id": 3})
                    grouped.setdefault(label, []).append([text, ent_id])
                    ent_id += 1
                else:
                    sent.append({"is_ent": False, "id": i, "text": words[i % len(words)], "ner": "O",
                                 "pos": "NOUN" if i != 3 else "VERB", "dep": "root" if i == 3 else "nmod",
                                 "head": words[2], "head_id": 0 if i == 3 else 3})
            sent.append({"is_ent": False, "id": 26, "text": ".", "ner": "O", "pos": "PUNCT",
                         "dep": "punct", "head": words[2], "head_id": 3})
            para.append(sent)
            sentences.append(sent)
        consolidated.append(para)
    return ["", sentences, consolidated, consolidated, grouped, []]


def pipeline_analysis(md_path):
    from semaino.ner_engine import process_markdown
    result = process_markdown(Path(md_path).read_text(encoding="utf-8"))
    return json.loads(json.dumps(result, default=list))


# --------------------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------------------
def find_chrome():
    candidates = [os.environ.get("SEMAINO_CHROME"),
                  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                  "/Applications/Chromium.app/Contents/MacOS/Chromium",
                  shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chromium-browser")]
    return next((c for c in candidates if c and Path(c).exists()), None)


def run_ui_probe(data, online_map=True, chrome=None, timeout=120):
    """Serve `data` from a real Flask server, load /results in headless Chrome, return (report, html).

    Chrome runs in real time (virtual-time mode skips animation frames and scroll events, which
    the page relies on). The in-page probe POSTs its report back to this server when finished.
    """
    from flask import request as flask_request
    from werkzeug.serving import make_server
    chrome = chrome or find_chrome()
    received, done = {}, threading.Event()
    with tempfile.TemporaryDirectory() as tmp:
        app = app_module.create_app(data_dir=tmp, online_map=online_map)

        @app.after_request
        def inject_probe(response):
            if response.mimetype == "text/html" and b'id="view-ner"' in response.get_data():
                response.set_data(response.get_data(as_text=True).replace("</head>", PROBE + "</head>", 1))
            return response

        def receive_report():
            received.update(flask_request.get_json(force=True))
            done.set()
            return "", 204
        app.add_url_rule("/__ui_report", "ui_report", receive_report, methods=["POST"])

        task_id = uuid.uuid4().hex
        app_module.tasks[task_id] = {"status": "complete", "progress": 1, "total": 1, "data": data, "error": None}
        server = make_server("127.0.0.1", 0, app, threaded=True)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_port}/results/{task_id}"
        browser = None
        try:
            page = app.test_client().get(f"/results/{task_id}").get_data(as_text=True)
            browser = subprocess.Popen(
                [chrome, "--headless=new", "--disable-gpu", "--no-first-run", "--disable-extensions",
                 "--window-size=1440,900", "--remote-debugging-port=0",   # keeps headless Chrome open
                 "--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE 127.0.0.1",   # never reach external hosts
                 url],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            done.wait(timeout)
        finally:
            if browser:
                browser.terminate()
                try:
                    browser.wait(10)
                except subprocess.TimeoutExpired:
                    browser.kill()
            server.shutdown()
            app_module.tasks.pop(task_id, None)
    report = received or {"checks": {"probe_completed": False}, "details": {"probe_completed": "no report received"}}
    return report, page


def offline_checks(page_html):
    return {
        "offline_mode_has_no_geomap_tab": 'id="tab-geomap"' not in page_html,
        "offline_mode_contains_no_external_endpoints":
            not any(s in page_html for s in ("openstreetmap.org", "nominatim", "leaflet.js")),
    }


def run_all(data):
    online, _ = run_ui_probe(data, online_map=True)
    offline, offline_html = run_ui_probe(data, online_map=False)
    checks = dict(online["checks"])
    checks.update({f"offline: {k}": v for k, v in offline["checks"].items()})
    checks.update(offline_checks(offline_html))
    return checks, {"online": online["details"], "offline": offline["details"]}


# --------------------------------------------------------------------------------------
# pytest entry point (synthetic data: fast, no NLP models required)
# --------------------------------------------------------------------------------------
def test_results_page_ui():
    if not find_chrome():
        import pytest
        pytest.skip("Chrome/Chromium not found (set SEMAINO_CHROME)")
    checks, details = run_all(synthetic_analysis())
    failed = {k: details["online"].get(k, details["offline"].get(k.replace("offline: ", "")))
              for k, ok in checks.items() if not ok}
    assert not failed, f"UI checks failed: {json.dumps(failed, ensure_ascii=False, indent=2)}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--md", help="Run the real NLP pipeline on this Markdown/text file first")
    src.add_argument("--json", help="Use a pipeline result previously saved as JSON")
    args = parser.parse_args()
    if not find_chrome():
        sys.exit("Chrome/Chromium not found (set SEMAINO_CHROME)")
    data = (pipeline_analysis(args.md) if args.md else
            json.loads(Path(args.json).read_text(encoding="utf-8")) if args.json else synthetic_analysis())
    checks, details = run_all(data)
    width = max(map(len, checks))
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name.ljust(width)}")
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        print("\nDetails:", json.dumps(details, ensure_ascii=False, indent=2))
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    sys.exit(1 if failed else 0)
