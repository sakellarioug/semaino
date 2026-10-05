"""Semaino Flask web application."""
import json
import os
import re
import shutil
import subprocess
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from flask import (Flask, flash, jsonify, redirect, render_template, request,
                   send_from_directory, url_for)
from werkzeug.utils import secure_filename

from . import __version__

# In-memory registry of background OCR/NLP tasks, keyed by task id.
tasks = {}

_HEX_ID = re.compile(r"^[0-9a-f]{32}$")


def default_data_dir() -> Path:
    """Where uploads and saved analyses live. Override with SEMAINO_DATA_DIR."""
    return Path(os.environ.get("SEMAINO_DATA_DIR", Path.home() / ".semaino")).expanduser()


def find_marker_bin():
    """Locate Marker's `marker_single` CLI."""
    import os
    import shutil
    from pathlib import Path
    
    if os.environ.get("SEMAINO_MARKER_BIN"):
        return os.environ.get("SEMAINO_MARKER_BIN")
        
    pipx_bin = str(Path.home() / ".local" / "bin" / "marker_single")
    if os.path.exists(pipx_bin):
        return pipx_bin
        
    return shutil.which("marker_single")


def _ensure_entity_ids(data):
    """Upgrade analyses saved by older versions in place.

    `data` is the pipeline result list: [ner_html, grouped_tokens, grouped_paragraphs,
    consolidated_paragraphs, grouped_entities, entity_order]. Older saves lack `ent_id` on
    entity tokens and may use the label PERSON, so the sidebar links would not match the text.
    Assign ids where missing and rebuild the sidebar list from the same tokens.
    """
    if not isinstance(data, list) or len(data) < 5:
        return
    next_id = 1
    grouped = {}
    for para in data[3]:
        for sent in para:
            for tok in sent:
                if not tok.get("is_ent"):
                    continue
                if tok.get("ner") == "PERSON":
                    tok["ner"] = "PER"
                if "ent_id" not in tok:
                    tok["ent_id"] = next_id
                next_id = max(next_id, tok["ent_id"]) + 1
                grouped.setdefault(tok["ner"], []).append([tok["text"], tok["ent_id"]])
    data[4] = grouped


def create_app(data_dir=None, online_map=True) -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24).hex())

    data_dir = Path(data_dir) if data_dir else default_data_dir()
    app.config["UPLOAD_FOLDER"] = str(data_dir / "uploads")
    app.config["SAVED_ANALYSES_FOLDER"] = str(data_dir / "saved_analyses")
    app.config["ONLINE_MAP"] = online_map
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["SAVED_ANALYSES_FOLDER"], exist_ok=True)

    @app.context_processor
    def inject_globals():
        return {"semaino_version": __version__, "online_map": app.config["ONLINE_MAP"]}

    @app.route('/')
    def index():
        return render_template('index.html', ocr_available=find_marker_bin() is not None)

    @app.route('/analyse', methods=['POST'])
    def analyse():
        text = None
        source_name = 'Text Input'
        if 'text' in request.form and request.form['text'].strip():
            text = request.form['text']
        elif 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            name = file.filename.lower()
            if name.endswith(('.md', '.txt')):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                except UnicodeDecodeError:
                    return jsonify({'error': 'The text file is not UTF-8 encoded. '
                                             'Please save it as UTF-8 and try again.'}), 400
                source_name = file.filename
            elif name.endswith('.docx'):
                from .docx_import import DocxError, docx_to_text
                try:
                    text = docx_to_text(file.stream.read())
                except DocxError as e:
                    return jsonify({'error': str(e)}), 400
                if not text.strip():
                    return jsonify({'error': 'No text was found in this .docx document.'}), 400
                source_name = file.filename
            elif name.endswith('.doc'):
                return jsonify({'error': "The old .doc format isn't supported. "
                                         "Please open it in Word and save it as .docx."}), 400

        if not text:
            return jsonify({'error': 'No text or valid .md, .txt or .docx file provided.'}), 400

        model_choice = request.form.get('model_choice', 'custom_el')

        task_id = uuid.uuid4().hex
        tasks[task_id] = {'status': 'processing', 'progress': 0, 'total': 0, 'data': None,
                          'error': None, 'filename': source_name}

        def process_task(tid, t, mc):
            # Imported lazily so the web UI starts instantly; models load on first use.
            from .ner_engine import process_markdown, process_markdown_spacy

            def progress_callback(current, total):
                tasks[tid]['progress'] = current
                tasks[tid]['total'] = total

            try:
                if mc == 'custom_el':
                    res = process_markdown(t, progress_callback=progress_callback)
                else:
                    res = process_markdown_spacy(t, mc, progress_callback=progress_callback)
                tasks[tid]['data'] = res
                tasks[tid]['status'] = 'complete'
            except Exception as e:
                traceback.print_exc()
                tasks[tid]['status'] = 'error'
                tasks[tid]['error'] = str(e)

        threading.Thread(target=process_task, args=(task_id, text, model_choice), daemon=True).start()
        return jsonify({'task_id': task_id})

    @app.route('/convert_pdf', methods=['POST'])
    def convert_pdf():
        marker_bin = find_marker_bin()
        if not marker_bin:
            return jsonify({'error': "PDF support is not installed. Install Marker separately "
                                     "(e.g. `pipx install marker-pdf`) and restart Semaino. "
                                     "See the README, section 'PDF / OCR support'."}), 501
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded.'}), 400
        file = request.files['file']
        if file.filename == '' or not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'No selected file or invalid type.'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        task_id = uuid.uuid4().hex
        tasks[task_id] = {'status': 'processing', 'progress': 0, 'total': 1, 'data': None, 'error': None, 'filename': filename}

        def process_pdf_task(tid, fp):
            try:
                output_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"out_{tid}")
                os.makedirs(output_dir, exist_ok=True)

                proc = subprocess.Popen(
                    [marker_bin, fp, '--output_dir', output_dir],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                # regex for tqdm progress like: 50%|██████████| 5/10
                progress_regex = re.compile(r"(\d+)/(\d+)")

                buffer = ""
                while True:
                    char = proc.stdout.read(1)
                    if not char:
                        break
                    if char in ('\r', '\n'):
                        if buffer:
                            match = progress_regex.search(buffer)
                            if match:
                                current, total = int(match.group(1)), int(match.group(2))
                                if total > 0:
                                    tasks[tid]['progress'] = current
                                    tasks[tid]['total'] = total
                        buffer = ""
                    else:
                        buffer += char

                proc.wait()

                if proc.returncode != 0:
                    tasks[tid]['status'] = 'error'
                    tasks[tid]['error'] = f"Marker OCR failed with exit code {proc.returncode}"
                    return

                base_name = os.path.splitext(os.path.basename(fp))[0]
                md_path = os.path.join(output_dir, base_name, f"{base_name}.md")

                if os.path.exists(md_path):
                    with open(md_path, 'r', encoding='utf-8') as f:
                        markdown_text = f.read()
                else:
                    markdown_text = "Error: Markdown file was not generated by marker."

                tasks[tid]['data'] = markdown_text
                tasks[tid]['progress'] = 1
                tasks[tid]['status'] = 'complete'
            except Exception as e:
                traceback.print_exc()
                tasks[tid]['status'] = 'error'
                tasks[tid]['error'] = str(e)

        threading.Thread(target=process_pdf_task, args=(task_id, filepath), daemon=True).start()
        return jsonify({'task_id': task_id})

    @app.route('/uploads/<filename>')
    def uploaded_file(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    @app.route('/status/<task_id>')
    def status(task_id):
        if task_id in tasks:
            t = tasks[task_id]
            return jsonify({'status': t['status'], 'progress': t['progress'],
                            'total': t['total'], 'error': t['error']})
        return jsonify({'status': 'not_found'}), 404

    @app.route('/inspect/<task_id>')
    def inspect(task_id):
        if task_id not in tasks:
            flash("Task not found.")
            return redirect(url_for('index'))
        task_info = tasks[task_id]
        if task_info['status'] != 'complete':
            flash("PDF conversion is not complete or failed.")
            return redirect(url_for('index'))
        return render_template('inspect.html', markdown_text=task_info['data'],
                               pdf_filename=task_info.get('filename'))

    @app.route('/results/<task_id>')
    def results(task_id):
        if task_id not in tasks:
            flash("Analysis task not found. Please try again.")
            return redirect(url_for('index'))
        task_info = tasks[task_id]
        if task_info['status'] != 'complete':
            flash("Analysis is either not complete or failed.")
            return redirect(url_for('index'))
        data = task_info['data']
        return render_template(
            'results.html',
            ner_html=data[0],
            grouped_tokens=data[1],
            grouped_paragraphs=data[2],
            consolidated_paragraphs=data[3],
            grouped_entities=data[4],
            entity_order=data[5],
            task_id=task_id,
        )

    @app.route('/save_analysis', methods=['POST'])
    def save_analysis():
        data = request.get_json(silent=True) or {}
        task_id = data.get('task_id')
        title = data.get('title', 'Untitled Analysis')

        if not task_id or task_id not in tasks:
            return jsonify({'error': 'Invalid or missing task_id'}), 400
        task_info = tasks[task_id]
        if task_info['status'] != 'complete':
            return jsonify({'error': 'Cannot save incomplete analysis'}), 400

        analysis_id = uuid.uuid4().hex
        filepath = os.path.join(app.config['SAVED_ANALYSES_FOLDER'], f"{analysis_id}.json")
        save_data = {
            'id': analysis_id,
            'title': title,
            'timestamp': datetime.now().isoformat(),
            'filename': task_info.get('filename', 'Text Input'),
            'semaino_version': __version__,
            'data': task_info['data'],
        }
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False)
            return jsonify({'status': 'success', 'id': analysis_id})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/saved_analyses', methods=['GET'])
    def list_saved_analyses():
        saved_list = []
        folder = app.config['SAVED_ANALYSES_FOLDER']
        for filename in os.listdir(folder):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
                        d = json.load(f)
                    saved_list.append({'id': d['id'], 'title': d['title'],
                                       'timestamp': d['timestamp'], 'filename': d['filename']})
                except Exception:
                    continue
        saved_list.sort(key=lambda x: x['timestamp'], reverse=True)
        return jsonify({'saved_analyses': saved_list})

    @app.route('/load_analysis/<analysis_id>', methods=['GET'])
    def load_analysis(analysis_id):
        if not _HEX_ID.match(analysis_id):
            return jsonify({'error': 'Invalid analysis id'}), 400
        filepath = os.path.join(app.config['SAVED_ANALYSES_FOLDER'], f"{analysis_id}.json")
        if not os.path.exists(filepath):
            return jsonify({'error': 'Analysis not found'}), 404
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                save_data = json.load(f)

            _ensure_entity_ids(save_data['data'])
            task_id = uuid.uuid4().hex
            tasks[task_id] = {'status': 'complete', 'progress': 1, 'total': 1,
                              'data': save_data['data'], 'error': None,
                              'filename': save_data.get('filename')}
            return jsonify({'status': 'success', 'task_id': task_id})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return app
