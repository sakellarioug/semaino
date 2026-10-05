import argparse
import sys
import os
import subprocess
from pathlib import Path
from .app import create_app, find_marker_bin
from . import __version__

def serve(args):
    """Start the Semaino Flask server."""
    print(f"Starting Semaino v{__version__} on {args.host}:{args.port}")
    if not args.debug:
        print("Note: Running with Werkzeug development server. For production, use WSGI (e.g. gunicorn).")
    
    app = create_app(data_dir=args.data_dir, online_map=not args.offline_map)
    app.run(host=args.host, port=args.port, debug=args.debug)

def download_models(args):
    """Pre-download Hugging Face and spaCy models."""
    print("Pre-downloading models for Semaino...")
    print("1. Installing Ancient Greek OdyCy model...")
    # Using the correct repo URL
    whl_url = "https://huggingface.co/chcaa/grc_odycy_joint_trf/resolve/main/grc_odycy_joint_trf-0.7.0-py3-none-any.whl"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", whl_url])
    
    print("\n2. Downloading spaCy Modern Greek core model...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "el_core_news_sm"])
    
    print("\n3. Downloading Hugging Face models...")
    # This downloads into the huggingface cache
    from transformers import AutoTokenizer, AutoModel, pipeline
    print("   -> nlpaueb/bert-base-greek-uncased-v1")
    AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
    AutoModel.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
    
    print("   -> UGARIT/grc-ner-bert")
    pipeline("token-classification", model="UGARIT/grc-ner-bert", aggregation_strategy="simple")
    
    # We also need to download gr-nlp-toolkit models
    print("\n4. Downloading gr-nlp-toolkit weights...")
    try:
        from gr_nlp_toolkit import Pipeline
        # Loading the pipeline will trigger the download of processors if they are not cached.
        Pipeline("pos,ner,dp")
    except Exception as e:
        print(f"Warning: gr-nlp-toolkit model cache may have encountered an issue: {e}")

    print("\n✅ All models downloaded successfully.")

def doctor(args):
    """Print diagnostic information."""
    print(f"Semaino v{__version__}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    
    try:
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"PyTorch: {torch.__version__} (Device: {device})")
    except ImportError:
        print("PyTorch: Not installed")
        
    marker = find_marker_bin()
    if marker:
        print(f"OCR Support (Marker): Found at {marker}")
    else:
        print("OCR Support (Marker): Not found (Install with `pip install semaino[ocr]` or `pipx install marker-pdf`)")

def main():
    parser = argparse.ArgumentParser(prog="semaino", description="Semaino: Hybrid Greek Language Analyser")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    subparsers.required = True

    # 'serve' command
    parser_serve = subparsers.add_parser("serve", help="Start the web interface")
    parser_serve.add_argument("--host", default="127.0.0.1", help="Host interface to bind to (default: 127.0.0.1)")
    parser_serve.add_argument("--port", type=int, default=5001, help="Port to listen on (default: 5001)")
    parser_serve.add_argument("--debug", action="store_true", help="Enable debug mode (warning: insecure)")
    parser_serve.add_argument("--data-dir", type=str, help="Directory for uploads and saved analyses (default: ~/.semaino)")
    parser_serve.add_argument("--offline-map", action="store_true", help="Disable the interactive geographic map to guarantee no external requests are made")
    parser_serve.set_defaults(func=serve)

    # 'download-models' command
    parser_download = subparsers.add_parser("download-models", help="Pre-download all required NLP models")
    parser_download.set_defaults(func=download_models)

    # 'doctor' command
    parser_doctor = subparsers.add_parser("doctor", help="Check system and dependencies")
    parser_doctor.set_defaults(func=doctor)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
