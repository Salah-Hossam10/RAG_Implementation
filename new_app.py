import os
import pdfplumber
import pytesseract
from PIL import Image
import faiss
import pickle
import numpy as np
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from functools import lru_cache, wraps
import logging
import time
import re
from werkzeug.middleware.proxy_fix import ProxyFix


# Logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)


# Configuration
PDF_DIR = "documents"
CHUNK_SIZE = 500
#EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
VECTOR_STORE = "vector_store"
CACHE_DIR = "/mnt/models"
API_KEYS = {os.getenv("ESG_API_KEY"): True}
os.makedirs(VECTOR_STORE, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


INDEX_PATH = os.path.join(VECTOR_STORE, "index.faiss")
CHUNKS_PATH = os.path.join(VECTOR_STORE, "chunks.pkl")
DOCS_PATH = os.path.join(VECTOR_STORE, "docs.pkl")


# Authentication decorator
def require_api_key(f):
    @wraps(f)  
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-KEY') or request.args.get('api_key')
        if api_key not in API_KEYS:
            return jsonify({"error": "Invalid or missing API key"}), 403
        return f(*args, **kwargs)
    return decorated_function


# OCR fallback extraction
def extract_text_from_page(page):
    text = page.extract_text()
    if text and len(text.strip()) > 20:
        return text
    try:
        image = page.to_image(resolution=300).original
        custom_config = r'--oem 3 --psm 6'
        ocr_text = pytesseract.image_to_string(image, config=custom_config)
        return ocr_text.strip() if ocr_text else ""
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return ""
    

# Load models
@lru_cache(maxsize=1)
def load_models():
    logger.info("🚀 Loading models...")
    embedder = SentenceTransformer(EMBED_MODEL, cache_folder=CACHE_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        cache_dir=CACHE_DIR
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id
    )
    return embedder, generator




