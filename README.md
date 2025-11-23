📄 PDF RAG Pipeline API

A lightweight Retrieval-Augmented Generation (RAG) API built with Flask, FAISS, Sentence Transformers, and Mistral-7B.
This service extracts text from PDFs (with OCR fallback), chunks & embeds content, stores it in FAISS, and serves semantic search + LLM responses.

🚀 Features
🔍 Document Processing

Extracts text from PDFs using pdfplumber

Falls back to Tesseract OCR when text extraction fails

Splits text into manageable chunks

Saves document chunks and embeddings to disk

🧠 Embeddings & Vector Search

Sentence Transformer: multi-qa-MiniLM-L6-cos-v1 (configurable)

FAISS vector index for fast semantic similarity search

🤖 LLM Generation

Uses Mistral-7B-Instruct (mistralai/Mistral-7B-Instruct-v0.3)

HuggingFace transformers pipeline for generation

🔐 API Security

Built-in API key authentication via X-API-KEY header or query parameter

Keys managed through environment variable: ESG_API_KEY

🧩 Modular & Extensible
| Component    | Technology                                  |
| ------------ | ------------------------------------------- |
| Framework    | Flask                                       |
| Vector Store | FAISS                                       |
| Embeddings   | Sentence Transformers                       |
| LLM          | HuggingFace Transformers (Mistral-7B)       |
| OCR          | Tesseract via pytesseract                   |
| PDF Parsing  | pdfplumber                                  |
| Dev Tools    | Logging, ProxyFix, Pickle-based persistence |

📁 Project Structure
.
│── documents/              # PDF uploads
│── vector_store/           # FAISS index + chunk metadata
│   ├── index.faiss
│   ├── chunks.pkl
│   └── docs.pkl
│
│── app.py                  # Main Flask API (this file)
│── README.md               # Documentation
└── requirements.txt

⚙️ Setup Instructions
1. Clone the Repo
git clone https://github.com/your-repo.git
cd your-repo

2. Install Dependencies
pip install -r requirements.txt

3. Install Tesseract (OCR)

Ubuntu

sudo apt install tesseract-ocr


Mac (Homebrew)

brew install tesseract

4. Set Environment Variable
export ESG_API_KEY="your-secret-key"

5. Run the Server
python app.py


Your API is now online at:
👉 http://localhost:5000

🔑 Authentication

Provide API key via:

Header

X-API-KEY: your-key


or

Query parameter

?api_key=your-key

📡 API Endpoints

Below are examples (customize if your repo has additional endpoints).

📤 Upload and Process PDFs
POST /upload


Processes a PDF, extracts text, chunks it, embeds it, and updates the FAISS index.

🔎 Semantic Search
GET /search?q=your query


Returns the most similar text chunks from your indexed PDFs.

🤖 Ask Questions (RAG)
POST /ask
{
  "query": "What does the document say about ESG reporting?"
}


The system retrieves relevant text from FAISS and uses Mistral-7B to generate a grounded response.

🧠 Model Loading

Models are loaded once using:

@lru_cache(maxsize=1)
def load_models():


This ensures fast inference after the initial load.

🗃 Vector Store Structure

index.faiss — the FAISS index

chunks.pkl — the text chunks

docs.pkl — metadata about processed PDFs

🧪 Example .env Setup
ESG_API_KEY=my-secret-key
PDF_DIR=documents
CACHE_DIR=/mnt/models
Caching for model loading

Clean architecture
