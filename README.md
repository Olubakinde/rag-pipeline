# RAG Pipeline

[![CI](https://github.com/Olubakinde/rag-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Olubakinde/rag-pipeline/actions/workflows/ci.yml)

Upload TXT, PDF and DOCX files and search them by meaning instead of exact
keywords. Documents are split into chunks, embedded with a local Sentence
Transformers model and stored in Qdrant. Everything runs on your machine, no
API keys needed.

## Setup

Requires Python 3.12+.

```bash
git clone https://github.com/Olubakinde/rag-pipeline.git
cd rag-pipeline
uv venv
source .venv/bin/activate
uv pip install -e ".[api,vectorstore,embeddings,cli]"
```

The embedding model (about 90 MB) downloads on first run.

## Command line

```bash
rag index notes.txt papers/        # index files or folders
rag search "how do robots see?"    # search
rag docs                           # list indexed documents
rag delete <document_id>           # remove a document
```

Search options: `--limit`, `--min-score`, `--source <file>`, `--diversity`.

```text
$ rag search "how do robots see" --limit 1
1. robots.txt (score 0.507)
   Robots use lidar sensors to map rooms.
```

## API

```bash
uvicorn rag_pipeline.api.main:app --reload
```

Open http://localhost:8000/docs to try the endpoints in your browser.

```bash
# upload
curl -F "file=@notes.txt" http://localhost:8000/documents

# search
curl http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "how do robots see?", "limit": 3}'
```

Endpoints:

- `POST /documents` upload a file
- `GET /documents` list documents
- `DELETE /documents/{document_id}` remove a document
- `POST /search` search (`query`, `limit`, `min_score`, `sources`, `diversity`)
- `GET /health` health check

The API and CLI use the same local database, and only one can have it open at
a time, so stop the server before using the CLI.

## Configuration

Copy `.env.example` to `.env` to change settings like the database path, chunk
size or embedding model. To turn on reranking for better result ordering, set:

```bash
RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
```

## Tests

```bash
uv pip install -e ".[dev,api,vectorstore,cli]"
pytest
ruff check .
```

## License

MIT
