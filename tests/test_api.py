from io import BytesIO

import pytest
from docx import Document as WordDocument
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from rag_pipeline.api.main import create_app
from rag_pipeline.chunking.boundaries import WordBoundaryChunker
from rag_pipeline.config import Settings
from rag_pipeline.ingestion import IngestionPipeline
from rag_pipeline.retrieval import RAGPipeline
from rag_pipeline.vectorstore.qdrant import QdrantVectorStore


class FakeEmbedder:
    """Deterministic embeddings for API testing."""

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, query):
        if "robot" in query.lower():
            return [1.0, 0.0]
        return [0.0, 1.0]


@pytest.fixture
def client():
    store = QdrantVectorStore(vector_size=2)

    pipeline = RAGPipeline(
        embedder=FakeEmbedder(),
        store=store,
        ingestion=IngestionPipeline(
            chunker=WordBoundaryChunker(
                chunk_size=200,
                overlap=0,
            )
        ),
    )

    application = create_app(pipeline=pipeline)

    with TestClient(application) as test_client:
        yield test_client

    store.client.close()


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_and_search_txt(client):
    response = client.post(
        "/documents",
        files={
            "file": (
                "notes.txt",
                b"Robots use sensors to navigate.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 201
    assert response.json()["filename"] == "notes.txt"
    assert response.json()["chunks_indexed"] == 1

    search_response = client.post(
        "/search",
        json={
            "query": "How do robots navigate?",
            "limit": 5,
        },
    )

    assert search_response.status_code == 200

    results = search_response.json()["results"]

    assert len(results) == 1
    assert results[0]["source"] == "notes.txt"
    assert "Robots use sensors" in results[0]["text"]
    assert results[0]["score"] == pytest.approx(1.0)


def test_upload_pdf(client):
    buffer = BytesIO()

    pdf = canvas.Canvas(buffer)
    pdf.drawString(72, 720, "Robots navigate using cameras.")
    pdf.save()

    response = client.post(
        "/documents",
        files={
            "file": (
                "research.pdf",
                buffer.getvalue(),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 201
    assert response.json()["chunks_indexed"] == 1

    search_response = client.post(
        "/search",
        json={"query": "robot navigation"},
    )

    result = search_response.json()["results"][0]

    assert result["source"] == "research.pdf"
    assert result["pages"] == [1]


def test_upload_docx(client):
    buffer = BytesIO()

    document = WordDocument()
    document.add_paragraph("Robots use sensors.")
    document.save(buffer)

    response = client.post(
        "/documents",
        files={
            "file": (
                "report.docx",
                buffer.getvalue(),
            (
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
            )
        },
    )

    assert response.status_code == 201
    assert response.json()["chunks_indexed"] == 1


def upload_text(client, filename, content):
    response = client.post(
        "/documents",
        files={"file": (filename, content, "text/plain")},
    )

    assert response.status_code == 201

    return response.json()["document_id"]


def test_list_documents(client):
    assert client.get("/documents").json() == {"documents": []}

    document_id = upload_text(client, "notes.txt", b"Robots navigate.")

    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == {
        "documents": [
            {
                "document_id": document_id,
                "source": "notes.txt",
                "chunk_count": 1,
            }
        ]
    }


def test_delete_document(client):
    document_id = upload_text(client, "notes.txt", b"Robots navigate.")

    response = client.delete(f"/documents/{document_id}")

    assert response.status_code == 200
    assert response.json() == {
        "document_id": document_id,
        "chunks_deleted": 1,
    }

    assert client.get("/documents").json() == {"documents": []}

    search_response = client.post("/search", json={"query": "robots"})

    assert search_response.json()["results"] == []


def test_delete_unknown_document(client):
    response = client.delete("/documents/missing")

    assert response.status_code == 404


def test_unsupported_file(client):
    response = client.post(
        "/documents",
        files={
            "file": ("image.png", b"example", "image/png")
        },
    )

    assert response.status_code == 415


def test_empty_file(client):
    response = client.post(
        "/documents",
        files={
            "file": ("empty.txt", b"", "text/plain")
        },
    )

    assert response.status_code == 400


def test_blank_search_query(client):
    response = client.post(
        "/search",
        json={"query": "   "},
    )

    assert response.status_code == 422


def test_invalid_search_limit(client):
    response = client.post(
        "/search",
        json={
            "query": "robotics",
            "limit": 0,
        },
    )

    assert response.status_code == 422


def test_configurable_file_size_limit():
    store = QdrantVectorStore(vector_size=2)
    pipeline = RAGPipeline(embedder=FakeEmbedder(), store=store)

    application = create_app(
        pipeline=pipeline,
        settings=Settings(max_upload_mb=1),
    )

    with TestClient(application) as test_client:
        response = test_client.post(
            "/documents",
            files={
                "file": (
                    "large.txt",
                    b"a" * (1024 * 1024 + 1),
                    "text/plain",
                )
            },
        )

    store.client.close()

    assert response.status_code == 413
    assert response.json()["detail"] == "File exceeds the 1 MB limit"


def test_file_size_limit(client):
    response = client.post(
        "/documents",
        files={
            "file": (
                "large.txt",
                b"a" * (10 * 1024 * 1024 + 1),
                "text/plain",
            )
        },
    )

    assert response.status_code == 413