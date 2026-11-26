from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_pipeline.config import Settings


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    # Keep a developer's local .env from leaking into these tests.
    monkeypatch.chdir(tmp_path)

    for name in [
        "RAG_QDRANT_PATH",
        "RAG_COLLECTION_NAME",
        "RAG_EMBEDDING_MODEL",
        "RAG_CHUNK_SIZE",
        "RAG_CHUNK_OVERLAP",
        "RAG_MAX_UPLOAD_MB",
    ]:
        monkeypatch.delenv(name, raising=False)


def test_defaults():
    settings = Settings()

    assert settings.qdrant_path == Path("data/qdrant")
    assert settings.collection_name == "documents"
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 150
    assert settings.max_upload_bytes == 10 * 1024 * 1024


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("RAG_QDRANT_PATH", "/tmp/vectors")
    monkeypatch.setenv("RAG_CHUNK_SIZE", "500")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "50")
    monkeypatch.setenv("RAG_MAX_UPLOAD_MB", "2")

    settings = Settings()

    assert settings.qdrant_path == Path("/tmp/vectors")
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50
    assert settings.max_upload_bytes == 2 * 1024 * 1024


def test_dotenv_file(tmp_path):
    (tmp_path / ".env").write_text(
        "RAG_COLLECTION_NAME=papers\n",
        encoding="utf-8",
    )

    assert Settings().collection_name == "papers"


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError):
        Settings(chunk_size=100, chunk_overlap=100)


@pytest.mark.parametrize(
    "field",
    ["chunk_size", "max_upload_mb"],
)
def test_positive_values_required(field):
    with pytest.raises(ValidationError):
        Settings(**{field: 0})
