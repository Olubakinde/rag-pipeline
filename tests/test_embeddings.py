import pytest

from rag_pipeline.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingProvider,
)


class FakeArray:
    """Provide the array interface needed by our tests."""

    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeModel:
    """Test double that avoids downloading an ML model."""

    def __init__(self):
        self.document_calls = []
        self.query_calls = []

    def encode_document(self, texts, **kwargs):
        self.document_calls.append((texts, kwargs))
        return FakeArray([[1.0, 0.0] for _ in texts])

    def encode_query(self, query, **kwargs):
        self.query_calls.append((query, kwargs))
        return FakeArray([0.0, 1.0])

    def get_embedding_dimension(self):
        return 2


def test_dimension_from_model():
    provider = SentenceTransformerEmbeddingProvider(model=FakeModel())

    assert provider.dimension == 2


def test_embed_documents():
    model = FakeModel()
    provider = SentenceTransformerEmbeddingProvider(model=model)

    vectors = provider.embed_documents([
        "Machine learning",
        "Robotics",
    ])

    assert len(vectors) == 2
    assert vectors == [[1.0, 0.0], [1.0, 0.0]]
    assert model.document_calls[0][0] == [
        "Machine learning",
        "Robotics",
    ]


def test_normalized_document_embeddings_requested():
    model = FakeModel()
    provider = SentenceTransformerEmbeddingProvider(model=model)

    provider.embed_documents(["Example document"])

    options = model.document_calls[0][1]

    assert options["normalize_embeddings"] is True
    assert options["convert_to_numpy"] is True


def test_embed_query():
    model = FakeModel()
    provider = SentenceTransformerEmbeddingProvider(model=model)

    vector = provider.embed_query("What is robotics?")

    assert vector == [0.0, 1.0]
    assert model.query_calls[0][0] == "What is robotics?"
    assert model.query_calls[0][1]["normalize_embeddings"] is True


def test_empty_document_batch():
    model = FakeModel()
    provider = SentenceTransformerEmbeddingProvider(model=model)

    assert provider.embed_documents([]) == []
    assert model.document_calls == []


@pytest.mark.parametrize(
    "texts",
    [
        [""],
        ["   "],
        ["Valid text", ""],
    ],
)
def test_invalid_document_texts(texts):
    provider = SentenceTransformerEmbeddingProvider(
        model=FakeModel()
    )

    with pytest.raises(ValueError):
        provider.embed_documents(texts)


@pytest.mark.parametrize("query", ["", "   "])
def test_invalid_query(query):
    provider = SentenceTransformerEmbeddingProvider(
        model=FakeModel()
    )

    with pytest.raises(ValueError):
        provider.embed_query(query)