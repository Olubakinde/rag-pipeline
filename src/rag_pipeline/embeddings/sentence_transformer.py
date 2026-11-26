from typing import Any


class SentenceTransformerEmbeddingProvider:
    """Generate local embeddings using Sentence Transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model: Any | None = None,
    ):
        self.model_name = model_name

        if model is not None:
            self.model = model
        else:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "Install the embedding dependencies using "
                    'uv pip install -e ".[embeddings]"'
                ) from exc

            self.model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        """Return the size of the vectors this model produces."""

        return self.model.get_embedding_dimension()

    def embed_documents(
        self, texts: list[str]
    ) -> list[list[float]]:
        """Generate normalized embeddings for document texts."""

        if not texts:
            return []

        if any(
            not isinstance(text, str) or not text.strip()
            for text in texts
        ):
            raise ValueError(
                "Document texts must be non-empty strings"
            )

        embeddings = self.model.encode_document(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Generate a normalized embedding for a search query."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")

        embedding = self.model.encode_query(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embedding.tolist()