from dataclasses import replace
from typing import Any

from rag_pipeline.vectorstore.qdrant import SearchHit


class CrossEncoderReranker:
    """Score query-passage pairs jointly with a cross-encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        model: Any | None = None,
    ):
        self.model_name = model_name

        if model is not None:
            self.model = model
        else:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "Install the embedding dependencies using "
                    'uv pip install -e ".[embeddings]"'
                ) from exc

            self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, hits: list[SearchHit]
    ) -> list[SearchHit]:
        """Attach a rerank score to each hit and sort by it."""

        if not hits:
            return []

        scores = self.model.predict(
            [(query, hit.chunk.text) for hit in hits],
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        reranked = [
            replace(hit, rerank_score=float(score))
            for hit, score in zip(hits, scores, strict=True)
        ]

        return sorted(
            reranked,
            key=lambda hit: hit.rerank_score,
            reverse=True,
        )
