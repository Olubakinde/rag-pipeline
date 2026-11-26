from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated
from zipfile import BadZipFile

from docx.opc.exceptions import PackageNotFoundError
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from pypdf.errors import PdfReadError

from rag_pipeline.chunking.boundaries import WordBoundaryChunker
from rag_pipeline.config import Settings
from rag_pipeline.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingProvider,
)
from rag_pipeline.ingestion import IngestionPipeline
from rag_pipeline.retrieval import RAGPipeline
from rag_pipeline.vectorstore.qdrant import QdrantVectorStore

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}


class IndexResponse(BaseModel):
    filename: str
    chunks_indexed: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=50)


class SearchResult(BaseModel):
    text: str
    score: float
    source: str
    document_id: str
    chunk_index: int
    start: int
    end: int
    pages: list[int]


class SearchResponse(BaseModel):
    results: list[SearchResult]


def create_app(
    pipeline: RAGPipeline | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Create the API with an injectable retrieval pipeline."""

    if settings is None:
        settings = Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owned_store = None

        if pipeline is None:
            embedder = SentenceTransformerEmbeddingProvider(
                model_name=settings.embedding_model,
            )

            # Persist vectors between server restarts.
            owned_store = QdrantVectorStore(
                collection_name=settings.collection_name,
                vector_size=embedder.dimension,
                path=settings.qdrant_path,
            )

            app.state.pipeline = RAGPipeline(
                embedder=embedder,
                store=owned_store,
                ingestion=IngestionPipeline(
                    chunker=WordBoundaryChunker(
                        chunk_size=settings.chunk_size,
                        overlap=settings.chunk_overlap,
                    )
                ),
            )
        else:
            # Tests can supply a lightweight fake embedding model.
            app.state.pipeline = pipeline

        try:
            yield
        finally:
            if owned_store is not None:
                owned_store.client.close()

    app = FastAPI(
        title="RAG Pipeline API",
        description="Document ingestion and semantic retrieval API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post(
        "/documents",
        response_model=IndexResponse,
        status_code=201,
    )
    def upload_document(file: Annotated[UploadFile, File()]):
        """Upload, parse, chunk, embed, and index a document."""

        filename = Path(
            (file.filename or "").replace("\\", "/")
        ).name

        extension = Path(filename).suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail="Supported formats: TXT, PDF, DOCX",
            )

        with TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / filename

            total_bytes = 0

            # Stream to disk without loading the full upload into memory.
            with file_path.open("wb") as output:
                while True:
                    block = file.file.read(1024 * 1024)

                    if not block:
                        break

                    total_bytes += len(block)

                    if total_bytes > settings.max_upload_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                "File exceeds the "
                                f"{settings.max_upload_mb} MB limit"
                            ),
                        )

                    output.write(block)

            if total_bytes == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded file is empty",
                )

            try:
                count = app.state.pipeline.index(file_path)
            except (
                ValueError,
                UnicodeError,
                PdfReadError,
                PackageNotFoundError,
                BadZipFile,
            ) as exc:
                raise HTTPException(
                    status_code=422,
                    detail="The document could not be parsed",
                ) from exc

        return IndexResponse(
            filename=filename,
            chunks_indexed=count,
        )

    @app.post(
        "/search",
        response_model=SearchResponse,
    )
    def search_documents(request: SearchRequest):
        """Search indexed documents by semantic similarity."""

        if not request.query.strip():
            raise HTTPException(
                status_code=422,
                detail="Search query cannot be blank",
            )

        hits = app.state.pipeline.search(
            query=request.query,
            limit=request.limit,
        )

        return SearchResponse(
            results=[
                SearchResult(
                    text=hit.chunk.text,
                    score=hit.score,
                    source=hit.source,
                    document_id=hit.chunk.document_id,
                    chunk_index=hit.chunk.index,
                    start=hit.chunk.start,
                    end=hit.chunk.end,
                    pages=hit.pages,
                )
                for hit in hits
            ]
        )

    return app


app = create_app()