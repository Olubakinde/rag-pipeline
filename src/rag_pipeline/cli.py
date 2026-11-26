from pathlib import Path
from typing import Annotated
from zipfile import BadZipFile

import typer
from docx.opc.exceptions import PackageNotFoundError
from pypdf.errors import PdfReadError

from rag_pipeline.config import Settings
from rag_pipeline.factory import open_pipeline

app = typer.Typer(
    help="Index documents and search them from the command line.",
    no_args_is_help=True,
)

SNIPPET_LENGTH = 300


def collect_files(
    paths: list[Path], extensions: set[str]
) -> list[Path]:
    """Expand directories into the supported files they contain."""

    files = []

    for path in paths:
        if path.is_dir():
            files.extend(
                sorted(
                    child
                    for child in path.rglob("*")
                    if child.is_file()
                    and child.suffix.lower() in extensions
                )
            )
        else:
            files.append(path)

    return files


@app.command()
def index(
    paths: Annotated[
        list[Path],
        typer.Argument(help="Files or directories to index."),
    ],
):
    """Parse, chunk, embed, and store documents."""

    failures = 0

    with open_pipeline(Settings()) as pipeline:
        files = collect_files(
            paths,
            pipeline.ingestion.registry.supported_extensions,
        )

        if not files:
            typer.echo("No supported documents found.", err=True)
            raise typer.Exit(code=1)

        for file_path in files:
            # Keep going so one bad file does not abort a batch.
            try:
                indexed = pipeline.index(file_path)
            except (
                OSError,
                ValueError,
                UnicodeError,
                PdfReadError,
                PackageNotFoundError,
                BadZipFile,
            ) as exc:
                failures += 1
                typer.echo(f"FAILED  {file_path}: {exc}", err=True)
                continue

            typer.echo(
                f"indexed {file_path} "
                f"({indexed.chunk_count} chunks, "
                f"id {indexed.document_id})"
            )

    if failures:
        raise typer.Exit(code=1)


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="What to look for.")],
    limit: Annotated[int, typer.Option(min=1, max=50)] = 5,
    min_score: Annotated[
        float | None, typer.Option(min=-1.0, max=1.0)
    ] = None,
    source: Annotated[
        list[str] | None,
        typer.Option(help="Only search this file. Repeatable."),
    ] = None,
    diversity: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.0,
):
    """Find the passages most relevant to a query."""

    if not query.strip():
        raise typer.BadParameter("Query cannot be blank")

    with open_pipeline(Settings()) as pipeline:
        hits = pipeline.search(
            query,
            limit=limit,
            min_score=min_score,
            sources=source,
            diversity=diversity,
        )

    if not hits:
        typer.echo("No results.")
        return

    for rank, hit in enumerate(hits, start=1):
        location = hit.source

        if hit.pages:
            location += f", pages {', '.join(map(str, hit.pages))}"

        score = f"score {hit.score:.3f}"

        if hit.rerank_score is not None:
            score += f", rerank {hit.rerank_score:.2f}"

        text = " ".join(hit.chunk.text.split())

        if len(text) > SNIPPET_LENGTH:
            text = text[:SNIPPET_LENGTH].rstrip() + "..."

        typer.echo(f"{rank}. {location} ({score})")
        typer.echo(f"   {text}")


@app.command()
def docs():
    """List indexed documents."""

    with open_pipeline(Settings()) as pipeline:
        documents = pipeline.list_documents()

    if not documents:
        typer.echo("No documents indexed.")
        return

    for document in documents:
        typer.echo(
            f"{document.document_id}  "
            f"{document.chunk_count:>4} chunks  "
            f"{document.source}"
        )


@app.command()
def delete(
    document_id: Annotated[
        str, typer.Argument(help="ID shown by the docs command.")
    ],
):
    """Remove a document and its chunks from the index."""

    with open_pipeline(Settings()) as pipeline:
        deleted = pipeline.delete_document(document_id)

    if not deleted:
        typer.echo(f"No document with ID {document_id}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Deleted {deleted} chunks from {document_id}")
