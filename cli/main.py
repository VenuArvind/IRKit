import typer
import uvicorn
from rich.console import Console
from irkit import IndexEngine, ArXivSource, WikipediaSource, NewsSource, HybridRanker, BM25Ranker, SemanticRanker, SentenceTransformerEmbedder, InMemoryStorage, CrossEncoderRanker
from irkit.serve.api import app as fastapi_app, set_engine

app = typer.Typer(help="irkit — Distributed Hybrid Information Retrieval Toolkit")
console = Console()

@app.command()
def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
    max_docs: int = 100,
    source: str = typer.Option("arxiv", help="Data source: arxiv, wikipedia, news"),
    quantization: Optional[str] = typer.Option(None, help="Quantization mode: sq8, pq, or None")
):
    """
    Start the IRKit Search Server.
    """
    console.print(f"[bold green]🚀 Starting IRKit Search Server on {host}:{port}...[/bold green]")
    
    embedder = SentenceTransformerEmbedder()
    semantic_ranker = SemanticRanker(embedder, quantization=quantization)
    ranker = HybridRanker(rankers=[BM25Ranker(), semantic_ranker])
    reranker = CrossEncoderRanker()
    engine = IndexEngine(ranker=ranker, storage=InMemoryStorage(), reranker=reranker)
    
    # 2. Select and ingest data
    try:
        data_source = get_source(source, max_docs=max_docs)
    except Exception as e:
        console.print(f"[bold red]Error resolving source '{source}': {e}. Defaulting to arxiv.[/bold red]")
        data_source = ArXivSource(max_docs=max_docs)

    engine.index(data_source, max_docs=max_docs)
    
    # 3. Inject engine into the API
    set_engine(engine)
    
    # 4. Run the web server
    uvicorn.run(fastapi_app, host=host, port=port)

@app.command()
def version():
    """ Show the version of IRKit. """
    console.print("IRKit v1.0.0")

if __name__ == "__main__":
    app()
