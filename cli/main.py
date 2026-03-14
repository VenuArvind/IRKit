import typer
import uvicorn
from rich.console import Console
from irkit import IndexEngine, ArxivSource, HybridRanker, BM25Ranker, SemanticRanker, HuggingFaceEmbedder, MemoryStorage
from irkit.serve.api import app as fastapi_app, set_engine

app = typer.Typer(help="irkit — Distributed Hybrid Information Retrieval Toolkit")
console = Console()

@app.command()
def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
    max_docs: int = 100
):
    """
    Start the IRKit Search Server.
    """
    console.print(f"[bold green]🚀 Starting IRKit Search Server on {host}:{port}...[/bold green]")
    
    embedder = HuggingFaceEmbedder()
    ranker = HybridRanker(rankers=[BM25Ranker(), SemanticRanker(embedder)])
    engine = IndexEngine(ranker=ranker, storage=MemoryStorage())
    
    # 2. Ingest some initial data
    engine.index(ArxivSource(), max_docs=max_docs)
    
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
