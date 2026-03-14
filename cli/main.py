import typer

app = typer.Typer(help="irkit — Distributed Hybrid Information Retrieval Toolkit")

@app.command()
def hello():
    print("Hello from irkit!")

if __name__ == "__main__":
    app()
