from __future__ import annotations

import typer
import uvicorn

app = typer.Typer(help="Run the hayfind local HTTP service")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8765, reload: bool = False) -> None:
    uvicorn.run("hayfind.service:app", host=host, port=port, reload=reload)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
