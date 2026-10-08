"""Stdio MCP server with a fixed, read-only project surface."""

import argparse
import json
import sqlite3
from pathlib import Path


class ReadOnlyProject:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve(strict=True)
        database = self.root / "metadata.sqlite"
        if not database.is_file():
            raise FileNotFoundError(database)
        self.db = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA query_only = ON")

    def close(self) -> None:
        self.db.close()

    def project(self) -> dict:
        row = self.db.execute("SELECT name, subject, created_at FROM project").fetchone()
        return dict(row)

    def models(self) -> list[dict]:
        return [{"id": row["id"], "algorithm": row["algorithm"], "subject": row["subject"],
                 "created_at": row["created_at"], "metrics": json.loads(row["metrics_json"])}
                for row in self.db.execute("SELECT id, algorithm, subject, created_at, metrics_json FROM models ORDER BY created_at")]

    def reports(self) -> list[str]:
        return [str(path.relative_to(self.root)) for path in (self.root / "reports").glob("*.html")
                if path.is_file() and path.resolve().is_relative_to(self.root)]

    def report(self, name: str) -> str:
        path = (self.root / "reports" / name).resolve()
        if not path.is_relative_to((self.root / "reports").resolve()) or path.suffix.lower() != ".html":
            raise ValueError("invalid report name")
        return path.read_text(encoding="utf-8")

    def logs(self, limit: int = 100) -> list[dict]:
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        return [{"created_at": row["created_at"], "action": row["action"],
                 "details": json.loads(row["details_json"])}
                for row in self.db.execute("SELECT created_at, action, details_json FROM audit ORDER BY id DESC LIMIT ?", (limit,))]


def create_server(root: str | Path):
    from mcp.server.fastmcp import FastMCP

    project = ReadOnlyProject(root)
    server = FastMCP("mi-bci-read-only")
    server.read_only_project = project

    @server.tool()
    def get_project() -> dict:
        """Read the current project's name, pseudonym and creation time."""
        return project.project()

    @server.tool()
    def list_models() -> list[dict]:
        """Read model identifiers, algorithms and evaluation metrics."""
        return project.models()

    @server.tool()
    def list_reports() -> list[str]:
        """List HTML reports stored in the project's reports directory."""
        return project.reports()

    @server.tool()
    def read_report(name: str) -> str:
        """Read a named HTML report in the project's reports directory."""
        return project.report(name)

    @server.tool()
    def read_logs(limit: int = 100) -> list[dict]:
        """Read recent audit events."""
        return project.logs(limit)

    return server


def main():
    parser = argparse.ArgumentParser(description="Read-only MI-BCI project MCP server")
    parser.add_argument("project", help="Existing project directory")
    args = parser.parse_args()
    create_server(args.project).run(transport="stdio")


if __name__ == "__main__":
    main()
