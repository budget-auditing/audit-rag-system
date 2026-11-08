"""FastAPI application entrypoint."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import text

from app.db import engine, Base, SessionLocal
from app.routers import health, documents, index, query, evidence, export, stats
from app.msg.graph import msg_graph

Base.metadata.create_all(bind=engine)


def _ensure_columns() -> None:
    """Lightweight SQLite migration: add columns introduced after first deploy."""
    wanted = {
        "query_logs": [("selected_mode", "VARCHAR"), ("rule_hint", "TEXT")],
        "documents": [("parse_seconds", "FLOAT")],
    }
    with engine.begin() as conn:
        for table, cols in wanted.items():
            existing = {row[1] for row in conn.execute(text(f"PRAGMA table_info({table})"))}
            for name, coltype in cols:
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {coltype}"))


_ensure_columns()

# Restore MSG-RAG graph from SQLite on startup.
with SessionLocal() as db:
    try:
        msg_graph.load(db)
    except Exception:
        # If no persisted graph yet, start empty; build will populate.
        pass

app = FastAPI(title="Audit RAG Platform", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(index.router, prefix="/api")
app.include_router(query.router, prefix="/api")
app.include_router(evidence.router, prefix="/api")
app.include_router(export.router, prefix="/api")
app.include_router(stats.router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
