"""PostgreSQL/pgvector-backed company interview-context retrieval."""

from typing import List

from app.core.database import db_pool
from app.core.logger import logger
from app.modules.ai.client import embed_text


def ensure_company_playbook_table(cur) -> None:
    """Create the RAG store when pgvector is available in PostgreSQL."""
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS company_playbook_chunks (
            id SERIAL PRIMARY KEY,
            company_name TEXT NOT NULL,
            source_section TEXT,
            chunk_text TEXT NOT NULL,
            embedding vector(768) NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_company_playbook_company
        ON company_playbook_chunks (LOWER(TRIM(company_name)))
        """
    )


async def retrieve_company_context(company_name: str, query: str, top_k: int = 5) -> List[str]:
    """Return the most relevant trusted playbook chunks for the chosen company.

    RAG is deliberately best-effort: a missing knowledge base must not prevent
    standard resume/JD question generation.
    """
    if not company_name or not query:
        return []

    try:
        embedding = await embed_text(query)
        vector_literal = "[" + ",".join(str(value) for value in embedding) + "]"
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH candidates AS MATERIALIZED (
                        SELECT chunk_text, embedding
                        FROM company_playbook_chunks
                        WHERE LOWER(TRIM(company_name)) = LOWER(TRIM(%s))
                           OR POSITION(LOWER(TRIM(%s)) IN LOWER(company_name)) > 0
                    )
                    SELECT chunk_text, embedding <=> %s::vector AS distance
                    FROM candidates
                    ORDER BY distance
                    LIMIT %s
                    """,
                    (
                        company_name,
                        company_name,
                        vector_literal,
                        max(1, min(top_k, 10)),
                    ),
                )
                rows = cur.fetchall()
                for chunk_text, distance in rows:
                    similarity = 1 - distance
                    logger.info(
                        "RAG match | company=%s | similarity=%.3f | chunk=%s...",
                        company_name,
                        similarity,
                        chunk_text[:80],
                    )
                return [row[0] for row in rows]
    except Exception as exc:
        logger.warning("Company-playbook RAG retrieval skipped for %s: %s", company_name, exc)
        return []
