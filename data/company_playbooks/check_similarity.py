"""
Demo/inspection tool for the company-playbook RAG similarity search.

Run this to SHOW (not just claim) how the similarity index works:
  - proves the pgvector ivfflat index actually exists on the table
  - embeds a real query and runs the same cosine-distance search the app uses
  - prints the ranked results with their real distance scores

Usage:
    python check_similarity.py "Design a URL shortener" GOOGLE
    python check_similarity.py "How do you handle conflict with a teammate" "MICROSOFT & AMAZON"

Lower distance = more similar (0 = identical meaning, 2 = opposite meaning).
"""
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.database import db_pool
from app.modules.ai.client import embed_text


def show_index_definition():
    print("=" * 70)
    print("1. PROOF THE SIMILARITY INDEX EXISTS")
    print("=" * 70)
    with db_pool.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'company_playbook_chunks'"
            )
            for name, definition in cur.fetchall():
                print(f"  {name}:\n    {definition}\n")


def show_chunk_counts():
    print("=" * 70)
    print("2. HOW MANY CHUNKS EXIST PER COMPANY")
    print("=" * 70)
    with db_pool.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT company_name, COUNT(*), AVG(LENGTH(chunk_text))::int
                FROM company_playbook_chunks
                GROUP BY company_name
                ORDER BY company_name
                """
            )
            for company, count, avg_len in cur.fetchall():
                print(f"  {company}: {count} chunks, avg {avg_len} chars each")
    print()


async def show_similarity_search(query: str, company: str, top_k: int = 5):
    print("=" * 70)
    print(f"3. LIVE SIMILARITY SEARCH")
    print(f"   Query:   \"{query}\"")
    print(f"   Company: {company}")
    print("=" * 70)

    embedding = await embed_text(query)
    print(f"  Query embedded into a {len(embedding)}-dimension vector.\n")

    vector_literal = "[" + ",".join(str(v) for v in embedding) + "]"
    with db_pool.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_text, embedding <=> %s::vector AS cosine_distance
                FROM company_playbook_chunks
                WHERE LOWER(TRIM(company_name)) = LOWER(TRIM(%s))
                ORDER BY cosine_distance
                LIMIT %s
                """,
                (vector_literal, company, top_k),
            )
            rows = cur.fetchall()

    if not rows:
        print(f"  No chunks found for company '{company}'. Check the name matches what's in the DB.")
        return

    print(f"  Top {len(rows)} most similar chunks (ranked closest first):\n")
    for rank, (text, distance) in enumerate(rows, start=1):
        preview = " ".join(text.split())[:110]
        print(f"  #{rank}  distance={distance:.4f}  {preview}...")
    print()


async def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "Design a URL shortening service"
    company = sys.argv[2] if len(sys.argv) > 2 else "GOOGLE"

    show_index_definition()
    show_chunk_counts()
    await show_similarity_search(query, company)


if __name__ == "__main__":
    asyncio.run(main())
