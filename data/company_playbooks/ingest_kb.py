import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

from psycopg2.extras import Json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.database import db_pool
from app.modules.ai.client import embed_text
from app.modules.rag.ingestion import chunk_playbook_section, extract_pdf_text
from app.modules.rag.service import ensure_company_playbook_table

def _resolve_pdf_paths() -> List[Path]:
    if len(sys.argv) > 1:
        return [Path(arg).resolve() for arg in sys.argv[1:]]

    search_dirs = [
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parents[1] / "company_playbooks",
        Path(__file__).resolve().parents[3] / "data" / "company_playbooks",
    ]
    seen = set()
    pdf_paths: List[Path] = []
    for directory in search_dirs:
        if not directory.exists():
            continue
        for pdf_path in sorted(directory.glob("*.pdf")):
            resolved = pdf_path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                pdf_paths.append(resolved)
    return pdf_paths


KNOWN_COMPANY_HEADINGS = {
    "GOOGLE",
    "MICROSOFT",
    "AMAZON",
    "META",
    "APPLE",
    "NETFLIX",
    "UBER",
    "AIRBNB",
    "SPOTIFY",
    "ADOBE",
    "INTUIT",
    "ORACLE",
    "SALESFORCE",
    "PALANTIR",
    "MICROSOFT & AMAZON",
    "ALIBABA CLOUD",
    "FLIPKART",
    "NVIDIA",
    "TESLA",
    "OPENAI",
    "BROADCOM",
}


def _normalize_company_name(raw: str) -> str:
    cleaned = re.sub(r"\s+", " ", (raw or "").strip())
    return cleaned.upper()


def _looks_like_company_heading(line: str) -> bool:
    normalized = re.sub(r"\s+", " ", (line or "").strip()).upper()
    if not normalized:
        return False
    if not re.fullmatch(r"[A-Z0-9&/\-\s]{2,60}", normalized):
        return False
    if normalized in KNOWN_COMPANY_HEADINGS:
        return True
    return any(company in normalized for company in KNOWN_COMPANY_HEADINGS)


def _split_by_company(text: str) -> List[Tuple[str, str]]:
    lines = [line.rstrip() for line in text.splitlines()]
    company_sections: List[Tuple[str, str]] = []
    current_company = None
    current_parts: List[str] = []

    for line in lines:
        normalized = line.strip()
        if not normalized:
            continue
        if _looks_like_company_heading(normalized):
            if current_company is not None:
                company_sections.append((current_company, "\n".join(current_parts).strip()))
            current_company = _normalize_company_name(normalized)
            current_parts = []
            continue
        if current_company is not None:
            current_parts.append(normalized)

    if current_company is not None and current_parts:
        company_sections.append((current_company, "\n".join(current_parts).strip()))

    return company_sections




async def ingest_company_playbook() -> None:
    pdf_paths = _resolve_pdf_paths()
    if not pdf_paths:
        print("No playbook PDFs found")
        return

    merged_sections: Dict[str, List[str]] = {}
    source_pdfs: Dict[str, List[str]] = {}
    for pdf_path in pdf_paths:
        pdf_text = extract_pdf_text(pdf_path)
        for company_name, section_text in _split_by_company(pdf_text):
            if not section_text:
                continue
            merged_sections.setdefault(company_name, []).append(section_text)
            source_pdfs.setdefault(company_name, []).append(pdf_path.name)
        print(f"Read {pdf_path.name}")

    if not merged_sections:
        print("No company sections found in playbook PDFs")
        return

    with db_pool.get_connection() as conn:
        if not conn:
            raise RuntimeError("Database connection unavailable")
        with conn.cursor() as cur:
            ensure_company_playbook_table(cur)
            for company_name, section_texts in merged_sections.items():
                cur.execute(
                    "DELETE FROM company_playbook_chunks WHERE LOWER(TRIM(company_name)) = LOWER(TRIM(%s))",
                    (company_name,),
                )
                combined_text = "\n\n".join(section_texts)
                chunks = chunk_playbook_section(combined_text)
                if not chunks:
                    print(f"{company_name}: 0 chunks")
                    continue

                for chunk in chunks:
                    text = str(chunk["text"])
                    embedding = await embed_text(text, task_type="retrieval_document")
                    cur.execute(
                        """
                        INSERT INTO company_playbook_chunks (company_name, source_section, chunk_text, embedding, metadata)
                        VALUES (%s, %s, %s, %s::vector, %s)
                        """,
                        (
                            company_name,
                            chunk.get("section"),
                            text,
                            embedding,
                            Json({"source_pdf": source_pdfs[company_name]}),
                        ),
                    )
                conn.commit()
                print(f"{company_name}: {len(chunks)} chunks ingested")


if __name__ == "__main__":
    asyncio.run(ingest_company_playbook())
