"""Shared PDF-extraction and chunking logic for company playbook ingestion.

Used by both the offline `data/company_playbooks/ingest_kb.py` script and the
admin "upload playbook" endpoint, so there is one implementation of each step.
"""

import re
from typing import BinaryIO, Dict, List, Tuple, Union

import pdfplumber

SECTION_TITLES = {
    "company overview",
    "interview process",
    "technical questions",
    "behavioral questions",
    "resume questions",
    "common mistakes",
    "hiring signals",
    "faq",
    "ai mock interview dataset",
}


def extract_pdf_text(pdf_source: Union[str, BinaryIO]) -> str:
    """Extract text from a PDF given a file path or a binary file-like object."""
    text = ""
    with pdfplumber.open(pdf_source) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text:
                text += page_text + "\n"
    return text


def chunk_playbook_section(section_text: str, max_chars: int = 1500) -> List[Dict[str, str]]:
    """Split a company's playbook text into section-title-aware chunks."""
    lines = section_text.splitlines()
    subsections: List[Tuple[str, str]] = []
    current_title = None
    current_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.lower() in SECTION_TITLES:
            if current_lines:
                subsections.append((current_title, "\n".join(current_lines).strip()))
            current_title = stripped
            current_lines = []
            continue
        current_lines.append(line)
    if current_lines:
        subsections.append((current_title, "\n".join(current_lines).strip()))

    chunks: List[Dict[str, str]] = []
    for section_title, text in subsections:
        text = text.strip()
        if not text:
            continue
        if len(text) <= max_chars:
            chunks.append({"text": text, "section": section_title})
            continue

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        buffer = ""
        for paragraph in paragraphs:
            candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
            if len(candidate) > max_chars and buffer:
                chunks.append({"text": buffer, "section": section_title})
                buffer = paragraph
            else:
                buffer = candidate
        if buffer:
            chunks.append({"text": buffer, "section": section_title})

    return chunks
