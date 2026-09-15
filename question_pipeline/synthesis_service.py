import json
import re
from typing import List, Dict, Any
from pathlib import Path
from .config import config
from .models import ResearchOutput, KnowledgeChunk
from .providers.factory import get_llm_provider
from .governor import ProviderBlockedException, QuotaExhaustedException
from .state_manager import state_manager

def slugify(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '_', text).lower()

class SynthesisService:
    """Synthesizes raw research into structured interview knowledge chunks using Gemini."""
    def __init__(self):
        self.llm = get_llm_provider()

    async def synthesize(self, research: ResearchOutput) -> List[KnowledgeChunk]:
        print(f"[SYNTHESIS] Synthesizing knowledge for role '{research.role}' via {self.llm.provider_name}...")

        prompt = (
            f"Synthesize the following web research dossier for '{research.role}' into 4-6 distinct, high-impact "
            "technical knowledge modules suitable for generating deep interview questions.\n\n"
            f"Research Content:\n{research.raw_summary}\n\n"
            "Return JSON matching this schema:\n"
            "{\n"
            '  "modules": [\n'
            '    {\n'
            '      "topic": "System Architecture & Scalability",\n'
            '      "content": "Detailed text highlighting architecture, trade-offs, and critical components."\n'
            '    }\n'
            '  ]\n'
            "}"
        )

        try:
            res = await self.llm.generate_json(
                prompt=prompt,
                system_prompt="You are a principal technical recruiter and systems architect.",
                task_name="synthesis"
            )
        except QuotaExhaustedException as qe:
            state_manager.mark_waiting_for_quota(research.role, f"Synthesis quota exhausted: {qe.message}")
            raise
        except ProviderBlockedException as pbe:
            state_manager.mark_blocked(research.role, f"Synthesis blocked: {pbe.message}")
            raise

        data = res.data or {}
        modules = data.get("modules", [])
        if not modules:
            # Fallback chunking from raw summary
            modules = [
                {"topic": "Core Competencies & Stack", "content": f"{research.role} modern stack and technologies."},
                {"topic": "Architecture & Engineering Practices", "content": research.raw_summary[:800]},
                {"topic": "Troubleshooting & High-Load Failures", "content": research.raw_summary[800:1600]}
            ]

        chunks: List[KnowledgeChunk] = []
        slug = slugify(research.role)
        for i, m in enumerate(modules):
            primary_ref = "; ".join(research.source_references) if research.source_references else f"Web Research Dossier: {research.role} ({research.actual_provider})"
            chunk = KnowledgeChunk(
                chunk_id=f"{slug}_chunk_{i+1}",
                role=research.role,
                topic=m.get("topic", f"Topic {i+1}"),
                text=m.get("content", ""),
                source_reference=primary_ref,
                knowledge_version=research.knowledge_version
            )
            chunks.append(chunk)

        # Persist synthesized knowledge chunks
        out_file = config.KNOWLEDGE_DIR / f"{slug}_synthesized.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in chunks], f, indent=2)

        print(f"[SYNTHESIS] Successfully synthesized {len(chunks)} knowledge modules for '{research.role}'.")
        return chunks

synthesis_service = SynthesisService()
