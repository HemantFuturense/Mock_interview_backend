import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from .config import config
from .models import QuestionObject, KnowledgeChunk
from .providers.factory import get_llm_provider
from .rag_service import rag_service
from .governor import ProviderBlockedException, QuotaExhaustedException
from .state_manager import state_manager

class GeneratorService:
    """Generates structured, role-invariant interview questions using Gemini Flash-Lite."""
    def __init__(self):
        self.llm = get_llm_provider()

    async def _generate_json_with_bounded_retry(self, prompt: str, role: str, experience_band: str):
        """Calls the LLM for a generation batch with exactly ONE retry if the
        response is malformed JSON (e.g. the model appends trailing content
        after a complete JSON value -- observed live as
        "Extra data: line 81 column 1"). The provider layer already applies
        robust extraction (providers/base.py: extract_json_object) before
        this is ever reached, so this only fires when that still isn't
        enough. The retry re-sends the IDENTICAL prompt -- it never edits,
        repairs, or fabricates question content, it just asks again.
        QuotaExhaustedException/ProviderBlockedException are never retried
        here; they propagate immediately so the caller's existing
        quota/blocked handling applies unchanged. If the retry also fails,
        the JSONDecodeError propagates to the caller, which already treats a
        failed batch as FAILED/PARTIAL and continues without blocking the
        rest of the pipeline -- no further retries are attempted."""
        try:
            return await self.llm.generate_json(
                prompt=prompt,
                system_prompt="You are a strict, world-class technical hiring manager.",
                task_name="generation",
            )
        except json.JSONDecodeError as je:
            print(
                f"[GENERATOR] Malformed JSON response for {role}/{experience_band} batch "
                f"({je}). Retrying once with the same request..."
            )
            return await self.llm.generate_json(
                prompt=prompt,
                system_prompt="You are a strict, world-class technical hiring manager.",
                task_name="generation",
            )

    async def generate_batch_for_band(
        self,
        role: str,
        experience_band: str,
        count: int = 4,
        target_scope: str = "UNIVERSAL",
        domain: Optional[str] = None
    ) -> List[QuestionObject]:
        """Generate a batch of questions for a specific role and experience band."""
        # Determine paradigm appropriate for experience band (shared with
        # validator_service via config.PARADIGM_BY_EXPERIENCE_BAND so the two
        # can never drift out of sync).
        paradigm = config.PARADIGM_BY_EXPERIENCE_BAND.get(experience_band, "ARCHITECTURE")

        # Retrieve RAG context chunks: role knowledge (as before) plus
        # whatever company knowledge is actually semantically relevant to
        # this role/band/paradigm query (rag_service.retrieve_company_context
        # ranks ALL indexed company chunks by cosine similarity and only
        # returns ones clearing config.COMPANY_CONTEXT_MIN_SIMILARITY -- a
        # company whose research doesn't relate to this query returns
        # nothing, so this never degenerates into injecting a fixed company
        # list). top_k=1 keeps company context supplementary, not dominant.
        chunks = await rag_service.retrieve_context(role, experience_band, paradigm, top_k=2)
        company_chunks = await rag_service.retrieve_company_context(role, experience_band, paradigm, top_k=1)

        source_refs = []
        for c in chunks:
            ref_entry = f"RAG Chunk: {c.chunk_id} | Topic: {c.topic}"
            if c.source_reference and c.source_reference != "Synthesized Research":
                ref_entry += f" | Provenance: {c.source_reference}"
            source_refs.append(ref_entry)
        for c in company_chunks:
            ref_entry = f"RAG Chunk: {c.chunk_id} | Company: {c.company} | Topic: {c.topic}"
            if c.source_reference and c.source_reference != "Synthesized Research":
                ref_entry += f" | Provenance: {c.source_reference}"
            source_refs.append(ref_entry)
        if not source_refs:
            source_refs = [f"Curated Knowledge Dossier for {role}"]

        rag_context = "\n\n".join([f"[{c.topic}]: {c.text}" for c in chunks]) if chunks else f"Curated industry knowledge and architectural patterns for {role}."
        if company_chunks:
            rag_context += "\n\n" + "\n\n".join(
                f"[Company Context -- {c.company}, topic: {c.topic}]: {c.text}" for c in company_chunks
            )

        prompt = (
            f"You are a Principal Engineering Interviewer at a top technology company.\n"
            f"Generate {count} unique, production-grade technical interview questions for the role: '{role}'.\n\n"
            f"CRITICAL CONSTRAINT - ROLE INVARIANCE:\n"
            f"The candidate's title/role is strictly '{role}'. Do NOT change or prefix the role name.\n\n"
            f"TARGET PARAMETERS:\n"
            f"- Role: '{role}'\n"
            f"- Experience Band: '{experience_band}' years\n"
            f"- Question Paradigm: '{paradigm}'\n"
            f"- Target Scope: '{target_scope}' (UNIVERSAL = applicable to most tech companies; DOMAIN = relevant to domain like FinTech/E-Commerce; COMPANY = company-specific)\n"
            f"- Domain: '{domain or 'GENERAL_TECH'}'\n\n"
            f"RELEVANT RAG KNOWLEDGE CONTEXT:\n"
            f"{rag_context}\n\n"
            f"SCOPE & COMPANY APPLICABILITY RULES (read carefully -- this is checked, not decorative):\n"
            f"- Determine 'applicable_companies' and 'domains' from the ACTUAL CONTENT of the question you just wrote. Never copy a default or habitual list.\n"
            f"- UNIVERSAL question (any competent tech company could ask it): 'applicable_companies' MUST be an empty list [].\n"
            f"- DOMAIN question (tied to a real industry/domain, e.g. FinTech, E-Commerce, Streaming, Healthcare): list 2-5 REAL companies that genuinely operate in that domain and would plausibly ask this exact question.\n"
            f"- COMPANY question (could only be meaningfully asked by ONE company because it depends on that company's specific scale, product, or proprietary system): list EXACTLY ONE real company, and the question text itself must make that dependency evident.\n"
            f"- NEVER invent a company that would not realistically ask this. NEVER pad the list with well-known names 'to be safe'. If you are unsure a company genuinely applies, leave it out.\n"
            f"- If the RAG context above includes a '[Company Context -- ...]' block, treat it as OPTIONAL supplementary evidence, not an instruction to write a company-specific question. "
            f"Use it only if it is genuinely relevant to what you're writing; a UNIVERSAL or DOMAIN target_scope question with company context present in the RAG should still end up UNIVERSAL/DOMAIN "
            f"(applicable_companies determined per the rules above) unless that company's context makes the question genuinely require it.\n\n"
            f"CALIBRATION GUIDELINES FOR EXPERIENCE BAND '{experience_band}':\n"
            f"- 0-1 yrs (EXECUTION): Junior-appropriate, but must still require REASONING or JUDGMENT, never pure recall.\n"
            f"    AVOID: definition/\"explain X\" questions, trivial CRUD/API wrapper instructions with no decision to make, "
            f"basic dataframe/data-cleaning steps with one obvious correct answer, generic \"how would you evaluate a model\" prompts.\n"
            f"    PREFER: debugging a realistic small ML issue from given symptoms/evidence, choosing between two reasonable approaches and justifying the choice, "
            f"interpreting a confusing model/evaluation result and identifying the likely cause, a small but genuine data-quality or feature-quality judgment call, "
            f"or a basic production consideration (e.g. why a metric looks good in training but degrades in production) that still requires applying judgment.\n"
            f"    Difficulty 3-5. Depth must stay genuinely junior -- do NOT introduce distributed systems, multi-service architecture, or org-level concerns here.\n"
            f"- 1-2 yrs (IMPLEMENTATION): Focus on concrete feature delivery, API handling, component integration, unit testing, error handling. Difficulty 4-6.\n"
            f"- 3-5 yrs (ARCHITECTURE): Focus on system architecture, database indexing, caching strategies, concurrency, microservice communication. Difficulty 6-8.\n"
            f"- 5-8 yrs (STRATEGY): Focus on end-to-end distributed system design, disaster recovery, zero-downtime migrations, cross-service bottlenecks. Difficulty 7-9.\n"
            f"- 8+ yrs (DOMAIN_OWNERSHIP): Focus on org-wide technical trade-offs, multi-region reliability, cost vs latency compromises, tech debt governance. Difficulty 8-10.\n\n"
            f"Every question, in every band, must require the candidate to reason, decide, diagnose, or interpret -- not merely recite a definition or follow one obvious instruction. "
            f"A strict automated quality gate will reject trivial or definition-only questions; this applies especially to the 0-1 year band.\n\n"
            f"Return a JSON object with this EXACT structure:\n"
            "{\n"
            '  "questions": [\n'
            "    {\n"
            '      "question": "Clear, detailed technical scenario or system question",\n'
            f'      "role": "{role}",\n'
            f'      "experience_band": "{experience_band}",\n'
            '      "difficulty": 7,\n'
            '      "technical_depth": 7,\n'
            '      "problem_complexity": 7,\n'
            '      "architecture_complexity": 7,\n'
            '      "troubleshooting": 6,\n'
            '      "business_complexity": 5,\n'
            '      "decision_making": 6,\n'
            '      "leadership_ownership": 4,\n'
            '      "question_type": "technical",\n'
            f'      "paradigm": "{paradigm}",\n'
            '      "mandatory_skills": ["Skill1", "Skill2"],\n'
            f'      "scope": "{target_scope}",\n'
            '      "domains": ["<genuine domain tag, or omit/empty if none>"],\n'
            '      "applicable_companies": []\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        try:
            res = await self._generate_json_with_bounded_retry(prompt, role, experience_band)
        except QuotaExhaustedException as qe:
            state_manager.mark_waiting_for_quota(role, f"Generation quota exhausted: {qe.message}")
            raise
        except ProviderBlockedException as pbe:
            state_manager.mark_blocked(role, f"Generation blocked: {pbe.message}")
            raise

        data = res.data or {}
        raw_list = data.get("questions", [])
        questions: List[QuestionObject] = []

        now_str = datetime.now(timezone.utc).isoformat()
        for item in raw_list:
            try:
                # Force role invariance
                item["role"] = role
                item["experience_band"] = experience_band
                item["paradigm"] = paradigm
                item["source_references"] = source_refs
                item["knowledge_version"] = "v1.0"
                item["generated_at"] = now_str
                q = QuestionObject(**item)
                questions.append(q)
            except Exception as e:
                print(f"[GENERATOR] Warning: Question validation parsing error: {e}")

        return questions

generator_service = GeneratorService()
