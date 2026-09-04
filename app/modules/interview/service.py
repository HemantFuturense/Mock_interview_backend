import asyncio
import difflib
import json
import re
import shutil
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import BackgroundTasks, HTTPException, UploadFile
from psycopg2.extras import Json

from app.config.constants import (
    BEHAVIORAL_COMPETENCIES,
    DIFFICULTY_WEIGHTS,
    HR_COMPETENCIES,
    QUESTION_TYPE_ALIAS_MAP,
    SCORING_GUIDE,
    TECHNICAL_COMPETENCIES,
)
from app.config.settings import settings
from app.core.cache import cached, questions_cache, session_cache
from app.core.database import db_pool
from app.core.logger import logger
from app.modules.ai.client import generate_content_with_fallback
from app.modules.ai.grading import (
    analyze_answer_and_generate_response,
    extract_scores_from_feedback,
    update_scores_from_feedback,
)
from app.modules.ai.prompts import jinja_env, render_template
from app.modules.ai.video import process_video_analysis_background
from app.modules.interview.repository import InterviewRepository
from app.modules.interview.schemas import InterviewResponse
from app.modules.rag.service import retrieve_company_context
from app.utils.text_utils import _clean_json_text, _parse_feedback_json


def _get_question_type_aliases(normalized_label: Optional[str]) -> List[str]:
    if not normalized_label:
        return []
    if normalized_label in QUESTION_TYPE_ALIAS_MAP:
        return sorted(list(QUESTION_TYPE_ALIAS_MAP[normalized_label]))
    for key, aliases in QUESTION_TYPE_ALIAS_MAP.items():
        if normalized_label in aliases:
            return sorted(list(aliases))
    return [normalized_label]


def _translate_difficulty_label(raw_difficulty: Optional[str], default: str = "medium") -> str:
    if not raw_difficulty:
        return default
    normalized = raw_difficulty.strip().lower()
    mapping = {
        "easy": "easy",
        "basic": "easy",
        "beginner": "easy",
        "medium": "medium",
        "intermediate": "medium",
        "hard": "difficult",
        "difficult": "difficult",
        "advanced": "difficult",
    }
    return mapping.get(normalized, default)


def _normalize_question_type_label(raw_label: Optional[str]) -> Optional[str]:
    if not raw_label:
        return None
    normalized = raw_label.strip().lower()
    mapping = {
        "standard": "standard",
        "general": "standard",
        "behavioral": "behavioral",
        "behavioural": "behavioral",
        "technical": "technical",
        "tech": "technical",
        "coding": "coding",
        "code": "coding",
        "system design": "system design",
        "system-design": "system design",
        "sql": "sql",
        "database": "sql",
        "speech": "speech",
        "speech based": "speech",
    }
    return mapping.get(normalized, normalized)


def _compute_running_average(current_avg: float, new_score: float, count: int) -> float:
    if count <= 1:
        return round(new_score, 2)
    updated = ((current_avg * (count - 1)) + new_score) / count
    return round(updated, 2)


def _determine_max_questions(difficulty_weights: List[int], answered_count: int = 0) -> int:
    try:
        average_weight = sum(difficulty_weights) / len(difficulty_weights) if difficulty_weights else 2.0
    except ZeroDivisionError:
        average_weight = 2.0

    if average_weight < 1.4:
        calculated_max = 5
    elif average_weight < 2.3:
        calculated_max = 6
    else:
        calculated_max = 7

    return max(calculated_max, answered_count)


def _extract_first_number(text: str) -> Optional[float]:
    if not text:
        return None
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None


def _normalize_video_analysis(raw_text: str) -> Tuple[float, str]:
    sentiment = 5.0
    demeanor = "neutral"
    if not raw_text:
        return sentiment, demeanor
    for line in raw_text.splitlines():
        line = line.strip()
        if line.upper().startswith("SENTIMENT:"):
            value_str = line.split(":", 1)[1].strip()
            extracted = _extract_first_number(value_str)
            if extracted is not None:
                sentiment = max(0.0, min(10.0, extracted))
        elif line.upper().startswith("DEMEANOR:"):
            demeanor = line.split(":", 1)[1].strip() or "neutral"
    return sentiment, demeanor


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def _is_system_design_question_type(question_type: Optional[str]) -> bool:
    if not question_type:
        return False
    normalized = question_type.strip().lower()
    return "system" in normalized and "design" in normalized


def _prepare_system_design_answer_payload(diagram_raw: Optional[str], summary: Optional[str]) -> str:
    base_payload: Dict[str, Any] = {}
    if diagram_raw:
        try:
            parsed = json.loads(diagram_raw)
            if isinstance(parsed, dict):
                base_payload = parsed
            else:
                base_payload["raw"] = diagram_raw
        except Exception:
            base_payload["raw"] = diagram_raw
    base_payload.setdefault("nodes", base_payload.get("nodes", []))
    base_payload.setdefault("edges", base_payload.get("edges", []))

    # The live frontend submits the SAME diagram JSON in both the diagram field and the
    # answer/summary field (InterviewScreen.js sends `systemDesignDiagram` as both `answer`
    # and `system_design_diagram`), not a separate free-text explanation. Blindly nesting
    # `summary` into the payload in that case double-encodes the diagram JSON inside itself.
    # Only attach `summary` when it's genuinely distinct free text, not a duplicate/second
    # copy of diagram-shaped JSON.
    if summary and summary.strip() != (diagram_raw or "").strip():
        summary_is_diagram_json = False
        try:
            summary_parsed = json.loads(summary)
            if isinstance(summary_parsed, dict) and ("nodes" in summary_parsed or "edges" in summary_parsed):
                summary_is_diagram_json = True
        except Exception:
            pass
        if not summary_is_diagram_json:
            base_payload["summary"] = summary

    return json.dumps(base_payload, ensure_ascii=False)


def _describe_system_design(answer_payload: Optional[str]) -> str:
    if not answer_payload:
        return "No system design diagram was provided."
    try:
        parsed = json.loads(answer_payload)
        if not isinstance(parsed, dict):
            return f"System design diagram payload: {answer_payload}"
    except Exception:
        return f"System design diagram payload: {answer_payload}"

    nodes = parsed.get("nodes") or []
    edges = parsed.get("edges") or []
    summary = parsed.get("summary")

    description_lines = []
    if summary:
        description_lines.append(f"Candidate summary: {summary}")

    if isinstance(nodes, list) and nodes:
        description_lines.append(f"The diagram has {len(nodes)} components:")
        for node in nodes[:20]:
            label = node.get("data", {}).get("label") or node.get("label") or node.get("id")
            desc = node.get("data", {}).get("description") or node.get("description")
            description_lines.append(f"- Component '{label}'{f' ({desc})' if desc else ''}")
        if len(nodes) > 20:
            description_lines.append(f"- ...and {len(nodes) - 20} more nodes.")
    else:
        description_lines.append("No components were provided in the diagram.")

    if isinstance(edges, list) and edges:
        description_lines.append(f"There are {len(edges)} connections defined.")
    else:
        description_lines.append("No connections were included.")

    return "\n".join(description_lines)


def _log_question_selection(
    source: str,
    role: str,
    company: str,
    difficulty: str,
    interview_type: Optional[str],
    work_experience: Optional[str],
    question_type: Optional[str],
    question_text: str,
) -> None:
    truncated = (question_text or "").replace("\n", " ").strip()
    if len(truncated) > 120:
        truncated = truncated[:117] + "..."
    logger.info(
        f"[QuestionSelect | {source}] role='{role}' company='{company}' difficulty='{difficulty}' "
        f"interview_type='{interview_type or 'any'}' work_exp='{work_experience or 'any'}' "
        f"question_type='{question_type or 'standard'}' text='{truncated}'"
    )


def _save_uploaded_video(session_id: str, question_number: int, upload: UploadFile | None) -> Optional[str]:
    if not upload or not upload.filename:
        return None
    try:
        upload_dir = settings.MEDIA_ROOT / session_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_ext = Path(upload.filename).suffix or ".webm"
        file_path = upload_dir / f"q_{question_number}{file_ext}"
        if hasattr(upload, "file") and upload.file:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(upload.file, buffer)
        else:
            content = upload.file.read() if hasattr(upload.file, "read") else None
            if content:
                with open(file_path, "wb") as buffer:
                    buffer.write(content)
        return str(file_path.relative_to(settings.MEDIA_ROOT.parent)) if settings.MEDIA_ROOT.parent in file_path.parents else str(file_path)
    except Exception as exc:
        logger.error(f"Failed saving video for session {session_id} Q#{question_number}: {exc}")
        return None


def _compose_conversation_excerpt(entries: List[str], char_limit: int, minimum_entries: int = 3) -> str:
    if not entries:
        return "No complete question-answer pairs recorded."
    if len(entries) <= minimum_entries:
        return "\n\n".join(entries)

    excerpt = []
    current_chars = 0
    for entry in entries:
        if current_chars + len(entry) + 2 <= char_limit or len(excerpt) < minimum_entries:
            excerpt.append(entry)
            current_chars += len(entry) + 2
        else:
            break

    if len(excerpt) < len(entries):
        excerpt.append(f"\n... plus {len(entries) - len(excerpt)} additional Q&A exchange(s) truncated for token economy ...")
    return "\n\n".join(excerpt)


def _resolve_feedback_template(interview_type: Optional[str]) -> Dict[str, Any]:
    normalized_type = (interview_type or "").strip().lower()
    is_coding_interview = any(
        keyword in normalized_type for keyword in ["coding", "technical", "software engineer", "developer"]
    )
    is_behavioral_interview = any(
        keyword in normalized_type for keyword in ["behavioral", "hr", "manager", "leadership", "culture"]
    )

    if is_coding_interview:
        return {
            "question_template": "technical_questions_prompt.j2",
            "competency_template": "technical_competencies_prompt.j2",
            "feedback_type": "technical_coding",
            "competencies": TECHNICAL_COMPETENCIES,
        }
    elif is_behavioral_interview:
        return {
            "question_template": "behavioral_questions_prompt.j2",
            "competency_template": "behavioral_competencies_prompt.j2",
            "feedback_type": "behavioral_leadership",
            "competencies": BEHAVIORAL_COMPETENCIES,
        }
    else:
        return {
            "question_template": "hr_questions_prompt.j2",
            "competency_template": "hr_competencies_prompt.j2",
            "feedback_type": "comprehensive_standard",
            "competencies": HR_COMPETENCIES,
        }


@cached(cache=questions_cache)
def get_questions_cached(
    role: str,
    industry: str,
    company: str,
    difficulty: str,
    limit: int = 10,
) -> list:
    try:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                filters = ["LOWER(TRIM(role)) = LOWER(TRIM(%s))"]
                params = [role]
                if industry and industry.lower() != "any":
                    filters.append("LOWER(TRIM(industry)) = LOWER(TRIM(%s))")
                    params.append(industry)
                if company and company.lower() != "any":
                    filters.append("LOWER(TRIM(company)) = LOWER(TRIM(%s))")
                    params.append(company)
                if difficulty and difficulty.lower() != "any":
                    filters.append("LOWER(TRIM(difficulty)) = LOWER(TRIM(%s))")
                    params.append(difficulty)

                where_clause = " AND ".join(filters)
                cur.execute(
                    f"""
                    SELECT question, mandatory_skills, difficulty, question_type, interview_type, work_experience
                    FROM interview_questions
                    WHERE {where_clause}
                    ORDER BY RANDOM()
                    LIMIT %s
                    """,
                    tuple(params + [limit]),
                )
                rows = cur.fetchall()
                return [
                    {
                        "question": row[0],
                        "mandatory_skills": row[1] or "Communication, Problem-solving",
                        "difficulty": row[2] or "medium",
                        "question_type": (row[3] or "standard").lower(),
                        "interview_type": (row[4] or "").lower(),
                        "work_experience": (row[5] or "").lower(),
                    }
                    for row in rows
                ]
    except Exception as exc:
        logger.error(f"Error fetching cached questions: {exc}")
        return []


def _build_company_question_prompt(
    job_role: str,
    industry_type: Optional[str],
    company_name: str,
    interview_type: Optional[str],
    work_experience: Optional[str],
    question_type_filter: Optional[str],
    preferred_difficulty: str,
    context_chunks: List[str],
    already_asked_questions: Optional[List[str]] = None,
) -> str:
    context_block = ""
    if context_chunks:
        joined_context = "\n---\n".join(context_chunks)
        context_block = (
            "\n\nCompany Interview Context (from internal knowledge base):\n"
            f"{joined_context}\n"
            "Use this context only when relevant to shape the question's style and focus areas. "
            "Do not invent facts beyond it.\n"
        )

    desired_type_line = (
        f"Desired question type: {question_type_filter}"
        if question_type_filter
        else "Desired question type: choose whichever of standard, behavioral, technical, coding, "
        "system design, sql, or speech best fits this role, company, and interview type."
    )

    avoid_block = ""
    if already_asked_questions:
        listed = "\n".join(f"- {q}" for q in already_asked_questions[:20])
        avoid_block = (
            "\n\nQuestions already asked earlier in this same interview -- you MUST NOT repeat "
            "any of these, and must not ask a close variant/rephrasing of any of them (pick a "
            "different topic, data structure, or problem area entirely):\n"
            f"{listed}\n"
        )

    return f"""You are an expert technical interviewer preparing a single interview question.

Role: {job_role}
Industry: {industry_type or 'General'}
Company: {company_name}
Interview type: {interview_type or 'General'}
Candidate experience level: {work_experience or 'Any'}
{desired_type_line}
Desired difficulty: {preferred_difficulty}
{context_block}{avoid_block}
Generate exactly ONE interview question appropriate for this role, company, and difficulty level.

CRITICAL classification rule: if the question asks the candidate to design, architect, or build a
system, service, or component (e.g. "design a system that...", "design an architecture for...",
"how would you build a service that..."), question_type MUST be "system design", never "technical"
or "coding", even if the underlying skills are technical.

Respond ONLY with a JSON object in this exact shape, no markdown fences, no extra commentary:
{{
  "question": "<the interview question text>",
  "mandatory_skills": "<comma-separated key skills this question assesses>",
  "difficulty": "easy | medium | difficult",
  "question_type": "standard | behavioral | technical | coding | system design | sql | speech"
}}"""


_SYSTEM_DESIGN_QUESTION_SIGNAL_RE = re.compile(
    r"^(?:design|architect)\s+(?:a|an|the)\b(?!.{0,40}\b(?:algorithm|function)\b)",
    re.IGNORECASE,
)


def _question_text_looks_like_system_design(question_text: str) -> bool:
    """Content-based safety net: Gemini's own question_type self-report during generation
    has been observed to mislabel obvious system-design questions as 'technical', even though
    a separate later Gemini call (feedback generation) correctly recognizes the same question
    text as system design. Questions that open with "Design a/an/the ..." are the standard
    system-design interview phrasing (cache, rate limiter, URL shortener, etc.) unless they're
    really asking for an algorithm/function (e.g. "Design an algorithm to..."), which is coding."""
    if not question_text:
        return False
    return bool(_SYSTEM_DESIGN_QUESTION_SIGNAL_RE.match(question_text.strip()))


def _is_duplicate_question(candidate: str, already_asked: List[str]) -> bool:
    """Catches exact repeats and near-duplicate rephrasings (Gemini has been observed to
    regenerate the same canonical question, e.g. "Valid Parentheses", across a session when
    it isn't told what's already been asked)."""
    if not candidate or not already_asked:
        return False
    normalized_candidate = candidate.strip().lower()
    for prior in already_asked:
        normalized_prior = (prior or "").strip().lower()
        if not normalized_prior:
            continue
        if normalized_candidate == normalized_prior:
            return True
        if difflib.SequenceMatcher(None, normalized_candidate, normalized_prior).ratio() > 0.8:
            return True
    return False
 

async def _generate_company_question_via_rag(
    job_role: str,
    industry_type: Optional[str],
    company_name: str,
    interview_type: Optional[str],
    work_experience: Optional[str],
    question_type_filter: Optional[str],
    preferred_difficulty: Optional[str],
    session_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """RAG+Gemini fallback used when a company-specific DB match is missing.
    Returns None (never raises) on any failure so callers fall through to
    find_generic_fallback_question."""
    from app.modules.rag.service import retrieve_company_context

    difficulty_label = preferred_difficulty or "medium"
    query = (
        f"{job_role} {question_type_filter or 'general interview'} interview question "
        f"at {difficulty_label} difficulty for {interview_type or 'general'} interview"
    )   

    try:
        context_chunks = await retrieve_company_context(company_name, query, top_k=5)
    except Exception as exc:
        logger.warning(f"RAG retrieval failed for company '{company_name}': {exc}")
        context_chunks = []

    already_asked = InterviewRepository.get_already_asked_questions(session_id)

    parsed = None
    question_text = ""
    for attempt in range(2):
        prompt = _build_company_question_prompt(
            job_role=job_role,
            industry_type=industry_type,
            company_name=company_name,
            interview_type=interview_type,
            work_experience=work_experience,
            question_type_filter=question_type_filter,
            preferred_difficulty=difficulty_label,
            context_chunks=context_chunks,
            already_asked_questions=already_asked,
        )
        try:
            gemini_response = await generate_content_with_fallback(
                prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.3 if attempt == 0 else 0.9,
                    "max_output_tokens": 500,
                },
                retry_label=f"Company question generation ({company_name}/{job_role}) attempt {attempt + 1}",
            )
            raw_text = gemini_response.text if gemini_response else ""
            parsed = json.loads(_clean_json_text(raw_text)) if raw_text and raw_text.strip() else None
        except Exception as exc:
            logger.warning(f"RAG+Gemini company question generation failed for '{company_name}'/{job_role}: {exc}")
            return None

        if not isinstance(parsed, dict):
            parsed = None
            continue

        question_text = str(parsed.get("question") or "").strip()
        if not question_text:
            continue

        if _is_duplicate_question(question_text, already_asked):
            logger.warning(
                f"RAG generated a duplicate/near-duplicate question for '{company_name}'/{job_role} "
                f"on attempt {attempt + 1}, retrying: {question_text[:80]!r}"
            )
            parsed = None
            question_text = ""
            continue

        break

    if not parsed or not question_text:
        return None

    mandatory_skills = parsed.get("mandatory_skills") or "Communication, Problem-solving"
    if isinstance(mandatory_skills, list):
        mandatory_skills = ", ".join(str(skill) for skill in mandatory_skills if skill)

    normalized_difficulty = _translate_difficulty_label(parsed.get("difficulty"), default=difficulty_label)
    normalized_question_type = (
        _normalize_question_type_label(parsed.get("question_type") or question_type_filter) or "standard"
    )
    if normalized_question_type != "system design" and _question_text_looks_like_system_design(question_text):
        normalized_question_type = "system design"

    generation_context = {
        "source": "rag_gemini",
        "trigger_reason": "company_specific_db_miss",
        "target_role": job_role,
        "company_context": company_name,
        "interview_type": interview_type,
        "work_experience": work_experience,
        "rag_context_used": bool(context_chunks),
        "rag_chunk_count": len(context_chunks),
    }

    return {
        "question": question_text,
        "mandatory_skills": mandatory_skills,
        "difficulty": normalized_difficulty,
        "question_type": normalized_question_type,
        "interview_type": interview_type,
        "work_experience": work_experience,
        "generation_context": generation_context,
    }


async def _resolve_next_question(
    job_role: str,
    industry_type: Optional[str],
    company_name: str,
    question_number: int,
    session_id: str,
    interview_type: Optional[str],
    work_experience: Optional[str],
    question_type_filter: Optional[str] = None,
    preferred_difficulty: Optional[str] = None,
) -> Dict[str, Any]:
    """Question resolution order: company-specific DB match -> RAG+Gemini generation
    (triggered as soon as the company-specific DB lookup misses) -> company-agnostic
    DB relaxation chain / hardcoded placeholder as the final safety net."""
    company_question = InterviewRepository.find_company_specific_question(
        job_role,
        industry_type,
        company_name,
        session_id,
        interview_type=interview_type,
        work_experience=work_experience,
        question_type_filter=question_type_filter,
        preferred_difficulty=preferred_difficulty,
    )
    if company_question:
        return company_question

    try:
        rag_question = await _generate_company_question_via_rag(
            job_role,
            industry_type,
            company_name,
            interview_type,
            work_experience,
            question_type_filter,
            preferred_difficulty,
            session_id=session_id,
        )
    except Exception as exc:
        logger.warning(f"RAG+Gemini generation raised for company '{company_name}'/{job_role}: {exc}")
        rag_question = None

    if rag_question:
        return rag_question

    return InterviewRepository.find_generic_fallback_question(
        job_role,
        industry_type,
        company_name,
        question_number,
        session_id,
        interview_type=interview_type,
        work_experience=work_experience,
        question_type_filter=question_type_filter,
    )


async def start_interview_service(
    request: Any,
    student_name: str,
    student_email: Optional[str],
    job_role: str,
    industry_type: str,
    company_name: str,
    interview_type: Optional[str],
    work_experience: Optional[str],
    job_description_id: Optional[int],
    job_description_text: Optional[str],
    job_description_raw_text: Optional[str],
    force_reattempt: bool,
    use_resume_questions: bool,
    resume_id: Optional[int],
    pre_generated_questions: Optional[str],
) -> Dict[str, Any]:
    normalized_interview_type = (interview_type or "").strip()
    normalized_work_experience = (work_experience or "").strip()
    job_description_id_value = job_description_id
    job_description_text_value = (job_description_text or "").strip() or None
    job_description_raw_value = (job_description_raw_text or "").strip() or None

    pre_generated_question_list: List[Dict[str, Any]] = []
    if pre_generated_questions:
        try:
            decoded = json.loads(pre_generated_questions)
            if isinstance(decoded, list):
                pre_generated_question_list = decoded
        except json.JSONDecodeError:
            logger.warning("Invalid pre_generated_questions payload provided; falling back to standard flow")

    resume_generation_enabled = bool(use_resume_questions and pre_generated_question_list)
    if resume_generation_enabled and pre_generated_question_list:
        first_payload_context = pre_generated_question_list[0].get("generation_context") or {}
        if not job_description_text_value:
            excerpt = (first_payload_context.get("job_description_excerpt") or "").strip()
            if excerpt:
                job_description_text_value = excerpt
        if not job_description_raw_value and job_description_text_value and first_payload_context.get("job_description_used"):
            job_description_raw_value = job_description_text_value

    if not force_reattempt and not resume_generation_enabled:
        is_company_card = False
        if hasattr(request, "headers") and hasattr(request, "query_params"):
            is_company_card = (
                request.headers.get("X-Request-Source") == "company-card"
                or request.query_params.get("source") == "company-card"
                or request.query_params.get("is_company_card", "").lower() == "true"
            )
        try:
            existing_sessions = InterviewRepository.find_existing_sessions(
                student_name=student_name,
                student_email=student_email,
                job_role=job_role,
                industry_type=industry_type,
                company_name=company_name,
                interview_type=normalized_interview_type,
                work_experience=normalized_work_experience,
                is_company_card=is_company_card,
            )
            if existing_sessions:
                return {
                    "requires_confirmation": True,
                    "existing_sessions": existing_sessions,
                    "message": "Existing interview attempts found. Confirm to start a reattempt.",
                }
        except Exception as exc:
            logger.error(f"Error finding existing sessions: {exc}", exc_info=True)

    session_id = InterviewRepository.create_enhanced_session(
        student_name,
        student_email,
        job_role,
        industry_type,
        company_name,
        normalized_interview_type,
        normalized_work_experience,
    )
    if not session_id:
        raise HTTPException(status_code=500, detail="Failed to create session")

    student_id = InterviewRepository.find_student_id(student_name, student_email)

    if job_description_id_value and student_id and not job_description_raw_value:
        try:
            with db_pool.get_connection() as conn:
                if conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT job_desc FROM job_descriptions WHERE id = %s AND (student_id = %s OR student_id IS NULL) LIMIT 1",
                            (job_description_id_value, student_id),
                        )
                        jd_lookup = cur.fetchone()
                        if jd_lookup and jd_lookup[0]:
                            fetched_raw = (jd_lookup[0] or "").strip()
                            if fetched_raw:
                                job_description_raw_value = fetched_raw
                                if not job_description_text_value:
                                    job_description_text_value = fetched_raw[:1000]
        except Exception as exc:
            logger.warning(f"Unable to fetch job description text for id {job_description_id_value}: {exc}")

    if job_description_text_value and not job_description_raw_value:
        job_description_raw_value = job_description_text_value

    if resume_generation_enabled:
        InterviewRepository.store_pre_generated_questions(
            session_id=session_id,
            student_id=student_id,
            resume_id=resume_id,
            job_description_id=job_description_id_value,
            job_description_text=job_description_text_value,
            job_desc=job_description_raw_value,
            company_name=company_name,
            job_role=job_role,
            interview_type=normalized_interview_type,
            work_experience=normalized_work_experience,
            questions=pre_generated_question_list,
        )

    if resume_generation_enabled and resume_id:
        InterviewRepository._attach_resume_metadata_to_session(session_id, resume_id)

    InterviewRepository._attach_job_description_metadata_to_session(
        session_id,
        job_description_id_value,
        job_description_text_value,
        job_description_raw_value,
    )

    payload_mandatory: Union[str, List[str]] = []
    if resume_generation_enabled:
        first_question_payload = pre_generated_question_list[0]
        first_question = first_question_payload.get("question", "Tell me about yourself.")
        payload_mandatory = first_question_payload.get("mandatory_skills") or []
        mandatory_skills = (
            payload_mandatory if isinstance(payload_mandatory, str) else ", ".join(payload_mandatory)
        )
        first_question_difficulty = first_question_payload.get("difficulty", "medium") or "medium"
        first_question_type = first_question_payload.get("question_type", "standard")
        first_question_context = first_question_payload.get("generation_context") or {}
        current_max_questions = max(len(pre_generated_question_list), 1)
    else:
        first_question_data = await _resolve_next_question(
            job_role,
            industry_type,
            company_name,
            1,
            session_id,
            interview_type=normalized_interview_type,
            work_experience=normalized_work_experience,
            preferred_difficulty="medium",
        )
        first_question = first_question_data["question"]
        mandatory_skills = first_question_data["mandatory_skills"]
        first_question_difficulty = first_question_data.get("difficulty", "medium") or "medium"
        first_question_type = first_question_data.get("question_type", "standard")
        first_question_context = first_question_data.get("generation_context") or {}
        first_difficulty_label = (first_question_difficulty or "medium").lower()
        first_weight = DIFFICULTY_WEIGHTS.get(first_difficulty_label, 2)
        current_max_questions = _determine_max_questions([first_weight], 1)

    key_skills = []
    if resume_generation_enabled:
        if isinstance(payload_mandatory, list):
            key_skills = [skill for skill in payload_mandatory if isinstance(skill, str)][:3]
        elif mandatory_skills:
            key_skills = [skill.strip() for skill in mandatory_skills.split(",") if skill.strip()][:3]
    elif mandatory_skills:
        key_skills = [skill.strip() for skill in mandatory_skills.split(",")[:3]]

    if not key_skills:
        key_skills = ["Problem Solving", "Communication"]

    InterviewRepository.save_first_question(
        session_id=session_id,
        job_role=job_role,
        industry_type=industry_type,
        company_name=company_name,
        interview_type=normalized_interview_type,
        work_experience=normalized_work_experience,
        question=first_question,
        difficulty=first_question_difficulty,
        mandatory_skills=mandatory_skills,
        question_type=first_question_type,
        job_description_id=job_description_id_value,
        job_description_text=job_description_text_value,
        job_desc=job_description_raw_value,
        resume_id=resume_id if resume_generation_enabled else None,
        generation_context=first_question_context,
    )

    return {
        "session_id": session_id,
        "first_question": first_question,
        "first_question_meta": {
            "question": first_question,
            "question_type": first_question_type,
            "mandatory_skills": payload_mandatory if resume_generation_enabled else mandatory_skills,
            "difficulty": first_question_difficulty,
        },
        "key_skills": key_skills,
        "job_role": job_role,
        "industry_type": industry_type,
        "company_name": company_name,
        "interview_type": normalized_interview_type,
        "work_experience": normalized_work_experience,
        "question_number": 1,
        "current_max_questions": current_max_questions,
        "job_description_id": job_description_id_value,
        "job_description_attached": bool(
            job_description_id_value or job_description_text_value or job_description_raw_value
        ),
        "message": "Interview started successfully",
    }


async def start_compatibility_interview_service(
    session_id: Optional[str],
    job_role: Optional[str],
    industry_type: Optional[str],
    company_name: Optional[str],
    interview_type: Optional[str],
    work_experience: Optional[str],
) -> InterviewResponse:
    if session_id and not job_role:
        ctx = None
        next_q_num = 1
        with db_pool.get_connection() as conn:
            if conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT MAX(question_number) FROM interview_data WHERE session_id = %s",
                        (session_id,),
                    )
                    next_q_num = (cur.fetchone()[0] or 0) + 1

                    cur.execute(
                        """
                        SELECT job_role, industry_type, company_name, interview_type, work_experience, difficulty
                        FROM interview_data WHERE session_id = %s ORDER BY question_number DESC LIMIT 1
                        """,
                        (session_id,),
                    )
                    ctx = cur.fetchone()
        if ctx:
            job_role, industry_type, company_name, interview_type, work_experience, prev_diff = ctx
            next_q_data = await _resolve_next_question(
                job_role,
                industry_type,
                company_name,
                next_q_num,
                session_id,
                interview_type=interview_type,
                work_experience=work_experience,
                preferred_difficulty=_translate_difficulty_label(prev_diff, "medium"),
            )
            return InterviewResponse(
                session_id=session_id,
                response=next_q_data["question"],
                mandatory_skills=next_q_data["mandatory_skills"],
                question_number=next_q_num,
                acknowledgment="Let's continue.",
            )

    if not all([job_role, industry_type, company_name]):
        raise HTTPException(
            status_code=400,
            detail="Missing required parameters: job_role, industry_type, and company_name are required.",
        )

    res = await start_interview_service(
        request=None,
        student_name=f"Candidate_{int(time.time())}",
        student_email=None,
        job_role=job_role or "Software Engineer",
        industry_type=industry_type or "Technology",
        company_name=company_name or "General",
        interview_type=interview_type,
        work_experience=work_experience,
        job_description_id=None,
        job_description_text=None,
        job_description_raw_text=None,
        force_reattempt=True,
        use_resume_questions=False,
        resume_id=None,
        pre_generated_questions=None,
    )
    return InterviewResponse(
        session_id=res["session_id"],
        response=res["first_question"],
        mandatory_skills=res["first_question_meta"]["mandatory_skills"],
        question_number=1,
        acknowledgment=None,
    )


async def submit_answer_service(
    background_tasks: BackgroundTasks,
    session_id: str,
    student_id: int,
    answer: str,
    question_type: Optional[str],
    code: Optional[str],
    stdin: Optional[str],
    stdout: Optional[str],
    stderr: Optional[str],
    runtime_error: Optional[str],
    execution_success: Optional[str],
    has_run: Optional[str],
    is_final: Optional[str],
    response_video: UploadFile | None,
    system_design_diagram: Optional[str],
) -> Dict[str, Any]:
    if not InterviewRepository.verify_session_belongs_to_student(session_id, student_id):
        raise HTTPException(status_code=403, detail="Session does not belong to this student")

    is_system_design = _is_system_design_question_type(question_type)
    stored_answer_value = answer
    if is_system_design:
        stored_answer_value = _prepare_system_design_answer_payload(system_design_diagram, answer)

    execution_success_bool = _parse_bool(execution_success)
    has_run_bool = _parse_bool(has_run)
    is_final_flag = _parse_bool(is_final)

    # Phase 1: DB preparation and saving candidate answer
    try:
        context = InterviewRepository.save_answer_and_get_context(
            session_id=session_id,
            answer=stored_answer_value,
            question_type=question_type or "standard",
            code=code,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            runtime_error=runtime_error,
            execution_success=bool(execution_success_bool),
            has_run=bool(has_run_bool),
            is_final=bool(is_final_flag),
            saved_video_path=None,
            is_system_design=is_system_design,
            system_design_diagram=system_design_diagram,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error(f"Database error saving answer for {session_id}: {exc}")
        raise HTTPException(status_code=500, detail="Database operation failed")

    current_q = context["current_q"]
    current_question = context["current_question"]
    mandatory_skills = context["mandatory_skills"]
    job_role = context["job_role"]
    industry_type = context["industry_type"]
    company_name = context["company_name"]
    db_difficulty = context["db_difficulty"]
    db_question_type = context["db_question_type"]
    db_interview_type = context["db_interview_type"]
    db_work_experience = context["db_work_experience"]
    jd_id = context["jd_id"]
    jd_text = context["jd_text"]
    jd_desc = context["jd_desc"]
    resume_id = context["resume_id"]
    generation_context = context["generation_context"]
    max_questions = context["max_questions"]

    # Save uploaded video if present (only meaningful for speech-based questions)
    is_speech_question = "speech" in (db_question_type or "").lower()
    saved_video_path = _save_uploaded_video(session_id, current_q, response_video) if is_speech_question else None
    if saved_video_path:
        InterviewRepository.update_video_upload_path(session_id, current_q, saved_video_path)
        background_tasks.add_task(process_video_analysis_background, session_id, current_q, saved_video_path)

    # Prepare system design evaluation payload if necessary
    effective_answer = answer
    if is_system_design or _is_system_design_question_type(db_question_type):
        is_system_design = True
        effective_answer = _describe_system_design(stored_answer_value)

    # Phase 2: AI evaluation with timeout protection
    try:
        ai_future = analyze_answer_and_generate_response(
            question=current_question,
            answer=effective_answer,
            mandatory_skills=mandatory_skills,
            job_role=job_role,
        )
        sentiment_score, acknowledgment, next_difficulty_raw = await asyncio.wait_for(ai_future, timeout=12.0)
    except asyncio.TimeoutError:
        logger.warning(f"AI evaluation timed out for {session_id} Q#{current_q}; using default fallback values.")
        sentiment_score = 5.0
        acknowledgment = "Thank you for sharing your experience. Let's move on to the next topic."
        next_difficulty_raw = "medium"
    except Exception as exc:
        logger.error(f"Error during AI evaluation for {session_id} Q#{current_q}: {exc}")
        sentiment_score = 5.0
        acknowledgment = "Got it. Let's continue with our discussion."
        next_difficulty_raw = "medium"

    # Translate next difficulty level
    fallback_difficulty = _translate_difficulty_label(next_difficulty_raw, default="medium")
    is_final_question = (is_final_flag is True) or (current_q >= max_questions)

    # Phase 3 onward: completion check, next-question resolution, and persistence.
    # Wrapped like Phase 1/2 so a DB or logic failure here returns a clean 500
    # instead of an unhandled exception.
    try:
        InterviewRepository.save_sentiment_and_check_final(
            session_id=session_id,
            question_number=current_q,
            sentiment_score=sentiment_score,
            is_final_question=is_final_question,
        )

        if is_final_question:
            InterviewRepository.mark_feedback_pending(session_id)
            background_tasks.add_task(generate_and_process_feedback_background, session_id)
            return {
                "session_id": session_id,
                "response": "Interview completed! Generating comprehensive feedback report...",
                "next_question": "Interview completed! Generating comprehensive feedback report...",
                "next_question_meta": {
                    "question": "Interview completed! Generating comprehensive feedback report...",
                    "text": "Interview completed! Generating comprehensive feedback report...",
                    "question_type": "completed",
                    "type": "completed",
                    "mandatory_skills": "Completed",
                    "difficulty": fallback_difficulty,
                },
                "mandatory_skills": "Completed",
                "question_number": current_q + 1,
                "acknowledgment": acknowledgment,
                "difficulty": fallback_difficulty,
                "question_type": "completed",
                "is_completed": True,
                "completed": True,
            }

        # Phase 4: Fetch next question and save
        next_q_num = current_q + 1
        next_q_data = None
        if resume_id:
            next_q_data = InterviewRepository.get_pre_generated_question(session_id, next_q_num)

        if not next_q_data:
            next_q_data = await _resolve_next_question(
                job_role,
                industry_type,
                company_name,
                next_q_num,
                session_id,
                interview_type=db_interview_type,
                work_experience=db_work_experience,
                question_type_filter=db_question_type,
                preferred_difficulty=fallback_difficulty,
            )

        InterviewRepository.save_next_question(
            session_id=session_id,
            job_role=job_role,
            industry_type=industry_type,
            company_name=company_name,
            interview_type=db_interview_type,
            work_experience=db_work_experience,
            next_q_num=next_q_num,
            next_q_data=next_q_data,
            fallback_difficulty=fallback_difficulty,
            jd_id=jd_id,
            jd_text=jd_text,
            jd_desc=jd_desc,
            resume_id=resume_id,
            generation_context=next_q_data.get("generation_context") or generation_context,
        )

        next_difficulty = next_q_data.get("difficulty") or fallback_difficulty or "medium"
        next_question_type = next_q_data.get("question_type") or "standard"
        next_mandatory_skills = next_q_data.get("mandatory_skills") or "Communication, Problem-solving"

        next_q_meta = {
            "question": next_q_data["question"],
            "text": next_q_data["question"],
            "question_type": next_question_type,
            "type": next_question_type,
            "mandatory_skills": next_mandatory_skills,
            "difficulty": next_difficulty,
        }

        return {
            "session_id": session_id,
            "response": next_q_data["question"],
            "next_question": next_q_data["question"],
            "next_question_meta": next_q_meta,
            "mandatory_skills": next_mandatory_skills,
            "question_number": next_q_num,
            "acknowledgment": acknowledgment,
            "difficulty": next_difficulty,
            "question_type": next_question_type,
            "is_completed": False,
            "completed": False,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error finalizing answer for session {session_id} Q#{current_q}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to process interview progression")


async def generate_and_process_feedback_background(session_id: str) -> None:
    try:
        max_wait_time = 300
        check_interval = 2
        elapsed_time = 0

        while elapsed_time < max_wait_time:
            pending_count, processing_count = InterviewRepository.get_pending_video_count(session_id)
            if pending_count == 0 and processing_count == 0:
                break
            await asyncio.sleep(check_interval)
            elapsed_time += check_interval

        InterviewRepository.mark_feedback_processing(session_id)
        qa_history, job_role, industry_type, company_name, interview_type = (
            InterviewRepository.get_qa_history_for_feedback(session_id)
        )

        if not qa_history:
            InterviewRepository.mark_feedback_failed(session_id, "No answered questions available to grade.")
            return

        formatted_exchanges = []
        unique_skills_set = set()
        answer_lookup = {}
        question_type_lookup = {}
        rag_lookup_pairs = []

        for row in qa_history:
            (
                q_num, q_text, ans_text, sent_score, mand_skills,
                vid_sent, vid_dem, code_sub, code_succ, code_run,
                q_type,
            ) = row
            # For system design questions, ans_text is raw diagram JSON (can be several KB,
            # heavy with quotes/braces). Asking Gemini to reproduce that verbatim inside its
            # own JSON "answer" field is what breaks the output JSON (unescaped/truncated
            # strings). The real diagram JSON is restored from the DB below via answer_lookup
            # regardless of what Gemini writes, so only a plain-language description is needed
            # here for scoring purposes.
            prompt_answer_text = ans_text
            if _is_system_design_question_type(q_type):
                prompt_answer_text = _describe_system_design(ans_text)
            exchange = f"Q{q_num}: {q_text}\nA: {prompt_answer_text}"
            if code_sub:
                exchange += f"\n[Code Submitted ({'Success' if code_succ else 'Failed/Unchecked'})]: {code_sub[:300]}"
            formatted_exchanges.append(exchange)
            if mand_skills:
                for sk in str(mand_skills).split(","):
                    sk_clean = sk.strip()
                    if sk_clean:
                        unique_skills_set.add(sk_clean)

            # Reconstruct the exact candidate answer for lookup
            candidate_answer = ans_text or ""
            normalized_qt = str(q_type or "").lower()
            if normalized_qt == "coding" or normalized_qt.startswith("coding "):
                code_snippet = code_sub or ""
                if code_snippet and code_snippet.strip():
                    candidate_answer = code_snippet.strip()

            answer_lookup[int(q_num)] = candidate_answer
            question_type_lookup[int(q_num)] = q_type
            rag_lookup_pairs.append((int(q_num), q_text))

        unique_skills = sorted(list(unique_skills_set))
        conversation_excerpt = _compose_conversation_excerpt(formatted_exchanges, char_limit=18000)
        template_info = _resolve_feedback_template(interview_type)
        resolved_template = template_info["question_template"]

        reference_answers = []
        if company_name:
            try:
                rag_results = await asyncio.gather(
                    *[
                        retrieve_company_context(company_name, q_text, top_k=1)
                        for _, q_text in rag_lookup_pairs
                    ]
                )
                for (q_num, _), chunks in zip(rag_lookup_pairs, rag_results):
                    if chunks:
                        reference_answers.append({"number": q_num, "answer": chunks[0]})
            except Exception as exc:
                logger.warning(
                    "RAG reference-answer lookup failed for session %s: %s", session_id, exc
                )
                reference_answers = []

        logger.info(
            "[RAG-FEEDBACK] session=%s company=%s reference_answers_found=%d sample=%s",
            session_id, company_name, len(reference_answers),
            (reference_answers[0]["answer"][:120] if reference_answers else None),
        )

        rendered_question_prompt = render_template(
            resolved_template,
            job_role=job_role,
            industry_type=industry_type,
            company_name=company_name,
            interview_type=interview_type or "standard",
            conversation_excerpt=conversation_excerpt,
            conversation_text=conversation_excerpt,
            unique_skills=unique_skills,
            reference_answers=reference_answers,
            scoring_guide=SCORING_GUIDE,
        )

        rendered_competency_prompt = render_template(
            template_info["competency_template"],
            job_role=job_role,
            industry_type=industry_type,
            company_name=company_name,
            interview_type=interview_type or "standard",
            conversation_excerpt=conversation_excerpt,
            conversation_text=conversation_excerpt,
            unique_skills=unique_skills,
            core_competencies=template_info["competencies"],
        )

        # Question-quality feedback and competency scoring are independent prompts over the
        # same transcript, so they run concurrently rather than paying two sequential Gemini
        # round-trips (same pattern as the RAG reference-answer lookups above).
        question_response, competency_response = await asyncio.gather(
            generate_content_with_fallback(
                rendered_question_prompt,
                retry_label=f"Feedback generation for {session_id}",
            ),
            generate_content_with_fallback(
                rendered_competency_prompt,
                retry_label=f"Competency scoring for {session_id}",
            ),
            return_exceptions=True,
        )

        if isinstance(question_response, Exception):
            raise question_response

        raw_feedback_text = question_response.text if question_response else ""
        if not raw_feedback_text or not raw_feedback_text.strip():
            raise RuntimeError("Gemini returned empty feedback report.")

        parsed_json = _parse_feedback_json(raw_feedback_text)

        # Post-process questions list to inject correct answers and types
        if parsed_json and isinstance(parsed_json, dict):
            questions_list = parsed_json.get("questions")
            if isinstance(questions_list, list):
                for entry in questions_list:
                    if not isinstance(entry, dict):
                        continue
                    num = entry.get("number")
                    if num is None:
                        continue
                    try:
                        idx = int(num)
                    except (ValueError, TypeError):
                        continue

                    original_answer = answer_lookup.get(idx, "")
                    if original_answer:
                        entry["answer"] = original_answer
                        entry["original_answer"] = original_answer

                    qt_value = question_type_lookup.get(idx)
                    if qt_value:
                        entry["question_type"] = qt_value
                        normalized_qt = str(qt_value).lower()
                        if normalized_qt == "coding" or normalized_qt.startswith("coding "):
                            entry["is_coding"] = True

        if not parsed_json:
            # Storing unparseable raw text and marking the session "completed" anyway made the
            # frontend poll forever: FeedbackScreen sees status=completed, tries to fetch
            # structured feedback, gets nothing back (same broken text fails to parse every
            # time), and re-polls -- forever, at whatever cadence the retry fires, hammering
            # this endpoint. Treat an unparseable AI response as a real failure instead, so the
            # frontend surfaces a "regenerate" error state rather than looping indefinitely.
            raise RuntimeError("Gemini returned malformed feedback JSON that could not be parsed.")

        # Competency scoring is best-effort: a failed/malformed response still lets the
        # question-level feedback save, just without a weighted overall_score (matching the
        # pre-fix behavior for that one session instead of failing the whole request).
        if isinstance(competency_response, Exception):
            logger.warning(
                "Competency scoring Gemini call failed for session %s: %s", session_id, competency_response
            )
        else:
            raw_competency_text = competency_response.text if competency_response else ""
            competency_parsed = _parse_feedback_json(raw_competency_text) if raw_competency_text.strip() else None
            if isinstance(competency_parsed, dict) and competency_parsed.get("core_competencies"):
                parsed_json["core_competencies"] = competency_parsed["core_competencies"]
                for summary_key in ("technical_summary", "communication_summary", "attitude_summary"):
                    if summary_key in competency_parsed:
                        parsed_json[summary_key] = competency_parsed[summary_key]
            else:
                logger.warning(
                    "Competency scoring response for session %s had no usable core_competencies", session_id
                )

        stored_payload = json.dumps(parsed_json, ensure_ascii=False)

        saved = InterviewRepository.save_detailed_feedback(session_id, stored_payload)
        if not saved:
            raise RuntimeError("Database error persisting detailed feedback.")

        update_scores_from_feedback(session_id, stored_payload)
        InterviewRepository.mark_feedback_completed(session_id)

    except Exception as exc:
        logger.error(f"Background feedback generation failed for session {session_id}: {exc}", exc_info=True)
        InterviewRepository.mark_feedback_failed(session_id, str(exc))


async def generate_and_process_feedback_service(session_id: str) -> Dict[str, Any]:
    await generate_and_process_feedback_background(session_id)
    return {"status": "processing", "session_id": session_id, "message": "Feedback generation initiated"}
