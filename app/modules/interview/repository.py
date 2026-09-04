import json
import re
import uuid
from datetime import datetime, date, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple, Union
from psycopg2.extras import Json
from app.core.database import db_pool, ensure_companies_table
from app.core.logger import logger
from app.config.constants import (
    DIFFICULTY_WEIGHTS,
    QUESTION_TYPE_ALIAS_MAP,
)
from app.utils.datetime_utils import IST_TZ, _ensure_datetime, _to_ist_datetime, format_datetime_ist
from app.utils.text_utils import _parse_feedback_json


class InterviewRepository:
    """Repository class for interview database operations."""

    @staticmethod
    def _ensure_job_descriptions_table(cur: Any) -> None:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS job_descriptions (
                id SERIAL PRIMARY KEY,
                student_id INTEGER REFERENCES students(student_id) ON DELETE CASCADE,
                job_role VARCHAR(255) NOT NULL,
                company_name VARCHAR(255),
                job_desc TEXT NOT NULL,
                file_name VARCHAR(255),
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    @staticmethod
    def _ensure_interview_data_columns(cur: Any) -> None:
        columns = [
            ("resume_id", "INTEGER REFERENCES resumes(id) ON DELETE SET NULL"),
            ("job_description_id", "INTEGER REFERENCES job_descriptions(id) ON DELETE SET NULL"),
            ("job_description_text", "TEXT"),
            ("job_desc", "TEXT"),
            ("generation_context", "JSONB"),
            ("question_type", "VARCHAR(50) DEFAULT 'standard'"),
            ("interview_type", "VARCHAR(100)"),
            ("work_experience", "VARCHAR(100)"),
            ("video_clip_path", "TEXT"),
            ("code_submission", "TEXT"),
            ("stdin_input", "TEXT"),
            ("stdout_output", "TEXT"),
            ("stderr_output", "TEXT"),
            ("runtime_error", "TEXT"),
            ("execution_success", "BOOLEAN"),
            ("manual_run", "BOOLEAN"),
            ("sentiment_score", "NUMERIC(4,2)"),
            ("is_system_design", "BOOLEAN DEFAULT FALSE"),
            ("system_design_diagram", "TEXT"),
            ("detailed_feedback", "TEXT"),
            ("video_sentiment_score", "NUMERIC(4,2)"),
            ("video_demeanor", "TEXT"),
            ("video_analysis_status", "VARCHAR(50)"),
        ]
        for col_name, col_type in columns:
            try:
                cur.execute(f"ALTER TABLE interview_data ADD COLUMN IF NOT EXISTS {col_name} {col_type}")
            except Exception as exc:
                logger.debug(f"Column {col_name} check/add noted: {exc}")

    @staticmethod
    def _ensure_session_metadata_columns(cur: Any) -> None:
        columns = [
            ("resume_id", "INTEGER REFERENCES resumes(id) ON DELETE SET NULL"),
            ("job_description_id", "INTEGER REFERENCES job_descriptions(id) ON DELETE SET NULL"),
            ("job_description_text", "TEXT"),
            ("job_desc", "TEXT"),
            ("overall_score", "NUMERIC(5,2)"),
            ("rubric_scores", "JSONB"),
            ("technical_score", "NUMERIC(5,2)"),
            ("communication_score", "NUMERIC(5,2)"),
            ("attitude_score", "NUMERIC(5,2)"),
            ("status", "VARCHAR(50) DEFAULT 'active'"),
            ("feedback_generated", "BOOLEAN DEFAULT FALSE"),
            ("duration_minutes", "INTEGER"),
            ("termination_reason", "TEXT"),
            ("terminated_at", "TIMESTAMP"),
            ("feedback_status", "VARCHAR(50)"),
            ("feedback_error", "TEXT"),
            ("feedback_requested_at", "TIMESTAMP"),
            ("feedback_ready_at", "TIMESTAMP"),
            ("student_id", "INTEGER REFERENCES students(student_id) ON DELETE SET NULL"),
            ("student_name", "VARCHAR(255)"),
            ("interview_type", "VARCHAR(100)"),
            ("work_experience", "VARCHAR(100)"),
        ]
        for col_name, col_type in columns:
            try:
                cur.execute(f"ALTER TABLE session_metadata ADD COLUMN IF NOT EXISTS {col_name} {col_type}")
            except Exception as exc:
                logger.debug(f"Column {col_name} check/add noted: {exc}")

    @staticmethod
    def _ensure_pre_generated_questions_table(cur: Any) -> None:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS session_pre_generated_questions (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) REFERENCES session_metadata(session_id) ON DELETE CASCADE,
                student_id INTEGER REFERENCES students(student_id) ON DELETE SET NULL,
                resume_id INTEGER REFERENCES resumes(id) ON DELETE SET NULL,
                job_description_id INTEGER REFERENCES job_descriptions(id) ON DELETE SET NULL,
                company_name VARCHAR(255),
                job_role VARCHAR(255),
                interview_type VARCHAR(100),
                work_experience VARCHAR(100),
                question_number INTEGER NOT NULL,
                question_text TEXT NOT NULL,
                mandatory_skills TEXT,
                difficulty VARCHAR(50),
                question_type VARCHAR(50) DEFAULT 'standard',
                generation_context JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (session_id, question_number)
            )
        """)

    @classmethod
    def _get_active_concurrent_sessions(cls) -> int:
        try:
            with db_pool.get_connection() as conn:
                if not conn:
                    return 0
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT COUNT(*)
                        FROM session_metadata
                        WHERE COALESCE(status, 'active') = 'active'
                          AND started_at >= NOW() - INTERVAL '30 minutes'
                    """)
                    row = cur.fetchone()
                    return int(row[0]) if row and row[0] is not None else 0
        except Exception as exc:
            logger.error(f"Error checking active concurrent sessions: {exc}")
            return 0

    @classmethod
    def find_student_id(cls, student_name: str, student_email: Optional[str]) -> Optional[int]:
        """Lookup an existing student_id without creating new records."""
        try:
            with db_pool.get_connection() as conn:
                if not conn:
                    return None
                with conn.cursor() as cur:
                    if student_email:
                        cur.execute(
                            "SELECT student_id FROM students WHERE LOWER(TRIM(email)) = LOWER(TRIM(%s))",
                            (student_email,),
                        )
                        row = cur.fetchone()
                        if row:
                            return row[0]

                    cur.execute(
                        "SELECT student_id FROM students WHERE LOWER(TRIM(name)) = LOWER(TRIM(%s))",
                        (student_name,),
                    )
                    row = cur.fetchone()
                    if row:
                        return row[0]
        except Exception as e:
            logger.error(f"Error finding student_id: {e}")
        return None

    @classmethod
    def find_existing_sessions(
        cls,
        student_name: str,
        student_email: Optional[str],
        job_role: str,
        industry_type: str,
        company_name: str,
        interview_type: Optional[str] = None,
        work_experience: Optional[str] = None,
        is_company_card: bool = False,
    ) -> List[Dict[str, Any]]:
        sessions: List[Dict[str, Any]] = []
        try:
            normalize_value = lambda value: (value or "").strip().lower()
            normalized_role = normalize_value(job_role)
            normalized_industry = normalize_value(industry_type)
            normalized_company = normalize_value(company_name)
            normalized_name = normalize_value(student_name)
            normalized_email = normalize_value(student_email)
            normalized_interview_type = normalize_value(interview_type)
            normalized_work_experience = normalize_value(work_experience)

            if not normalized_role or not normalized_company:
                return sessions

            student_id = cls.find_student_id(student_name, student_email)
            student_filters: List[str] = []
            filter_params: List[Any] = []

            if student_id:
                student_filters.append("sm.student_id = %s")
                filter_params.append(student_id)

            if normalized_name:
                student_filters.append("LOWER(TRIM(sm.student_name)) = LOWER(TRIM(%s))")
                filter_params.append(normalized_name)

            if normalized_email:
                student_filters.append(
                    """
                    EXISTS (
                        SELECT 1 FROM students st
                        WHERE st.student_id = sm.student_id
                          AND LOWER(TRIM(st.email)) = LOWER(TRIM(%s))
                    )
                    """
                )
                filter_params.append(normalized_email)

            if not student_filters:
                return sessions

            student_predicate = "(" + " OR ".join(student_filters) + ")"

            with db_pool.get_connection() as conn:
                if not conn:
                    return sessions
                with conn.cursor() as cur:
                    query = f"""
                        SELECT 
                            sm.session_id,
                            sm.status,
                            sm.started_at,
                            sm.completed_at,
                            sm.overall_score,
                            MAX(LOWER(TRIM(NULLIF(id.job_role, '')))) AS job_role_key,
                            MAX(LOWER(TRIM(NULLIF(id.company_name, '')))) AS company_key,
                            MAX(LOWER(TRIM(NULLIF(id.industry_type, '')))) AS industry_key,
                            MAX(LOWER(TRIM(NULLIF(id.interview_type, '')))) AS interview_type_key,
                            MAX(LOWER(TRIM(NULLIF(id.work_experience, '')))) AS work_experience_key
                        FROM session_metadata sm
                        LEFT JOIN interview_data id ON id.session_id = sm.session_id
                        WHERE {student_predicate}
                          AND LOWER(TRIM(COALESCE(sm.status, ''))) = 'completed'
                        GROUP BY sm.session_id, sm.status, sm.started_at, sm.completed_at, sm.overall_score
                        ORDER BY sm.started_at DESC
                    """
                    cur.execute(query, tuple(filter_params))
                    rows = cur.fetchall()

                    for row in rows:
                        (
                            session_id,
                            status,
                            started_at,
                            completed_at,
                            overall_score,
                            job_role_key,
                            company_key,
                            industry_key,
                            interview_type_key,
                            work_experience_key,
                        ) = row

                        if not company_key or company_key != normalized_company:
                            continue
                        if not job_role_key or job_role_key != normalized_role:
                            continue

                        if normalized_work_experience:
                            if not work_experience_key or work_experience_key != normalized_work_experience:
                                continue
                        elif work_experience_key:
                            continue

                        if normalized_interview_type:
                            if not interview_type_key or interview_type_key != normalized_interview_type:
                                continue
                        elif interview_type_key:
                            continue

                        if not is_company_card:
                            if not industry_key or industry_key != normalized_industry:
                                continue

                        normalized_started = _ensure_datetime(started_at)
                        started_ist_dt = _to_ist_datetime(normalized_started, assume_tz=timezone.utc)
                        normalized_completed = _ensure_datetime(completed_at)
                        completed_ist_dt: Optional[datetime] = None
                        if normalized_completed:
                            if normalized_started and started_ist_dt:
                                try:
                                    candidate_from_utc = _to_ist_datetime(normalized_completed, assume_tz=timezone.utc)
                                    candidate_from_ist = _to_ist_datetime(normalized_completed, assume_tz=IST_TZ)

                                    def _duration(candidate: Optional[datetime]) -> float:
                                        if not candidate:
                                            return float("inf")
                                        return abs((candidate - started_ist_dt).total_seconds())

                                    duration_utc = _duration(candidate_from_utc)
                                    duration_ist = _duration(candidate_from_ist)
                                    if candidate_from_utc and (candidate_from_utc - started_ist_dt).total_seconds() >= 0 and duration_utc <= duration_ist:
                                        completed_ist_dt = candidate_from_utc
                                    elif candidate_from_ist and (candidate_from_ist - started_ist_dt).total_seconds() >= 0:
                                        completed_ist_dt = candidate_from_ist
                                    else:
                                        completed_ist_dt = candidate_from_utc or candidate_from_ist
                                except Exception:
                                    completed_ist_dt = _to_ist_datetime(normalized_completed, assume_tz=timezone.utc)
                            else:
                                completed_ist_dt = _to_ist_datetime(normalized_completed, assume_tz=timezone.utc)

                        sessions.append({
                            "session_id": session_id,
                            "status": status,
                            "started_at": started_ist_dt.isoformat() if started_ist_dt else None,
                            "completed_at": completed_ist_dt.isoformat() if completed_ist_dt else None,
                            "overall_score": float(overall_score) if overall_score is not None else None,
                        })
        except Exception as e:
            logger.error(f"Error finding existing sessions: {e}")
        return sessions

    @classmethod
    def can_reattempt_interview(
        cls,
        student_id: int,
        resume_id: Optional[int],
        job_description_id: Optional[int],
        company_name: str,
        job_role: str,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        try:
            with db_pool.get_connection() as conn:
                if not conn:
                    return (True, "Database connection unavailable; allowing reattempt.", {})
                with conn.cursor() as cur:
                    cls._ensure_session_metadata_columns(cur)
                    cls._ensure_interview_data_columns(cur)

                    if resume_id and job_description_id:
                        query = """
                            SELECT sm.session_id, sm.started_at, sm.status,
                                   COUNT(id.question_number) AS answered_count
                            FROM session_metadata sm
                            LEFT JOIN interview_data id ON sm.session_id = id.session_id
                                AND id.answer IS NOT NULL AND TRIM(id.answer) != ''
                            WHERE sm.student_id = %s
                              AND sm.resume_id = %s
                              AND sm.job_description_id = %s
                              AND COALESCE(sm.status, '') = 'completed'
                            GROUP BY sm.session_id, sm.started_at, sm.status
                            ORDER BY sm.started_at DESC
                            LIMIT 1
                        """
                        params = (student_id, resume_id, job_description_id)
                    else:
                        query = """
                            SELECT sm.session_id, sm.started_at, sm.status,
                                   COUNT(id.question_number) AS answered_count
                            FROM session_metadata sm
                            JOIN interview_data id ON sm.session_id = id.session_id
                            WHERE sm.student_id = %s
                              AND LOWER(TRIM(id.company_name)) = LOWER(TRIM(%s))
                              AND LOWER(TRIM(id.job_role)) = LOWER(TRIM(%s))
                              AND COALESCE(sm.status, '') = 'completed'
                            GROUP BY sm.session_id, sm.started_at, sm.status
                            ORDER BY sm.started_at DESC
                            LIMIT 1
                        """
                        params = (student_id, company_name, job_role)

                    cur.execute(query, params)
                    prior_session = cur.fetchone()

                    if not prior_session:
                        return (True, "No completed prior session found.", {})

                    session_id, started_at, status, answered_count = prior_session
                    meta = {
                        "previous_session_id": session_id,
                        "started_at": format_datetime_ist(started_at),
                        "status": status,
                        "answered_questions": answered_count,
                    }

                    if answered_count and answered_count >= 1:
                        return (False, "You have already completed an interview for this role/setup.", meta)
                    else:
                        return (True, "Prior session incomplete or has no answered questions; reattempt permitted.", meta)
        except Exception as exc:
            logger.error(f"Error checking reattempt capability: {exc}")
            return (True, "Error verifying prior attempts; proceeding by default.", {})

    @classmethod
    def create_enhanced_session(
        cls,
        student_name: str,
        student_email: Optional[str],
        job_role: str,
        industry_type: str,
        company_name: str,
        interview_type: Optional[str] = None,
        work_experience: Optional[str] = None,
    ) -> Optional[str]:
        try:
            with db_pool.get_connection() as conn:
                if not conn:
                    return None
                with conn.cursor() as cur:
                    student_id = cls.find_student_id(student_name, student_email)
                    if not student_id:
                        cur.execute(
                            "INSERT INTO students (name, email) VALUES (%s, %s) RETURNING student_id",
                            (student_name, student_email)
                        )
                        student_id = cur.fetchone()[0]

                    session_id = str(uuid.uuid4())
                    cur.execute(
                        """
                        INSERT INTO session_metadata (session_id, student_id, student_name, status, interview_type, work_experience)
                        VALUES (%s, %s, %s, 'active', %s, %s)
                        """,
                        (session_id, student_id, student_name, interview_type, work_experience)
                    )
                    conn.commit()
                    return session_id
        except Exception as e:
            logger.error(f"Error creating enhanced session: {e}")
            return None

    @classmethod
    def store_pre_generated_questions(
        cls,
        session_id: str,
        student_id: Optional[int],
        resume_id: Optional[int],
        job_description_id: Optional[int],
        job_description_text: Optional[str],
        job_desc: Optional[str],
        company_name: str,
        job_role: str,
        interview_type: Optional[str],
        work_experience: Optional[str],
        questions: List[Dict[str, Any]],
    ) -> None:
        if not questions:
            return
        try:
            with db_pool.get_connection() as conn:
                if not conn:
                    return
                with conn.cursor() as cur:
                    cls._ensure_pre_generated_questions_table(cur)
                    for index, item in enumerate(questions):
                        q_num = item.get("question_number") or (index + 1)
                        question_text = (item.get("question") or "").strip()
                        if not question_text:
                            continue
                        mandatory_skills = item.get("mandatory_skills")
                        if isinstance(mandatory_skills, list):
                            mandatory_skills = ", ".join([str(s) for s in mandatory_skills if s])
                        difficulty = item.get("difficulty", "medium")
                        question_type = item.get("question_type", "standard")
                        generation_context = item.get("generation_context") or {}

                        cur.execute(
                            """
                            INSERT INTO session_pre_generated_questions (
                                session_id, student_id, resume_id, job_description_id, company_name, job_role,
                                interview_type, work_experience, question_number, question_text, mandatory_skills,
                                difficulty, question_type, generation_context
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (session_id, question_number) DO UPDATE SET
                                question_text = EXCLUDED.question_text,
                                mandatory_skills = EXCLUDED.mandatory_skills,
                                difficulty = EXCLUDED.difficulty,
                                question_type = EXCLUDED.question_type,
                                generation_context = EXCLUDED.generation_context
                            """,
                            (
                                session_id, student_id, resume_id, job_description_id, company_name, job_role,
                                interview_type, work_experience, q_num, question_text, mandatory_skills,
                                difficulty, question_type, Json(generation_context)
                            )
                        )
                    conn.commit()
        except Exception as exc:
            logger.error(f"Failed to store pre-generated questions for session {session_id}: {exc}")

    @classmethod
    def _attach_resume_metadata_to_session(cls, session_id: str, resume_id: int) -> None:
        try:
            with db_pool.get_connection() as conn:
                if not conn:
                    return
                with conn.cursor() as cur:
                    cls._ensure_session_metadata_columns(cur)
                    cls._ensure_interview_data_columns(cur)
                    cur.execute(
                        "UPDATE session_metadata SET resume_id = %s WHERE session_id = %s",
                        (resume_id, session_id)
                    )
                    cur.execute(
                        "UPDATE interview_data SET resume_id = %s WHERE session_id = %s",
                        (resume_id, session_id)
                    )
                    conn.commit()
        except Exception as exc:
            logger.warning(f"Unable to attach resume_id {resume_id} to session {session_id}: {exc}")

    @classmethod
    def _attach_job_description_metadata_to_session(
        cls,
        session_id: str,
        job_description_id: Optional[int],
        job_description_text: Optional[str],
        job_desc: Optional[str],
    ) -> None:
        if not any([job_description_id, job_description_text, job_desc]):
            return
        try:
            with db_pool.get_connection() as conn:
                if not conn:
                    return
                with conn.cursor() as cur:
                    cls._ensure_session_metadata_columns(cur)
                    cls._ensure_interview_data_columns(cur)
                    cur.execute(
                        """
                        UPDATE session_metadata
                        SET job_description_id = COALESCE(%s, job_description_id),
                            job_description_text = COALESCE(%s, job_description_text),
                            job_desc = COALESCE(%s, job_desc)
                        WHERE session_id = %s
                        """,
                        (job_description_id, job_description_text, job_desc, session_id)
                    )
                    cur.execute(
                        """
                        UPDATE interview_data
                        SET job_description_id = COALESCE(%s, job_description_id),
                            job_description_text = COALESCE(%s, job_description_text),
                            job_desc = COALESCE(%s, job_desc)
                        WHERE session_id = %s
                        """,
                        (job_description_id, job_description_text, job_desc, session_id)
                    )
                    conn.commit()
        except Exception as exc:
            logger.warning(f"Unable to attach JD metadata to session {session_id}: {exc}")

    @classmethod
    def save_first_question(
        cls,
        session_id: str,
        job_role: str,
        industry_type: str,
        company_name: str,
        interview_type: Optional[str],
        work_experience: Optional[str],
        question: str,
        difficulty: str,
        mandatory_skills: str,
        question_type: str,
        job_description_id: Optional[int],
        job_description_text: Optional[str],
        job_desc: Optional[str],
        resume_id: Optional[int],
        generation_context: dict,
    ) -> None:
        try:
            with db_pool.get_connection() as conn:
                if not conn:
                    return
                with conn.cursor() as cur:
                    cls._ensure_interview_data_columns(cur)
                    insert_params = {
                        "session_id": session_id,
                        "job_role": job_role,
                        "industry_type": industry_type,
                        "company_name": company_name,
                        "interview_type": interview_type or None,
                        "work_experience": work_experience or None,
                        "question_number": 1,
                        "question": question,
                        "answer": "",
                        "analysis_status": "pending",
                        "difficulty": difficulty,
                        "mandatory_skills": mandatory_skills,
                        "question_type": question_type,
                        "job_description_id": job_description_id,
                        "job_description_text": job_description_text,
                        "job_desc": job_desc,
                        "resume_id": resume_id,
                        "generation_context": Json(generation_context or {}),
                    }
                    cur.execute("""
                        INSERT INTO interview_data 
                        (session_id, job_role, industry_type, company_name, interview_type, work_experience, question_number, 
                         question, answer, analysis_status, difficulty, mandatory_skills, question_type, job_description_id, job_description_text, job_desc, resume_id, generation_context, timestamp)
                        VALUES (%(session_id)s, %(job_role)s, %(industry_type)s, %(company_name)s, %(interview_type)s, %(work_experience)s, %(question_number)s,
                                %(question)s, %(answer)s, %(analysis_status)s, %(difficulty)s, %(mandatory_skills)s, %(question_type)s,
                                %(job_description_id)s, %(job_description_text)s, %(job_desc)s, %(resume_id)s, %(generation_context)s, CURRENT_TIMESTAMP)
                        ON CONFLICT (session_id, question_number) 
                        DO UPDATE SET question = EXCLUDED.question,
                                      question_type = EXCLUDED.question_type,
                                      interview_type = EXCLUDED.interview_type,
                                      work_experience = EXCLUDED.work_experience,
                                      job_description_id = EXCLUDED.job_description_id,
                                      job_description_text = EXCLUDED.job_description_text,
                                      job_desc = EXCLUDED.job_desc,
                                      resume_id = EXCLUDED.resume_id,
                                      generation_context = EXCLUDED.generation_context
                    """, insert_params)
                    conn.commit()
        except Exception as e:
            logger.error(f"Error storing first question: {e}")

    @classmethod
    def get_question_from_db_with_difficulty(
        cls,
        job_role: str,
        industry_type: str,
        company_name: str,
        question_number: int,
        session_id: str = None,
        preferred_difficulty: str = "medium",
        interview_type: Optional[str] = None,
        work_experience: Optional[str] = None,
        question_type_filter: Optional[str] = None,
    ) -> Dict[str, str]:
        """Legacy entry point retained for compatibility; new callers should use
        find_company_specific_question + find_generic_fallback_question via the
        service-layer _resolve_next_question, which also gives RAG generation a
        chance between the two."""
        result = cls.find_company_specific_question(
            job_role, industry_type, company_name, session_id,
            interview_type=interview_type, work_experience=work_experience,
            question_type_filter=question_type_filter, preferred_difficulty=preferred_difficulty,
        )
        if result:
            return result
        return cls.find_generic_fallback_question(
            job_role, industry_type, company_name, question_number, session_id,
            interview_type=interview_type, work_experience=work_experience, question_type_filter=question_type_filter,
        )

    @classmethod
    def get_already_asked_questions(cls, session_id: Optional[str]) -> List[str]:
        """Question texts already asked in this session, so RAG+Gemini generation can be told
        to avoid repeating/near-duplicating them (the DB lookup path already excludes these via
        its own NOT IN clause; the generation path has no such guard without this)."""
        if not session_id:
            return []
        try:
            with db_pool.get_connection() as conn:
                if not conn:
                    return []
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT DISTINCT question
                        FROM interview_data
                        WHERE session_id = %s AND question IS NOT NULL AND question != ''
                    """, (session_id,))
                    return [row[0] for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching already-asked questions for session {session_id}: {e}")
            return []

    @classmethod
    def find_company_specific_question(
        cls,
        job_role: str,
        industry_type: str,
        company_name: str,
        session_id: str = None,
        interview_type: Optional[str] = None,
        work_experience: Optional[str] = None,
        question_type_filter: Optional[str] = None,
        preferred_difficulty: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        """Company-required lookup only (old get_question_from_db_with_difficulty/get_question_from_db
        steps 1-2). Returns None on a miss instead of relaxing further or falling back to a
        placeholder -- callers should try RAG+Gemini generation next, then
        find_generic_fallback_question as the final safety net."""
        from app.modules.interview.service import _normalize_question_type_label, _get_question_type_aliases, _log_question_selection
        try:
            with db_pool.get_connection() as conn:
                if not conn:
                    return None
                with conn.cursor() as cur:
                    normalized_company = (company_name or '').strip().lower()
                    if not normalized_company or normalized_company in ('any', 'general', 'n/a'):
                        return None

                    already_asked = []
                    if session_id:
                        cur.execute("""
                            SELECT DISTINCT question
                            FROM interview_data
                            WHERE session_id = %s AND question IS NOT NULL AND question != ''
                        """, (session_id,))
                        already_asked = [row[0] for row in cur.fetchall()]

                    exclusion_clause = ""
                    exclusion_params = []
                    if already_asked:
                        placeholders = ','.join(['%s'] * len(already_asked))
                        exclusion_clause = f" AND question NOT IN ({placeholders})"
                        exclusion_params = already_asked

                    normalized_role = (job_role or '').strip().lower()
                    normalized_industry = (industry_type or '').strip().lower()
                    normalized_interview = (interview_type or '').strip().lower()
                    normalized_experience = (work_experience or '').strip().lower()
                    normalized_question_type = _normalize_question_type_label(question_type_filter)
                    question_type_aliases = _get_question_type_aliases(normalized_question_type)

                    include_interview_filter = normalized_interview and normalized_interview not in ('any', 'n/a', '')
                    include_experience_filter = normalized_experience and normalized_experience not in ('any', 'n/a', '')
                    include_question_type_filter = bool(question_type_aliases)

                    base_filters = ["LOWER(TRIM(role)) = %s", "LOWER(TRIM(company)) = %s"]
                    base_params = [normalized_role, normalized_company]
                    if normalized_industry and normalized_industry not in ('any', 'n/a', ''):
                        base_filters.append("LOWER(TRIM(industry)) = %s")
                        base_params.append(normalized_industry)

                    full_filters = list(base_filters)
                    full_params = list(base_params)
                    if include_interview_filter:
                        full_filters.append("LOWER(TRIM(interview_type)) = %s")
                        full_params.append(normalized_interview)
                    if include_experience_filter:
                        full_filters.append("LOWER(TRIM(work_experience)) = %s")
                        full_params.append(normalized_experience)
                    if include_question_type_filter:
                        full_filters.append("LOWER(TRIM(question_type)) = ANY(%s)")
                        full_params.append(question_type_aliases)

                    result = None
                    difficulty_source = None

                    if preferred_difficulty:
                        where_clause = " AND ".join(full_filters)
                        query = f"""
                            SELECT question, mandatory_skills, difficulty, question_type, interview_type, work_experience
                            FROM interview_questions
                            WHERE {where_clause} {exclusion_clause} AND LOWER(TRIM(difficulty)) = %s
                            ORDER BY RANDOM() LIMIT 1
                        """
                        cur.execute(query, tuple(full_params + exclusion_params + [preferred_difficulty.lower()]))
                        result = cur.fetchone()
                        difficulty_source = "difficulty_lookup"

                    if not result:
                        where_clause = " AND ".join(full_filters)
                        query = f"""
                            SELECT question, mandatory_skills, difficulty, question_type, interview_type, work_experience
                            FROM interview_questions
                            WHERE {where_clause} {exclusion_clause}
                            ORDER BY RANDOM() LIMIT 1
                        """
                        cur.execute(query, tuple(full_params + exclusion_params))
                        result = cur.fetchone()
                        difficulty_source = "company_specific_lookup"

                    if (not result) and (include_interview_filter or include_experience_filter):
                        relaxed_filters = list(base_filters)
                        relaxed_params = list(base_params)
                        if include_question_type_filter:
                            relaxed_filters.append("LOWER(TRIM(question_type)) = ANY(%s)")
                            relaxed_params.append(question_type_aliases)
                        relaxed_clause = " AND ".join(relaxed_filters)
                        query = f"""
                            SELECT question, mandatory_skills, difficulty, question_type, interview_type, work_experience
                            FROM interview_questions
                            WHERE {relaxed_clause} {exclusion_clause}
                            ORDER BY RANDOM() LIMIT 1
                        """
                        cur.execute(query, tuple(relaxed_params + exclusion_params))
                        result = cur.fetchone()
                        difficulty_source = "company_specific_relaxed_lookup"

                    if not result:
                        return None

                    question_text, mandatory_skills_value, difficulty_value, question_type_value, db_interview_type, db_work_experience = result
                    _log_question_selection(
                        source=difficulty_source,
                        role=job_role, company=company_name,
                        difficulty=difficulty_value or preferred_difficulty or 'medium',
                        interview_type=db_interview_type or normalized_interview,
                        work_experience=db_work_experience or normalized_experience,
                        question_type=question_type_value, question_text=question_text,
                    )
                    return {
                        'question': question_text,
                        'mandatory_skills': mandatory_skills_value or 'Communication, Problem-solving',
                        'difficulty': (difficulty_value or preferred_difficulty or 'medium'),
                        'question_type': (question_type_value or 'standard').lower(),
                        'interview_type': (db_interview_type or normalized_interview or '').lower() or normalized_interview,
                        'work_experience': (db_work_experience or normalized_experience or '').lower() or normalized_experience,
                    }
        except Exception as e:
            logger.error(f"Error finding company-specific question: {e}")
            return None

    @classmethod
    def find_generic_fallback_question(
        cls,
        job_role: str,
        industry_type: str,
        company_name: str,
        question_number: int,
        session_id: str = None,
        interview_type: Optional[str] = None,
        work_experience: Optional[str] = None,
        question_type_filter: Optional[str] = None,
    ) -> Dict[str, str]:
        """Company-agnostic relaxation chain (old get_question_from_db steps 3-5) plus the
        hardcoded placeholder safety net. Used only when neither a company-specific DB match
        nor RAG+Gemini generation produced a usable question."""
        from app.modules.interview.service import _normalize_question_type_label, _get_question_type_aliases, _log_question_selection
        try:
            with db_pool.get_connection() as conn:
                if not conn:
                    raise Exception("No DB connection")
                with conn.cursor() as cur:
                    already_asked = []
                    if session_id:
                        cur.execute("""
                            SELECT DISTINCT question, question_type
                            FROM interview_data
                            WHERE session_id = %s AND question IS NOT NULL AND question != ''
                        """, (session_id,))
                        already_asked = [row[0] for row in cur.fetchall()]

                    exclusion_clause = ""
                    exclusion_params = []
                    if already_asked:
                        placeholders = ','.join(['%s'] * len(already_asked))
                        exclusion_clause = f" AND question NOT IN ({placeholders})"
                        exclusion_params = already_asked

                    normalized_role = (job_role or '').strip().lower()
                    normalized_industry = (industry_type or '').strip().lower()
                    normalized_question_type = _normalize_question_type_label(question_type_filter)
                    question_type_aliases = _get_question_type_aliases(normalized_question_type)
                    include_question_type_filter = bool(question_type_aliases)

                    result = None
                    if normalized_industry and normalized_industry not in ('any', 'n/a', ''):
                        filters = ["LOWER(TRIM(role)) = %s", "LOWER(TRIM(industry)) = %s"]
                        params = [normalized_role, normalized_industry]
                        if include_question_type_filter:
                            filters.append("LOWER(TRIM(question_type)) = ANY(%s)")
                            params.append(question_type_aliases)
                        where_clause = " AND ".join(filters)
                        query = f"""
                            SELECT question, mandatory_skills, difficulty, question_type, interview_type, work_experience
                            FROM interview_questions
                            WHERE {where_clause} {exclusion_clause}
                            ORDER BY RANDOM() LIMIT 1
                        """
                        cur.execute(query, tuple(params + exclusion_params))
                        result = cur.fetchone()

                    if not result:
                        base_filters = ["LOWER(TRIM(role)) = %s"]
                        base_params = [normalized_role]
                        if include_question_type_filter:
                            base_filters.append("LOWER(TRIM(question_type)) = ANY(%s)")
                            base_params.append(question_type_aliases)
                        where_clause = " AND ".join(base_filters)
                        query = f"""
                            SELECT question, mandatory_skills, difficulty, question_type, interview_type, work_experience
                            FROM interview_questions
                            WHERE {where_clause} {exclusion_clause}
                            ORDER BY RANDOM() LIMIT 1
                        """
                        cur.execute(query, tuple(base_params + exclusion_params))
                        result = cur.fetchone()

                    if not result and include_question_type_filter:
                        final_filters = ["LOWER(TRIM(role)) = %s"]
                        final_params = [normalized_role]
                        where_clause = " AND ".join(final_filters)
                        query = f"""
                            SELECT question, mandatory_skills, difficulty, question_type, interview_type, work_experience
                            FROM interview_questions
                            WHERE {where_clause} {exclusion_clause}
                            ORDER BY RANDOM() LIMIT 1
                        """
                        cur.execute(query, tuple(final_params + exclusion_params))
                        result = cur.fetchone()

                    if result:
                        question_text, mandatory_skills_value, difficulty_value, question_type_value, db_interview_type, db_work_experience = result
                        _log_question_selection(
                            source="generic_fallback_lookup",
                            role=job_role, company=company_name,
                            difficulty=difficulty_value,
                            interview_type=db_interview_type or (interview_type or ''),
                            work_experience=db_work_experience or (work_experience or ''),
                            question_type=question_type_value, question_text=question_text,
                        )
                        return {
                            'question': question_text,
                            'mandatory_skills': mandatory_skills_value or 'Communication, Problem-solving',
                            'difficulty': difficulty_value or 'medium',
                            'question_type': (question_type_value or 'standard').lower(),
                            'interview_type': (db_interview_type or interview_type or '').lower() or interview_type,
                            'work_experience': (db_work_experience or work_experience or '').lower() or work_experience,
                        }
                    else:
                        _log_question_selection(
                            source="fallback_placeholder",
                            role=job_role, company=company_name,
                            difficulty='medium', interview_type=interview_type, work_experience=work_experience,
                            question_type='standard', question_text=f"Question {question_number}: Tell me about your experience with {job_role} responsibilities.",
                        )
                        return {
                            'question': f"Question {question_number}: Tell me about your experience with {job_role} responsibilities.",
                            'mandatory_skills': 'Communication, Problem-solving',
                            'difficulty': 'medium',
                            'question_type': 'standard',
                            'interview_type': interview_type,
                            'work_experience': work_experience,
                        }
        except Exception as e:
            logger.error(f"Error getting generic fallback question: {e}")
            return {
                'question': f"Question {question_number}: Tell me about your experience with {job_role} responsibilities.",
                'mandatory_skills': 'Communication, Problem-solving',
                'difficulty': 'medium',
                'question_type': 'standard',
                'interview_type': interview_type,
                'work_experience': work_experience,
            }

    @classmethod
    def get_question_from_db(
        cls,
        job_role: str,
        industry_type: str,
        company_name: str,
        question_number: int,
        session_id: str = None,
        interview_type: Optional[str] = None,
        work_experience: Optional[str] = None,
        question_type_filter: Optional[str] = None,
    ) -> Dict[str, str]:
        from app.modules.interview.service import _normalize_question_type_label, _get_question_type_aliases, _log_question_selection
        try:
            with db_pool.get_connection() as conn:
                if not conn:
                    raise Exception("No DB connection")
                with conn.cursor() as cur:
                    already_asked = []
                    if session_id:
                        cur.execute("""
                            SELECT DISTINCT question, question_type
                            FROM interview_data 
                            WHERE session_id = %s AND question IS NOT NULL AND question != ''
                        """, (session_id,))
                        already_asked = [row[0] for row in cur.fetchall()]

                    exclusion_clause = ""
                    exclusion_params = []
                    if already_asked:
                        placeholders = ','.join(['%s'] * len(already_asked))
                        exclusion_clause = f" AND question NOT IN ({placeholders})"
                        exclusion_params = already_asked

                    normalized_role = (job_role or '').strip().lower()
                    normalized_industry = (industry_type or '').strip().lower()
                    normalized_company = (company_name or '').strip().lower()
                    normalized_interview = (interview_type or '').strip().lower()
                    normalized_experience = (work_experience or '').strip().lower()
                    normalized_question_type = _normalize_question_type_label(question_type_filter)
                    question_type_aliases = _get_question_type_aliases(normalized_question_type)

                    include_interview_filter = normalized_interview and normalized_interview not in ('any', 'n/a', '')
                    include_experience_filter = normalized_experience and normalized_experience not in ('any', 'n/a', '')
                    include_question_type_filter = bool(question_type_aliases)

                    filters = ["LOWER(TRIM(role)) = %s"]
                    params = [normalized_role]

                    if normalized_industry and normalized_industry not in ('any', 'n/a', ''):
                        filters.append("LOWER(TRIM(industry)) = %s")
                        params.append(normalized_industry)

                    if normalized_company and normalized_company not in ('any', 'general', ''):
                        filters.append("LOWER(TRIM(company)) = %s")
                        params.append(normalized_company)

                    if include_interview_filter:
                        filters.append("LOWER(TRIM(interview_type)) = %s")
                        params.append(normalized_interview)

                    if include_experience_filter:
                        filters.append("LOWER(TRIM(work_experience)) = %s")
                        params.append(normalized_experience)

                    if include_question_type_filter:
                        filters.append("LOWER(TRIM(question_type)) = ANY(%s)")
                        params.append(question_type_aliases)

                    result = None
                    if len(filters) > 1:
                        where_clause = " AND ".join(filters)
                        query = f"""
                            SELECT question, mandatory_skills, difficulty, question_type, interview_type, work_experience
                            FROM interview_questions
                            WHERE {where_clause} {exclusion_clause}
                            ORDER BY RANDOM() LIMIT 1
                        """
                        cur.execute(query, tuple(params + exclusion_params))
                        result = cur.fetchone()

                    if (not result) and (include_interview_filter or include_experience_filter):
                        relaxed_filters = ["LOWER(TRIM(role)) = %s"]
                        relaxed_params = [normalized_role]
                        if normalized_industry and normalized_industry not in ('any', 'n/a', ''):
                            relaxed_filters.append("LOWER(TRIM(industry)) = %s")
                            relaxed_params.append(normalized_industry)
                        if normalized_company and normalized_company not in ('any', 'general', ''):
                            relaxed_filters.append("LOWER(TRIM(company)) = %s")
                            relaxed_params.append(normalized_company)
                        if include_question_type_filter:
                            relaxed_filters.append("LOWER(TRIM(question_type)) = ANY(%s)")
                            relaxed_params.append(question_type_aliases)

                        relaxed_clause = " AND ".join(relaxed_filters)
                        query = f"""
                            SELECT question, mandatory_skills, difficulty, question_type, interview_type, work_experience
                            FROM interview_questions
                            WHERE {relaxed_clause} {exclusion_clause}
                            ORDER BY RANDOM() LIMIT 1
                        """
                        cur.execute(query, tuple(relaxed_params + exclusion_params))
                        result = cur.fetchone()

                    if not result and normalized_industry and normalized_industry not in ('any', 'n/a', ''):
                        filters = ["LOWER(TRIM(role)) = %s", "LOWER(TRIM(industry)) = %s"]
                        params = [normalized_role, normalized_industry]
                        if include_question_type_filter:
                            filters.append("LOWER(TRIM(question_type)) = ANY(%s)")
                            params.append(question_type_aliases)
                        where_clause = " AND ".join(filters)
                        query = f"""
                            SELECT question, mandatory_skills, difficulty, question_type, interview_type, work_experience
                            FROM interview_questions
                            WHERE {where_clause} {exclusion_clause}
                            ORDER BY RANDOM() LIMIT 1
                        """
                        cur.execute(query, tuple(params + exclusion_params))
                        result = cur.fetchone()

                    if not result:
                        base_filters = ["LOWER(TRIM(role)) = %s"]
                        base_params = [normalized_role]
                        if include_question_type_filter:
                            base_filters.append("LOWER(TRIM(question_type)) = ANY(%s)")
                            base_params.append(question_type_aliases)
                        where_clause = " AND ".join(base_filters)
                        query = f"""
                            SELECT question, mandatory_skills, difficulty, question_type, interview_type, work_experience
                            FROM interview_questions
                            WHERE {where_clause} {exclusion_clause}
                            ORDER BY RANDOM() LIMIT 1
                        """
                        cur.execute(query, tuple(base_params + exclusion_params))
                        result = cur.fetchone()

                    if not result and include_question_type_filter:
                        final_filters = ["LOWER(TRIM(role)) = %s"]
                        final_params = [normalized_role]
                        where_clause = " AND ".join(final_filters)
                        query = f"""
                            SELECT question, mandatory_skills, difficulty, question_type, interview_type, work_experience
                            FROM interview_questions
                            WHERE {where_clause} {exclusion_clause}
                            ORDER BY RANDOM() LIMIT 1
                        """
                        cur.execute(query, tuple(final_params + exclusion_params))
                        result = cur.fetchone()

                    if result:
                        question_text, mandatory_skills_value, difficulty_value, question_type_value, db_interview_type, db_work_experience = result
                        _log_question_selection(
                            source="role_lookup",
                            role=job_role, company=company_name,
                            difficulty=difficulty_value,
                            interview_type=db_interview_type or normalized_interview,
                            work_experience=db_work_experience or normalized_experience,
                            question_type=question_type_value, question_text=question_text,
                        )
                        return {
                            'question': question_text,
                            'mandatory_skills': mandatory_skills_value or 'Communication, Problem-solving',
                            'difficulty': difficulty_value or 'medium',
                            'question_type': (question_type_value or 'standard').lower(),
                            'interview_type': (db_interview_type or normalized_interview or '').lower() or normalized_interview,
                            'work_experience': (db_work_experience or normalized_experience or '').lower() or normalized_experience,
                        }
                    else:
                        _log_question_selection(
                            source="fallback_placeholder",
                            role=job_role, company=company_name,
                            difficulty='medium', interview_type=interview_type, work_experience=work_experience,
                            question_type='standard', question_text=f"Question {question_number}: Tell me about your experience with {job_role} responsibilities.",
                        )
                        return {
                            'question': f"Question {question_number}: Tell me about your experience with {job_role} responsibilities.",
                            'mandatory_skills': 'Communication, Problem-solving',
                            'difficulty': 'medium',
                            'question_type': 'standard',
                            'interview_type': interview_type,
                            'work_experience': work_experience,
                        }
        except Exception as e:
            logger.error(f"Error getting question from database: {e}")
            return {
                'question': f"Question {question_number}: Tell me about your experience with {job_role} responsibilities.",
                'mandatory_skills': 'Communication, Problem-solving',
                'difficulty': 'medium',
                'question_type': 'standard',
                'interview_type': interview_type,
                'work_experience': work_experience,
            }

    @classmethod
    def get_pre_generated_question(cls, session_id: str, question_number: int) -> Optional[Dict[str, Any]]:
        try:
            with db_pool.get_connection() as conn:
                if not conn:
                    return None
                with conn.cursor() as cur:
                    cls._ensure_pre_generated_questions_table(cur)
                    cur.execute(
                        """
                        SELECT question_text, mandatory_skills, difficulty, question_type, generation_context
                        FROM session_pre_generated_questions
                        WHERE session_id = %s AND question_number = %s
                        LIMIT 1
                        """,
                        (session_id, question_number)
                    )
                    row = cur.fetchone()
                    if not row:
                        return None
                    return {
                        "question": row[0],
                        "mandatory_skills": row[1] or "Communication, Problem-solving",
                        "difficulty": row[2] or "medium",
                        "question_type": (row[3] or "standard").lower(),
                        "generation_context": row[4] or {},
                    }
        except Exception as exc:
            logger.error(f"Error checking pre-generated questions for session {session_id}, q#{question_number}: {exc}")
        return None

    @classmethod
    def get_active_session_repo(cls, student_email: str) -> Optional[Dict[str, Any]]:
        try:
            with db_pool.get_connection() as conn:
                if not conn:
                    return None
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT sm.session_id, sm.status, sm.interview_type, sm.work_experience, sm.started_at
                        FROM session_metadata sm
                        JOIN students s ON sm.student_id = s.student_id
                        WHERE s.email = %s
                          AND LOWER(TRIM(COALESCE(sm.status, ''))) = 'active'
                        ORDER BY sm.started_at DESC
                        LIMIT 1
                        """,
                        (student_email,)
                    )
                    active_session = cur.fetchone()
                    if not active_session:
                        return None

                    session_id, status, interview_type, work_experience, started_at = active_session

                    cur.execute(
                        """
                        SELECT 
                            MAX(question_number) as current_question,
                            MAX(CASE WHEN question_number = (SELECT MAX(question_number) FROM interview_data WHERE session_id = %s) 
                                THEN question END) as current_question_text,
                            MAX(CASE WHEN question_number = (SELECT MAX(question_number) FROM interview_data WHERE session_id = %s) 
                                THEN question_type END) as current_question_type,
                            MAX(CASE WHEN question_number = (SELECT MAX(question_number) FROM interview_data WHERE session_id = %s) 
                                THEN mandatory_skills END) as mandatory_skills,
                            MAX(CASE WHEN question_number = (SELECT MAX(question_number) FROM interview_data WHERE session_id = %s) 
                                THEN difficulty END) as difficulty,
                            MAX(job_role) as job_role,
                            MAX(industry_type) as industry_type,
                            MAX(company_name) as company_name
                        FROM interview_data
                        WHERE session_id = %s
                        """,
                        (session_id, session_id, session_id, session_id, session_id)
                    )
                    question_data = cur.fetchone()
                    if not question_data or not question_data[0]:
                        return None

                    current_question, question_text, question_type, mandatory_skills, difficulty, job_role, industry_type, company_name = question_data

                    cur.execute(
                        "SELECT answer FROM interview_data WHERE session_id = %s AND question_number = %s",
                        (session_id, current_question)
                    )
                    answer_row = cur.fetchone()
                    current_answer = answer_row[0] if answer_row else None
                    has_answered_current = bool(current_answer and current_answer.strip())

                    cur.execute(
                        "SELECT difficulty FROM interview_data WHERE session_id = %s AND difficulty IS NOT NULL ORDER BY question_number",
                        (session_id,)
                    )
                    difficulty_rows = cur.fetchall()
                    difficulty_weights = [DIFFICULTY_WEIGHTS.get((row[0] or "medium").lower(), 2) for row in difficulty_rows]
                    from app.modules.interview.service import _determine_max_questions
                    current_max_questions = _determine_max_questions(difficulty_weights, current_question)

                    return {
                        "has_active_session": True,
                        "session_id": session_id,
                        "current_question_number": current_question,
                        "has_answered_current": has_answered_current,
                        "current_question": {
                            "question": question_text,
                            "question_type": question_type or "standard",
                            "mandatory_skills": mandatory_skills,
                            "difficulty": difficulty or "medium"
                        },
                        "job_role": job_role,
                        "industry_type": industry_type,
                        "company_name": company_name,
                        "interview_type": interview_type,
                        "work_experience": work_experience,
                        "max_questions": current_max_questions,
                        "started_at": format_datetime_ist(started_at) if started_at else None
                    }
        except Exception as e:
            logger.error(f"Failed checking active session repo: {e}")
            raise

    @classmethod
    def terminate_session_repo(cls, session_id: str, reason: str) -> bool:
        truncated_reason = reason[:500]
        with db_pool.get_connection() as conn:
            if not conn:
                raise Exception("Database connection failed")
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE session_metadata
                    SET status = 'terminated',
                        termination_reason = %s,
                        terminated_at = CURRENT_TIMESTAMP,
                        completed_at = COALESCE(completed_at, CURRENT_TIMESTAMP)
                    WHERE session_id = %s
                    RETURNING 1
                    """,
                    (truncated_reason, session_id),
                )
                updated = cur.fetchone()
                if not updated:
                    return False
            conn.commit()
        return True

    @classmethod
    def get_session_rating_repo(cls, session_id: str, student_email: str) -> Dict[str, Any]:
        with db_pool.get_connection() as conn:
            if not conn:
                raise Exception("Database connection unavailable")
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT sr.rating, sr.comments, sr.created_at
                    FROM session_ratings sr
                    JOIN students s ON sr.student_id = s.student_id
                    WHERE sr.session_id = %s AND LOWER(TRIM(s.email)) = LOWER(TRIM(%s))
                    """,
                    (session_id, student_email),
                )
                result = cur.fetchone()
                if not result:
                    return {"session_id": session_id, "rating": None, "comments": None}
                return {
                    "session_id": session_id,
                    "rating": int(result[0]),
                    "comments": result[1],
                    "created_at": format_datetime_ist(result[2]),
                }

    @classmethod
    def submit_session_rating_repo(cls, session_id: str, student_email: str, rating: int, comments: Optional[str]) -> Tuple[int, Optional[str]]:
        with db_pool.get_connection() as conn:
            if not conn:
                raise Exception("Database connection unavailable")
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT student_id FROM students WHERE LOWER(TRIM(email)) = LOWER(TRIM(%s))",
                    (student_email,),
                )
                student_row = cur.fetchone()
                if not student_row:
                    raise KeyError("Student not found")
                student_id = student_row[0]

                cur.execute(
                    "SELECT 1 FROM session_metadata WHERE session_id = %s AND student_id = %s",
                    (session_id, student_id),
                )
                if not cur.fetchone():
                    raise PermissionError("Session does not belong to this student")

                cur.execute(
                    """
                    INSERT INTO session_ratings (session_id, student_id, rating, comments)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (session_id, student_id)
                    DO UPDATE SET rating = EXCLUDED.rating, comments = EXCLUDED.comments, created_at = CURRENT_TIMESTAMP
                    RETURNING id, created_at
                    """,
                    (session_id, student_id, rating, comments),
                )
                inserted = cur.fetchone()
                conn.commit()
                created_at_str = format_datetime_ist(inserted[1]) if inserted else None
                return student_id, created_at_str

    @classmethod
    def verify_session_belongs_to_student(cls, session_id: str, student_id: int) -> bool:
        """Check that the given session is owned by the given student."""
        with db_pool.get_connection() as conn:
            if not conn:
                raise Exception("Database connection unavailable")
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM session_metadata WHERE session_id = %s AND student_id = %s",
                    (session_id, student_id),
                )
                return cur.fetchone() is not None

    @classmethod
    def save_answer_and_get_context(
        cls,
        session_id: str,
        answer: str,
        question_type: str,
        code: Optional[str],
        stdin: Optional[str],
        stdout: Optional[str],
        stderr: Optional[str],
        runtime_error: Optional[str],
        execution_success: bool,
        has_run: bool,
        is_final: bool,
        saved_video_path: Optional[str],
        is_system_design: bool,
        system_design_diagram: Optional[str],
    ) -> Dict[str, Any]:
        with db_pool.get_connection() as conn:
            if not conn:
                raise Exception("Database connection unavailable")
            with conn.cursor() as cur:
                cls._ensure_interview_data_columns(cur)

                cur.execute(
                    "SELECT MAX(question_number) FROM interview_data WHERE session_id = %s",
                    (session_id,),
                )
                max_row = cur.fetchone()
                current_q = max_row[0] if max_row and max_row[0] else 1

                cur.execute(
                    """
                    SELECT question, mandatory_skills, job_role, industry_type, company_name,
                           difficulty, question_type, interview_type, work_experience,
                           job_description_id, job_description_text, job_desc, resume_id, generation_context
                    FROM interview_data
                    WHERE session_id = %s AND question_number = %s
                    """,
                    (session_id, current_q),
                )
                row = cur.fetchone()
                if not row:
                    raise KeyError(f"Question #{current_q} not found for session {session_id}")

                (
                    current_question, mandatory_skills, job_role, industry_type, company_name,
                    db_difficulty, db_question_type, db_interview_type, db_work_experience,
                    jd_id, jd_text, jd_desc, resume_id, generation_context,
                ) = row

                cur.execute(
                    """
                    UPDATE interview_data
                    SET answer = %s,
                        analysis_status = %s,
                        code_submission = %s,
                        stdin_input = %s,
                        stdout_output = %s,
                        stderr_output = %s,
                        runtime_error = %s,
                        execution_success = %s,
                        manual_run = %s,
                        video_clip_path = COALESCE(%s, video_clip_path),
                        is_system_design = %s,
                        system_design_diagram = %s
                    WHERE session_id = %s AND question_number = %s
                    """,
                    (
                        answer,
                        "processing" if not is_final else "completed",
                        code,
                        stdin,
                        stdout,
                        stderr,
                        runtime_error,
                        execution_success,
                        has_run,
                        saved_video_path,
                        is_system_design,
                        system_design_diagram,
                        session_id,
                        current_q,
                    ),
                )
                conn.commit()

                cur.execute(
                    "SELECT difficulty FROM interview_data WHERE session_id = %s AND difficulty IS NOT NULL ORDER BY question_number",
                    (session_id,),
                )
                diff_rows = cur.fetchall()
                diff_weights = [DIFFICULTY_WEIGHTS.get((r[0] or "medium").lower(), 2) for r in diff_rows]
                from app.modules.interview.service import _determine_max_questions
                max_questions = _determine_max_questions(diff_weights, current_q)

                return {
                    "current_q": current_q,
                    "current_question": current_question,
                    "mandatory_skills": mandatory_skills or "Communication, Problem-solving",
                    "job_role": job_role or "Software Engineer",
                    "industry_type": industry_type or "Technology",
                    "company_name": company_name or "General",
                    "db_difficulty": db_difficulty or "medium",
                    "db_question_type": db_question_type or question_type or "standard",
                    "db_interview_type": db_interview_type,
                    "db_work_experience": db_work_experience,
                    "jd_id": jd_id,
                    "jd_text": jd_text,
                    "jd_desc": jd_desc,
                    "resume_id": resume_id,
                    "generation_context": generation_context or {},
                    "max_questions": max_questions,
                }

    @classmethod
    def save_sentiment_and_check_final(
        cls,
        session_id: str,
        question_number: int,
        sentiment_score: float,
        is_final_question: bool,
    ) -> None:
        with db_pool.get_connection() as conn:
            if not conn:
                return
            with conn.cursor() as cur:
                cls._ensure_interview_data_columns(cur)
                cls._ensure_session_metadata_columns(cur)

                if is_final_question:
                    cur.execute(
                        """
                        UPDATE interview_data
                        SET sentiment_score = %s,
                            analysis_status = 'completed'
                        WHERE session_id = %s AND question_number = %s
                        """,
                        (sentiment_score, session_id, question_number),
                    )
                    cur.execute(
                        """
                        UPDATE session_metadata
                        SET status = 'completed',
                            completed_at = CURRENT_TIMESTAMP
                        WHERE session_id = %s
                        """,
                        (session_id,),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE interview_data
                        SET sentiment_score = %s
                        WHERE session_id = %s AND question_number = %s
                        """,
                        (sentiment_score, session_id, question_number),
                    )
                conn.commit()

    @classmethod
    def save_next_question(
        cls,
        session_id: str,
        job_role: str,
        industry_type: str,
        company_name: str,
        interview_type: Optional[str],
        work_experience: Optional[str],
        next_q_num: int,
        next_q_data: Dict[str, Any],
        fallback_difficulty: str,
        jd_id: Optional[int] = None,
        jd_text: Optional[str] = None,
        jd_desc: Optional[str] = None,
        resume_id: Optional[int] = None,
        generation_context: Optional[dict] = None,
    ) -> None:
        with db_pool.get_connection() as conn:
            if not conn:
                return
            with conn.cursor() as cur:
                cls._ensure_interview_data_columns(cur)
                next_difficulty = next_q_data.get("difficulty") or fallback_difficulty or "medium"
                next_question_type = next_q_data.get("question_type") or "standard"
                next_mandatory_skills = next_q_data.get("mandatory_skills") or "Communication, Problem-solving"
                if isinstance(next_mandatory_skills, list):
                    next_mandatory_skills = ", ".join(next_mandatory_skills)

                insert_params = {
                    "session_id": session_id,
                    "job_role": job_role,
                    "industry_type": industry_type,
                    "company_name": company_name,
                    "interview_type": interview_type or None,
                    "work_experience": work_experience or None,
                    "question_number": next_q_num,
                    "question": next_q_data["question"],
                    "answer": "",
                    "analysis_status": "pending",
                    "difficulty": next_difficulty,
                    "mandatory_skills": next_mandatory_skills,
                    "question_type": next_question_type,
                    "job_description_id": jd_id,
                    "job_description_text": jd_text,
                    "job_desc": jd_desc,
                    "resume_id": resume_id,
                    "generation_context": Json(generation_context or {}),
                }
                cur.execute(
                    """
                    INSERT INTO interview_data 
                    (session_id, job_role, industry_type, company_name, interview_type, work_experience, question_number, 
                     question, answer, analysis_status, difficulty, mandatory_skills, question_type, job_description_id, job_description_text, job_desc, resume_id, generation_context, timestamp)
                    VALUES (%(session_id)s, %(job_role)s, %(industry_type)s, %(company_name)s, %(interview_type)s, %(work_experience)s, %(question_number)s,
                            %(question)s, %(answer)s, %(analysis_status)s, %(difficulty)s, %(mandatory_skills)s, %(question_type)s,
                            %(job_description_id)s, %(job_description_text)s, %(job_desc)s, %(resume_id)s, %(generation_context)s, CURRENT_TIMESTAMP)
                    ON CONFLICT (session_id, question_number) 
                    DO UPDATE SET question = EXCLUDED.question,
                                  question_type = EXCLUDED.question_type,
                                  interview_type = EXCLUDED.interview_type,
                                  work_experience = EXCLUDED.work_experience,
                                  job_description_id = EXCLUDED.job_description_id,
                                  job_description_text = EXCLUDED.job_description_text,
                                  job_desc = EXCLUDED.job_desc,
                                  resume_id = EXCLUDED.resume_id,
                                  generation_context = EXCLUDED.generation_context
                    """,
                    insert_params,
                )
                conn.commit()

    @classmethod
    def update_video_upload_path(cls, session_id: str, question_number: int, saved_video_path: str) -> None:
        try:
            with db_pool.get_connection() as conn:
                if not conn:
                    return
                with conn.cursor() as cur:
                    cls._ensure_interview_data_columns(cur)
                    cur.execute(
                        """
                        UPDATE interview_data
                        SET video_clip_path = %s,
                            video_analysis_status = COALESCE(video_analysis_status, 'pending')
                        WHERE session_id = %s AND question_number = %s
                        """,
                        (saved_video_path, session_id, question_number)
                    )
                    conn.commit()
        except Exception as exc:
            logger.error(f"Error updating video path for {session_id} Q#{question_number}: {exc}")

    @classmethod
    def get_pending_video_count(cls, session_id: str) -> Tuple[int, int]:
        try:
            with db_pool.get_connection() as conn:
                if not conn:
                    return (0, 0)
                with conn.cursor() as cur:
                    cls._ensure_interview_data_columns(cur)
                    cur.execute(
                        """
                        SELECT COUNT(*)
                        FROM interview_data
                        WHERE session_id = %s
                          AND video_clip_path IS NOT NULL AND video_clip_path != ''
                          AND COALESCE(video_analysis_status, 'pending') != 'completed'
                        """,
                        (session_id,)
                    )
                    pending_count = cur.fetchone()[0]

                    cur.execute(
                        """
                        SELECT COUNT(*)
                        FROM interview_data
                        WHERE session_id = %s
                          AND video_clip_path IS NOT NULL AND video_clip_path != ''
                          AND COALESCE(video_analysis_status, 'pending') = 'processing'
                        """,
                        (session_id,)
                    )
                    processing_count = cur.fetchone()[0]
                    return (pending_count, processing_count)
        except Exception as exc:
            logger.error(f"Error checking pending video analysis status for {session_id}: {exc}")
            return (0, 0)

    @classmethod
    def get_qa_history_for_feedback(cls, session_id: str) -> Tuple[List[tuple], str, str, str, Optional[str]]:
        with db_pool.get_connection() as conn:
            if not conn:
                raise Exception("Database connection failed")
            with conn.cursor() as cur:
                cls._ensure_interview_data_columns(cur)
                cur.execute(
                    """
                    SELECT question_number, question, answer, sentiment_score, mandatory_skills,
                           video_sentiment_score, video_demeanor, code_submission, execution_success, manual_run,
                           question_type
                    FROM interview_data
                    WHERE session_id = %s AND answer IS NOT NULL AND TRIM(answer) != ''
                    ORDER BY question_number
                    """,
                    (session_id,),
                )
                qa_history = cur.fetchall()

                cur.execute(
                    """
                    SELECT job_role, industry_type, company_name, interview_type
                    FROM interview_data
                    WHERE session_id = %s
                    LIMIT 1
                    """,
                    (session_id,),
                )
                role_info = cur.fetchone()
                job_role = role_info[0] if role_info and role_info[0] else "Software Engineer"
                industry_type = role_info[1] if role_info and role_info[1] else "Technology"
                company_name = role_info[2] if role_info and role_info[2] else "General"
                interview_type = role_info[3] if role_info and role_info[3] else None
                return qa_history, job_role, industry_type, company_name, interview_type

    @classmethod
    def save_detailed_feedback(cls, session_id: str, stored_payload: str) -> bool:
        try:
            with db_pool.get_connection() as conn:
                if not conn:
                    return False
                with conn.cursor() as cur:
                    cls._ensure_interview_data_columns(cur)
                    cur.execute(
                        """
                        UPDATE interview_data
                        SET detailed_feedback = %s,
                            analysis_status = 'completed'
                        WHERE session_id = %s
                        """,
                        (stored_payload, session_id),
                    )
                    conn.commit()
            return True
        except Exception as exc:
            logger.error(f"Failed storing detailed feedback for {session_id}: {exc}")
            return False

    @classmethod
    def get_session_scores(cls, session_id: str) -> Optional[tuple]:
        try:
            with db_pool.get_connection() as conn:
                if not conn:
                    return None
                with conn.cursor() as cur:
                    cls._ensure_session_metadata_columns(cur)
                    cur.execute(
                        """
                        SELECT overall_score, rubric_scores, status, completed_at
                        FROM session_metadata
                        WHERE session_id = %s
                        """,
                        (session_id,),
                    )
                    return cur.fetchone()
        except Exception as exc:
            logger.error(f"Error fetching scores for {session_id}: {exc}")
            return None

    @classmethod
    def _update_feedback_status(cls, query: str, params: tuple) -> None:
        try:
            with db_pool.get_connection() as conn:
                if not conn:
                    return
                with conn.cursor() as cur:
                    cls._ensure_session_metadata_columns(cur)
                    cur.execute(query, params)
                conn.commit()
        except Exception as exc:
            logger.error(f"Failed to update feedback status: {exc}")

    @classmethod
    def mark_feedback_pending(cls, session_id: str) -> None:
        requested_at = datetime.utcnow()
        cls._update_feedback_status(
            """
            UPDATE session_metadata
            SET feedback_status = 'pending',
                feedback_error = NULL,
                feedback_requested_at = %s,
                feedback_ready_at = NULL,
                feedback_generated = FALSE
            WHERE session_id = %s
            """,
            (requested_at, session_id),
        )

    @classmethod
    def mark_feedback_processing(cls, session_id: str) -> None:
        requested_at = datetime.utcnow()
        cls._update_feedback_status(
            """
            UPDATE session_metadata
            SET feedback_status = 'processing',
                feedback_error = NULL,
                feedback_requested_at = COALESCE(feedback_requested_at, %s)
            WHERE session_id = %s
            """,
            (requested_at, session_id),
        )

    @classmethod
    def mark_feedback_failed(cls, session_id: str, error_message: Optional[str]) -> None:
        ready_at = datetime.now(timezone.utc)
        truncated = error_message[:500] if error_message else None
        cls._update_feedback_status(
            """
            UPDATE session_metadata
            SET feedback_status = 'failed',
                feedback_error = %s,
                feedback_ready_at = %s,
                feedback_generated = FALSE
            WHERE session_id = %s
            """,
            (truncated, ready_at, session_id),
        )

    @classmethod
    def mark_feedback_completed(cls, session_id: str) -> None:
        ready_at = datetime.utcnow()
        cls._update_feedback_status(
            """
            UPDATE session_metadata
            SET feedback_status = 'completed',
                feedback_error = NULL,
                feedback_ready_at = %s,
                feedback_generated = TRUE
            WHERE session_id = %s
            """,
            (ready_at, session_id),
        )

    @classmethod
    def get_feedback_payload(cls, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            with db_pool.get_connection() as conn:
                if not conn:
                    return None
                with conn.cursor() as cur:
                    cls._ensure_interview_data_columns(cur)
                    cur.execute(
                        """
                        SELECT detailed_feedback
                        FROM interview_data
                        WHERE session_id = %s AND detailed_feedback IS NOT NULL
                        ORDER BY question_number DESC
                        LIMIT 1
                        """,
                        (session_id,)
                    )
                    result = cur.fetchone()
                    if not result or not result[0]:
                        return None
                    payload_text = result[0]
                    parsed = _parse_feedback_json(payload_text)

                    # The Gemini feedback JSON schema does not include question_type per
                    # question, so the frontend has no reliable way to detect system-design
                    # questions once the candidate's own answer isn't diagram JSON (e.g. they
                    # typed "I don't know"). Merge in the ground-truth question_type already
                    # stored per question_number so the UI can render diagrams correctly.
                    if isinstance(parsed, dict) and isinstance(parsed.get("questions"), list):
                        cur.execute(
                            """
                            SELECT question_number, question_type
                            FROM interview_data
                            WHERE session_id = %s
                            """,
                            (session_id,),
                        )
                        question_type_by_number = {row[0]: row[1] for row in cur.fetchall() if row[1]}
                        for item in parsed["questions"]:
                            if not isinstance(item, dict):
                                continue
                            question_number = item.get("number")
                            if question_number in question_type_by_number and not item.get("question_type"):
                                item["question_type"] = question_type_by_number[question_number]

                    return {
                        "structured": parsed,
                        "raw": payload_text,
                    }
        except Exception as exc:
            logger.error(f"Error retrieving feedback payload for {session_id}: {exc}")
        return None

    @classmethod
    def get_feedback_status_data(cls, session_id: str) -> Optional[tuple]:
        try:
            with db_pool.get_connection() as conn:
                if not conn:
                    return None
                with conn.cursor() as cur:
                    cls._ensure_session_metadata_columns(cur)
                    cur.execute(
                        """
                        SELECT feedback_status, feedback_error, feedback_requested_at, feedback_ready_at, feedback_generated
                        FROM session_metadata
                        WHERE session_id = %s
                        """,
                        (session_id,),
                    )
                    return cur.fetchone()
        except Exception as exc:
            logger.error(f"Error checking feedback status for {session_id}: {exc}")
            return None

    @classmethod
    def list_interview_industries(cls) -> List[str]:
        with db_pool.get_connection() as conn:
            if not conn:
                return []
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT TRIM(industry) AS industry_name
                    FROM interview_questions
                    WHERE industry IS NOT NULL AND TRIM(industry) <> ''
                    ORDER BY industry_name ASC
                    """
                )
                return [row[0] for row in cur.fetchall()]

    @classmethod
    def list_companies_by_industry(cls, industry: Optional[str]) -> List[str]:
        with db_pool.get_connection() as conn:
            if not conn:
                return []
            with conn.cursor() as cur:
                if industry:
                    cur.execute(
                        """
                        SELECT DISTINCT TRIM(company) AS company_name
                        FROM interview_questions
                        WHERE company IS NOT NULL AND TRIM(company) <> ''
                          AND LOWER(TRIM(industry)) = LOWER(TRIM(%s))
                        ORDER BY company_name ASC
                        """,
                        (industry,),
                    )
                else:
                    cur.execute(
                        """
                        SELECT DISTINCT TRIM(company) AS company_name
                        FROM interview_questions
                        WHERE company IS NOT NULL AND TRIM(company) <> ''
                        ORDER BY company_name ASC
                        """
                    )
                return [row[0] for row in cur.fetchall()]

    @classmethod
    def list_trending_companies(cls) -> List[Dict[str, Any]]:
        with db_pool.get_connection() as conn:
            if not conn:
                return []
            with conn.cursor() as cur:
                ensure_companies_table(cur)
                cur.execute(
                    """
                    SELECT name, logo_url, difficulty_tag, work_experience_tag
                    FROM companies
                    WHERE is_active = TRUE
                    ORDER BY id
                    """
                )
                return [
                    {
                        "name": row[0],
                        "logo": row[1],
                        "difficulty": row[2],
                        "experience": row[3],
                    }
                    for row in cur.fetchall()
                ]

    @classmethod
    def list_interview_types(cls, industry: Optional[str], company: Optional[str]) -> List[str]:
        with db_pool.get_connection() as conn:
            if not conn:
                return []
            with conn.cursor() as cur:
                filters = ["interview_type IS NOT NULL", "TRIM(interview_type) <> ''"]
                params: List[str] = []
                if industry:
                    filters.append("LOWER(TRIM(industry)) = LOWER(TRIM(%s))")
                    params.append(industry)
                if company:
                    filters.append("LOWER(TRIM(company)) = LOWER(TRIM(%s))")
                    params.append(company)
                where_clause = " AND ".join(filters)
                cur.execute(
                    f"""
                    SELECT DISTINCT TRIM(interview_type) AS interview_type_name
                    FROM interview_questions
                    WHERE {where_clause}
                    ORDER BY interview_type_name ASC
                    """,
                    tuple(params),
                )
                return [row[0] for row in cur.fetchall()]

    @classmethod
    def list_work_experience_levels(cls, industry: Optional[str], company: Optional[str], interview_type: Optional[str]) -> List[str]:
        with db_pool.get_connection() as conn:
            if not conn:
                return []
            with conn.cursor() as cur:
                filters = ["work_experience IS NOT NULL", "TRIM(work_experience) <> ''"]
                params: List[str] = []
                if industry:
                    filters.append("LOWER(TRIM(industry)) = LOWER(TRIM(%s))")
                    params.append(industry)
                if company:
                    filters.append("LOWER(TRIM(company)) = LOWER(TRIM(%s))")
                    params.append(company)
                if interview_type:
                    filters.append("LOWER(TRIM(interview_type)) = LOWER(TRIM(%s))")
                    params.append(interview_type)
                where_clause = " AND ".join(filters)
                cur.execute(
                    f"""
                    SELECT DISTINCT TRIM(work_experience) AS work_experience_level
                    FROM interview_questions
                    WHERE {where_clause}
                    ORDER BY work_experience_level ASC
                    """,
                    tuple(params),
                )
                return [row[0] for row in cur.fetchall()]

    @classmethod
    def list_job_roles_for_selection(
        cls,
        industry: Optional[str],
        company: Optional[str],
        interview_type: Optional[str],
        work_experience: Optional[str],
        program_name: Optional[str],
    ) -> List[str]:
        with db_pool.get_connection() as conn:
            if not conn:
                return []
            with conn.cursor() as cur:
                filters = ["role IS NOT NULL", "TRIM(role) <> ''"]
                params: List[str] = []
                if industry:
                    filters.append("LOWER(TRIM(industry)) = LOWER(TRIM(%s))")
                    params.append(industry)
                if company:
                    filters.append("LOWER(TRIM(company)) = LOWER(TRIM(%s))")
                    params.append(company)
                if interview_type:
                    filters.append("LOWER(TRIM(interview_type)) = LOWER(TRIM(%s))")
                    params.append(interview_type)
                if work_experience:
                    filters.append("LOWER(TRIM(work_experience)) = LOWER(TRIM(%s))")
                    params.append(work_experience)
                if program_name:
                    try:
                        cur.execute(
                            """
                            SELECT EXISTS (
                                SELECT 1
                                FROM information_schema.columns
                                WHERE table_name = 'interview_questions' AND column_name = 'program_name'
                            )
                            """
                        )
                        if cur.fetchone()[0]:
                            filters.append("LOWER(TRIM(program_name)) = LOWER(TRIM(%s))")
                            params.append(program_name)
                    except Exception:
                        pass
                where_clause = " AND ".join(filters)
                cur.execute(
                    f"""
                    SELECT DISTINCT TRIM(role) AS job_role_name
                    FROM interview_questions
                    WHERE {where_clause}
                    ORDER BY job_role_name ASC
                    """,
                    tuple(params),
                )
                return [row[0] for row in cur.fetchall()]

    @classmethod
    def list_job_roles_by_work_experience(cls, work_experience: Optional[str], program_name: Optional[str]) -> List[str]:
        with db_pool.get_connection() as conn:
            if not conn:
                return []
            with conn.cursor() as cur:
                filters = ["role IS NOT NULL", "TRIM(role) <> ''"]
                params: List[str] = []
                if work_experience:
                    filters.append("LOWER(TRIM(work_experience)) = LOWER(TRIM(%s))")
                    params.append(work_experience)
                if program_name:
                    try:
                        cur.execute(
                            """
                            SELECT EXISTS (
                                SELECT 1
                                FROM information_schema.columns
                                WHERE table_name = 'interview_questions' AND column_name = 'program_name'
                            )
                            """
                        )
                        if cur.fetchone()[0]:
                            filters.append("LOWER(TRIM(program_name)) = LOWER(TRIM(%s))")
                            params.append(program_name)
                    except Exception:
                        pass
                where_clause = " AND ".join(filters)
                cur.execute(
                    f"""
                    SELECT DISTINCT TRIM(role) AS job_role_name
                    FROM interview_questions
                    WHERE {where_clause}
                    ORDER BY job_role_name ASC
                    """,
                    tuple(params),
                )
                return [row[0] for row in cur.fetchall()]

    @classmethod
    def fetch_distinct_question_metadata(cls, column: str) -> List[str]:
        valid_columns = {"interview_type", "work_experience", "question_type"}
        if column not in valid_columns:
            return []
        with db_pool.get_connection() as conn:
            if not conn:
                return []
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT DISTINCT TRIM({column}) AS value
                    FROM interview_questions
                    WHERE {column} IS NOT NULL AND TRIM({column}) <> ''
                    ORDER BY value ASC
                    """
                )
                return [row[0] for row in cur.fetchall()]

    @classmethod
    def get_debug_session_data(cls, session_id: str) -> Dict[str, Any]:
        with db_pool.get_connection() as conn:
            if not conn:
                raise Exception("No database connection")
            with conn.cursor() as cur:
                cls._ensure_session_metadata_columns(cur)
                cls._ensure_interview_data_columns(cur)
                cur.execute("""
                    SELECT session_id, student_name, job_role, company_name, 
                           industry_type, status, started_at
                    FROM session_metadata 
                    WHERE session_id = %s
                """, (session_id,))
                session_data = cur.fetchone()
                if not session_data:
                    raise KeyError("Session not found")

                cur.execute("""
                    SELECT question_number, question, answer, sentiment_score, 
                           difficulty, mandatory_skills, analysis_status
                    FROM interview_data 
                    WHERE session_id = %s
                    ORDER BY question_number
                """, (session_id,))
                questions_data = cur.fetchall()

                return {
                    "session_id": session_id,
                    "session_metadata": {
                        "student_name": session_data[1],
                        "job_role": session_data[2],
                        "company_name": session_data[3],
                        "industry_type": session_data[4],
                        "status": session_data[5],
                        "started_at": format_datetime_ist(session_data[6])
                    },
                    "questions": [
                        {
                            "question_number": q[0],
                            "question": q[1],
                            "answer": q[2],
                            "sentiment_score": float(q[3]) if q[3] else None,
                            "difficulty": q[4],
                            "mandatory_skills": q[5],
                            "analysis_status": q[6]
                        } for q in questions_data
                    ]
                }

    @classmethod
    def get_latest_feedback_for_debug(cls, session_id: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        with db_pool.get_connection() as conn:
            if not conn:
                raise Exception("Database connection failed")
            with conn.cursor() as cur:
                cls._ensure_interview_data_columns(cur)
                cls._ensure_session_metadata_columns(cur)
                cur.execute("""
                    SELECT detailed_feedback
                    FROM interview_data
                    WHERE session_id = %s AND detailed_feedback IS NOT NULL
                    ORDER BY question_number DESC
                    LIMIT 1
                """, (session_id,))
                result = cur.fetchone()
                feedback_text = result[0] if result and result[0] else None

                cur.execute("SELECT * FROM session_metadata WHERE session_id = %s", (session_id,))
                metadata = cur.fetchone()
                meta_dict = dict(zip([desc[0] for desc in cur.description], metadata)) if metadata else None
                return feedback_text, meta_dict

    @classmethod
    def get_latest_answer_for_debug(cls, session_id: str) -> Optional[Tuple[str, str, Optional[str], Optional[str]]]:
        with db_pool.get_connection() as conn:
            if not conn:
                raise Exception("Database connection failed")
            with conn.cursor() as cur:
                cls._ensure_interview_data_columns(cur)
                cur.execute("""
                    SELECT question, answer, job_role, mandatory_skills
                    FROM interview_data
                    WHERE session_id = %s AND answer IS NOT NULL AND TRIM(answer) != ''
                    ORDER BY question_number DESC
                    LIMIT 1
                """, (session_id,))
                return cur.fetchone()

