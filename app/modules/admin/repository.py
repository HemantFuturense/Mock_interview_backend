import logging
import re
from datetime import datetime, date, time
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from fastapi import HTTPException
from psycopg2.errors import UniqueViolation
from psycopg2.extras import Json

from app.core.database import db_pool, ensure_companies_table
from app.modules.ai.client import embed_text
from app.modules.rag.service import ensure_company_playbook_table
from app.utils.datetime_utils import format_datetime_ist

logger = logging.getLogger(__name__)

QUESTION_INSERT_COLUMNS = [
    "industry",
    "company",
    "role",
    "question",
    "mandatory_skills",
    "pre_def_answer",
    "difficulty",
    "question_type",
    "interview_type",
    "work_experience",
]


class AdminRepository:
    @staticmethod
    def get_admin_by_email(email: str) -> Optional[Dict[str, Any]]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT admin_id, email, password_hash, display_name, is_active
                    FROM admin_users
                    WHERE LOWER(email) = LOWER(%s)
                    """,
                    (email.strip(),),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "admin_id": row[0],
                    "email": row[1],
                    "password_hash": row[2],
                    "display_name": row[3],
                    "is_active": row[4],
                }

    @staticmethod
    def get_admin_by_id(admin_id: str | int) -> Optional[Dict[str, Any]]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT admin_id, email, display_name, is_active FROM admin_users WHERE admin_id = %s",
                    (admin_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "admin_id": row[0],
                    "email": row[1],
                    "display_name": row[2],
                    "is_active": row[3],
                }

    @staticmethod
    def update_admin_password_hash(admin_id: str | int, password_hash: str) -> None:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE admin_users SET password_hash = %s WHERE admin_id = %s",
                    (password_hash, admin_id),
                )
                conn.commit()

    @staticmethod
    def resolve_ubp_id_for_admin(university_name: str, program_name: str, batch_label: str) -> Optional[int]:
        u_clean = university_name.strip()
        p_clean = program_name.strip()
        b_clean = batch_label.strip()

        if not u_clean or not p_clean or not b_clean:
            return None

        query = """
            SELECT ubp_id
            FROM university_batch_program
            WHERE LOWER(TRIM(university_name)) = LOWER(%s)
              AND LOWER(TRIM(program_name)) = LOWER(%s)
              AND LOWER(TRIM(batch_label)) = LOWER(%s)
            LIMIT 1
        """
        try:
            with db_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (u_clean, p_clean, b_clean))
                    row = cur.fetchone()
                    if row:
                        return row[0]
        except Exception as exc:
            logger.error(f"Error resolving UBP id in admin repository: {exc}")
        return None

    @staticmethod
    def insert_interview_questions(records: List[Dict[str, Any]]) -> List[int]:
        if not records:
            return []

        values = [[record.get(col) for col in QUESTION_INSERT_COLUMNS] for record in records]
        insert_sql = f"""
            INSERT INTO interview_questions ({', '.join(QUESTION_INSERT_COLUMNS)})
            VALUES %s
            RETURNING id
        """
        try:
            from psycopg2 import extras
            with db_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    extras.execute_values(cur, insert_sql, values)
                    rows = cur.fetchall()
                conn.commit()
            return [row[0] for row in rows]
        except Exception as exc:
            logger.error("Error batch inserting questions: %s", exc)
            raise HTTPException(status_code=500, detail="Failed to save interview questions") from exc

    @staticmethod
    def get_dashboard_stats_repo() -> Dict[str, Any]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM students")
                total_students = cur.fetchone()[0] or 0

                cur.execute("SELECT COUNT(*) FROM session_metadata WHERE status = 'completed'")
                completed_sessions = cur.fetchone()[0] or 0

                total_sessions = completed_sessions
                active_sessions = 0

                cur.execute("SELECT AVG(overall_score) FROM session_metadata WHERE status = 'completed' AND overall_score IS NOT NULL")
                avg_score_result = cur.fetchone()[0]
                avg_score = float(avg_score_result) if avg_score_result else 0.0

                cur.execute("""
                    SELECT COUNT(*) FROM session_metadata 
                    WHERE status = 'completed' AND DATE(completed_at) = CURRENT_DATE
                """)
                today_sessions = cur.fetchone()[0] or 0

                return {
                    "total_students": total_students,
                    "total_sessions": total_sessions,
                    "active_sessions": active_sessions,
                    "completed_sessions": completed_sessions,
                    "avg_score": round(avg_score, 2),
                    "today_sessions": today_sessions,
                }

    @staticmethod
    def get_students_paginated_repo(
        page: int,
        limit: int,
        name: Optional[str],
        email: Optional[str],
        status: Optional[str],
        university_name: Optional[str],
        program_name: Optional[str],
        batch_label: Optional[str],
    ) -> Dict[str, Any]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                base_query = """
                    FROM students s
                    LEFT JOIN session_metadata sm ON s.student_id = sm.student_id AND sm.status = 'completed'
                    LEFT JOIN programs p ON s.program_id = p.id
                    LEFT JOIN LATERAL (
                        SELECT selected.university_name, selected.program_name, selected.batch_label
                        FROM (
                            SELECT u.university_name, u.program_name, u.batch_label, 0 AS priority
                            FROM university_batch_program u
                            WHERE u.ubp_id = s.program_id
                            UNION ALL
                            SELECT u2.university_name, u2.program_name, u2.batch_label, 1 AS priority
                            FROM university_batch_program u2
                            WHERE s.program_id IS NOT NULL
                              AND p.program_name IS NOT NULL
                              AND NOT EXISTS (
                                  SELECT 1
                                  FROM university_batch_program u_exact
                                  WHERE u_exact.ubp_id = s.program_id
                              )
                              AND LOWER(TRIM(u2.program_name)) = LOWER(TRIM(p.program_name))
                        ) selected
                        ORDER BY selected.priority
                        LIMIT 1
                    ) ubp ON TRUE
                """
                where_clauses = []
                params = []

                if name:
                    where_clauses.append("s.name ILIKE %s")
                    params.append(f"%{name}%")
                if email:
                    where_clauses.append("s.email ILIKE %s")
                    params.append(f"%{email}%")
                if university_name:
                    where_clauses.append("COALESCE(ubp.university_name, '') ILIKE %s")
                    params.append(f"%{university_name}%")
                if program_name:
                    where_clauses.append("COALESCE(ubp.program_name, p.program_name, '') ILIKE %s")
                    params.append(f"%{program_name}%")
                if batch_label:
                    where_clauses.append("COALESCE(ubp.batch_label, '') ILIKE %s")
                    params.append(f"%{batch_label}%")

                having_clauses = []
                if status:
                    if status == "Completed":
                        having_clauses.append("COUNT(sm.session_id) > 0")
                    elif status == "No Sessions":
                        having_clauses.append("COUNT(sm.session_id) = 0")

                where_clause = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
                having_clause = " HAVING " + " AND ".join(having_clauses) if having_clauses else ""

                count_query = f"""
                    SELECT COUNT(*) FROM (
                        SELECT s.student_id
                        {base_query}
                        {where_clause}
                        GROUP BY s.student_id
                        {having_clause}
                    ) as filtered_students
                """
                cur.execute(count_query, tuple(params))
                total_count = cur.fetchone()[0]

                offset = (page - 1) * limit
                query = f"""
                    SELECT 
                        s.student_id,
                        s.name,
                        s.email,
                        COUNT(sm.session_id) as total_sessions,
                        AVG(sm.overall_score) as avg_score,
                        MAX(sm.started_at) as last_session,
                        CASE 
                            WHEN COUNT(sm.session_id) = 0 THEN 'No Sessions'
                            ELSE 'Completed'
                        END as status,
                        COALESCE(ubp.program_name, p.program_name) AS resolved_program_name,
                        ubp.university_name,
                        ubp.batch_label
                    {base_query}
                    {where_clause}
                    GROUP BY s.student_id, s.name, s.email, p.program_name, ubp.program_name, ubp.university_name, ubp.batch_label
                    {having_clause}
                    ORDER BY s.student_id DESC
                    LIMIT %s OFFSET %s
                """
                final_params = tuple(params + [limit, offset])
                cur.execute(query, final_params)

                students = []
                for row in cur.fetchall():
                    last_sess = format_datetime_ist(row[5]) if row[5] else None
                    students.append({
                        "student_id": row[0],
                        "name": row[1],
                        "email": row[2],
                        "total_sessions": row[3],
                        "avg_score": float(row[4]) if row[4] else None,
                        "last_session": last_sess,
                        "status": row[6],
                        "program_name": row[7],
                        "university_name": row[8],
                        "batch_label": row[9],
                    })

                return {
                    "students": students,
                    "pagination": {
                        "page": page,
                        "limit": limit,
                        "total": total_count,
                        "pages": (total_count + limit - 1) // limit if limit > 0 else 0,
                    },
                }

    @staticmethod
    def get_sessions_paginated_repo(
        page: int,
        limit: int,
        status: Optional[str],
        student_name: Optional[str],
        student_email: Optional[str],
        job_role: Optional[str],
        company_name: Optional[str],
        min_score: Optional[float],
        max_score: Optional[float],
        university_name: Optional[str],
        program_name: Optional[str],
        batch_label: Optional[str],
    ) -> Dict[str, Any]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                where_conditions = []
                params: List[Any] = []

                where_conditions.append("sm.status = 'completed'")
                if status:
                    where_conditions.append("sm.status = %s")
                    params.append(status)
                if student_name:
                    where_conditions.append("sm.student_name ILIKE %s")
                    params.append(f"%{student_name}%")
                if student_email:
                    where_conditions.append("s.email ILIKE %s")
                    params.append(f"%{student_email}%")
                if job_role:
                    where_conditions.append("id.job_role ILIKE %s")
                    params.append(f"%{job_role}%")
                if company_name:
                    where_conditions.append("id.company_name ILIKE %s")
                    params.append(f"%{company_name}%")
                if min_score is not None:
                    where_conditions.append("sm.overall_score >= %s")
                    params.append(min_score)
                if max_score is not None:
                    where_conditions.append("sm.overall_score <= %s")
                    params.append(max_score)
                if university_name:
                    where_conditions.append("COALESCE(ubp.university_name, '') ILIKE %s")
                    params.append(f"%{university_name}%")
                if program_name:
                    where_conditions.append("COALESCE(ubp.program_name, p.program_name, '') ILIKE %s")
                    params.append(f"%{program_name}%")
                if batch_label:
                    where_conditions.append("COALESCE(ubp.batch_label, '') ILIKE %s")
                    params.append(f"%{batch_label}%")

                where_clause = ""
                if where_conditions:
                    where_clause = "WHERE " + " AND ".join(where_conditions)

                count_query = f"""
                    SELECT COUNT(DISTINCT sm.session_id)
                    FROM session_metadata sm
                    LEFT JOIN interview_data id ON sm.session_id = id.session_id
                    LEFT JOIN students s ON sm.student_id = s.student_id
                    LEFT JOIN programs p ON s.program_id = p.id
                    LEFT JOIN LATERAL (
                        SELECT selected.university_name, selected.program_name, selected.batch_label
                        FROM (
                            SELECT u.university_name, u.program_name, u.batch_label, 0 AS priority
                            FROM university_batch_program u
                            WHERE u.ubp_id = s.program_id
                            UNION ALL
                            SELECT u2.university_name, u2.program_name, u2.batch_label, 1 AS priority
                            FROM university_batch_program u2
                            WHERE s.program_id IS NOT NULL
                              AND p.program_name IS NOT NULL
                              AND NOT EXISTS (
                                  SELECT 1
                                  FROM university_batch_program u_exact
                                  WHERE u_exact.ubp_id = s.program_id
                              )
                              AND LOWER(TRIM(u2.program_name)) = LOWER(TRIM(p.program_name))
                        ) selected
                        ORDER BY selected.priority
                        LIMIT 1
                    ) ubp ON TRUE
                    {where_clause}
                """
                cur.execute(count_query, params)
                total_count = cur.fetchone()[0]

                offset = (page - 1) * limit
                sessions_query = f"""
                    WITH RankedSessions AS (
                        SELECT 
                            sm.session_id,
                            sm.student_id,
                            sm.student_name,
                            id.job_role,
                            id.company_name,
                            sm.status,
                            sm.overall_score,
                            sm.technical_score,
                            sm.communication_score,
                            sm.attitude_score,
                            sm.started_at,
                            sm.completed_at,
                            sm.duration_minutes,
                            COALESCE(sm.interview_type, id.interview_type) as interview_type,
                            COALESCE(sm.work_experience, id.work_experience) as work_experience,
                            ROW_NUMBER() OVER(PARTITION BY sm.session_id ORDER BY id.timestamp DESC) as rn,
                            DENSE_RANK() OVER (
                                PARTITION BY sm.student_id, 
                                    COALESCE(id.job_role, ''), 
                                    COALESCE(id.company_name, ''),
                                    COALESCE(sm.interview_type, id.interview_type, ''),
                                    COALESCE(sm.work_experience, id.work_experience, '')
                                ORDER BY sm.started_at
                            ) as attempt_number
                        FROM session_metadata sm
                        LEFT JOIN interview_data id ON sm.session_id = id.session_id
                        LEFT JOIN students s ON sm.student_id = s.student_id
                        LEFT JOIN programs p ON s.program_id = p.id
                        LEFT JOIN LATERAL (
                            SELECT selected.university_name, selected.program_name, selected.batch_label
                            FROM (
                                SELECT u.university_name, u.program_name, u.batch_label, 0 AS priority
                                FROM university_batch_program u
                                WHERE u.ubp_id = s.program_id
                                UNION ALL
                                SELECT u2.university_name, u2.program_name, u2.batch_label, 1 AS priority
                                FROM university_batch_program u2
                                WHERE s.program_id IS NOT NULL
                                  AND p.program_name IS NOT NULL
                                  AND NOT EXISTS (
                                      SELECT 1
                                      FROM university_batch_program u_exact
                                      WHERE u_exact.ubp_id = s.program_id
                                  )
                                  AND LOWER(TRIM(u2.program_name)) = LOWER(TRIM(p.program_name))
                            ) selected
                            ORDER BY selected.priority
                            LIMIT 1
                        ) ubp ON TRUE
                        {where_clause}
                    )
                    SELECT 
                        rs.session_id,
                        rs.student_id,
                        rs.student_name,
                        COALESCE(rs.job_role, 'Unknown') as job_role,
                        COALESCE(rs.company_name, 'Unknown') as company_name,
                        rs.status,
                        rs.overall_score,
                        rs.technical_score,
                        rs.communication_score,
                        rs.attitude_score,
                        sm.rubric_scores,
                        rs.started_at,
                        rs.completed_at,
                        rs.duration_minutes,
                        rs.interview_type,
                        rs.work_experience,
                        rs.attempt_number
                    FROM RankedSessions rs
                    LEFT JOIN session_metadata sm ON rs.session_id = sm.session_id
                    WHERE rs.rn = 1
                    ORDER BY rs.started_at DESC
                    LIMIT %s OFFSET %s
                """
                cur.execute(sessions_query, params + [limit, offset])
                sessions = []
                for row in cur.fetchall():
                    sessions.append({
                        "session_id": row[0],
                        "student_id": row[1],
                        "student_name": row[2] or "Unknown",
                        "job_role": row[3] or "Unknown",
                        "company_name": row[4] or "Unknown",
                        "status": row[5],
                        "overall_score": float(row[6]) if row[6] is not None else None,
                        "technical_score": float(row[7]) if row[7] is not None else None,
                        "communication_score": float(row[8]) if row[8] is not None else None,
                        "attitude_score": float(row[9]) if row[9] is not None else None,
                        "rubric_scores": row[10],
                        "started_at": format_datetime_ist(row[11]),
                        "completed_at": format_datetime_ist(row[12]),
                        "duration_minutes": row[13],
                        "interview_type": row[14],
                        "work_experience": row[15],
                        "attempt_number": row[16],
                    })

                return {
                    "sessions": sessions,
                    "pagination": {
                        "page": page,
                        "limit": limit,
                        "total": total_count,
                        "pages": (total_count + limit - 1) // limit if limit > 0 else 0,
                    },
                }

    @staticmethod
    def get_session_details_repo(session_id: str) -> Optional[Dict[str, Any]]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT session_id, student_name, job_role, company_name, industry_type,
                           status, overall_score, technical_score, communication_score, 
                           attitude_score, started_at, completed_at, duration_minutes
                    FROM interview_sessions 
                    WHERE session_id = %s
                    """,
                    (session_id,),
                )
                session_row = cur.fetchone()
                if not session_row:
                    return None

                cur.execute(
                    """
                    SELECT question_number, question, answer, sentiment_score,
                           technical_score, communication_score, difficulty,
                           acknowledgment, detailed_feedback, timestamp
                    FROM interview_data 
                    WHERE session_id = %s
                    ORDER BY question_number
                    """,
                    (session_id,),
                )
                qa_data = []
                for row in cur.fetchall():
                    qa_data.append({
                        "question_number": row[0],
                        "question": row[1],
                        "answer": row[2],
                        "sentiment_score": float(row[3]) if row[3] else None,
                        "technical_score": float(row[4]) if row[4] else None,
                        "communication_score": float(row[5]) if row[5] else None,
                        "difficulty": row[6],
                        "acknowledgment": row[7],
                        "detailed_feedback": row[8],
                        "timestamp": row[9].isoformat() if row[9] else None,
                    })

                return {
                    "session": {
                        "session_id": session_row[0],
                        "student_name": session_row[1],
                        "job_role": session_row[2],
                        "company_name": session_row[3],
                        "industry_type": session_row[4],
                        "status": session_row[5],
                        "overall_score": float(session_row[6]) if session_row[6] else None,
                        "technical_score": float(session_row[7]) if session_row[7] else None,
                        "communication_score": float(session_row[8]) if session_row[8] else None,
                        "attitude_score": float(session_row[9]) if session_row[9] else None,
                        "started_at": session_row[10].isoformat() if session_row[10] else None,
                        "completed_at": session_row[11].isoformat() if session_row[11] else None,
                        "duration_minutes": session_row[12],
                    },
                    "questions_and_answers": qa_data,
                }

    @staticmethod
    def get_job_roles_repo() -> List[str]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT role FROM interview_questions WHERE role IS NOT NULL ORDER BY role")
                return [row[0] for row in cur.fetchall()]

    @staticmethod
    def get_industry_types_repo() -> List[str]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT industry FROM interview_questions WHERE industry IS NOT NULL ORDER BY industry")
                return [row[0] for row in cur.fetchall()]

    @staticmethod
    def get_interview_types_repo() -> List[str]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT interview_type
                    FROM interview_questions
                    WHERE interview_type IS NOT NULL AND TRIM(interview_type) <> ''
                    ORDER BY interview_type
                    """
                )
                return [row[0] for row in cur.fetchall()]

    @staticmethod
    def get_work_experience_levels_repo() -> List[str]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT work_experience
                    FROM interview_questions
                    WHERE work_experience IS NOT NULL AND TRIM(work_experience) <> ''
                    ORDER BY work_experience
                    """
                )
                return [row[0] for row in cur.fetchall()]

    @staticmethod
    def map_program_roles_repo(program: str, work_exp: str, roles: List[str]) -> Tuple[int, int]:
        inserted = 0
        skipped = 0
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                for role in roles:
                    cur.execute(
                        """
                        INSERT INTO programs (program_name, job_role, work_experience)
                        SELECT %s, %s, %s
                        WHERE NOT EXISTS (
                            SELECT 1
                            FROM programs
                            WHERE LOWER(TRIM(program_name)) = LOWER(TRIM(%s))
                              AND LOWER(TRIM(job_role)) = LOWER(TRIM(%s))
                              AND LOWER(TRIM(work_experience)) = LOWER(TRIM(%s))
                        )
                        """,
                        (program, role, work_exp, program, role, work_exp),
                    )
                    if cur.rowcount > 0:
                        inserted += 1
                    else:
                        skipped += 1
            conn.commit()
        return inserted, skipped

    @staticmethod
    def get_question_types_repo() -> List[str]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT question_type
                    FROM interview_questions
                    WHERE question_type IS NOT NULL AND TRIM(question_type) <> ''
                    ORDER BY question_type
                    """
                )
                return [row[0] for row in cur.fetchall()]

    @staticmethod
    def get_sessions_by_email_repo(email: str, exclude_resume: bool) -> List[Dict[str, Any]]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT student_id FROM students WHERE email = %s", (email,))
                student_row = cur.fetchone()
                if not student_row:
                    return []
                student_id = student_row[0]

                query = """
                    SELECT 
                        sm.session_id, sm.student_name, sm.status, sm.overall_score,
                        sm.technical_score, sm.communication_score, sm.attitude_score,
                        sm.rubric_scores, sm.started_at, sm.completed_at, sm.duration_minutes,
                        (SELECT id.job_role FROM interview_data id WHERE id.session_id = sm.session_id ORDER BY id.timestamp DESC LIMIT 1) as job_role,
                        (SELECT id.company_name FROM interview_data id WHERE id.session_id = sm.session_id ORDER BY id.timestamp DESC LIMIT 1) as company_name,
                        COALESCE(sm.interview_type, (SELECT id.interview_type FROM interview_data id WHERE id.session_id = sm.session_id ORDER BY id.timestamp DESC LIMIT 1)) as interview_type,
                        COALESCE(sm.work_experience, (SELECT id.work_experience FROM interview_data id WHERE id.session_id = sm.session_id ORDER BY id.timestamp DESC LIMIT 1)) as work_experience
                    FROM session_metadata sm
                    WHERE sm.student_id = %s
                """
                params: List[Any] = [student_id]
                if exclude_resume:
                    query += "\n                    AND COALESCE(sm.question_generation_type, 'standard') != 'resume'"
                query += "\n                    ORDER BY sm.started_at DESC"

                cur.execute(query, tuple(params))
                sessions = []
                for row in cur.fetchall():
                    sessions.append({
                        "session_id": row[0],
                        "student_name": row[1] or "Unknown",
                        "status": row[2],
                        "overall_score": float(row[3]) if row[3] is not None else None,
                        "technical_score": float(row[4]) if row[4] is not None else None,
                        "communication_score": float(row[5]) if row[5] is not None else None,
                        "attitude_score": float(row[6]) if row[6] is not None else None,
                        "rubric_scores": row[7],
                        "started_at": format_datetime_ist(row[8]),
                        "completed_at": format_datetime_ist(row[9]),
                        "duration_minutes": row[10],
                        "job_role": row[11] or "N/A",
                        "company_name": row[12] or "N/A",
                        "interview_type": row[13] or "N/A",
                        "work_experience": row[14] or "N/A",
                    })
                return sessions

    @staticmethod
    def get_company_names_repo() -> List[str]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT company FROM interview_questions WHERE company IS NOT NULL ORDER BY company")
                return [row[0] for row in cur.fetchall()]

    @staticmethod
    def get_bulk_upload_options_repo() -> Dict[str, List[str]]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT industry FROM interview_questions
                    WHERE industry IS NOT NULL AND TRIM(industry) <> ''
                    ORDER BY industry
                    """
                )
                industries = [row[0] for row in cur.fetchall() if row and row[0]]

                cur.execute(
                    """
                    SELECT DISTINCT company FROM interview_questions
                    WHERE company IS NOT NULL AND TRIM(company) <> ''
                    ORDER BY company
                    """
                )
                companies = [row[0] for row in cur.fetchall() if row and row[0]]

                cur.execute(
                    """
                    SELECT DISTINCT job_role
                    FROM programs
                    WHERE job_role IS NOT NULL AND TRIM(job_role) <> ''
                    ORDER BY job_role
                    """
                )
                roles = [row[0] for row in cur.fetchall() if row and row[0]]

                cur.execute(
                    """
                    SELECT DISTINCT interview_type FROM interview_questions
                    WHERE interview_type IS NOT NULL AND TRIM(interview_type) <> ''
                    ORDER BY interview_type
                    """
                )
                interview_types = [row[0] for row in cur.fetchall() if row and row[0]]

                cur.execute(
                    """
                    SELECT DISTINCT work_experience FROM interview_questions
                    WHERE work_experience IS NOT NULL AND TRIM(work_experience) <> ''
                    ORDER BY work_experience
                    """
                )
                work_experiences = [row[0] for row in cur.fetchall() if row and row[0]]

                cur.execute(
                    """
                    SELECT DISTINCT question_type FROM interview_questions
                    WHERE question_type IS NOT NULL AND TRIM(question_type) <> ''
                    ORDER BY question_type
                    """
                )
                question_types = [row[0] for row in cur.fetchall() if row and row[0]]

                return {
                    "industries": industries,
                    "companies": companies,
                    "roles": roles,
                    "interview_types": interview_types,
                    "work_experiences": work_experiences,
                    "question_types": question_types,
                }

    @staticmethod
    def get_admin_companies_by_industry_repo(industry: Optional[str]) -> List[str]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                filters = ["company IS NOT NULL", "TRIM(company) <> ''"]
                params: List[Any] = []

                if industry is None or str(industry).strip() == "":
                    filters.append("(industry IS NULL OR TRIM(industry) = '')")
                else:
                    filters.append("LOWER(TRIM(industry)) = LOWER(TRIM(%s))")
                    params.append(industry)

                where_clause = " AND ".join(filters)
                query = f"""
                    SELECT DISTINCT company
                    FROM interview_questions
                    WHERE {where_clause}
                    ORDER BY company
                """
                cur.execute(query, tuple(params))
                return [row[0] for row in cur.fetchall() if row and row[0]]

    @staticmethod
    def get_admin_interview_types_for_selection_repo(industry: Optional[str], company: str) -> List[str]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                params: List[Any] = [company]
                base_filters = [
                    "interview_type IS NOT NULL",
                    "TRIM(interview_type) <> ''",
                    "LOWER(TRIM(company)) = LOWER(TRIM(%s))",
                ]
                if industry is None or str(industry).strip() == "":
                    base_filters.append("(industry IS NULL OR TRIM(industry) = '')")
                else:
                    base_filters.append("LOWER(TRIM(industry)) = LOWER(TRIM(%s))")
                    params.append(industry)

                where_clause = " AND ".join(base_filters)
                query = f"""
                    SELECT DISTINCT interview_type
                    FROM interview_questions
                    WHERE {where_clause}
                    ORDER BY interview_type
                """
                cur.execute(query, tuple(params))
                return [row[0] for row in cur.fetchall() if row and row[0]]

    @staticmethod
    def get_admin_work_experience_for_selection_repo(industry: Optional[str], company: str, interview_type: str) -> List[str]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                params: List[Any] = [company, interview_type]
                base_filters = [
                    "work_experience IS NOT NULL",
                    "TRIM(work_experience) <> ''",
                    "LOWER(TRIM(company)) = LOWER(TRIM(%s))",
                    "LOWER(TRIM(interview_type)) = LOWER(TRIM(%s))",
                ]
                if industry is None or str(industry).strip() == "":
                    base_filters.append("(industry IS NULL OR TRIM(industry) = '')")
                else:
                    base_filters.append("LOWER(TRIM(industry)) = LOWER(TRIM(%s))")
                    params.append(industry)

                where_clause = " AND ".join(base_filters)
                query = f"""
                    SELECT DISTINCT work_experience
                    FROM interview_questions
                    WHERE {where_clause}
                    ORDER BY work_experience
                """
                cur.execute(query, tuple(params))
                return [row[0] for row in cur.fetchall() if row and row[0]]

    @staticmethod
    def get_admin_job_roles_for_selection_repo(
        industry: Optional[str], company: str, interview_type: str, work_experience: str
    ) -> List[str]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                params: List[Any] = [company, interview_type, work_experience]
                base_filters = [
                    "role IS NOT NULL",
                    "TRIM(role) <> ''",
                    "LOWER(TRIM(company)) = LOWER(TRIM(%s))",
                    "LOWER(TRIM(interview_type)) = LOWER(TRIM(%s))",
                    "LOWER(TRIM(work_experience)) = LOWER(TRIM(%s))",
                ]
                if industry is None or str(industry).strip() == "":
                    base_filters.append("(industry IS NULL OR TRIM(industry) = '')")
                else:
                    base_filters.append("LOWER(TRIM(industry)) = LOWER(TRIM(%s))")
                    params.append(industry)

                where_clause = " AND ".join(base_filters)
                query = f"""
                    SELECT DISTINCT role
                    FROM interview_questions
                    WHERE {where_clause}
                    ORDER BY role
                """
                cur.execute(query, tuple(params))
                return [row[0] for row in cur.fetchall() if row and row[0]]

    @staticmethod
    def get_performance_analytics_repo(days: int) -> Dict[str, Any]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        DATE(started_at) as date,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as sessions,
                        AVG(overall_score) as avg_score,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed
                    FROM session_metadata 
                    WHERE started_at >= CURRENT_DATE - INTERVAL %s AND status = 'completed'
                    GROUP BY DATE(started_at)
                    ORDER BY date
                    """,
                    (f"{days} days",),
                )
                daily_trends = []
                for row in cur.fetchall():
                    date_obj = row[0]
                    if isinstance(date_obj, date) and not isinstance(date_obj, datetime):
                        date_obj = datetime.combine(date_obj, time.min)
                    daily_trends.append({
                        "date": format_datetime_ist(date_obj),
                        "sessions": row[1],
                        "avg_score": float(row[2]) if row[2] else 0,
                        "completed": row[3],
                    })

                cur.execute(
                    """
                    SELECT 
                        CASE 
                            WHEN overall_score >= 4.5 THEN 'Excellent (4.5-5.0)'
                            WHEN overall_score >= 3.5 THEN 'Good (3.5-4.4)'
                            WHEN overall_score >= 2.5 THEN 'Average (2.5-3.4)'
                            WHEN overall_score >= 1.5 THEN 'Below Average (1.5-2.4)'
                            ELSE 'Poor (0-1.4)'
                        END as score_range,
                        COUNT(*) as count
                    FROM session_metadata 
                    WHERE status = 'completed' AND overall_score IS NOT NULL
                    GROUP BY score_range
                    ORDER BY MIN(overall_score) DESC
                    """
                )
                score_distribution = [{"range": row[0], "count": row[1]} for row in cur.fetchall()]

                return {
                    "daily_trends": daily_trends,
                    "score_distribution": score_distribution,
                    "summary": {
                        "total_days": days,
                        "total_sessions": sum(d["sessions"] for d in daily_trends),
                        "avg_daily_sessions": sum(d["sessions"] for d in daily_trends) / len(daily_trends) if daily_trends else 0,
                    },
                }

    @staticmethod
    def get_admin_insights_repo() -> Dict[str, Any]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM students")
                total_students_row = cur.fetchone()
                total_students = total_students_row[0] if total_students_row and total_students_row[0] else 0

                cur.execute(
                    """
                    SELECT COUNT(DISTINCT student_id)
                    FROM session_metadata
                    WHERE started_at >= CURRENT_DATE - INTERVAL '30 days'
                    """
                )
                active_students_row = cur.fetchone()
                active_students = active_students_row[0] if active_students_row and active_students_row[0] else 0

                cur.execute(
                    """
                    SELECT AVG(session_count)
                    FROM (
                        SELECT COUNT(*) AS session_count
                        FROM session_metadata
                        WHERE started_at >= CURRENT_DATE - INTERVAL '30 days'
                        AND status = 'completed'
                        GROUP BY student_id
                    ) sub
                    """
                )
                avg_sessions_active_row = cur.fetchone()
                avg_sessions_per_active = (
                    float(avg_sessions_active_row[0]) if avg_sessions_active_row and avg_sessions_active_row[0] else 0.0
                )

                cur.execute("SELECT COUNT(*) FROM session_metadata WHERE status = 'completed'")
                completed_sessions_row = cur.fetchone()
                total_completed_sessions = (
                    completed_sessions_row[0] if completed_sessions_row and completed_sessions_row[0] else 0
                )

                cur.execute(
                    """
                    SELECT
                        COALESCE(ubp.program_name, p.program_name, 'Unassigned') AS program_name,
                        COUNT(DISTINCT s.student_id) AS student_count,
                        COUNT(CASE WHEN sm.status = 'completed' THEN 1 END) AS completed_sessions,
                        AVG(CASE WHEN sm.status = 'completed' THEN sm.overall_score END) AS avg_overall
                    FROM students s
                    LEFT JOIN session_metadata sm ON sm.student_id = s.student_id
                    LEFT JOIN programs p ON s.program_id = p.id
                    LEFT JOIN university_batch_program ubp ON s.program_id = ubp.ubp_id
                    GROUP BY COALESCE(ubp.program_name, p.program_name, 'Unassigned')
                    ORDER BY student_count DESC
                    """
                )
                program_performance = []
                for row in cur.fetchall():
                    program_performance.append({
                        "program_name": row[0],
                        "student_count": row[1] or 0,
                        "completed_sessions": row[2] or 0,
                        "avg_overall": round(float(row[3]), 2) if row[3] is not None else 0.0,
                    })

                cur.execute(
                    """
                    SELECT
                        COALESCE(NULLIF(TRIM(sm.work_experience), ''), 'Not Provided') AS work_experience,
                        COUNT(*) AS sessions,
                        AVG(sm.overall_score) AS avg_overall
                    FROM session_metadata sm
                    WHERE sm.status = 'completed'
                    GROUP BY work_experience
                    ORDER BY sessions DESC
                    """
                )
                experience_breakdown = []
                for row in cur.fetchall():
                    experience_breakdown.append({
                        "work_experience": row[0],
                        "sessions": row[1] or 0,
                        "avg_overall": round(float(row[2]), 2) if row[2] is not None else 0.0,
                    })

                cur.execute(
                    """
                    SELECT
                        COALESCE(NULLIF(TRIM(id.industry_type), ''), 'Unknown') AS industry,
                        COALESCE(NULLIF(TRIM(id.company_name), ''), 'Unknown') AS company,
                        COUNT(DISTINCT sm.session_id) AS total_sessions,
                        AVG(sm.overall_score) AS avg_overall
                    FROM session_metadata sm
                    JOIN interview_data id ON sm.session_id = id.session_id
                    WHERE sm.status = 'completed'
                    GROUP BY industry, company
                    ORDER BY total_sessions DESC
                    LIMIT 8
                    """
                )
                industry_company_hotspots = []
                for row in cur.fetchall():
                    industry_company_hotspots.append({
                        "industry": row[0],
                        "company": row[1],
                        "total_sessions": row[2] or 0,
                        "avg_overall": round(float(row[3]), 2) if row[3] is not None else 0.0,
                    })

                cur.execute(
                    """
                    SELECT
                        COALESCE(NULLIF(TRIM(id.job_role), ''), 'Unknown') AS job_role,
                        COUNT(DISTINCT sm.session_id) AS total_sessions,
                        AVG(sm.overall_score) AS avg_overall
                    FROM session_metadata sm
                    JOIN interview_data id ON sm.session_id = id.session_id
                    WHERE sm.status = 'completed'
                    GROUP BY job_role
                    ORDER BY total_sessions DESC
                    LIMIT 8
                    """
                )
                trending_roles = []
                for row in cur.fetchall():
                    trending_roles.append({
                        "job_role": row[0],
                        "total_sessions": row[1] or 0,
                        "avg_overall": round(float(row[2]), 2) if row[2] is not None else 0.0,
                    })

                cur.execute(
                    """
                    WITH combo_attempts AS (
                        SELECT
                            sm.student_id,
                            COALESCE(NULLIF(TRIM(id.job_role), ''), 'Unknown') AS job_role,
                            COALESCE(NULLIF(TRIM(id.company_name), ''), 'Unknown') AS company,
                            COUNT(*) AS attempts
                        FROM session_metadata sm
                        JOIN interview_data id ON sm.session_id = id.session_id
                        WHERE sm.status = 'completed'
                        GROUP BY sm.student_id, id.job_role, id.company_name
                    )
                    SELECT
                        COUNT(*) FILTER (WHERE attempts > 1) AS repeat_combos,
                        COALESCE(SUM(attempts) FILTER (WHERE attempts > 1), 0) AS total_attempts
                    FROM combo_attempts
                    """
                )
                reattempt_summary_row = cur.fetchone() or (0, 0)
                repeat_combos = reattempt_summary_row[0] or 0
                repeat_attempts = reattempt_summary_row[1] or 0

                cur.execute(
                    """
                    WITH combo_attempts AS (
                        SELECT
                            sm.student_id,
                            COALESCE(NULLIF(TRIM(id.job_role), ''), 'Unknown') AS job_role,
                            COALESCE(NULLIF(TRIM(id.company_name), ''), 'Unknown') AS company,
                            COUNT(*) AS attempts
                        FROM session_metadata sm
                        JOIN interview_data id ON sm.session_id = id.session_id
                        WHERE sm.status = 'completed'
                        GROUP BY sm.student_id, id.job_role, id.company_name
                    ),
                    repeated AS (
                        SELECT
                            job_role,
                            company,
                            COUNT(*) AS repeat_students
                        FROM combo_attempts
                        WHERE attempts > 1
                        GROUP BY job_role, company
                    )
                    SELECT job_role, company, repeat_students
                    FROM repeated
                    ORDER BY repeat_students DESC
                    LIMIT 6
                    """
                )
                reattempt_hotspots = []
                for row in cur.fetchall():
                    reattempt_hotspots.append({
                        "job_role": row[0],
                        "company": row[1],
                        "repeat_students": row[2] or 0,
                    })

                engagement_summary = {
                    "total_students": total_students,
                    "active_students_30_days": active_students,
                    "inactive_students_30_days": max(total_students - active_students, 0),
                    "avg_sessions_per_active": round(avg_sessions_per_active, 2),
                    "total_completed_sessions": total_completed_sessions,
                    "repeat_combos": repeat_combos,
                    "repeat_attempts": repeat_attempts,
                }

                return {
                    "engagement_summary": engagement_summary,
                    "program_performance": program_performance,
                    "experience_breakdown": experience_breakdown,
                    "industry_company_hotspots": industry_company_hotspots,
                    "trending_roles": trending_roles,
                    "reattempt_hotspots": reattempt_hotspots,
                }

    @staticmethod
    def get_ubp_performance_repo() -> Dict[str, Any]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        COALESCE(ubp.university_name, 'Unknown') AS university_name,
                        COALESCE(ubp.program_name, p.program_name, 'Unassigned') AS program_name,
                        COALESCE(ubp.batch_label, 'Unknown') AS batch_label,
                        COUNT(DISTINCT s.student_id) AS student_count,
                        COUNT(CASE WHEN sm.status = 'completed' THEN 1 END) AS completed_sessions,
                        AVG(CASE WHEN sm.status = 'completed' THEN sm.overall_score END) AS avg_overall
                    FROM students s
                    LEFT JOIN session_metadata sm ON sm.student_id = s.student_id
                    LEFT JOIN programs p ON s.program_id = p.id
                    LEFT JOIN university_batch_program ubp ON s.program_id = ubp.ubp_id
                    GROUP BY
                        COALESCE(ubp.university_name, 'Unknown'),
                        COALESCE(ubp.program_name, p.program_name, 'Unassigned'),
                        COALESCE(ubp.batch_label, 'Unknown')
                    ORDER BY university_name, program_name, batch_label
                    """
                )
                cohorts = []
                for row in cur.fetchall():
                    cohorts.append({
                        "university_name": row[0],
                        "program_name": row[1],
                        "batch_label": row[2],
                        "student_count": row[3] or 0,
                        "completed_sessions": row[4] or 0,
                        "avg_overall": round(float(row[5]), 2) if row[5] is not None else 0.0,
                    })
                return {"cohorts": cohorts}

    @staticmethod
    def get_retention_analytics_repo() -> Dict[str, Any]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM students")
                total_row = cur.fetchone()
                total_students = total_row[0] if total_row and total_row[0] else 0

                cur.execute(
                    """
                    SELECT
                        COUNT(DISTINCT CASE WHEN status = 'completed' AND started_at >= CURRENT_DATE - INTERVAL '7 days' THEN student_id END) AS active_7d,
                        COUNT(DISTINCT CASE WHEN status = 'completed' AND started_at >= CURRENT_DATE - INTERVAL '30 days' THEN student_id END) AS active_30d
                    FROM session_metadata
                    """
                )
                act_row = cur.fetchone() or (0, 0)
                active_7d = act_row[0] or 0
                active_30d = act_row[1] or 0

                cur.execute(
                    """
                    SELECT AVG(session_count)
                    FROM (
                        SELECT COUNT(*) AS session_count
                        FROM session_metadata
                        WHERE status = 'completed'
                        GROUP BY student_id
                    ) sub
                    """
                )
                avg_sessions_row = cur.fetchone()
                avg_sessions_per_student = (
                    float(avg_sessions_row[0]) if avg_sessions_row and avg_sessions_row[0] is not None else 0.0
                )

                cur.execute(
                    """
                    WITH ordered_sessions AS (
                        SELECT
                            student_id,
                            started_at,
                            LAG(started_at) OVER (PARTITION BY student_id ORDER BY started_at) AS prev_started_at
                        FROM session_metadata
                        WHERE status = 'completed'
                    ),
                    diffs AS (
                        SELECT EXTRACT(EPOCH FROM (started_at - prev_started_at)) / 86400.0 AS days_diff
                        FROM ordered_sessions
                        WHERE prev_started_at IS NOT NULL
                    )
                    SELECT
                        AVG(days_diff) AS avg_days_between_sessions,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY days_diff) AS median_days_between_sessions
                    FROM diffs
                    """
                )
                tb_row = cur.fetchone() or (None, None)
                avg_days_between = float(tb_row[0]) if tb_row[0] is not None else None
                median_days_between = float(tb_row[1]) if tb_row[1] is not None else None

                return {
                    "total_students": total_students,
                    "active_7d": active_7d,
                    "active_30d": active_30d,
                    "retention_7d": round(active_7d / total_students, 4) if total_students else 0.0,
                    "retention_30d": round(active_30d / total_students, 4) if total_students else 0.0,
                    "avg_sessions_per_student": round(avg_sessions_per_student, 2),
                    "avg_days_between_sessions": round(avg_days_between, 2) if avg_days_between is not None else None,
                    "median_days_between_sessions": round(median_days_between, 2) if median_days_between is not None else None,
                }

    @staticmethod
    def get_student_analytics_repo(student_id: int) -> Optional[Dict[str, Any]]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT s.name, s.email, s.program_id, ubp.program_name, ubp.batch_label
                    FROM students s
                    LEFT JOIN university_batch_program ubp ON s.program_id = ubp.ubp_id
                    WHERE s.student_id = %s
                    """,
                    (student_id,),
                )
                student_info = cur.fetchone()
                if not student_info:
                    return None

                cur.execute(
                    """
                    SELECT 
                        sm.session_id,
                        sm.student_name,
                        sm.status,
                        sm.overall_score,
                        sm.technical_score,
                        sm.communication_score,
                        sm.attitude_score,
                        sm.rubric_scores,
                        sm.started_at,
                        sm.completed_at,
                        sm.duration_minutes,
                        id.job_role,
                        id.company_name,
                        id.industry_type,
                        COALESCE(sm.interview_type, id.interview_type) AS interview_type,
                        COALESCE(sm.work_experience, id.work_experience) AS work_experience
                    FROM session_metadata sm
                    LEFT JOIN interview_data id ON sm.session_id = id.session_id
                    WHERE sm.student_id = %s
                    GROUP BY 
                        sm.session_id, sm.student_name, sm.status, sm.overall_score,
                        sm.technical_score, sm.communication_score, sm.attitude_score,
                        sm.rubric_scores, sm.started_at, sm.completed_at, sm.duration_minutes,
                        id.job_role, id.company_name, id.industry_type,
                        COALESCE(sm.interview_type, id.interview_type),
                        COALESCE(sm.work_experience, id.work_experience)
                    ORDER BY sm.started_at DESC
                    """,
                    (student_id,),
                )
                sessions = []
                for row in cur.fetchall():
                    sessions.append({
                        "session_id": row[0],
                        "student_name": row[1],
                        "status": row[2],
                        "overall_score": float(row[3]) if row[3] is not None else None,
                        "technical_score": float(row[4]) if row[4] is not None else None,
                        "communication_score": float(row[5]) if row[5] is not None else None,
                        "attitude_score": float(row[6]) if row[6] is not None else None,
                        "rubric_scores": row[7],
                        "started_at": format_datetime_ist(row[8]),
                        "completed_at": format_datetime_ist(row[9]),
                        "duration_minutes": row[10],
                        "job_role": row[11],
                        "company_name": row[12],
                        "industry_type": row[13],
                        "interview_type": row[14],
                        "work_experience": row[15],
                    })

                completed_sessions = [s for s in sessions if s["status"] == "completed"]
                overall_values = [s["overall_score"] for s in completed_sessions if s["overall_score"] is not None]
                technical_values = [s["technical_score"] for s in completed_sessions if s["technical_score"] is not None]
                communication_values = [s["communication_score"] for s in completed_sessions if s["communication_score"] is not None]
                attitude_values = [s["attitude_score"] for s in completed_sessions if s["attitude_score"] is not None]

                avg_scores = {
                    "overall": sum(overall_values) / len(overall_values) if overall_values else 0,
                    "technical": sum(technical_values) / len(technical_values) if technical_values else 0,
                    "communication": sum(communication_values) / len(communication_values) if communication_values else 0,
                    "attitude": sum(attitude_values) / len(attitude_values) if attitude_values else 0,
                }

                role_performance: Dict[str, List[float]] = {}
                company_performance: Dict[str, List[float]] = {}

                for session in completed_sessions:
                    if session["job_role"] and session["overall_score"] is not None:
                        role_performance.setdefault(session["job_role"], []).append(session["overall_score"])
                    if session["company_name"] and session["overall_score"] is not None:
                        company_performance.setdefault(session["company_name"], []).append(session["overall_score"])

                role_averages = {role: sum(scores) / len(scores) for role, scores in role_performance.items()}
                company_averages = {company: sum(scores) / len(scores) for company, scores in company_performance.items()}

                return {
                    "student_info": {
                        "student_id": student_id,
                        "name": student_info[0],
                        "email": student_info[1],
                        "program_id": student_info[2],
                        "program_name": student_info[3],
                        "batch_label": student_info[4],
                    },
                    "sessions": sessions,
                    "statistics": {
                        "total_sessions": len(sessions),
                        "completed_sessions": len(completed_sessions),
                        "completion_rate": len(completed_sessions) / len(sessions) * 100 if sessions else 0,
                        "average_scores": avg_scores,
                        "performance_by_role": role_averages,
                        "performance_by_company": company_averages,
                    },
                }

    @staticmethod
    def get_leaderboard_repo(
        role: Optional[str],
        company: Optional[str],
        program: Optional[str],
        university: Optional[str],
        batch: Optional[str],
    ) -> List[Dict[str, Any]]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                base_query = """
                    WITH StudentScores AS (
                        SELECT 
                            s.student_id,
                            s.name,
                            AVG(sm.overall_score) as avg_score,
                            COUNT(DISTINCT sm.session_id) as total_sessions
                        FROM students s
                        JOIN session_metadata sm ON s.student_id = sm.student_id
                        LEFT JOIN programs p ON s.program_id = p.id
                        LEFT JOIN LATERAL (
                            SELECT selected.university_name, selected.program_name, selected.batch_label
                            FROM (
                                SELECT u.university_name, u.program_name, u.batch_label, 0 AS priority
                                FROM university_batch_program u
                                WHERE u.ubp_id = s.program_id
                                UNION ALL
                                SELECT u2.university_name, u2.program_name, u2.batch_label, 1 AS priority
                                FROM university_batch_program u2
                                WHERE s.program_id IS NOT NULL
                                  AND p.program_name IS NOT NULL
                                  AND NOT EXISTS (
                                      SELECT 1
                                      FROM university_batch_program u_exact
                                      WHERE u_exact.ubp_id = s.program_id
                                  )
                                  AND LOWER(TRIM(u2.program_name)) = LOWER(TRIM(p.program_name))
                            ) selected
                            ORDER BY selected.priority
                            LIMIT 1
                        ) ubp ON TRUE
                        {join_clause}
                        {where_clause}
                        GROUP BY s.student_id, s.name
                    )
                    SELECT 
                        student_id,
                        name,
                        avg_score,
                        total_sessions
                    FROM StudentScores
                    ORDER BY avg_score DESC
                """

                join_clause = ""
                where_clauses: List[str] = ["sm.overall_score IS NOT NULL"]
                params: List[Any] = []

                if role or company:
                    join_clause = "JOIN interview_data id ON sm.session_id = id.session_id"
                if role:
                    where_clauses.append("id.job_role = %s")
                    params.append(role)
                if company:
                    where_clauses.append("id.company_name = %s")
                    params.append(company)
                if program:
                    where_clauses.append("COALESCE(ubp.program_name, p.program_name) = %s")
                    params.append(program)
                if university:
                    where_clauses.append("ubp.university_name = %s")
                    params.append(university)
                if batch:
                    where_clauses.append("ubp.batch_label = %s")
                    params.append(batch)

                where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
                final_query = base_query.format(join_clause=join_clause, where_clause=where_clause)

                cur.execute(final_query, tuple(params))
                leaderboard = []
                for i, row in enumerate(cur.fetchall()):
                    leaderboard.append({
                        "rank": i + 1,
                        "student_id": row[0],
                        "student_name": row[1],
                        "avg_score": round(float(row[2]), 2),
                        "total_sessions": row[3],
                    })
                return leaderboard

    @staticmethod
    def get_detailed_session_analytics_repo(session_id: str) -> Optional[Dict[str, Any]]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT sm.session_id, sm.student_name, sm.status, sm.overall_score,
                           sm.technical_score, sm.communication_score, sm.attitude_score,
                           sm.started_at, sm.completed_at, sm.duration_minutes, sm.student_id
                    FROM session_metadata sm
                    WHERE sm.session_id = %s
                    """,
                    (session_id,),
                )
                session_data = cur.fetchone()
                if not session_data:
                    return None

                cur.execute(
                    """
                    SELECT detailed_feedback 
                    FROM interview_data 
                    WHERE session_id = %s 
                    ORDER BY question_number DESC 
                    LIMIT 1
                    """,
                    (session_id,),
                )
                overall_feedback_row = cur.fetchone()
                overall_feedback = overall_feedback_row[0] if overall_feedback_row else None

                cur.execute(
                    """
                    SELECT question_number, question, answer, sentiment_score, difficulty,
                           acknowledgment, detailed_feedback, timestamp, mandatory_skills,
                           job_role, company_name, industry_type
                    FROM interview_data
                    WHERE session_id = %s
                    ORDER BY question_number
                    """,
                    (session_id,),
                )
                questions = []
                difficulty_progression = []
                sentiment_progression = []

                for row in cur.fetchall():
                    question_data = {
                        "question_number": row[0],
                        "question": row[1],
                        "answer": row[2],
                        "sentiment_score": float(row[3]) if row[3] else None,
                        "difficulty": row[4],
                        "acknowledgment": row[5],
                        "detailed_feedback": row[6],
                        "timestamp": row[7].isoformat() if row[7] else None,
                        "mandatory_skills": row[8],
                        "job_role": row[9],
                        "company_name": row[10],
                        "industry_type": row[11],
                    }

                    feedback_text = question_data.get("detailed_feedback") or ""
                    score_match = re.search(r"Technical Score: (\d(\.\d)?)", feedback_text)
                    question_data["technical_score"] = float(score_match.group(1)) if score_match else None

                    questions.append(question_data)
                    if row[4]:
                        difficulty_progression.append({"question_number": row[0], "difficulty": row[4]})
                    if row[3]:
                        sentiment_progression.append({"question_number": row[0], "sentiment_score": float(row[3])})

                job_role = questions[0]["job_role"] if questions else None
                company_name = questions[0]["company_name"] if questions else None

                role_avg = None
                company_avg = None

                if job_role:
                    cur.execute(
                        """
                        SELECT AVG(sm.overall_score)
                        FROM session_metadata sm
                        JOIN interview_data id ON sm.session_id = id.session_id
                        WHERE id.job_role = %s AND sm.overall_score IS NOT NULL
                        """,
                        (job_role,),
                    )
                    role_avg_result = cur.fetchone()
                    role_avg = float(role_avg_result[0]) if role_avg_result and role_avg_result[0] else None

                if company_name:
                    cur.execute(
                        """
                        SELECT AVG(sm.overall_score)
                        FROM session_metadata sm
                        JOIN interview_data id ON sm.session_id = id.session_id
                        WHERE id.company_name = %s AND sm.overall_score IS NOT NULL
                        """,
                        (company_name,),
                    )
                    company_avg_result = cur.fetchone()
                    company_avg = float(company_avg_result[0]) if company_avg_result and company_avg_result[0] else None

                return {
                    "session": {
                        "session_id": session_data[0],
                        "student_name": session_data[1],
                        "status": session_data[2],
                        "overall_score": float(session_data[3]) if session_data[3] else None,
                        "technical_score": float(session_data[4]) if session_data[4] else None,
                        "communication_score": float(session_data[5]) if session_data[5] else None,
                        "attitude_score": float(session_data[6]) if session_data[6] else None,
                        "started_at": session_data[7].isoformat() if session_data[7] else None,
                        "completed_at": session_data[8].isoformat() if session_data[8] else None,
                        "duration_minutes": session_data[9],
                        "student_id": session_data[10],
                        "overall_feedback": overall_feedback,
                    },
                    "questions": questions,
                    "progression": {"difficulty": difficulty_progression, "sentiment": sentiment_progression},
                    "comparative_analysis": {
                        "role_average": role_avg,
                        "company_average": company_avg,
                        "performance_vs_role": (float(session_data[3]) - role_avg) if session_data[3] and role_avg else None,
                        "performance_vs_company": (float(session_data[3]) - company_avg) if session_data[3] and company_avg else None,
                    },
                }

    @staticmethod
    def get_comparative_analytics_repo() -> Dict[str, Any]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id.job_role, 
                           COUNT(sm.session_id) as total_sessions,
                           AVG(sm.overall_score) as avg_overall,
                           AVG(sm.technical_score) as avg_technical,
                           AVG(sm.communication_score) as avg_communication,
                           AVG(sm.attitude_score) as avg_attitude
                    FROM session_metadata sm
                    JOIN interview_data id ON sm.session_id = id.session_id
                    WHERE sm.overall_score IS NOT NULL AND id.job_role IS NOT NULL
                    GROUP BY id.job_role
                    ORDER BY avg_overall DESC
                    """
                )
                role_performance = [
                    {
                        "role": row[0],
                        "total_sessions": row[1],
                        "avg_overall": round(float(row[2]), 2) if row[2] else 0,
                        "avg_technical": round(float(row[3]), 2) if row[3] else 0,
                        "avg_communication": round(float(row[4]), 2) if row[4] else 0,
                        "avg_attitude": round(float(row[5]), 2) if row[5] else 0,
                    }
                    for row in cur.fetchall()
                ]

                cur.execute(
                    """
                    SELECT id.company_name, 
                           COUNT(sm.session_id) as total_sessions,
                           AVG(sm.overall_score) as avg_overall,
                           AVG(sm.technical_score) as avg_technical,
                           AVG(sm.communication_score) as avg_communication,
                           AVG(sm.attitude_score) as avg_attitude
                    FROM session_metadata sm
                    JOIN interview_data id ON sm.session_id = id.session_id
                    WHERE sm.overall_score IS NOT NULL AND id.company_name IS NOT NULL
                    GROUP BY id.company_name
                    ORDER BY avg_overall DESC
                    """
                )
                company_performance = [
                    {
                        "company": row[0],
                        "total_sessions": row[1],
                        "avg_overall": round(float(row[2]), 2) if row[2] else 0,
                        "avg_technical": round(float(row[3]), 2) if row[3] else 0,
                        "avg_communication": round(float(row[4]), 2) if row[4] else 0,
                        "avg_attitude": round(float(row[5]), 2) if row[5] else 0,
                    }
                    for row in cur.fetchall()
                ]

                cur.execute(
                    """
                    SELECT difficulty, COUNT(*) as count
                    FROM interview_data
                    WHERE difficulty IS NOT NULL
                    GROUP BY difficulty
                    ORDER BY count DESC
                    """
                )
                difficulty_distribution = [{"difficulty": row[0], "count": row[1]} for row in cur.fetchall()]

                return {
                    "role_performance": role_performance,
                    "company_performance": company_performance,
                    "difficulty_distribution": difficulty_distribution,
                }

    @staticmethod
    def get_filter_options_repo() -> Dict[str, List[str]]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT job_role FROM interview_data WHERE job_role IS NOT NULL ORDER BY job_role")
                roles = [row[0] for row in cur.fetchall()]

                cur.execute("SELECT DISTINCT company_name FROM interview_data WHERE company_name IS NOT NULL ORDER BY company_name")
                companies = [row[0] for row in cur.fetchall()]

                cur.execute(
                    "SELECT DISTINCT university_name FROM university_batch_program WHERE university_name IS NOT NULL ORDER BY university_name"
                )
                universities = [row[0] for row in cur.fetchall()]

                cur.execute(
                    "SELECT DISTINCT program_name FROM university_batch_program WHERE program_name IS NOT NULL ORDER BY program_name"
                )
                programs = [row[0] for row in cur.fetchall()]

                cur.execute(
                    "SELECT DISTINCT batch_label FROM university_batch_program WHERE batch_label IS NOT NULL ORDER BY batch_label"
                )
                batches = [row[0] for row in cur.fetchall()]

                return {
                    "roles": roles,
                    "companies": companies,
                    "universities": universities,
                    "programs": programs,
                    "batches": batches,
                }

    @staticmethod
    def get_role_filter_options_repo(
        university: Optional[str], program: Optional[str], batch: Optional[str]
    ) -> List[str]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                base_query = """
                    SELECT DISTINCT id.job_role
                    FROM session_metadata sm
                    LEFT JOIN interview_data id ON sm.session_id = id.session_id
                    LEFT JOIN students s ON sm.student_id = s.student_id
                    LEFT JOIN programs p ON s.program_id = p.id
                    LEFT JOIN LATERAL (
                        SELECT selected.university_name, selected.program_name, selected.batch_label
                        FROM (
                            SELECT u.university_name, u.program_name, u.batch_label, 0 AS priority
                            FROM university_batch_program u
                            WHERE u.ubp_id = s.program_id
                            UNION ALL
                            SELECT u2.university_name, u2.program_name, u2.batch_label, 1 AS priority
                            FROM university_batch_program u2
                            WHERE s.program_id IS NOT NULL
                              AND p.program_name IS NOT NULL
                              AND NOT EXISTS (
                                  SELECT 1
                                  FROM university_batch_program u_exact
                                  WHERE u_exact.ubp_id = s.program_id
                              )
                              AND LOWER(TRIM(u2.program_name)) = LOWER(TRIM(p.program_name))
                        ) selected
                        ORDER BY selected.priority
                        LIMIT 1
                    ) ubp ON TRUE
                """
                where_clauses: List[str] = ["id.job_role IS NOT NULL", "TRIM(id.job_role) <> ''"]
                params: List[Any] = []

                if university:
                    where_clauses.append("ubp.university_name = %s")
                    params.append(university)
                if program:
                    where_clauses.append("COALESCE(ubp.program_name, p.program_name) = %s")
                    params.append(program)
                if batch:
                    where_clauses.append("ubp.batch_label = %s")
                    params.append(batch)

                where_clause = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
                query = base_query + where_clause + " ORDER BY id.job_role"
                cur.execute(query, tuple(params))
                return [row[0] for row in cur.fetchall() if row and row[0]]

    @staticmethod
    def get_export_sessions_repo() -> List[Tuple]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        s.session_id, s.student_name, s.job_role, s.company_name,
                        s.status, s.overall_score, s.technical_score, s.communication_score,
                        s.attitude_score, s.started_at, s.completed_at, s.duration_minutes
                    FROM interview_sessions s
                    ORDER BY s.started_at DESC
                    """
                )
                return cur.fetchall()

    @staticmethod
    def check_admin_db_health_repo() -> bool:
        try:
            with db_pool.get_connection() as conn:
                if not conn:
                    return False
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return True
        except Exception:
            return False

    @staticmethod
    def _company_row_to_dict(row: Tuple) -> Dict[str, Any]:
        return {
            "id": row[0],
            "name": row[1],
            "logo_url": row[2],
            "industry": row[3],
            "difficulty_tag": row[4],
            "work_experience_tag": row[5],
            "is_active": row[6],
            "created_at": row[7].isoformat() if row[7] else None,
        }

    @staticmethod
    def insert_company(record: Dict[str, Any]) -> Dict[str, Any]:
        try:
            with db_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    ensure_companies_table(cur)
                    cur.execute(
                        """
                        INSERT INTO companies (name, logo_url, industry, difficulty_tag, work_experience_tag)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING id, name, logo_url, industry, difficulty_tag, work_experience_tag, is_active, created_at
                        """,
                        (
                            record["name"],
                            record.get("logo_url"),
                            record.get("industry"),
                            record.get("difficulty_tag"),
                            record.get("work_experience_tag"),
                        ),
                    )
                    row = cur.fetchone()
                    conn.commit()
                    return AdminRepository._company_row_to_dict(row)
        except UniqueViolation as exc:
            raise HTTPException(status_code=409, detail="A company with this name already exists") from exc
        except Exception as exc:
            logger.error("Error inserting company: %s", exc)
            raise HTTPException(status_code=500, detail="Failed to create company") from exc

    @staticmethod
    def update_company_logo(company_id: int, logo_url: str) -> Dict[str, Any]:
        try:
            with db_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE companies SET logo_url = %s
                        WHERE id = %s
                        RETURNING id, name, logo_url, industry, difficulty_tag, work_experience_tag, is_active, created_at
                        """,
                        (logo_url, company_id),
                    )
                    row = cur.fetchone()
                    conn.commit()
                    return AdminRepository._company_row_to_dict(row)
        except Exception as exc:
            logger.error("Error updating company logo for %s: %s", company_id, exc)
            raise HTTPException(status_code=500, detail="Failed to update company logo") from exc

    @staticmethod
    def list_companies() -> List[Dict[str, Any]]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                ensure_companies_table(cur)
                cur.execute(
                    """
                    SELECT id, name, logo_url, industry, difficulty_tag, work_experience_tag, is_active, created_at
                    FROM companies
                    ORDER BY name
                    """
                )
                return [AdminRepository._company_row_to_dict(row) for row in cur.fetchall()]

    @staticmethod
    def get_company_by_id(company_id: int) -> Optional[Dict[str, Any]]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                ensure_companies_table(cur)
                cur.execute(
                    """
                    SELECT id, name, logo_url, industry, difficulty_tag, work_experience_tag, is_active, created_at
                    FROM companies
                    WHERE id = %s
                    """,
                    (company_id,),
                )
                row = cur.fetchone()
                return AdminRepository._company_row_to_dict(row) if row else None

    @staticmethod
    async def replace_company_playbook_chunks(company_name: str, chunks: List[Dict[str, str]]) -> int:
        if not chunks:
            return 0
        try:
            embedded_chunks = []
            for chunk in chunks:
                text = str(chunk["text"])
                embedding = await embed_text(text, task_type="retrieval_document")
                embedded_chunks.append((chunk.get("section"), text, embedding))

            with db_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    ensure_company_playbook_table(cur)
                    cur.execute(
                        "DELETE FROM company_playbook_chunks WHERE LOWER(TRIM(company_name)) = LOWER(TRIM(%s))",
                        (company_name,),
                    )
                    for section, text, embedding in embedded_chunks:
                        cur.execute(
                            """
                            INSERT INTO company_playbook_chunks (company_name, source_section, chunk_text, embedding, metadata)
                            VALUES (%s, %s, %s, %s::vector, %s)
                            """,
                            (company_name, section, text, embedding, Json({})),
                        )
                conn.commit()
            return len(chunks)
        except Exception as exc:
            logger.error("Error replacing playbook chunks for %s: %s", company_name, exc)
            raise HTTPException(status_code=500, detail="Failed to ingest company playbook") from exc

    @staticmethod
    def get_playbook_chunk_counts() -> List[Dict[str, Any]]:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                ensure_companies_table(cur)
                ensure_company_playbook_table(cur)
                cur.execute(
                    """
                    SELECT c.id AS company_id, c.name, COALESCE(COUNT(p.id), 0) AS chunk_count
                    FROM companies c
                    LEFT JOIN company_playbook_chunks p ON LOWER(TRIM(p.company_name)) = LOWER(TRIM(c.name))
                    GROUP BY c.id, c.name
                    ORDER BY c.name
                    """
                )
                return [
                    {"company_id": row[0], "name": row[1], "chunk_count": row[2]}
                    for row in cur.fetchall()
                ]
