from typing import Any, Dict, List, Optional
from fastapi import HTTPException

from app.core.cache import session_cache
from app.core.database import db_pool
from app.core.logger import logger
from app.modules.auth.service import create_student_jwt_token, generate_temp_password, hash_password
from app.modules.students.schemas import StudentRegistration, StudentRegistrationResponse


def find_student_id(student_name: str, student_email: Optional[str]) -> Optional[int]:
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


def resolve_ubp_id(
    university_name: Optional[str], program_name: Optional[str], batch_label: Optional[str]
) -> Optional[int]:
    """Resolve ubp_id from university_batch_program by composite keys."""
    if not (university_name and program_name and batch_label):
        return None
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                return None
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT ubp_id
                    FROM university_batch_program
                    WHERE LOWER(TRIM(university_name)) = LOWER(TRIM(%s))
                      AND LOWER(TRIM(program_name)) = LOWER(TRIM(%s))
                      AND LOWER(TRIM(batch_label)) = LOWER(TRIM(%s))
                    LIMIT 1
                    """,
                    (university_name, program_name, batch_label),
                )
                row = cur.fetchone()
                return row[0] if row else None
    except Exception as exc:
        logger.error(f"Error resolving UBP id: {exc}")
    return None


def program_metadata_by_ubp(ubp_id: int) -> Optional[Dict[str, Any]]:
    """Get program metadata (name, batch, university) from UBP by id."""
    if not ubp_id:
        return None
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                return None
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT program_name, batch_label, university_name
                    FROM university_batch_program
                    WHERE ubp_id = %s
                    """,
                    (ubp_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return {"program_name": row[0], "batch_label": row[1], "university_name": row[2]}
    except Exception as exc:
        logger.error(f"Error fetching program metadata by UBP {ubp_id}: {exc}")
    return None


def resolve_program_id_by_name(program_name: str) -> Optional[int]:
    """Resolve a canonical program_id by program name."""
    if not program_name:
        return None

    try:
        with db_pool.get_connection() as conn:
            if not conn:
                return None

            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT MIN(id)
                    FROM programs
                    WHERE LOWER(TRIM(program_name)) = LOWER(TRIM(%s))
                """,
                    (program_name,),
                )
                row = cur.fetchone()
                return row[0] if row and row[0] else None
    except Exception as exc:
        logger.error(f"Error resolving program id for {program_name}: {exc}")
    return None


def fetch_program_info(program_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve program details treating program_id as ubp_id.
    1) Resolve program_name via university_batch_program by ubp_id.
    2) Fallback to legacy programs.id lookup if not found.
    3) Fetch job_roles by program_name from programs.
    """
    if not program_id:
        return None

    try:
        program_meta = program_metadata_by_ubp(program_id)
        program_name = program_meta.get("program_name") if program_meta else None
        batch_label = program_meta.get("batch_label") if program_meta else None
        university_name = program_meta.get("university_name") if program_meta else None

        with db_pool.get_connection() as conn:
            if not conn:
                return None

            with conn.cursor() as cur:
                if not program_name:
                    cur.execute(
                        """
                        SELECT program_name
                        FROM programs
                        WHERE id = %s
                        """,
                        (program_id,),
                    )
                    row = cur.fetchone()
                    if not row:
                        return None
                    program_name = row[0]

                cur.execute(
                    """
                    SELECT job_role
                    FROM programs
                    WHERE program_name = %s
                    ORDER BY job_role
                    """,
                    (program_name,),
                )
                job_roles = [row[0] for row in cur.fetchall() if row and row[0]]

                return {
                    "program_id": program_id,
                    "program_name": program_name,
                    "batch_label": batch_label,
                    "university_name": university_name,
                    "job_roles": job_roles,
                }
    except Exception as exc:
        logger.error(f"Error fetching program info for id {program_id}: {exc}")
    return None


def fetch_all_programs() -> List[Dict[str, Any]]:
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection unavailable")

            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT MIN(id) AS program_id,
                           program_name,
                           ARRAY_AGG(DISTINCT job_role ORDER BY job_role) AS job_roles
                    FROM programs
                    WHERE program_name IS NOT NULL AND TRIM(program_name) <> ''
                    GROUP BY program_name
                    ORDER BY program_name
                    """
                )
                rows = cur.fetchall()
                return [
                    {"program_id": row[0], "program_name": row[1], "job_roles": [role for role in row[2] if role]}
                    for row in rows
                    if row
                ]
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching programs: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch programs") from exc


def ensure_job_descriptions_table(cur) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS job_descriptions (
            id SERIAL PRIMARY KEY,
            student_id INTEGER REFERENCES students(student_id) ON DELETE SET NULL,
            filename VARCHAR(255) NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            parsed_data JSONB,
            job_desc TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    column_statements = [
        "ALTER TABLE job_descriptions ADD COLUMN IF NOT EXISTS parsed_data JSONB",
        "ALTER TABLE job_descriptions ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE",
        "ALTER TABLE job_descriptions ADD COLUMN IF NOT EXISTS upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        "ALTER TABLE job_descriptions ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        "ALTER TABLE job_descriptions ADD COLUMN IF NOT EXISTS job_desc TEXT",
    ]
    for statement in column_statements:
        try:
            cur.execute(statement)
        except Exception as column_error:
            logger.warning(f"Failed ensuring job_descriptions column with statement '{statement}': {column_error}")


def register_student_repo(payload: StudentRegistration) -> StudentRegistrationResponse:
    try:
        program_id = payload.program_id

        if not program_id and (payload.university_name and payload.program_name and payload.batch_label):
            program_id = resolve_ubp_id(
                payload.university_name.strip(), payload.program_name.strip(), payload.batch_label.strip()
            )

        if not program_id and payload.program_name:
            program_id = resolve_program_id_by_name(payload.program_name.strip())

        if not program_id:
            raise HTTPException(status_code=422, detail="Unknown program specified")

        program_info = fetch_program_info(program_id)
        if not program_info:
            raise HTTPException(status_code=422, detail="Unknown program specified")

        temporary_password: Optional[str] = None
        password_to_store: Optional[str] = None
        if payload.password is not None:
            password_value = payload.password.strip()
            if password_value:
                password_to_store = hash_password(password_value)
        else:
            temporary_password = generate_temp_password()
            password_to_store = hash_password(temporary_password)

        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection unavailable")

            with conn.cursor() as cur:
                email = payload.student_email.strip()
                name = payload.student_name.strip()

                cur.execute(
                    """
                    SELECT student_id
                    FROM students
                    WHERE LOWER(TRIM(email)) = LOWER(TRIM(%s))
                    """,
                    (email,),
                )
                existing = cur.fetchone()

                if existing:
                    cur.execute(
                        """
                        UPDATE students
                        SET name = %s,
                            program_id = %s,
                            password = COALESCE(%s, password),
                            last_active = CURRENT_TIMESTAMP
                        WHERE student_id = %s
                        RETURNING student_id
                        """,
                        (name, program_id, password_to_store, existing[0]),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO students (name, email, program_id, password, last_active)
                        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                        RETURNING student_id
                        """,
                        (name, email, program_id, password_to_store),
                    )

                student_id = cur.fetchone()[0]
                conn.commit()

        access_token = create_student_jwt_token({"sub": str(student_id), "email": email})

        return StudentRegistrationResponse(
            access_token=access_token,
            student_id=student_id,
            name=name,
            email=email,
            program_id=program_id,
            program_name=program_info.get("program_name"),
            batch_label=program_info.get("batch_label"),
            university_name=program_info.get("university_name"),
            job_roles=program_info.get("job_roles", []),
            temporary_password=temporary_password,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error registering student: {exc}")
        raise HTTPException(status_code=500, detail="Failed to register student") from exc


def get_student_profile_repo(email: str) -> Dict[str, Any]:
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection unavailable")

            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT student_id, name, email, program_id
                    FROM students
                    WHERE email = %s
                    """,
                    (email,),
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Student not found")

                program_details = {}
                if row[3]:
                    program_info = fetch_program_info(row[3])
                    if program_info:
                        program_details = program_info

                return {
                    "student_id": row[0],
                    "name": row[1],
                    "email": row[2],
                    "program_id": row[3],
                    **program_details,
                }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching student profile for {email}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch profile")


def list_universities_repo() -> List[str]:
    session_cache.remove("ubp_universities")
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection unavailable")
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT university_name
                    FROM university_batch_program
                    WHERE university_name IS NOT NULL AND TRIM(university_name) <> ''
                    ORDER BY university_name
                    """
                )
                return [r[0] for r in cur.fetchall() if r and r[0]]
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching universities: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch universities") from exc


def list_programs_by_university_repo(university: str) -> List[str]:
    session_cache.remove(f"ubp_programs_{university}")
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection unavailable")
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT program_name
                    FROM university_batch_program
                    WHERE LOWER(TRIM(university_name)) = LOWER(TRIM(%s))
                      AND program_name IS NOT NULL AND TRIM(program_name) <> ''
                    ORDER BY program_name
                    """,
                    (university,),
                )
                return [r[0] for r in cur.fetchall() if r and r[0]]
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching programs for university {university}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch programs") from exc


def list_batches_repo(university: str, program: str) -> List[str]:
    session_cache.remove(f"ubp_batches_{university}_{program}")
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection unavailable")
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT batch_label
                    FROM university_batch_program
                    WHERE LOWER(TRIM(university_name)) = LOWER(TRIM(%s))
                      AND LOWER(TRIM(program_name)) = LOWER(TRIM(%s))
                      AND batch_label IS NOT NULL AND TRIM(batch_label) <> ''
                    ORDER BY batch_label
                    """,
                    (university, program),
                )
                return [r[0] for r in cur.fetchall() if r and r[0]]
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching batches for {university} / {program}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch batches") from exc


def get_student_performance_summary_repo(student_id: int) -> Dict[str, Any]:
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection unavailable")

            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        COUNT(DISTINCT pgq.session_id) AS total_sessions,
                        ROUND(AVG(sm.overall_score), 2) AS avg_score,
                        ROUND(
                            COALESCE(SUM(
                                EXTRACT(EPOCH FROM (sm.completed_at - sm.started_at)) / 3600
                            ), 0), 2
                        ) AS hours_practiced
                    FROM pre_generated_questions pgq
                    LEFT JOIN session_metadata sm ON pgq.session_id = sm.session_id
                    WHERE pgq.student_id = %s
                    """,
                    (student_id,),
                )
                row = cur.fetchone()
                if not row:
                    return {
                        "avg_score": 0.0,
                        "total_sessions": 0,
                        "hours_practiced": 0.0,
                        "completion_dates": [],
                    }

                cur.execute(
                    """
                    SELECT DISTINCT DATE(sm.completed_at)
                    FROM pre_generated_questions pgq
                    LEFT JOIN session_metadata sm ON pgq.session_id = sm.session_id
                    WHERE pgq.student_id = %s
                      AND sm.completed_at IS NOT NULL
                    ORDER BY DATE(sm.completed_at) DESC
                    """,
                    (student_id,),
                )
                completion_dates = [r[0] for r in cur.fetchall() if r and r[0]]

                return {
                    "avg_score": float(row[1] or 0.0),
                    "total_sessions": int(row[0] or 0),
                    "hours_practiced": float(row[2] or 0.0),
                    "completion_dates": completion_dates,
                }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching performance summary for student {student_id}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch performance summary") from exc


def get_student_interview_history_repo(student_id: int, page: int = 1, page_size: int = 10) -> Dict[str, Any]:
    try:
        offset = (page - 1) * page_size
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection unavailable")

            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(DISTINCT pgq.session_id)
                    FROM pre_generated_questions pgq
                    LEFT JOIN session_metadata sm ON pgq.session_id = sm.session_id
                    WHERE pgq.student_id = %s
                    """,
                    (student_id,),
                )
                total_sessions = cur.fetchone()[0] or 0

                cur.execute(
                    """
                    SELECT DISTINCT
                        pgq.session_id,
                        pgq.company_name,
                        pgq.job_role,
                        pgq.interview_type,
                        pgq.work_experience,
                        sm.started_at,
                        sm.completed_at,
                        sm.status,
                        sm.overall_score
                    FROM pre_generated_questions pgq
                    LEFT JOIN session_metadata sm ON pgq.session_id = sm.session_id
                    WHERE pgq.student_id = %s
                    ORDER BY sm.started_at DESC NULLS LAST
                    LIMIT %s OFFSET %s
                    """,
                    (student_id, page_size, offset),
                )
                rows = cur.fetchall()

                items = [
                    {
                        "session_id": row[0],
                        "company_name": row[1],
                        "job_role": row[2],
                        "interview_type": row[3],
                        "work_experience": row[4],
                        "started_at": row[5],
                        "completed_at": row[6],
                        "status": row[7],
                        "score": row[8],
                    }
                    for row in rows
                ]

                total_pages = (total_sessions + page_size - 1) // page_size if total_sessions else 0
                return {
                    "items": items,
                    "page": page,
                    "page_size": page_size,
                    "total": int(total_sessions),
                    "pages": total_pages,
                }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching interview history for student {student_id}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch interview history") from exc
