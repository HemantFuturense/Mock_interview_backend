import logging
from contextlib import contextmanager
from typing import Optional
import psycopg2
from psycopg2 import pool

from app.config.settings import config
from app.core.logger import logger

_INTERVIEW_DATA_COLUMNS_ENSURED = False


class DatabasePool:
    """Singleton thread-safe connection pool for PostgreSQL using psycopg2."""
    _instance = None
    _pool = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._pool is None:
            try:
                self._pool = psycopg2.pool.ThreadedConnectionPool(
                    config.MIN_CONNECTIONS,
                    config.MAX_CONNECTIONS,
                    **config.DB_CONFIG
                )
                logger.info("Database pool initialized")
            except Exception as e:
                logger.error(f"Failed to create database pool: {e}")

    @contextmanager
    def get_connection(self):
        """Context manager to lease and return database connections safely."""
        conn = None
        try:
            conn = self._pool.getconn()

            # Simple check: if connection is closed or broken, get a fresh one
            if conn.closed != 0:
                self._pool.putconn(conn, close=True)
                conn = self._pool.getconn()

            yield conn

            # Return valid connection to pool on clean exit
            if conn and not conn.closed:
                self._pool.putconn(conn)
                conn = None

        except psycopg2.OperationalError as e:
            logger.error(f"Database operational error (connection possibly dropped): {e}")
            if conn and not conn.closed:
                try:
                    conn.rollback()
                except Exception:
                    pass
                self._pool.putconn(conn, close=True)
                conn = None
            raise

        except Exception as e:
            logger.error(f"Database error: {e}")
            if conn and not conn.closed:
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise

        finally:
            if conn and not conn.closed:
                self._pool.putconn(conn)


# Create global connection pool instance
db_pool = DatabasePool()


def ensure_job_descriptions_table(cur) -> None:
    """Ensure job_descriptions table exists with required schema."""
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


def ensure_interview_data_columns(cur) -> None:
    """Ensure interview_data table has all required columns for enhanced AI interview tracking."""
    global _INTERVIEW_DATA_COLUMNS_ENSURED
    if _INTERVIEW_DATA_COLUMNS_ENSURED:
        return

    ensure_job_descriptions_table(cur)

    column_statements = [
        "ALTER TABLE interview_data ADD COLUMN IF NOT EXISTS question_type TEXT",
        "ALTER TABLE interview_data ADD COLUMN IF NOT EXISTS code_submission TEXT",
        "ALTER TABLE interview_data ADD COLUMN IF NOT EXISTS stdin_input TEXT",
        "ALTER TABLE interview_data ADD COLUMN IF NOT EXISTS stdout_output TEXT",
        "ALTER TABLE interview_data ADD COLUMN IF NOT EXISTS stderr_output TEXT",
        "ALTER TABLE interview_data ADD COLUMN IF NOT EXISTS runtime_error TEXT",
        "ALTER TABLE interview_data ADD COLUMN IF NOT EXISTS execution_success BOOLEAN",
        "ALTER TABLE interview_data ADD COLUMN IF NOT EXISTS manual_run BOOLEAN",
        "ALTER TABLE interview_data ADD COLUMN IF NOT EXISTS interview_type TEXT",
        "ALTER TABLE interview_data ADD COLUMN IF NOT EXISTS work_experience TEXT",
        "ALTER TABLE interview_data ADD COLUMN IF NOT EXISTS video_analysis TEXT",
        "ALTER TABLE interview_data ADD COLUMN IF NOT EXISTS video_analysis_status TEXT DEFAULT 'not_applicable'",
        "ALTER TABLE interview_data ADD COLUMN IF NOT EXISTS video_clip_path TEXT",
        "ALTER TABLE interview_data ADD COLUMN IF NOT EXISTS resume_id INTEGER REFERENCES student_resumes(id) ON DELETE SET NULL",
        "ALTER TABLE interview_data ADD COLUMN IF NOT EXISTS job_description_id INTEGER REFERENCES job_descriptions(id) ON DELETE SET NULL",
        "ALTER TABLE interview_data ADD COLUMN IF NOT EXISTS job_description_text TEXT",
        "ALTER TABLE interview_data ADD COLUMN IF NOT EXISTS job_desc TEXT",
        "ALTER TABLE interview_data ADD COLUMN IF NOT EXISTS generation_context JSONB"
    ]

    for statement in column_statements:
        try:
            cur.execute(statement)
        except Exception as column_error:
            logger.warning(f"Failed ensuring interview_data column with statement '{statement}': {column_error}")

    # Ensure columns on pre_generated_questions if table exists
    pre_gen_column_statements = [
        "ALTER TABLE pre_generated_questions ADD COLUMN IF NOT EXISTS job_description_id INTEGER REFERENCES job_descriptions(id) ON DELETE SET NULL",
        "ALTER TABLE pre_generated_questions ADD COLUMN IF NOT EXISTS job_description_text TEXT",
        "ALTER TABLE pre_generated_questions ADD COLUMN IF NOT EXISTS job_desc TEXT"
    ]
    for statement in pre_gen_column_statements:
        try:
            cur.execute(statement)
        except Exception as column_error:
            logger.debug(f"Pre-gen table column check (expected if not yet created): {column_error}")

    _INTERVIEW_DATA_COLUMNS_ENSURED = True


def ensure_session_metadata_columns(cur) -> None:
    """Ensure session_metadata table has all required columns."""
    column_statements = [
        "ALTER TABLE session_metadata ADD COLUMN IF NOT EXISTS resume_id INTEGER REFERENCES student_resumes(id) ON DELETE SET NULL",
        "ALTER TABLE session_metadata ADD COLUMN IF NOT EXISTS question_generation_type VARCHAR(20) DEFAULT 'standard'",
        "ALTER TABLE session_metadata ADD COLUMN IF NOT EXISTS questions_generated_at TIMESTAMP",
        "ALTER TABLE session_metadata ADD COLUMN IF NOT EXISTS job_description_id INTEGER REFERENCES job_descriptions(id) ON DELETE SET NULL",
        "ALTER TABLE session_metadata ADD COLUMN IF NOT EXISTS job_description_text TEXT",
        "ALTER TABLE session_metadata ADD COLUMN IF NOT EXISTS job_desc TEXT",
        "ALTER TABLE session_metadata ADD COLUMN IF NOT EXISTS termination_reason TEXT",
        "ALTER TABLE session_metadata ADD COLUMN IF NOT EXISTS terminated_at TIMESTAMP"
    ]

    for statement in column_statements:
        try:
            cur.execute(statement)
        except Exception as column_error:
            logger.warning(f"Failed ensuring session_metadata column: {column_error}")


def ensure_pre_generated_questions_table(cur) -> None:
    """Ensure pre_generated_questions table exists with required schema."""
    ensure_job_descriptions_table(cur)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS pre_generated_questions (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(255) NOT NULL,
            student_id INTEGER REFERENCES students(student_id) ON DELETE SET NULL,
            resume_id INTEGER REFERENCES student_resumes(id) ON DELETE SET NULL,
            job_description_id INTEGER REFERENCES job_descriptions(id) ON DELETE SET NULL,
            company_name VARCHAR(255),
            job_role VARCHAR(255),
            interview_type VARCHAR(100),
            work_experience VARCHAR(100),
            job_description_text TEXT,
            job_desc TEXT,
            question_number INTEGER NOT NULL,
            question TEXT NOT NULL,
            question_type VARCHAR(100),
            difficulty VARCHAR(50),
            mandatory_skills JSONB,
            generation_context JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(session_id, question_number)
        )
        """
    )


def ensure_companies_table(cur) -> None:
    """Ensure companies table exists, matching its current live schema exactly."""
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS companies (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL UNIQUE,
            logo_url TEXT,
            industry VARCHAR(255),
            difficulty_tag VARCHAR(50),
            work_experience_tag VARCHAR(100),
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
