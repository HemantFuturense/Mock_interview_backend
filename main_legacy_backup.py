from fastapi import FastAPI, HTTPException, Depends, Query, Form, File, UploadFile, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, conint
import asyncio
import google.generativeai as genai
import uuid
import logging
import psycopg2
from psycopg2 import pool
from psycopg2.extras import Json
import random
from contextlib import contextmanager
import time
from functools import lru_cache
import inspect
from typing import Optional, Dict, List, Any, Tuple, Union
from itertools import combinations
from datetime import datetime, timedelta, timezone, date
import csv
import io
import secrets
import smtplib
import ssl
from urllib.parse import quote_plus
from email.message import EmailMessage
from dotenv import load_dotenv
import json
import os
import shutil
import traceback
import re
import textwrap
from pathlib import Path
from zoneinfo import ZoneInfo
from jinja2 import Environment, FileSystemLoader, select_autoescape
import urllib.request
import urllib.error
import pdfplumber
import docx
from collections import defaultdict

from api_router import admin_router

# Load environment
load_dotenv()


def _log_question_selection(
    source: str,
    role: Optional[str],
    company: Optional[str],
    difficulty: Optional[str],
    interview_type: Optional[str],
    work_experience: Optional[str],
    question_type: Optional[str],
    question_text: Optional[str],
) -> None:
    """Emit a concise log/print describing the question that was fetched."""
    summary = (question_text or "").replace("\n", " ").strip()
    if len(summary) > 140:
        summary = summary[:137] + "..."
    message = (
        f"[QUESTION FETCH][{source}] role={role or '-'} company={company or '-'} "
        f"difficulty={difficulty or '-'} interview_type={interview_type or '-'} "
        f"work_experience={work_experience or '-'} question_type={question_type or '-'} "
        f"question=\"{summary or 'N/A'}\""
    )
    logger.info(message)
    print(message)

# Enhanced Configuration
class Config:
    # Database
    DB_CONFIG = {
        "dbname": os.getenv("DB_NAME", "ai_mock_interviews"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "hemant851311420"),
        "host": os.getenv("DB_HOST", "ai-chatbot.cxuzaqkzgcfx.ap-south-1.rds.amazonaws.com"),
        "port": os.getenv("DB_PORT", "5432"),
    }

    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    # Email / SMTP (optional)
    SMTP_SETTINGS = {
        "host": os.getenv("SMTP_HOST"),
        "port": int(os.getenv("SMTP_PORT", "587")),
        "username": os.getenv("SMTP_USERNAME"),
        "password": os.getenv("SMTP_PASSWORD"),
        "use_tls": os.getenv("SMTP_USE_TLS", "true").lower() == "true",
    }
    EMAIL_SENDER = os.getenv("EMAIL_SENDER")
    PASSWORD_RESET_URL_BASE = os.getenv("PASSWORD_RESET_URL_BASE")

    CACHE_TTL = int(os.getenv("CACHE_TTL", "900"))  # seconds
    MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "128"))

    # Connection Pool
    MIN_CONNECTIONS = int(os.getenv("MIN_DB_CONNECTIONS", "5"))
    MAX_CONNECTIONS = int(os.getenv("MAX_DB_CONNECTIONS", "50"))

    # Self-hosted Piston configuration
    # PISTON_BASE_URL = os.getenv("PISTON_BASE_URL", "http://localhost:2000/api/v2")
    PISTON_BASE_URL = os.getenv("PISTON_BASE_URL", "https://emkc.org/api/v2/piston")
    

    # Media storage
    MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", Path(__file__).resolve().parent / "media"))
    INTERVIEW_VIDEO_DIR = MEDIA_ROOT / "interview_videos"

    # Gemini models
    GEMINI_VIDEO_MODEL = os.getenv("GEMINI_VIDEO_MODEL", "gemini-3.5-flash-lite")

# Create config instance
config = Config()
config.MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
config.INTERVIEW_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# Configure Gemini
genai.configure(api_key=config.GEMINI_API_KEY)


async def embed_text(text: str) -> list[float]:
    """Generate a 768-dim embedding using Gemini's gemini-embedding-001 model."""
    result = await asyncio.to_thread(
        genai.embed_content,
        model="models/gemini-embedding-001",
        content=text,
        output_dimensionality=768,
    )
    return result["embedding"]


# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PYTHON_VISUALIZATION_BOOTSTRAP = """
import io
import base64

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

def __emit_all_figures_for_ai_mock_interview():
    if plt is None:
        return
    try:
        fig_nums = plt.get_fignums()
        for num in fig_nums:
            fig = plt.figure(num)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("ascii")
            print("__IMAGE_PNG__:" + img_b64)
        plt.close("all")
    except Exception:
        pass

__emit_all_figures_for_ai_mock_interview()
"""

def _inject_python_visualization_bootstrap(payload: Dict[str, Any]) -> Dict[str, Any]:
    language = str(payload.get("language", "")).lower()
    if language not in {"python", "py", "py3", "python3"}:
        return payload
    files = payload.get("files")
    if not isinstance(files, list):
        return payload
    main_index: Optional[int] = None
    for i, item in enumerate(files):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if not name.endswith(".py"):
            continue
        if name == "main.py" or main_index is None:
            main_index = i
    if main_index is None:
        if files:
            main_index = 0
        else:
            return payload
    content = files[main_index].get("content")
    if not isinstance(content, str):
        return payload
    if "__IMAGE_PNG__:" in content:
        return payload
    files[main_index]["content"] = content + "\n\n" + PYTHON_VISUALIZATION_BOOTSTRAP
    return payload

async def execute_with_retries(func, *args, max_retries: int = 3, base_delay: float = 1.0, backoff_factor: float = 2.0,
                               retry_label: str = "Gemini call", **kwargs):
    """Execute a callable with retry support and exponential backoff."""
    for attempt in range(max_retries):
        try:
            if inspect.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = await asyncio.to_thread(func, *args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result
        except Exception as exc:
            logger.warning(f"{retry_label} attempt {attempt + 1} failed: {exc}")
            if attempt >= max_retries - 1:
                raise
            await asyncio.sleep(base_delay * (backoff_factor ** attempt))

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates" / "feedback"
jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(disabled_extensions=("j2",), default_for_string=False, default=True),
    trim_blocks=True,
    lstrip_blocks=True,
)

TECHNICAL_COMPETENCIES = [
    {
        "name": "Problem Solving & Logical Thinking",
        "weight": 30,
        "description": "Ability to analyse problems, structure approaches, and consider efficiency trade-offs.",
    },
    {
        "name": "Technical Concepts & Domain Knowledge",
        "weight": 25,
        "description": "Depth of fundamentals across the relevant technical stack and correctness of reasoning.",
    },
    {
        "name": "Application & Project Readiness",
        "weight": 20,
        "description": "Real-world experience, project articulation, and tool proficiency.",
    },
    {
        "name": "Communication & STAR Response",
        "weight": 15,
        "description": "Clarity and structure while explaining using STAR/PEEL frameworks.",
    },
    {
        "name": "Aptitude & Interview Readiness",
        "weight": 10,
        "description": "Overall composure, analytical reasoning, and industry awareness.",
    },
]

HR_COMPETENCIES = [
    {
        "name": "Career Motivation & Goal Alignment",
        "weight": 25,
        "description": "Understanding of career choices and alignment with company goals.",
    },
    {
        "name": "Cultural Fit & Value Alignment",
        "weight": 25,
        "description": "Alignment with organisational values, adaptability, and integrity.",
    },
    {
        "name": "Communication & Presentation",
        "weight": 20,
        "description": "Professional communication, confidence, and listening skills.",
    },
    {
        "name": "Emotional Intelligence & Interpersonal Skills",
        "weight": 15,
        "description": "Empathy, collaboration, and conflict management.",
    },
    {
        "name": "Learning Agility & Growth Orientation",
        "weight": 15,
        "description": "Curiosity, openness to feedback, and continuous learning.",
    },
]

BEHAVIORAL_COMPETENCIES = [
    {
        "name": "Ownership & Accountability",
        "weight": 25,
        "description": "Taking responsibility, initiative, and learning from mistakes.",
    },
    {
        "name": "Teamwork & Collaboration",
        "weight": 20,
        "description": "Working well with diverse teams, contributing ideas, and respecting others.",
    },
    {
        "name": "Problem-Solving in Real Situations",
        "weight": 20,
        "description": "Handling real-world challenges with creativity, resilience, and structure.",
    },
    {
        "name": "Adaptability & Resilience",
        "weight": 20,
        "description": "Responding positively to change, setbacks, and ambiguity.",
    },
    {
        "name": "Ethical Judgment & Professionalism",
        "weight": 15,
        "description": "Acting with integrity, honesty, and professionalism in decisions.",
    },
]

SCORING_GUIDE = [
    {"level": "Excellent", "range": "85-100", "description": "Highly ready; recommend strongly."},
    {"level": "Good", "range": "70-84", "description": "Ready with minor gaps."},
    {"level": "Average", "range": "55-69", "description": "Needs guided support before job readiness."},
    {"level": "Below Average", "range": "0-54", "description": "Requires significant improvement."},
]

MAX_PROMPT_CHARS_QUESTIONS = 48000
MAX_PROMPT_CHARS_COMPETENCIES = 64000


def _compose_conversation_excerpt(entries: List[str], char_limit: int, minimum_entries: int = 3) -> str:
    if not entries:
        return ""

    joined = "\n".join(entries)
    if len(joined) <= char_limit:
        return joined

    selected: List[str] = []
    running_length = 0
    for entry in reversed(entries):
        selected.insert(0, entry)
        running_length += len(entry) + 1  # account for newline joiner
        if running_length >= char_limit and len(selected) >= minimum_entries:
            break

    notice = (
        "[Context trimmed to most recent responses to fit AI model limits. "
        "Earlier questions were omitted; rely on the latest entries below.]"
    )
    return "\n".join([notice, *selected])


def _resolve_feedback_template(interview_type: Optional[str]) -> Dict[str, Any]:
    normalized = (interview_type or "technical").strip().lower()

    if normalized.startswith("behav"):
        return {
            "key": "behavioral",
            "question_template": "behavioral_questions_prompt.j2",
            "competency_template": "behavioral_competencies_prompt.j2",
            "competencies": BEHAVIORAL_COMPETENCIES,
        }

    if normalized.startswith("hr") or "human" in normalized:
        return {
            "key": "hr",
            "question_template": "hr_questions_prompt.j2",
            "competency_template": "hr_competencies_prompt.j2",
            "competencies": HR_COMPETENCIES,
        }

    return {
        "key": "technical",
        "question_template": "technical_questions_prompt.j2",
        "competency_template": "technical_competencies_prompt.j2",
        "competencies": TECHNICAL_COMPETENCIES,
    }


_INTERVIEW_DATA_COLUMNS_ENSURED = False


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def _ensure_job_descriptions_table(cur) -> None:
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
        "ALTER TABLE pre_generated_questions ADD COLUMN IF NOT EXISTS job_description_id INTEGER REFERENCES job_descriptions(id) ON DELETE SET NULL",
        "ALTER TABLE pre_generated_questions ADD COLUMN IF NOT EXISTS job_description_text TEXT",
        "ALTER TABLE pre_generated_questions ADD COLUMN IF NOT EXISTS job_desc TEXT"
    ]

    for statement in column_statements:
        try:
            cur.execute(statement)
        except Exception as column_error:
            logger.warning(
                "Failed ensuring pre_generated_questions column with statement '%s': %s",
                statement,
                column_error,
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


def _ensure_company_playbook_table(cur) -> None:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS company_playbook_chunks (
            id SERIAL PRIMARY KEY,
            company_name TEXT NOT NULL,
            source_section TEXT,
            chunk_text TEXT NOT NULL,
            embedding vector(768),
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_company_playbook_embedding
            ON company_playbook_chunks USING ivfflat (embedding vector_cosine_ops)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_company_playbook_company
            ON company_playbook_chunks (LOWER(TRIM(company_name)))
        """
    )


def _ensure_interview_data_columns(cur) -> None:
    global _INTERVIEW_DATA_COLUMNS_ENSURED
    if _INTERVIEW_DATA_COLUMNS_ENSURED:
        return

    _ensure_job_descriptions_table(cur)

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

    _INTERVIEW_DATA_COLUMNS_ENSURED = True

def _ensure_session_metadata_columns(cur) -> None:
    """Ensure session_metadata has all required columns for resume functionality"""
    column_statements = [
        "ALTER TABLE session_metadata ADD COLUMN IF NOT EXISTS resume_id INTEGER REFERENCES student_resumes(id) ON DELETE SET NULL",
        "ALTER TABLE session_metadata ADD COLUMN IF NOT EXISTS question_generation_type VARCHAR(20) DEFAULT 'standard'",
        "ALTER TABLE session_metadata ADD COLUMN IF NOT EXISTS questions_generated_at TIMESTAMP",
        "ALTER TABLE session_metadata ADD COLUMN IF NOT EXISTS job_description_id INTEGER REFERENCES job_descriptions(id) ON DELETE SET NULL",
        "ALTER TABLE session_metadata ADD COLUMN IF NOT EXISTS job_description_text TEXT",
        "ALTER TABLE session_metadata ADD COLUMN IF NOT EXISTS job_desc TEXT",
        "ALTER TABLE session_metadata ADD COLUMN IF NOT EXISTS termination_reason TEXT",
        "ALTER TABLE session_metadata ADD COLUMN IF NOT EXISTS terminated_at TIMESTAMP",
        "ALTER TABLE session_metadata ADD COLUMN IF NOT EXISTS feedback_generated BOOLEAN DEFAULT FALSE",
        "ALTER TABLE session_metadata ADD COLUMN IF NOT EXISTS feedback_status VARCHAR(30) DEFAULT 'not_requested'",
        "ALTER TABLE session_metadata ADD COLUMN IF NOT EXISTS feedback_error TEXT",
        "ALTER TABLE session_metadata ADD COLUMN IF NOT EXISTS feedback_requested_at TIMESTAMP",
        "ALTER TABLE session_metadata ADD COLUMN IF NOT EXISTS feedback_ready_at TIMESTAMP"
    ]
    for statement in column_statements:
        try:
            cur.execute(statement)
        except Exception as column_error:
            logger.warning(f"Failed ensuring session_metadata column: {column_error}")

def _ensure_pre_generated_questions_table(cur) -> None:
    _ensure_job_descriptions_table(cur)
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


def _attach_resume_metadata_to_session(session_id: str, resume_id: Optional[int]) -> None:
    if not resume_id:
        return
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                return
            with conn.cursor() as cur:
                _ensure_session_metadata_columns(cur)
                cur.execute(
                    """
                    UPDATE session_metadata
                    SET resume_id = %s,
                        question_generation_type = 'resume',
                        questions_generated_at = CURRENT_TIMESTAMP
                    WHERE session_id = %s
                    """,
                    (resume_id, session_id),
                )
                conn.commit()
    except Exception as exc:
        logger.error(f"Failed to attach resume metadata to session {session_id}: {exc}")


def _attach_job_description_metadata_to_session(
    session_id: str,
    job_description_id: Optional[int],
    job_description_text: Optional[str],
    job_desc: Optional[str] = None,
) -> None:
    if not (job_description_id or (job_description_text and job_description_text.strip()) or (job_desc and job_desc.strip())):
        return

    normalized_text = (job_description_text or "").strip()
    normalized_desc = (job_desc or "").strip()

    try:
        with db_pool.get_connection() as conn:
            if not conn:
                return
            with conn.cursor() as cur:
                _ensure_session_metadata_columns(cur)
                cur.execute(
                    """
                    UPDATE session_metadata
                    SET job_description_id = COALESCE(%s, job_description_id),
                        job_description_text = CASE WHEN %s <> '' THEN %s ELSE job_description_text END,
                        job_desc = CASE WHEN %s <> '' THEN %s ELSE job_desc END
                    WHERE session_id = %s
                    """,
                    (job_description_id, normalized_text, normalized_text, normalized_desc, normalized_desc, session_id),
                )
                conn.commit()
    except Exception as exc:
        logger.error(f"Failed to attach job description metadata to session {session_id}: {exc}")


def store_pre_generated_questions(
    *,
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
                _ensure_pre_generated_questions_table(cur)
                for idx, question in enumerate(questions, start=1):
                    mandatory = question.get("mandatory_skills")
                    if isinstance(mandatory, list):
                        mandatory_payload = Json(mandatory)
                    else:
                        mandatory_payload = Json([mandatory] if mandatory else [])

                    cur.execute(
                        """
                        INSERT INTO pre_generated_questions (
                            session_id, student_id, resume_id, job_description_id, company_name, job_role,
                            interview_type, work_experience, job_description_text, job_desc, question_number, question,
                            question_type, difficulty, mandatory_skills, generation_context
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (session_id, question_number) DO UPDATE SET
                            student_id = EXCLUDED.student_id,
                            resume_id = EXCLUDED.resume_id,
                            job_description_id = EXCLUDED.job_description_id,
                            company_name = EXCLUDED.company_name,
                            job_role = EXCLUDED.job_role,
                            interview_type = EXCLUDED.interview_type,
                            work_experience = EXCLUDED.work_experience,
                            job_description_text = EXCLUDED.job_description_text,
                            job_desc = EXCLUDED.job_desc,
                            question = EXCLUDED.question,
                            question_type = EXCLUDED.question_type,
                            difficulty = EXCLUDED.difficulty,
                            mandatory_skills = EXCLUDED.mandatory_skills,
                            generation_context = EXCLUDED.generation_context
                        """,
                        (
                            session_id,
                            student_id,
                            resume_id,
                            job_description_id,
                            company_name,
                            job_role,
                            interview_type,
                            work_experience,
                            job_description_text,
                            job_desc,
                            idx,
                            question.get("question"),
                            question.get("question_type"),
                            question.get("difficulty"),
                            mandatory_payload,
                            Json(question.get("generation_context")),
                        ),
                    )
                conn.commit()
    except Exception as exc:
        logger.error(f"Failed to store pre-generated questions for session {session_id}: {exc}")

# FastAPI App
app = FastAPI(title="Enhanced AI Interview API with Admin Support")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(admin_router)

# Database Pool
class DatabasePool:
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
    
    # @contextmanager
    # def get_connection(self):
    #     conn = None
    #     try:
    #         conn = self._pool.getconn()
    #         yield conn
    #     except Exception as e:
    #         logger.error(f"Database error: {e}")
    #         if conn:
    #             conn.rollback()
    #         raise e  # Re-raise the exception instead of yielding None
    #     finally:
    #         if conn:
    #             self._pool.putconn(conn)

    @contextmanager
    def get_connection(self):
        conn = None
        try:
            conn = self._pool.getconn()
            
            # Simple check: if connection is closed, get a fresh one
            if conn.closed:
                logger.warning("Got closed connection from pool, requesting fresh connection")
                self._pool.putconn(conn, close=True)  # Return bad connection and mark it for closure
                conn = self._pool.getconn()  # Get a new one
                
                # If still closed (unlikely), raise error
                if conn.closed:
                    raise Exception("Unable to get open connection from pool")
            
            yield conn
            
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            # Connection-level errors - log and re-raise
            logger.error(f"Connection error: {e}")
            if conn:
                self._pool.putconn(conn, close=True)  # Force close bad connection
            raise
            
        except Exception as e:
            # Other errors - rollback and re-raise
            logger.error(f"Database error: {e}")
            if conn:
                conn.rollback()
            raise
            
        finally:
            # Return connection to pool if it's still valid
            if conn and not conn.closed:
                self._pool.putconn(conn)            
    

db_pool = DatabasePool()

# In-Memory Caching System
class InMemoryCache:
    def __init__(self, ttl: int = config.CACHE_TTL):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache with current timestamp"""
        self.cache[key] = (value, time.time())
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
    
    def remove(self, key: str) -> None:
        """Remove specific key from cache"""
        self.cache.pop(key, None)

# Initialize caches
session_cache = InMemoryCache()
questions_cache = InMemoryCache()

# Enhanced Pydantic Models
class StudentProfile(BaseModel):
    student_id: Optional[int] = None
    name: str
    email: str
    phone: Optional[str] = None
    created_at: Optional[datetime] = None

class StudentRegistration(BaseModel):
    student_name: str
    student_email: EmailStr
    # Backward-compat fields (treated as ubp_id/program_name if present)
    program_id: Optional[int] = None
    program_name: Optional[str] = None
    # New UBP-driven fields
    university_name: Optional[str] = None
    batch_label: Optional[str] = None

class StudentLoginRequest(BaseModel):
    email: EmailStr
    password: str
    # Optional override on login; treated as ubp_id if provided
    program_id: Optional[int] = None

class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    email: EmailStr
    token: str
    new_password: str

class StudentImportError(BaseModel):
    row: int
    email: Optional[EmailStr] = None
    reason: str


class StudentDuplicateInfo(BaseModel):
    row: int
    email: EmailStr


class StudentImportResult(BaseModel):
    imported: int
    email_sent: int
    total_rows: int
    duplicates_ignored: List[StudentDuplicateInfo] = Field(default_factory=list)
    errors: List[StudentImportError] = Field(default_factory=list)

class InterviewSession(BaseModel):
    session_id: str
    student_id: Optional[int] = None
    student_name: Optional[str] = None
    job_role: str
    company_name: str
    industry_type: str
    status: str = "active"
    answered_questions: int = 0
    overall_score: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class InterviewResponse(BaseModel):
    session_id: str
    response: str
    mandatory_skills: Optional[str] = None
    question_number: Optional[int] = None
    acknowledgment: Optional[str] = None


class TerminateSessionRequest(BaseModel):
    reason: Optional[str] = Field(
        default="Session terminated by system.",
        description="Short note describing why the interview was terminated.",
        max_length=500,
    )

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    database_connected: bool
    cache_size: int

class FeedbackResponse(BaseModel):
    session_id: str
    full_feedback: Optional[str] = None
    question_wise_feedback: Optional[List[dict[str, Any]]] = None
    technical_summary: Optional[str] = None
    communication_summary: Optional[str] = None
    attitude_summary: Optional[str] = None
    completed: bool = False


@app.get("/piston/runtimes")
async def get_piston_runtimes() -> Any:
    """Proxy to self-hosted Piston runtimes endpoint.

    This allows the frontend to fetch runtimes from our backend (port 8001)
    instead of calling Piston directly from the browser, avoiding CORS issues.
    """
    base_url = config.PISTON_BASE_URL.rstrip("/")
    url = f"{base_url}/runtimes"

    def _fetch() -> Any:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:  # nosec B310
            data = resp.read()
            return json.loads(data.decode("utf-8"))

    try:
        data = await asyncio.to_thread(_fetch)
        return data
    except urllib.error.HTTPError as http_err:
        logger.error("Piston runtimes HTTP error: %s", http_err)
        raise HTTPException(status_code=http_err.code, detail="Failed to fetch runtimes from code runner")
    except Exception as exc:  # pragma: no cover - network edge cases
        logger.error("Piston runtimes error: %s", exc)
        raise HTTPException(status_code=502, detail="Error contacting code runner service")


@app.post("/piston/execute")
async def execute_piston_code(request: Request) -> Any:
    """Proxy to self-hosted Piston execute endpoint.

    The request body is forwarded as-is to Piston, and the JSON response is
    returned to the frontend.
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    payload = _inject_python_visualization_bootstrap(payload)

    base_url = config.PISTON_BASE_URL.rstrip("/")
    url = f"{base_url}/execute"

    def _post() -> Any:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310
            data = resp.read()
            return json.loads(data.decode("utf-8"))

    try:
        data = await asyncio.to_thread(_post)
        return data
    except urllib.error.HTTPError as http_err:
        logger.error("Piston execute HTTP error: %s", http_err)
        raise HTTPException(status_code=http_err.code, detail="Failed to execute code in runner")
    except Exception as exc:  # pragma: no cover - network edge cases
        logger.error("Piston execute error: %s", exc)
        raise HTTPException(status_code=502, detail="Error contacting code runner service")

class ReattemptCheckRequest(BaseModel):
    student_name: str
    student_email: Optional[str] = None
    job_role: str
    industry_type: str
    company_name: str

class SessionRatingPayload(BaseModel):
    rating: conint(ge=1, le=5)  # type: ignore[valid-type]
    comments: Optional[str] = None

# Resume-related Pydantic Models
class ResumeUpload(BaseModel):
    student_id: int
    filename: str
    file_path: str
    file_size: int

class ResumeData(BaseModel):
    skills: List[str]
    experience: List[Dict[str, Any]]
    education: List[Dict[str, Any]]
    parsed_text: str

class ResumeQuestionRequest(BaseModel):
    student_id: int
    company_name: str
    job_role: Optional[str] = None
    interview_type: Optional[str] = None
    work_experience: Optional[str] = None
    resume_id: Optional[int] = None
    job_description_id: Optional[int] = None
    job_description_text: Optional[str] = None

class PreGeneratedQuestion(BaseModel):
    session_id: str
    student_id: int
    resume_id: Optional[int]
    company_name: str
    job_role: str
    interview_type: Optional[str]
    work_experience: Optional[str]
    question_number: int
    question: str
    question_type: str  # "Coding (Python)", "Coding (SQL)", "Speech Based"
    difficulty: str
    mandatory_skills: List[str]
    generation_context: Dict[str, Any]

# Enhanced Database Functions - Works with existing tables
def init_enhanced_db():
    """Initialize only NEW admin tables, don't touch existing ones"""
    try:
        print("trying to connect")
        with db_pool.get_connection() as conn:
            if not conn:
                return False
            
            with conn.cursor() as cur:
                # Only create NEW tables for admin functionality
                # Students table (NEW)
                print("connected")
                _ensure_company_playbook_table(cur)
                # cur.execute("""
                #     CREATE TABLE IF NOT EXISTS students (
                #         student_id SERIAL PRIMARY KEY,
                #         name VARCHAR(255) NOT NULL,
                #         email VARCHAR(255) UNIQUE,
                #         phone VARCHAR(20),
                #         program_id INTEGER,
                #         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                #         last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                #         total_sessions INTEGER DEFAULT 0,
                #         avg_score FLOAT DEFAULT 0.0
                #     );
                # """
                # )

                # # Ensure legacy constraints/columns are aligned with new programs table
                # cur.execute("""ALTER TABLE students DROP CONSTRAINT IF EXISTS fk_students_ubp;""")
                # cur.execute("""ALTER TABLE students DROP COLUMN IF EXISTS ubp_id;""")
                # cur.execute("""ALTER TABLE students ADD COLUMN IF NOT EXISTS program_id INTEGER;""")
                # cur.execute(
                #     """
                #     DO $$
                #     BEGIN
                #         IF NOT EXISTS (
                #             SELECT 1
                #             FROM   information_schema.table_constraints
                #             WHERE  constraint_name = 'fk_students_program'
                #             AND    table_name = 'students'
                #         ) THEN
                #             ALTER TABLE students
                #             ADD CONSTRAINT fk_students_program
                #             FOREIGN KEY (program_id)
                #             REFERENCES programs(id)
                #             ON DELETE SET NULL;
                #         END IF;
                #     END
                #     $$;
                #     """
                # )

                # # Session metadata table (NEW) - links to existing interview_data
                # cur.execute("""
                #     CREATE TABLE IF NOT EXISTS session_metadata (
                #         id SERIAL PRIMARY KEY,
                #         session_id VARCHAR(255) UNIQUE NOT NULL,
                #         student_id INTEGER REFERENCES students(student_id) ON DELETE SET NULL,
                #         student_name VARCHAR(255),
                #         status VARCHAR(50) DEFAULT 'active',
                #         overall_score FLOAT,
                #         technical_score FLOAT,
                #         communication_score FLOAT,
                #         attitude_score FLOAT,
                #         started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                #         completed_at TIMESTAMP,
                #         duration_minutes INTEGER,
                #         feedback_generated BOOLEAN DEFAULT FALSE,
                #         feedback_status VARCHAR(30) DEFAULT 'not_requested',
                #         feedback_error TEXT,
                #         feedback_requested_at TIMESTAMP,
                #         feedback_ready_at TIMESTAMP
                #     );
                # """)

                # cur.execute("""
                #     ALTER TABLE session_metadata
                #         ADD COLUMN IF NOT EXISTS feedback_status VARCHAR(30) DEFAULT 'not_requested'
                # """)

                # cur.execute("""
                #     ALTER TABLE session_metadata
                #         ADD COLUMN IF NOT EXISTS feedback_error TEXT
                # """)

                # cur.execute("""
                #     ALTER TABLE session_metadata
                #         ADD COLUMN IF NOT EXISTS feedback_requested_at TIMESTAMP
                # """)

                # cur.execute("""
                #     ALTER TABLE session_metadata
                #         ADD COLUMN IF NOT EXISTS feedback_ready_at TIMESTAMP
                # """)

                # cur.execute("""
                #     CREATE TABLE IF NOT EXISTS password_reset_tokens (
                #         id SERIAL PRIMARY KEY,
                #         email VARCHAR(255) NOT NULL,
                #         token VARCHAR(255) NOT NULL,
                #         expires_at TIMESTAMP NOT NULL,
                #         used BOOLEAN DEFAULT FALSE,
                #         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                #         used_at TIMESTAMP
                #     );
                # """)

                # cur.execute("""
                #     CREATE TABLE IF NOT EXISTS session_ratings (
                #         id SERIAL PRIMARY KEY,
                #         session_id VARCHAR(255) REFERENCES session_metadata(session_id) ON DELETE CASCADE,
                #         student_id INTEGER REFERENCES students(student_id) ON DELETE SET NULL,
                #         rating SMALLINT NOT NULL CHECK (rating BETWEEN 1 AND 5),
                #         comments TEXT,
                #         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                #         UNIQUE(session_id, student_id)
                #     );
                # """)

                # # DON'T modify existing interview_data table
                # # DON'T modify existing interview_questions table

                # # Create indexes for new tables only
                # cur.execute("""
                #     CREATE INDEX IF NOT EXISTS idx_students_email ON students(email);
                #     CREATE INDEX IF NOT EXISTS idx_session_meta_session_id ON session_metadata(session_id);
                #     CREATE INDEX IF NOT EXISTS idx_session_meta_student ON session_metadata(student_id);
                #     CREATE INDEX IF NOT EXISTS idx_session_meta_status ON session_metadata(status);
                #     CREATE INDEX IF NOT EXISTS idx_password_reset_email_token ON password_reset_tokens(email, token);
                # """)
                
                # Ensure existing tables have required columns for resume functionality
                _ensure_interview_data_columns(cur)
                _ensure_session_metadata_columns(cur)
                
                conn.commit()
                logger.info("Enhanced admin tables initialized successfully (existing tables untouched)")
                return True
                
            if not conn:
                return None
            
            with conn.cursor() as cur:
                # Check if student exists by email (preferred) or name fallback
                student_id = None
                lookup_email = student_email if student_email else None

                if lookup_email:
                    cur.execute("SELECT student_id FROM students WHERE email = %s", (lookup_email,))
                    student_row = cur.fetchone()
                    if student_row:
                        student_id = student_row[0]

                if student_id is None:
                    cur.execute("SELECT student_id FROM students WHERE name = %s", (student_name,))
                    student_row = cur.fetchone()
                    if student_row:
                        student_id = student_row[0]

                # Create student if not found
                if student_id is None:
                    generated_email = lookup_email or f"{student_name.lower().replace(' ', '.')}@temp.com"
                    cur.execute("""
                        INSERT INTO students (name, email) 
                        VALUES (%s, %s) 
                        RETURNING student_id
                    """, (student_name, generated_email))
                    student_id = cur.fetchone()[0]
                
                # Create session metadata (NEW table)
                cur.execute("""
                    INSERT INTO session_metadata 
                    (session_id, student_id, student_name)
                    VALUES (%s, %s, %s)
                """, (session_id, student_id, student_name))
                
                # Don't create interview_data entry here - let the compatibility endpoint handle it
                # This avoids duplicate key conflicts
                
                conn.commit()
                return session_id
                
    except Exception as e:
        logger.error(f"Error creating enhanced session: {e}")
        return None

def _resolve_ubp_id(university_name: Optional[str], program_name: Optional[str], batch_label: Optional[str]) -> Optional[int]:
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
                    (university_name, program_name, batch_label)
                )
                row = cur.fetchone()
                return row[0] if row else None
    except Exception as exc:
        logger.error(f"Error resolving UBP id: {exc}")
    return None


def _program_metadata_by_ubp(ubp_id: int) -> Optional[Dict[str, Any]]:
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
                    (ubp_id,)
                )
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "program_name": row[0],
                    "batch_label": row[1],
                    "university_name": row[2],
                }
    except Exception as exc:
        logger.error(f"Error fetching program metadata by UBP {ubp_id}: {exc}")
    return None


def _fetch_program_info(program_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve program details treating program_id as ubp_id.
    1) Resolve program_name via university_batch_program by ubp_id.
    2) Fallback to legacy programs.id lookup if not found.
    3) Fetch job_roles by program_name from programs.
    """
    if not program_id:
        return None

    try:
        # Step 1: Try UBP
        program_meta = _program_metadata_by_ubp(program_id)
        program_name = program_meta.get("program_name") if program_meta else None
        batch_label = program_meta.get("batch_label") if program_meta else None
        university_name = program_meta.get("university_name") if program_meta else None

        with db_pool.get_connection() as conn:
            if not conn:
                return None

            with conn.cursor() as cur:
                if not program_name:
                    # Step 2: Legacy fallback
                    cur.execute(
                        """
                        SELECT program_name
                        FROM programs
                        WHERE id = %s
                        """,
                        (program_id,)
                    )
                    row = cur.fetchone()
                    if not row:
                        return None
                    program_name = row[0]

                # Step 3: Roles by program_name
                cur.execute(
                    """
                    SELECT job_role
                    FROM programs
                    WHERE program_name = %s
                    ORDER BY job_role
                    """,
                    (program_name,)
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


def _fetch_all_programs() -> List[Dict[str, Any]]:
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
                    {
                        "program_id": row[0],
                        "program_name": row[1],
                        "job_roles": [role for role in row[2] if role],
                    }
                    for row in rows
                    if row
                ]
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching programs: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch programs") from exc


@app.get("/programs")
async def list_programs() -> List[Dict[str, Any]]:
    return _fetch_all_programs()


@app.get("/programs/{program_id}/job_roles")
async def list_program_job_roles(program_id: int) -> Dict[str, Any]:
    program_info = _fetch_program_info(program_id)
    if not program_info:
        raise HTTPException(status_code=404, detail="Program not found")
    return program_info


@app.get("/ubp/universities")
async def list_universities() -> List[str]:
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


@app.get("/ubp/programs")
async def list_programs_by_university(university: str = Query(..., alias="university_name")) -> List[str]:
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
                    (university,)
                )
                return [r[0] for r in cur.fetchall() if r and r[0]]
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching programs for university {university}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch programs") from exc


@app.get("/ubp/batches")
async def list_batches(university: str = Query(..., alias="university_name"), program: str = Query(..., alias="program_name")) -> List[str]:
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
                    (university, program)
                )
                return [r[0] for r in cur.fetchall() if r and r[0]]
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching batches for {university} / {program}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch batches") from exc


@app.get("/ubp/resolve")
async def resolve_ubp(university: str = Query(..., alias="university_name"), program: str = Query(..., alias="program_name"), batch: str = Query(..., alias="batch_label")) -> Dict[str, Any]:
    ubp_id = _resolve_ubp_id(university.strip(), program.strip(), batch.strip())
    if not ubp_id:
        raise HTTPException(status_code=404, detail="UBP combination not found")
    return {"ubp_id": ubp_id}


@app.get("/interview-options/industries")
async def list_interview_industries() -> List[str]:
    """Return distinct industries from interview_questions."""
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection unavailable")
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT industry
                    FROM interview_questions
                    WHERE industry IS NOT NULL AND TRIM(industry) <> ''
                    ORDER BY industry
                    """
                )
                return [row[0] for row in cur.fetchall() if row and row[0]]
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching industries: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch industries") from exc


@app.get("/interview-options/companies")
async def list_companies_by_industry(industry: str = Query(...)) -> List[str]:
    """Return distinct company names for a given industry from interview_questions."""
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection unavailable")
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT company
                    FROM interview_questions
                    WHERE LOWER(TRIM(industry)) = LOWER(TRIM(%s))
                      AND company IS NOT NULL AND TRIM(company) <> ''
                    ORDER BY company
                    """,
                    (industry,)
                )
                companies = [row[0] for row in cur.fetchall() if row and row[0]]
                if not companies:
                    raise HTTPException(status_code=404, detail="No companies found for industry")
                return companies
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching companies for industry {industry}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch companies") from exc


@app.get("/interview-options/interview-types")
async def list_interview_types(industry: str = Query(...), company: str = Query(...)) -> List[str]:
    """Return distinct interview types for the selected industry/company."""
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection unavailable")
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT interview_type
                    FROM interview_questions
                    WHERE LOWER(TRIM(industry)) = LOWER(TRIM(%s))
                      AND LOWER(TRIM(company)) = LOWER(TRIM(%s))
                      AND interview_type IS NOT NULL AND TRIM(interview_type) <> ''
                    ORDER BY interview_type
                    """,
                    (industry, company)
                )
                interview_types = [row[0] for row in cur.fetchall() if row and row[0]]
                if not interview_types:
                    raise HTTPException(status_code=404, detail="No interview types found for selection")
                return interview_types
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching interview types for {industry}/{company}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch interview types") from exc


@app.get("/interview-options/work-experience")
async def list_work_experience_levels(industry: str = Query(...), company: str = Query(...), interview_type: str = Query(...)) -> List[str]:
    """Return distinct work experience levels for the selected context."""
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection unavailable")
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT work_experience
                    FROM interview_questions
                    WHERE LOWER(TRIM(industry)) = LOWER(TRIM(%s))
                      AND LOWER(TRIM(company)) = LOWER(TRIM(%s))
                      AND LOWER(TRIM(interview_type)) = LOWER(TRIM(%s))
                      AND work_experience IS NOT NULL AND TRIM(work_experience) <> ''
                    ORDER BY work_experience
                    """,
                    (industry, company, interview_type)
                )
                work_levels = [row[0] for row in cur.fetchall() if row and row[0]]
                if not work_levels:
                    raise HTTPException(status_code=404, detail="No work experience levels found for selection")
                return work_levels
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching work experience for {industry}/{company}/{interview_type}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch work experience levels") from exc


@app.get("/interview-options/job-roles")
async def list_job_roles_for_selection(
    industry: str = Query(...),
    company: str = Query(...),
    interview_type: str = Query(...),
    work_experience: str = Query(...),
    program_name: Optional[str] = Query(None)
) -> List[str]:
    """Return distinct job roles filtered by context, optionally constrained by program name via programs table."""
    try:
        normalized_program = (program_name or '').strip()
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection unavailable")
            with conn.cursor() as cur:
                # Base roles from interview_questions
                cur.execute(
                    """
                    SELECT DISTINCT role
                    FROM interview_questions
                    WHERE LOWER(TRIM(industry)) = LOWER(TRIM(%s))
                      AND LOWER(TRIM(company)) = LOWER(TRIM(%s))
                      AND LOWER(TRIM(interview_type)) = LOWER(TRIM(%s))
                      AND LOWER(TRIM(work_experience)) = LOWER(TRIM(%s))
                      AND role IS NOT NULL AND TRIM(role) <> ''
                    ORDER BY role
                    """,
                    (industry, company, interview_type, work_experience)
                )
                question_roles = [row[0] for row in cur.fetchall() if row and row[0]]

                if not normalized_program:
                    return question_roles

                # Optional alignment with programs table job roles (program_name override)
                cur.execute(
                    """
                    SELECT DISTINCT job_role
                    FROM programs
                    WHERE LOWER(TRIM(program_name)) = LOWER(TRIM(%s))
                      AND job_role IS NOT NULL AND TRIM(job_role) <> ''
                    ORDER BY job_role
                    """,
                    (normalized_program,)
                )
                program_roles = {row[0] for row in cur.fetchall() if row and row[0]}

                if program_roles:
                    return [role for role in question_roles if role in program_roles] or list(program_roles)
                return question_roles
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Error fetching job roles for %s/%s/%s/%s (program=%s): %s",
            industry,
            company,
            interview_type,
            work_experience,
            program_name,
            exc,
        )
        raise HTTPException(status_code=500, detail="Failed to fetch job roles") from exc


@app.get("/companies/trending")
async def list_trending_companies() -> List[Dict[str, Any]]:
    """Return active companies for the homepage trending-companies section."""
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection unavailable")
            with conn.cursor() as cur:
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
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching trending companies: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch trending companies") from exc


@app.get("/job-roles/by-work-experience")
async def list_job_roles_by_work_experience(
    work_experience: str,
    program_name: Optional[str] = Query(None, alias="program_name"),
) -> List[str]:
    """Fetch job roles filtered by work experience (and optional program) from the programs table"""
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection unavailable")
            with conn.cursor() as cur:
                filters = ["LOWER(TRIM(work_experience)) = LOWER(TRIM(%s))", "job_role IS NOT NULL", "TRIM(job_role) <> ''"]
                params: List[Any] = [work_experience]

                if program_name and program_name.strip():
                    filters.append("LOWER(TRIM(program_name)) = LOWER(TRIM(%s))")
                    params.append(program_name)

                where_clause = " AND ".join(filters)
                query = f"""
                    SELECT DISTINCT job_role
                    FROM programs
                    WHERE {where_clause}
                    ORDER BY job_role
                """

                cur.execute(query, tuple(params))
                job_roles = [row[0] for row in cur.fetchall()]
                logger.info(
                    "Fetched %d job roles for work_experience='%s'%s: %s",
                    len(job_roles),
                    work_experience,
                    f" and program_name='{program_name}'" if program_name else "",
                    job_roles,
                )
                return job_roles
    except Exception as exc:
        logger.error(f"Error fetching job roles by work experience '{work_experience}': {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch job roles") from exc


def _fetch_distinct_question_metadata(column: str) -> List[str]:
    allowed_columns = {
        "interview_type": "interview_type",
        "work_experience": "work_experience",
        "question_type": "question_type",
    }

    column_name = allowed_columns.get(column)
    if not column_name:
        raise ValueError("Unsupported interview metadata column")

    query = f"""
        SELECT DISTINCT {column_name}
        FROM interview_questions
        WHERE {column_name} IS NOT NULL AND TRIM({column_name}) <> ''
        ORDER BY {column_name}
    """

    with db_pool.get_connection() as conn:
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection unavailable")

        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            return [row[0] for row in rows if row and row[0]]


@app.get("/metadata/interview-types")
async def get_public_interview_types() -> List[str]:
    try:
        return _fetch_distinct_question_metadata("interview_type")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching interview types: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch interview types") from exc


@app.get("/metadata/work-experience-levels")
async def get_public_work_experience_levels() -> List[str]:
    try:
        return _fetch_distinct_question_metadata("work_experience")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching work experience levels: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch work experience levels") from exc


@app.get("/metadata/question-types")
async def get_public_question_types() -> List[str]:
    try:
        return _fetch_distinct_question_metadata("question_type")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching question types: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch question types") from exc


def _resolve_program_id_by_name(program_name: str) -> Optional[int]:
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
                    (program_name,)
                )
                row = cur.fetchone()
                return row[0] if row and row[0] else None
    except Exception as exc:
        logger.error(f"Error resolving program id for {program_name}: {exc}")
    return None

OPEN_ENDPOINTS: set[str] = {
    "/",
    "/health",
    "/students/register",
    "/ubp/universities",
    "/ubp/programs",
    "/ubp/batches",
    "/ubp/resolve",
    "/interview-options/industries",
    "/interview-options/companies",
    "/interview-options/interview-types",
    "/interview-options/work-experience",
    "/interview-options/job-roles",
    "/job-roles/by-work-experience",
    "/companies/trending",
    "/metadata/interview-types",
    "/metadata/work-experience-levels",
    "/metadata/question-types",
}

@app.post("/students/register")
async def register_student(payload: StudentRegistration):
    """Register or update a student and capture their program mapping."""
    try:
        # Resolve UBP id (ubp_id) as the canonical program_id to store on students table
        program_id = payload.program_id  # treat as ubp_id if present

        # Prefer UBP composite if provided
        if not program_id and (payload.university_name and payload.program_name and payload.batch_label):
            program_id = _resolve_ubp_id(
                payload.university_name.strip(),
                payload.program_name.strip(),
                payload.batch_label.strip(),
            )

        # Backward compatibility: if only program_name is provided, try legacy resolver (programs table id)
        if not program_id and payload.program_name:
            program_id = _resolve_program_id_by_name(payload.program_name.strip())

        if not program_id:
            raise HTTPException(status_code=422, detail="Unknown program specified")

        program_info = _fetch_program_info(program_id)
        if not program_info:
            raise HTTPException(status_code=422, detail="Unknown program specified")

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
                    (email,)
                )
                existing = cur.fetchone()

                if existing:
                    cur.execute(
                        """
                        UPDATE students
                        SET name = %s,
                            program_id = %s,
                            last_active = CURRENT_TIMESTAMP
                        WHERE student_id = %s
                        RETURNING student_id
                        """,
                        (name, program_id, existing[0])
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO students (name, email) 
                        VALUES (%s, %s) 
                        RETURNING student_id
                        """,
                        (name, email)
                    )

                student_id = cur.fetchone()[0]
                conn.commit()

        return {
            "student_id": student_id,
            "name": payload.student_name.strip(),
            "email": payload.student_email.strip(),
            "program": program_info,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error registering student: {exc}")
        raise HTTPException(status_code=500, detail="Failed to register student") from exc

@app.get("/students/profile/{email}")
async def get_student_profile(email: EmailStr):
    """Fetch a student's complete profile by email."""
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
                    (email,)
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Student not found")
                
                program_details = {}
                if row[3]:
                    program_info = _fetch_program_info(row[3])
                    if program_info:
                        program_details = program_info

                return {
                    "student_id": row[0],
                    "name": row[1],
                    "email": row[2],
                    "program_id": row[3],
                    **program_details
                }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching student profile for {email}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch profile")

@app.post("/students/login")
async def login_student(payload: StudentLoginRequest):
    """Authenticate a student and keep their program mapping in sync."""
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection unavailable")

            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT student_id, name, email, program_id, password
                    FROM students
                    WHERE LOWER(TRIM(email)) = LOWER(TRIM(%s))
                    """,
                    (payload.email,)
                )
                row = cur.fetchone()

                if not row:
                    raise HTTPException(status_code=401, detail="Invalid email or password")

                student_id, student_name, student_email, current_program_id, stored_password = row

                if (stored_password or "") != payload.password:
                    raise HTTPException(status_code=401, detail="Invalid email or password")

                final_program_id = current_program_id

                # Allow overriding program context (treated as ubp_id) on login
                if payload.program_id and payload.program_id != current_program_id:
                    cur.execute(
                        "UPDATE students SET program_id = %s WHERE student_id = %s",
                        (payload.program_id, student_id)
                    )
                    conn.commit()
                    final_program_id = payload.program_id

                program_details = {}
                if final_program_id:
                    program_info = _fetch_program_info(final_program_id)
                    if program_info:
                        program_details = program_info

                return {
                    "student_id": student_id,
                    "name": student_name,
                    "email": student_email,
                    "program_id": final_program_id,
                    **program_details,
                }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error logging in student {payload.email}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to log in student") from exc

def _send_password_reset_email(email: str, token: str, expires_at: datetime) -> None:
    smtp_cfg = config.SMTP_SETTINGS
    sender = config.EMAIL_SENDER

    if not (smtp_cfg.get("host") and smtp_cfg.get("port") and sender):
        logger.error("SMTP configuration incomplete; password reset email cannot be sent")
        raise HTTPException(status_code=500, detail="Password reset email service not configured")

    base_url = config.PASSWORD_RESET_URL_BASE
    if not base_url:
        base_url = "/login"

    reset_url = base_url.rstrip('/')
    reset_url = f"{reset_url}?email={quote_plus(email)}&token={quote_plus(token)}"

    token_inline = token
    token_html = token

    subject = "Password reset instructions"
    expires_dt = expires_at
    if expires_dt.tzinfo is None:
        expires_dt = expires_dt.replace(tzinfo=timezone.utc)
    expires_ist = expires_dt.astimezone(IST_TZ)
    expires_str = expires_ist.strftime('%Y-%m-%d %H:%M:%S IST')
    plain_body = textwrap.dedent(
        f"""
        Hello,

        You recently requested to reset your password for the AI Mock Interview platform.

        Your one-time reset token is: {token_inline}
        This token expires at {expires_str}.

        Copy the token above and enter it on the password reset screen,
        or open this link in your browser: {reset_url}

        If you did not request this change, please ignore this email.

        Thank you.
        """
    ).strip()

    html_body = f"""
    <html>
      <body style="margin:0;padding:0;background:#f5f5ff;font-family:'Segoe UI',Arial,sans-serif;color:#1f1b3a;">
        <table role="presentation" cellpadding="0" cellspacing="0" width="100%">
          <tr>
            <td align="center" style="padding:32px 16px;">
              <table width="480" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:16px;box-shadow:0 18px 48px rgba(31,27,58,0.12);overflow:hidden;">
                <tr>
                  <td style="background:linear-gradient(135deg,#6c63ff,#5d40f5);padding:28px 32px;color:#ffffff;">
                    <h1 style="margin:0;font-size:24px;font-weight:700;">Reset your AI Mock Interview password</h1>
                    <p style="margin:8px 0 0;font-size:14px;opacity:0.9;">Use the secure token below to finish resetting.</p>
                  </td>
                </tr>
                <tr>
                  <td style="padding:32px;">
                    <p style="margin:0 0 16px;font-size:16px;">Hello,</p>
                    <p style="margin:0 0 16px;font-size:15px;line-height:1.6;">We received a request to reset the password for your AI Mock Interview account. Use the details below to continue.</p>
                    <table role="presentation" cellpadding="0" cellspacing="0" style="width:100%;margin:0 0 20px;border-collapse:separate;border-spacing:0 6px;">
                      <tr>
                        <td style="padding:8px 12px;background:#f5f3ff;font-weight:600;border-radius:10px 0 0 10px;white-space:nowrap;">Reset token</td>
                        <td style="padding:6px 10px;border:1px solid #ece8ff;border-left:none;border-radius:0 10px 10px 0;font-size:11px;font-family:'Roboto Mono','Courier New',Courier,monospace;letter-spacing:0;white-space:nowrap;">
                          <span style="display:inline-block;white-space:nowrap;word-break:keep-all;overflow-wrap:normal;line-height:1;">{token_html}</span>
                        </td>
                      </tr>
                      <tr>
                        <td style="padding:8px 12px;background:#f5f3ff;font-weight:600;border-radius:10px 0 0 10px;white-space:nowrap;">Expires</td>
                        <td style="padding:8px 12px;border:1px solid #ece8ff;border-left:none;border-radius:0 10px 10px 0;font-size:14px;white-space:nowrap;">{expires_str}</td>
                      </tr>
                    </table>
                    <p style="margin:0 0 20px;font-size:15px;line-height:1.6;">Copy the token above and paste it into the password reset screen, or click the button below to jump straight there.</p>
                    <p style="margin:0 0 28px;text-align:center;">
                      <a href="{reset_url}" style="display:inline-block;background:#6c63ff;color:#ffffff;text-decoration:none;padding:12px 24px;border-radius:999px;font-weight:600;font-size:15px;">Open reset page</a>
                    </p>
                    <p style="margin:24px 0 0;font-size:15px;line-height:1.6;">Thank you</p>
                  </td>
                </tr>
              </table>
              <p style="margin:24px 0 0;font-size:12px;color:#7a7794;">If the button doesn't work, copy and paste this link into your browser:<br/><span style="word-break:break-all;">{reset_url}</span></p>
            </td>
          </tr>
        </table>
      </body>
    </html>
    """

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = email
    message.set_content(plain_body)
    message.add_alternative(html_body, subtype="html")

    try:
        if smtp_cfg.get("use_tls", True):
            context = ssl.create_default_context()
            with smtplib.SMTP(smtp_cfg["host"], smtp_cfg["port"]) as server:
                server.starttls(context=context)
                if smtp_cfg.get("username") and smtp_cfg.get("password"):
                    server.login(smtp_cfg["username"], smtp_cfg["password"])
                server.send_message(message)
        else:
            with smtplib.SMTP(smtp_cfg["host"], smtp_cfg["port"]) as server:
                if smtp_cfg.get("username") and smtp_cfg.get("password"):
                    server.login(smtp_cfg["username"], smtp_cfg["password"])
                server.send_message(message)
    except Exception as exc:
        logger.error(f"Failed to send password reset email to {email}: {exc}")
        raise HTTPException(status_code=500, detail="Unable to send password reset email") from exc


def _create_password_reset_token(email: str) -> Dict[str, Any]:
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

    with db_pool.get_connection() as conn:
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection unavailable")

        with conn.cursor() as cur:
            cur.execute(
                "SELECT student_id FROM students WHERE LOWER(TRIM(email)) = LOWER(TRIM(%s))",
                (email.strip(),)
            )
            student_row = cur.fetchone()
            if not student_row:
                raise HTTPException(status_code=404, detail="No account found for this email")

            cur.execute(
                """
                INSERT INTO password_reset_tokens (email, token, expires_at)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (email.strip(), token, expires_at)
            )
            conn.commit()

    _send_password_reset_email(email.strip(), token, expires_at)

    return {
        "token": token,
        "expires_at": expires_at.isoformat()
    }


def _generate_temp_password(length: int = 10) -> str:
    # Use urlsafe token and trim to requested length
    token = secrets.token_urlsafe(max(8, length))
    return token[:length]


def _send_credentials_email(name: str, email: str, temp_password: str) -> None:
    smtp_cfg = config.SMTP_SETTINGS
    sender = config.EMAIL_SENDER

    if not (smtp_cfg.get("host") and smtp_cfg.get("port") and sender):
        logger.error("SMTP configuration incomplete; credential email cannot be sent")
        raise HTTPException(status_code=500, detail="Email service not configured")

    login_url = config.PASSWORD_RESET_URL_BASE or "/login"
    reset_hint = (
        "If you wish to set your own password, visit the login page and choose 'Forgot password?'."
    )

    subject = "Your AI Mock Interview login credentials"
    plain_body = textwrap.dedent(
        f"""
        Hello {name or 'there'},

        You have been registered for the AI Mock Interview platform.

        Email: {email}
        Temporary password: {temp_password}
        Login here: {login_url}

        {reset_hint}
        After logging in, please reset your password for security.

        Thank you
        """
    ).strip()

    html_body = f"""
    <html>
      <body style="margin:0;padding:0;background:#f5f5ff;font-family:'Segoe UI',Arial,sans-serif;color:#1f1b3a;">
        <table role="presentation" cellpadding="0" cellspacing="0" width="100%">
          <tr>
            <td align="center" style="padding:32px 16px;">
              <table width="480" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:16px;box-shadow:0 18px 48px rgba(31,27,58,0.12);overflow:hidden;">
                <tr>
                  <td style="background:linear-gradient(135deg,#6c63ff,#5d40f5);padding:28px 32px;color:#ffffff;">
                    <h1 style="margin:0;font-size:24px;font-weight:700;">Welcome to AI Mock Interview Platform</h1>
                    <p style="margin:8px 0 0;font-size:14px;opacity:0.9;">Your interview practice space is ready.</p>
                  </td>
                </tr>
                <tr>
                  <td style="padding:32px;">
                    <p style="margin:0 0 16px;font-size:16px;">Hi {name or 'there'},</p>
                    <p style="margin:0 0 16px;font-size:15px;line-height:1.6;">
                      You're now registered for the AI Mock Interview platform. Use the credentials below to sign in and start practising.
                    </p>
                    <table role="presentation" cellpadding="0" cellspacing="0" style="width:100%;margin:0 0 10px;border-collapse:separate;border-spacing:0 4px;">
                      <tr>
                        <td style="padding:8px 12px;background:#f5f3ff;font-weight:600;border-radius:10px 0 0 10px;width:160px;">Email</td>
                        <td style="padding:8px 12px;border:1px solid #ece8ff;border-left:none;border-radius:0 10px 10px 0;font-size:15px;">{email}</td>
                      </tr>
                      <tr>
                        <td style="padding:8px 12px;background:#f5f3ff;font-weight:600;border-radius:10px 0 0 10px;white-space:nowrap;">Password</td>
                        <td style="padding:8px 12px;border:1px solid #ece8ff;border-left:none;border-radius:0 10px 10px 0;font-size:15px;letter-spacing:0.6px;">{temp_password}</td>
                      </tr>
                    </table>
                    <p style="margin:0 0 20px;font-size:15px;line-height:1.6;">Click below to launch the app.</p>
                    <p style="margin:0 0 28px;text-align:center;">
                      <a href="{login_url}" style="display:inline-block;background:#6c63ff;color:#ffffff;text-decoration:none;padding:12px 24px;border-radius:999px;font-weight:600;font-size:15px;">Go to login</a>
                    </p>
                    <p style="margin:0 0 16px;font-size:15px;line-height:1.6;">{reset_hint}</p>
                    <p style="margin:24px 0 0;font-size:15px;line-height:1.6;">Thank you</p>
                  </td>
                </tr>
              </table>
              <p style="margin:24px 0 0;font-size:12px;color:#7a7794;">If the button doesn't work, copy and paste this link into your browser:<br/><span style="word-break:break-all;">{login_url}</span></p>
            </td>
          </tr>
        </table>
      </body>
    </html>
    """

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = email
    message.set_content(plain_body)
    message.add_alternative(html_body, subtype="html")

    try:
        if smtp_cfg.get("use_tls", True):
            context = ssl.create_default_context()
            with smtplib.SMTP(smtp_cfg["host"], smtp_cfg["port"]) as server:
                server.starttls(context=context)
                if smtp_cfg.get("username") and smtp_cfg.get("password"):
                    server.login(smtp_cfg["username"], smtp_cfg["password"])
                server.send_message(message)
        else:
            with smtplib.SMTP(smtp_cfg["host"], smtp_cfg["port"]) as server:
                if smtp_cfg.get("username") and smtp_cfg.get("password"):
                    server.login(smtp_cfg["username"], smtp_cfg["password"])
                server.send_message(message)
    except Exception as exc:
        logger.error(f"Failed to send credentials email to {email}: {exc}")
        raise HTTPException(status_code=500, detail="Unable to send credential email") from exc

def _update_student_password(email: str, new_password: str) -> None:
    with db_pool.get_connection() as conn:
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection unavailable")

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE students
                SET password = %s, last_active = CURRENT_TIMESTAMP
                WHERE LOWER(TRIM(email)) = LOWER(TRIM(%s))
                RETURNING student_id
                """,
                (new_password, email)
            )
            result = cur.fetchone()
            if not result:
                raise HTTPException(status_code=404, detail="No account found for this email")
            conn.commit()

def _validate_password_reset_token(email: str, token: str) -> int:
    with db_pool.get_connection() as conn:
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection unavailable")

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, expires_at, used
                FROM password_reset_tokens
                WHERE LOWER(TRIM(email)) = LOWER(TRIM(%s))
                  AND token = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (email, token)
            )
            row = cur.fetchone()

            if not row:
                raise HTTPException(status_code=400, detail="Invalid or expired token")

            token_id, expires_at, used = row
            if used:
                raise HTTPException(status_code=400, detail="This reset link has already been used")

            now_utc = datetime.now(timezone.utc)

            if not isinstance(expires_at, datetime):
                raise HTTPException(status_code=500, detail="Invalid token expiry timestamp")

            if expires_at.tzinfo is None:
                expires_utc = expires_at.replace(tzinfo=timezone.utc)
                if expires_utc < now_utc:
                    try:
                        expires_ist = expires_at.replace(tzinfo=IST_TZ).astimezone(timezone.utc)
                    except Exception:
                        expires_ist = expires_utc
                    if expires_ist >= now_utc:
                        expires_utc = expires_ist
                expires_at_utc = expires_utc
            else:
                expires_at_utc = expires_at.astimezone(timezone.utc)

            if expires_at_utc < now_utc:
                raise HTTPException(status_code=400, detail="This reset link has expired")

            return token_id

def _mark_token_used(token_id: int) -> None:
    with db_pool.get_connection() as conn:
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection unavailable")

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE password_reset_tokens
                SET used = TRUE, used_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (token_id,)
            )
            conn.commit()

@app.post("/students/password/forgot")
async def request_password_reset(payload: PasswordResetRequest):
    try:
        token_info = _create_password_reset_token(payload.email.strip())

        logger.info(
            "Password reset requested for %s, token expires at %s",
            payload.email,
            token_info["expires_at"],
        )

        return {
            "message": "Password reset instructions have been emailed if the account exists.",
            "expires_at": token_info["expires_at"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error generating password reset token: {exc}")
        raise HTTPException(status_code=500, detail="Failed to generate password reset token")

@app.post("/students/password/reset")
async def reset_password(payload: PasswordResetConfirm):
    try:
        if len(payload.new_password.strip()) < 6:
            raise HTTPException(status_code=422, detail="Password must be at least 6 characters long")

        token_id = _validate_password_reset_token(payload.email.strip(), payload.token.strip())

        _update_student_password(payload.email.strip(), payload.new_password.strip())

        _mark_token_used(token_id)

        return {"message": "Password updated successfully"}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error resetting password for {payload.email}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to reset password")


@app.post("/students/upload-resume")
async def upload_student_resume(
    file: UploadFile = File(...),
    student_id: int = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Upload and parse student resume"""
    
    if not file.filename.lower().endswith(('.pdf', '.doc', '.docx')):
        raise HTTPException(status_code=400, detail="Only PDF, DOC, and DOCX files are allowed")
    
    try:
        # 1. Save file
        file_extension = file.filename.split('.')[-1].lower()
        unique_filename = f"{student_id}_{int(time.time())}.{file_extension}"
        file_path = f"uploads/resumes/{unique_filename}"
        
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads/resumes", exist_ok=True)
        
        # Save file to disk
        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        # 2. Store in database
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO student_resumes 
                    (student_id, filename, file_path, file_size)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (student_id, file.filename, file_path, len(content)))
                
                resume_id = cur.fetchone()[0]
                conn.commit()
        
        # 3. Parse resume in background
        background_tasks.add_task(parse_resume_background, resume_id, file_path)
        
        return {
            "message": "Resume uploaded successfully",
            "resume_id": resume_id,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Resume upload failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload resume")


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_text_from_docx(docx_path: str) -> str:
    document = docx.Document(docx_path)
    return "\n".join([para.text for para in document.paragraphs if para.text.strip()])


def extract_text_from_resume(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    if ext == ".docx":
        return extract_text_from_docx(file_path)
    raise ValueError("Unsupported file format. Use PDF or DOCX.")


def structure_resume_text(raw_text: str) -> Dict[str, List[str]]:
    sections = {
        "summary": ["summary", "profile", "objective"],
        "skills": ["skills", "technical skills", "core competencies"],
        "experience": ["experience", "work experience", "professional experience"],
        "education": ["education", "academic"],
        "projects": ["projects"],
        "certifications": ["certifications", "certificates"],
        "achievements": ["achievements", "awards", "recognitions"],
        "languages": ["languages", "language skills"],
        "interests": ["interests", "hobbies"],
        "references": ["references", "reference"],
        "contact": ["contact", "contact information"],
        "others": [],
    }

    structured_data = defaultdict(list)
    current_section = "others"

    for line in raw_text.splitlines():
        clean_line = line.strip()
        if not clean_line:
            continue

        matched = False
        for section, keywords in sections.items():
            for keyword in keywords:
                if re.fullmatch(keyword, clean_line.lower()):
                    current_section = section
                    matched = True
                    break
            if matched:
                break

        if not matched:
            structured_data[current_section].append(clean_line)

    return dict(structured_data)


async def parse_resume_background(resume_id: int, file_path: str):
    """Background task to parse resume"""
    try:
        raw_text = extract_text_from_resume(file_path)
        structured_data = structure_resume_text(raw_text)
        parsed_data = {
            "raw_text": raw_text,
            "structured_sections": structured_data,
        }

        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                        UPDATE student_resumes 
                        SET parsed_data = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """,
                    (json.dumps(parsed_data), resume_id),
                )
                conn.commit()

    except Exception as e:
        logger.error(f"Resume parsing failed for resume_id {resume_id}: {e}")


async def parse_job_description_background(job_description_id: int, file_path: str):
    """Background task to parse job description files into text"""
    try:
        raw_text = extract_text_from_resume(file_path)
        parsed_data = {"raw_text": raw_text}

        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                        UPDATE job_descriptions
                        SET parsed_data = %s,
                            job_desc = %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """,
                    (json.dumps(parsed_data), raw_text, job_description_id),
                )
                conn.commit()

    except Exception as e:
        logger.error(f"Job description parsing failed for id {job_description_id}: {e}")


@app.get("/students/{student_id}/resume")
async def get_student_resume(student_id: int):
    """Get student's active resume"""
    
    try:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, filename, file_path, file_size, upload_date, 
                           parsed_data, is_active, created_at, updated_at
                    FROM student_resumes 
                    WHERE student_id = %s AND is_active = true 
                    ORDER BY created_at DESC LIMIT 1
                """, (student_id,))
                
                resume = cur.fetchone()
                if not resume:
                    return {"message": "No resume found", "has_resume": False}
                
                return {
                    "resume_id": resume[0],
                    "filename": resume[1],
                    "file_path": resume[2],
                    "file_size": resume[3],
                    "upload_date": resume[4],
                    "parsed_data": resume[5],
                    "is_active": resume[6],
                    "created_at": resume[7],
                    "updated_at": resume[8],
                    "has_resume": True
                }
                
    except Exception as e:
        logger.error(f"Failed to get student resume: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve resume")


@app.post("/students/upload-job-description")
async def upload_job_description(
    file: UploadFile = File(...),
    student_id: int = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Upload and parse a job description file for tailored interviews"""

    if not file.filename.lower().endswith((".pdf", ".doc", ".docx", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF, DOC, DOCX, or TXT files are allowed")

    try:
        file_extension = file.filename.split('.')[-1].lower()
        unique_filename = f"{student_id}_{int(time.time())}_jd.{file_extension}"
        file_path = f"uploads/job_descriptions/{unique_filename}"

        os.makedirs("uploads/job_descriptions", exist_ok=True)

        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)

        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                _ensure_job_descriptions_table(cur)
                cur.execute(
                    """
                        UPDATE job_descriptions
                        SET is_active = FALSE
                        WHERE student_id = %s
                    """,
                    (student_id,),
                )

                cur.execute(
                    """
                        INSERT INTO job_descriptions (student_id, filename, file_path, file_size, is_active)
                        VALUES (%s, %s, %s, %s, TRUE)
                        RETURNING id
                    """,
                    (student_id, file.filename, file_path, len(content))
                )
                job_description_id = cur.fetchone()[0]
                conn.commit()

        background_tasks.add_task(parse_job_description_background, job_description_id, file_path)

        return {
            "message": "Job description uploaded successfully",
            "job_description_id": job_description_id,
            "status": "processing"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job description upload failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload job description")


@app.get("/students/{student_id}/job-description")
async def get_student_job_description(student_id: int):
    """Fetch the latest active job description for a student"""

    try:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                        SELECT id, filename, file_path, file_size, upload_date,
                               parsed_data, is_active, created_at, updated_at, job_desc
                        FROM job_descriptions
                        WHERE student_id = %s AND is_active = TRUE
                        ORDER BY created_at DESC
                        LIMIT 1
                    """,
                    (student_id,),
                )
                job_description = cur.fetchone()
                if not job_description:
                    return {"message": "No job description found", "has_job_description": False}

                return {
                    "job_description_id": job_description[0],
                    "filename": job_description[1],
                    "file_path": job_description[2],
                    "file_size": job_description[3],
                    "upload_date": job_description[4],
                    "parsed_data": job_description[5],
                    "is_active": job_description[6],
                    "created_at": job_description[7],
                    "updated_at": job_description[8],
                    "job_desc": job_description[9],
                    "has_job_description": True,
                }

    except Exception as e:
        logger.error(f"Failed to get student job description: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve job description")


@app.get("/students/{student_id}/resume-interview-sessions")
async def get_student_resume_interview_sessions(student_id: int):
    """Fetch all resume-based interview sessions with their pre-generated questions for a student"""
    try:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                # Get all unique sessions for this student that have pre-generated questions
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
                            sm.overall_score,
                            sm.rubric_scores
                        FROM pre_generated_questions pgq
                        LEFT JOIN session_metadata sm ON pgq.session_id = sm.session_id
                        WHERE pgq.student_id = %s
                        GROUP BY pgq.session_id, pgq.company_name, pgq.job_role, 
                                 pgq.interview_type, pgq.work_experience,
                                 sm.started_at, sm.completed_at, sm.status, sm.overall_score,
                                 sm.rubric_scores
                        ORDER BY sm.started_at DESC NULLS LAST
                    """,
                    (student_id,),
                )
                sessions = cur.fetchall()
                
                if not sessions:
                    return {"sessions": [], "message": "No resume-based interview sessions found"}
                
                result = []
                for session in sessions:
                    session_id = session[0]
                    rubric_scores = session[9]
                    if isinstance(rubric_scores, str):
                        try:
                            rubric_scores = json.loads(rubric_scores)
                        except json.JSONDecodeError:
                            logger.warning("Invalid rubric_scores JSON for session %s", session_id)
                            rubric_scores = None

                    # Fetch all pre-generated questions for this session
                    cur.execute(
                        """
                            SELECT question_number, question, question_type, difficulty, mandatory_skills
                            FROM pre_generated_questions
                            WHERE session_id = %s
                            ORDER BY question_number
                        """,
                        (session_id,),
                    )
                    questions = cur.fetchall()
                    
                    result.append({
                        "session_id": session_id,
                        "company_name": session[1],
                        "job_role": session[2],
                        "interview_type": session[3],
                        "work_experience": session[4],
                        "started_at": session[5],
                        "completed_at": session[6],
                        "status": session[7],
                        "overall_score": session[8],
                        "rubric_scores": rubric_scores,
                        "questions": [
                            {
                                "question_number": q[0],
                                "question": q[1],
                                "question_type": q[2],
                                "difficulty": q[3],
                                "mandatory_skills": q[4]
                            }
                            for q in questions
                        ]
                    })
                
                return {"sessions": result}
                
    except Exception as e:
        logger.error(f"Failed to get resume interview sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve resume interview sessions")


@app.post("/api/generate-resume-questions")
async def generate_resume_based_questions(request: ResumeQuestionRequest):
    """Generate resume- or JD-aware questions using Gemini"""
    try:
        resume_data: Optional[dict] = None
        job_description_data: Optional[dict] = None
        job_description_text: Optional[str] = (request.job_description_text or "").strip() or None
        job_description_id_used: Optional[int] = request.job_description_id

        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                # Load resume context (required for now)
                if request.resume_id:
                    cur.execute(
                        """
                        SELECT parsed_data FROM student_resumes 
                        WHERE id = %s AND student_id = %s AND is_active = true
                        """,
                        (request.resume_id, request.student_id),
                    )
                else:
                    cur.execute(
                        """
                        SELECT parsed_data FROM student_resumes 
                        WHERE student_id = %s AND is_active = true
                        ORDER BY created_at DESC LIMIT 1
                        """,
                        (request.student_id,),
                    )

                resume_result = cur.fetchone()
                if not resume_result:
                    raise HTTPException(status_code=404, detail="No resume found for student")

                resume_data = resume_result[0]

                # Load job description context if provided/available
                jd_row: Optional[Tuple[int, Any, Optional[str]]] = None
                if request.job_description_id:
                    cur.execute(
                        """
                        SELECT id, parsed_data, job_desc
                        FROM job_descriptions
                        WHERE id = %s AND (student_id = %s OR student_id IS NULL)
                        LIMIT 1
                        """,
                        (request.job_description_id, request.student_id),
                    )
                    jd_row = cur.fetchone()
                elif job_description_text is None:
                    cur.execute(
                        """
                        SELECT id, parsed_data, job_desc
                        FROM job_descriptions
                        WHERE student_id = %s AND is_active = TRUE
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        (request.student_id,),
                    )
                    jd_row = cur.fetchone()

                if jd_row:
                    job_description_id_used = jd_row[0]
                    job_description_data = jd_row[1]
                    job_description_text = job_description_text or jd_row[2]

        if not (request.job_role or job_description_text or job_description_data):
            raise HTTPException(status_code=400, detail="Provide either a job role or a job description to tailor questions")

        questions = await generate_resume_questions_gemini(
            resume_data=resume_data,
            job_role=request.job_role,
            company=request.company_name,
            interview_type=request.interview_type,
            work_experience=request.work_experience,
            job_description_data=job_description_data,
            job_description_text=job_description_text,
        )

        if not questions:
            raise HTTPException(status_code=500, detail="Question generation returned empty result")

        return {
            "message": "Resume/JD-based questions generated successfully",
            "questions": questions,
            "resume_context": resume_data,
            "job_description_context": {
                "id": job_description_id_used,
                "parsed_data": job_description_data,
                "raw_text": job_description_text,
            },
            "job_description_id": job_description_id_used,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resume question generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate resume questions")


async def retrieve_company_context(company_name: str, query: str, top_k: int = 5) -> list[str]:
    """Embed the query and retrieve the top-k most relevant chunks for a company."""
    if not company_name:
        return []
    try:
        query_embedding = await embed_text(query)
        vector_literal = "[" + ",".join(str(value) for value in query_embedding) + "]"
        with db_pool.get_connection() as conn:
            if not conn:
                return []
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_text FROM company_playbook_chunks
                    WHERE LOWER(TRIM(company_name)) = LOWER(TRIM(%s))
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (company_name, vector_literal, top_k),
                )
                return [row[0] for row in cur.fetchall()]
    except Exception as exc:
        logger.warning(f"Company playbook retrieval failed for {company_name}: {exc}")
        return []


async def generate_resume_questions_gemini(
    *,
    resume_data: dict,
    job_role: Optional[str],
    company: str,
    interview_type: Optional[str],
    work_experience: Optional[str],
    job_description_data: Optional[dict],
    job_description_text: Optional[str],
    question_count: int = 10,
) -> List[Dict[str, Any]]:
    effective_role = (job_role or "the role described in the job description").strip()
    jd_text = (job_description_text or "").strip()
    if not jd_text and isinstance(job_description_data, dict):
        jd_text = (job_description_data.get("raw_text") or "").strip()

    jd_excerpt = (jd_text[:1200] + "...") if jd_text and len(jd_text) > 1200 else jd_text
    jd_json_snippet = json.dumps(job_description_data, ensure_ascii=False) if job_description_data else None

    company_context_chunks = []
    if company:
        company_context_chunks = await retrieve_company_context(
            company,
            f"{effective_role} interview questions and evaluation style",
        )

    prompt_sections = [
        f"You are an AI interview coach helping candidates prepare for a {effective_role} position at {company}.",
        f"Use the candidate's resume data, job description, and context below to craft {question_count} personalized interview questions.",
        "",
        "Resume JSON:",
        json.dumps(resume_data, ensure_ascii=False),
    ]

    if jd_excerpt:
        prompt_sections.extend([
            "",
            "Job Description Text:",
            jd_excerpt,
        ])

    if jd_json_snippet:
        prompt_sections.extend([
            "",
            "Job Description JSON:",
            jd_json_snippet,
        ])



    prompt_sections.extend(
    [
        "",
        "Context:",
        f"- Target Role: {effective_role}",
        f"- Company: {company}",
        f"- Interview Type: {interview_type or 'Not specified'}",
        f"- Work Experience Level: {work_experience or 'Not specified'}",
    ]
)

    if company_context_chunks:
        prompt_sections.extend(
            [
                "",
                "Company-Specific Interview Style (from internal knowledge base):",
                "Use the following company-specific interview context to shape question style, difficulty, and evaluation emphasis where relevant. Follow all the rules below, but adapt where the company's actual interview patterns differ from generic guidance.",
                "\n\n".join(company_context_chunks),
            ]
        )

    prompt_sections.extend(
    [
        "",
        "Critical Alignment Instructions:",
        "- Treat the candidate’s resume and the provided job description as the PRIMARY source of truth for question generation.",
        "- Every question MUST be explicitly grounded in at least one concrete item from the resume (skills, tools, projects, responsibilities, achievements) AND aligned with expectations from the job description.",
        "- Avoid generic, role-agnostic questions. Each question should feel custom-written for this exact candidate applying to this exact role.",
        "",
        "Resume-Driven Questioning Rules:",
        "- Prioritize technologies, frameworks, tools, and domains explicitly mentioned in the resume.",
        "- For candidates with projects listed, generate scenario-based questions that probe design decisions, trade-offs, scaling, failures, and optimizations from those projects.",
        "- If the resume shows depth in a particular area, increase the difficulty and depth of questions in that area.",
        "",
        "Job-Description-Driven Questioning Rules:",
        "- Map required and preferred skills from the job description directly to question topics.",
        "- If the job description emphasizes specific responsibilities (e.g., data pipelines, APIs, ML models, cloud, performance, security), ensure multiple questions directly test those responsibilities.",
        "- Mirror real interview expectations for the company and role described in the job description.",
        "",
        "Technical Question Generation Rules:",
        "- Coding (Python) questions MUST be based on problems the candidate is likely to face in the target role, using tools, data types, and problem patterns reflected in the resume and job description.",
        "- SQL questions MUST be framed around realistic datasets, tables, and business problems inferred from the candidate’s past experience and the job description domain.",
        "- System Design questions MUST be scoped to systems the candidate could realistically have worked on or will work on in this role (based on resume projects, tech stack, and job requirements).",
        "- Avoid abstract textbook-style technical questions; all technical questions should be contextual, applied, and role-specific.",
        "",
        "Requirements:",
        "- Make it highly relevant and aligned with both the resume and the job description so the questions feel like real, live interview questions asked by an interviewer.",
        "- Always return 10 total questions (fill remaining slots with the most relevant speech-based scenarios if a category lacks coverage).",
        "- If the interview type is technical ensure at least 3 speech-based, 1 coding (Python), 1 SQL, and add 1 system design question when work_experience exceeds 2 years. Use the remaining slots for the strongest mix of speech/coding/SQL/system design aligned to the role.",
        "- If the interview type is behavioral/HR prioritize speech-based questions but still output 10 unique prompts.",
        "- While asking coding questions do not involve questions that might require high level packages or tools since the system given to the user to code will not support importing them. The system supports only basic packages like numpy pandas scikit-learn",
        "- Each question must specify:",
        "    * question_type (types available: \"Coding (Python)\", \"Coding (SQL)\", \"Speech Based\", \"System Design\")",
        "    * question text (clear and concise, but context-aware)",
        "    * difficulty (Easy/Medium/Hard)",
        "    * mandatory_skills (array of at least minimum of two and maximum of 4 skills tied directly to resume and job description keywords)",
        "- Respond ONLY in strict JSON using this schema:",
        "{",
        '    "questions": [',
        '        {',
        '            "question_type": "...",',
        '            "question": "...",',
        '            "difficulty": "Easy|Medium|Hard",',
        '            "mandatory_skills": ["skill1", "skill2", ...]',
        '        }',
        '    ]',
        '}',
    ]
)
        
    # prompt_sections.extend(
    #     [
    #         "",
    #         "Context:",
    #         f"- Target Role: {effective_role}",
    #         f"- Company: {company}",
    #         f"- Interview Type: {interview_type or 'Not specified'}",
    #         f"- Work Experience Level: {work_experience or 'Not specified'}",
    #         "",
    #         "Requirements:",
    #         "- Make it highly relevant and aligned with both the resume and the job description so the questions feel like real, live interview questions asked by an interviewer.",
    #         "- Always return 10 total questions (fill remaining slots with the most relevant speech-based scenarios if a category lacks coverage).",
    #         "- If the interview type is technical ensure at least 3 speech-based, 1 coding (Python), 1 SQL, and add 1 system design question when work_experience exceeds 2 years. Use the remaining slots for the strongest mix of speech/coding/SQL/system design aligned to the role.",
    #         "- If the interview type is behavioral/HR prioritize speech-based questions but still output 10 unique prompts.",
    #         "- While asking coding questions do not involve questions that might require high level packages or tools since the system given to the user to code will not support importing them. The system supports only basic packages like numpy pandas scikit-learn",
    #         "- Each question must specify:",
    #         "    * question_type (types available: \"Coding (Python)\", \"Coding (SQL)\", \"Speech Based\", \"System Design\")",
    #         "    * question text (clear and concise)",
    #         "    * difficulty (Easy/Medium/Hard)",
    #         "    * mandatory_skills (array of at least minimum of two and maximum of 4 skills tied to the question)",
    #         "- Respond ONLY in strict JSON using this schema:",
    #         "{",
    #         '    "questions": [',
    #         '        {',
    #         '            "question_type": "...",',
    #         '            "question": "...",',
    #         '            "difficulty": "Easy|Medium|Hard",',
    #         '            "mandatory_skills": ["skill1", "skill2", ...]',
    #         '        }',
    #         '    ]',
    #         '}',
    #     ]
    # )

    prompt = "\n".join(prompt_sections).strip()

    response = await execute_with_retries(
        lambda: genai.GenerativeModel('gemini-3.5-flash-lite').generate_content(prompt),
        retry_label="Gemini resume question generation",
    )

    raw_text = response.text.strip()
    questions_payload = _extract_question_list_from_response(raw_text)

    normalized_questions: List[Dict[str, Any]] = []
    for idx, item in enumerate(questions_payload):
        if not isinstance(item, dict):
            continue
        question_text = (item.get("question") or "").strip()
        if not question_text:
            continue
        question_type = item.get("question_type") or "General"
        difficulty = (item.get("difficulty") or "Medium").title()
        mandatory_skills = item.get("mandatory_skills") or []
        if isinstance(mandatory_skills, str):
            mandatory_skills = [skill.strip() for skill in mandatory_skills.split(',') if skill.strip()]
        if not isinstance(mandatory_skills, list):
            mandatory_skills = []
        if len(mandatory_skills) < 2:
            mandatory_skills.extend([job_role, "Communication"])

        normalized_questions.append(
            {
                "question": question_text,
                "question_type": question_type,
                "difficulty": difficulty,
                "mandatory_skills": mandatory_skills[:4],
                "generation_context": {
                    "target_role": effective_role,
                    "company_context": company,
                    "interview_type": interview_type,
                    "work_experience": work_experience,
                    "job_description_used": bool(jd_excerpt or jd_json_snippet),
                    "job_description_excerpt": jd_excerpt[:400] if jd_excerpt else None,
                    "source": "gemini",
                },
            }
        )

    return normalized_questions[:question_count]


def _extract_question_list_from_response(raw_text: str) -> List[Dict[str, Any]]:
    try:
        parsed = json.loads(raw_text)
        questions = parsed.get("questions") if isinstance(parsed, dict) else parsed
        if isinstance(questions, list):
            return questions
    except json.JSONDecodeError:
        pass

    start = raw_text.find("[")
    end = raw_text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw_text[start : end + 1])
        except Exception:
            pass

    logger.warning("Gemini resume question response not parseable, returning empty list")
    return []


@app.post("/mentors/students/import", response_model=StudentImportResult)
async def import_students_from_csv(
    file: UploadFile = File(...),
    program_id: Optional[int] = Form(None),
    ubp_id: Optional[int] = Form(None),
    university_name: Optional[str] = Form(None),
    program_name: Optional[str] = Form(None),
    batch_label: Optional[str] = Form(None),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    raw_bytes = await file.read()
    try:
        decoded = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="CSV must be UTF-8 encoded")

    reader = csv.DictReader(io.StringIO(decoded))
    fieldnames = {(field or "").strip().lower() for field in (reader.fieldnames or [])}
    # New requirement: only name and email in CSV; program context comes from form dropdowns
    required_fields = {"name", "email"}

    if not required_fields.issubset(fieldnames):
        raise HTTPException(status_code=400, detail="CSV must include columns: name, email")

    total_rows = 0
    imported = 0
    email_sent = 0
    duplicates_ignored: List[StudentDuplicateInfo] = []
    errors: List[StudentImportError] = []

    with db_pool.get_connection() as conn:
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection unavailable")

        with conn.cursor() as cur:
            for row_index, row in enumerate(reader, start=2):  # header counts as row 1
                total_rows += 1

                normalized_row = {
                    (key or "").strip().lower(): (value or "")
                    for key, value in row.items()
                }
                name = normalized_row.get("name", "").strip()
                email = normalized_row.get("email", "").strip().lower()
                # CSV no longer carries program; keep backwards compatibility if present
                program_from_csv = normalized_row.get("program_name", "").strip()

                try:
                    if not name or not email:
                        raise ValueError("Missing required fields")

                    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
                        raise ValueError("Invalid email format")

                    # Resolve UBP/program id
                    resolved_program_id = ubp_id or program_id
                    if not resolved_program_id and (university_name and program_name and batch_label):
                        resolved_program_id = _resolve_ubp_id(
                            (university_name or '').strip(),
                            (program_name or '').strip(),
                            (batch_label or '').strip(),
                        )
                    # Backward compatibility: allow program_name column in CSV to resolve legacy program id
                    if not resolved_program_id and program_from_csv:
                        resolved_program_id = _resolve_program_id_by_name(program_from_csv)
                    if not resolved_program_id:
                        raise ValueError("Program context not provided or not found (select University/Program/Batch)")

                    cur.execute(
                        """
                        SELECT student_id
                        FROM students
                        WHERE LOWER(TRIM(email)) = LOWER(TRIM(%s))
                        """,
                        (email,)
                    )
                    existing_student = cur.fetchone()

                    if existing_student:
                        duplicates_ignored.append(StudentDuplicateInfo(row=row_index, email=email))
                        continue

                    temp_password = _generate_temp_password()

                    cur.execute(
                        """
                        INSERT INTO students (name, email, program_id, password, last_active)
                        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                        RETURNING student_id
                        """,
                        (name, email, resolved_program_id, temp_password)
                    )

                    cur.fetchone()
                    imported += 1

                    try:
                        _send_credentials_email(name, email, temp_password)
                        email_sent += 1
                    except HTTPException as email_exc:
                        errors.append(
                            StudentImportError(row=row_index, email=email, reason=email_exc.detail)
                        )

                except ValueError as validation_exc:
                    errors.append(
                        StudentImportError(row=row_index, email=email or None, reason=str(validation_exc))
                    )
                except Exception as exc:
                    logger.error(f"Error importing row {row_index}: {exc}")
                    errors.append(
                        StudentImportError(
                            row=row_index,
                            email=email or None,
                            reason=str(exc) if str(exc) else "Unexpected error",
                        )
                    )

            conn.commit()

    return StudentImportResult(
        imported=imported,
        email_sent=email_sent,
        total_rows=total_rows,
        duplicates_ignored=duplicates_ignored,
        errors=errors,
    )

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

IST_TZ = ZoneInfo("Asia/Kolkata")


def _ensure_datetime(value: Optional[Union[datetime, date]]) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.combine(value, datetime.min.time())


def _to_ist_datetime(value: Optional[Union[datetime, date]], assume_tz: timezone = timezone.utc) -> Optional[datetime]:
    dt_value = _ensure_datetime(value)
    if not dt_value:
        return None
    if dt_value.tzinfo is None:
        dt_value = dt_value.replace(tzinfo=assume_tz)
    return dt_value.astimezone(IST_TZ)


def format_datetime_ist(value: Optional[Union[datetime, date]], assume_tz: timezone = timezone.utc) -> Optional[str]:
    ist_dt = _to_ist_datetime(value, assume_tz)
    return ist_dt.isoformat() if ist_dt else None


def find_existing_sessions(
    student_name: str,
    student_email: Optional[str],
    job_role: str,
    industry_type: str,
    company_name: str,
    interview_type: Optional[str] = None,
    work_experience: Optional[str] = None,
    is_company_card: bool = False,
) -> List[Dict[str, Any]]:
    """Fetch prior sessions for the same student and contextual combination.
    
    Args:
        is_company_card: If True, only requires company_name to match for reattempt check
    """
    sessions: List[Dict[str, Any]] = []
    
    logger.info(
        f"Finding existing sessions for {student_name} ({student_email}) - "
        f"Role: {job_role}, Company: {company_name}, Industry: {industry_type}, "
        f"Interview Type: {interview_type}, Work Exp: {work_experience}, "
        f"is_company_card: {is_company_card}"
    )

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

        student_id = find_student_id(student_name, student_email)

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
                logger.error("Failed to get database connection")
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

                logger.info(
                    "Executing reattempt query with %d student params",
                    len(filter_params)
                )
                logger.debug("Query: %s", query)
                logger.debug("Params: %s", tuple(filter_params))
                
                cur.execute(query, tuple(filter_params))
                rows = cur.fetchall()
                
                logger.info(f"Found {len(rows)} existing completed sessions")
                
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
                        logger.debug(
                            "Skipping session %s because existing work_experience='%s' but request missing value",
                            session_id,
                            work_experience_key,
                        )
                        continue

                    if normalized_interview_type:
                        if not interview_type_key or interview_type_key != normalized_interview_type:
                            continue
                    elif interview_type_key:
                        logger.debug(
                            "Skipping session %s because existing interview_type='%s' but request missing value",
                            session_id,
                            interview_type_key,
                        )
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

                                # Prefer interpretation that keeps duration non-negative and minimal
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

def _clean_json_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    return cleaned


def _parse_feedback_json(raw_text: str) -> Optional[Dict[str, Any]]:
    cleaned = _clean_json_text(raw_text)
    if not cleaned:
        return None

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse structured feedback JSON: {exc}")

        # Fallback: try trimming any trailing non-JSON content (e.g., extra commentary)
        last_brace = cleaned.rfind("}")
        if last_brace != -1 and last_brace < len(cleaned) - 1:
            trimmed = cleaned[: last_brace + 1]
            try:
                parsed = json.loads(trimmed)
                logger.info("Recovered structured feedback JSON after trimming trailing content.")
                return parsed
            except json.JSONDecodeError:
                logger.error("Fallback JSON trimming also failed; giving up on structured feedback.")

        return None


def extract_scores_from_feedback(feedback_text: str) -> Dict[str, Any]:
    """Derive overall score and rubric breakdown from structured feedback."""
    try:
        parsed = _parse_feedback_json(feedback_text)
        if parsed and "core_competencies" in parsed:
            competencies = parsed.get("core_competencies", [])
            if not competencies:
                logger.warning("core_competencies array is empty")
                return {"overall_score": 0.0}

            total_weighted_score = 0.0
            total_weight = 0.0
            for comp in competencies:
                if isinstance(comp, dict):
                    score = float(comp.get("score", 0.0) or 0.0)
                    weight = float(comp.get("weight", 0.0) or 0.0)
                    total_weighted_score += score * weight
                    total_weight += weight

            if total_weight > 0:
                overall_score = total_weighted_score / total_weight
                logger.info(
                    f"Weighted score extraction: Overall={overall_score:.2f} (from {len(competencies)} competencies)"
                )
                rubric_entries = []
                for comp in competencies:
                    if not isinstance(comp, dict):
                        continue
                    rubric_entries.append({
                        "name": comp.get("name"),
                        "score": comp.get("score"),
                    })
                interview_type = (
                    parsed.get("metadata", {}).get("interview_type")
                    or parsed.get("metadata", {}).get("interviewType")
                    or "unknown"
                )
                return {
                    "overall_score": overall_score,
                    "rubric_scores": {
                        "interview_type": interview_type,
                        "rubric": rubric_entries,
                    },
                }

            logger.warning("Total weight is zero, cannot calculate weighted average")
            return {"overall_score": 0.0}

        summary_scores: List[float] = []
        if parsed:
            for key in ("technical_summary", "communication_summary", "attitude_summary"):
                score_value = parsed.get(key, {}).get("score") if isinstance(parsed.get(key), dict) else None
                if score_value is not None:
                    try:
                        summary_scores.append(float(score_value))
                    except (TypeError, ValueError):
                        logger.warning(f"Unable to parse {key} score from feedback payload")

            if summary_scores:
                overall_score = sum(summary_scores) / len(summary_scores)
                logger.info(
                    f"Legacy structured score extraction produced overall score {overall_score:.2f}"
                )
                return {"overall_score": overall_score}

        section_patterns = (
            r"##\s*`?TECHNICAL_SKILLS_SUMMARY`?.*?###\s*Score:\s*`?([0-9]+(?:\.[0-9]+)?)/5`?",
            r"##\s*`?COMMUNICATION_STAR_RESPONSE`?.*?###\s*Score:\s*`?([0-9]+(?:\.[0-9]+)?)/5`?",
            r"##\s*`?ATTITUDE_INTERVIEW_READINESS`?.*?###\s*Score:\s*`?([0-9]+(?:\.[0-9]+)?)/5`?",
        )

        markdown_scores: List[float] = []
        for pattern in section_patterns:
            match = re.search(pattern, feedback_text, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    markdown_scores.append(float(match.group(1)))
                except (TypeError, ValueError):
                    logger.warning("Failed to parse markdown score while extracting overall score")

        if markdown_scores:
            overall_score = sum(markdown_scores) / len(markdown_scores)
            logger.info(
                f"Markdown score extraction (legacy) produced overall score {overall_score:.2f}"
            )
            return {"overall_score": overall_score}

        logger.warning("No score signals detected in feedback; defaulting overall score to 0.0")
        return {"overall_score": 0.0}

    except Exception as e:
        logger.error(f"Error extracting scores from feedback: {e}")
        return {"overall_score": 0.0}


def update_session_scores(session_id: str, overall_score: float, rubric_scores: Optional[Dict[str, Any]] = None) -> bool:
    """Persist the overall score for a session, along with rubric breakdown if provided."""
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                return False

            with conn.cursor() as cur:
                cur.execute(
                    "SELECT started_at FROM session_metadata WHERE session_id = %s",
                    (session_id,),
                )
                started_at = cur.fetchone()
                duration_minutes = None
                if started_at and started_at[0]:
                    duration_seconds = (datetime.now() - started_at[0]).total_seconds()
                    duration_minutes = int(duration_seconds / 60)

                cur.execute(
                    """
                        UPDATE session_metadata
                        SET overall_score = %s,
                            rubric_scores = %s,
                            technical_score = NULL,
                            communication_score = NULL,
                            attitude_score = NULL,
                            completed_at = CURRENT_TIMESTAMP,
                            status = 'completed',
                            feedback_generated = TRUE,
                            duration_minutes = %s
                        WHERE session_id = %s
                    """,
                    (
                        overall_score,
                        Json(rubric_scores) if rubric_scores is not None else None,
                        duration_minutes,
                        session_id,
                    ),
                )

                conn.commit()
                logger.info(
                    f"Updated overall score for session {session_id}: Overall={overall_score:.2f}"
                )
                return True

    except Exception as e:
        logger.error(f"Error updating session scores: {e}")
        return False


def update_scores_from_feedback(session_id: str, feedback_text: str) -> bool:
    """Extract the overall score from feedback and persist it."""
    try:
        logger.info(f"Starting score extraction for session {session_id}")

        scores = extract_scores_from_feedback(feedback_text)
        overall_score = scores.get("overall_score")
        if overall_score is None:
            logger.error("Overall score missing after extraction; aborting DB update")
            return False

        with db_pool.get_connection() as conn:
            if conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT session_id FROM session_metadata WHERE session_id = %s",
                        (session_id,),
                    )
                    exists = cur.fetchone()

                    if not exists:
                        logger.warning(
                            f"Session {session_id} not found in session_metadata table, creating it..."
                        )
                        cur.execute(
                            """
                                INSERT INTO session_metadata (session_id, status, created_at)
                                VALUES (%s, 'active', CURRENT_TIMESTAMP)
                                ON CONFLICT (session_id) DO NOTHING
                            """,
                            (session_id,),
                        )
                        conn.commit()

        rubric_scores = scores.get("rubric_scores")

        success = update_session_scores(session_id, float(overall_score), rubric_scores)
        if success:
            logger.info(
                f"Successfully updated overall score from feedback for session {session_id}"
            )
        else:
            logger.error(f"Failed to update overall score for session {session_id}")

        return success

    except Exception as e:
        logger.error(f"Error updating scores from feedback: {e}")
        return False

def _update_feedback_status(query: str, params: tuple[Any, ...]) -> None:
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                return
            with conn.cursor() as cur:
                cur.execute(query, params)
            conn.commit()
    except Exception as exc:
        logger.error(
            f"Failed to update feedback status for session {params[-1] if params else '?'}: {exc}"
        )


def _mark_feedback_pending(session_id: str) -> None:
    requested_at = datetime.utcnow()
    _update_feedback_status(
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


def _mark_feedback_processing(session_id: str) -> None:
    requested_at = datetime.utcnow()
    _update_feedback_status(
        """
        UPDATE session_metadata
        SET feedback_status = 'processing',
            feedback_error = NULL,
            feedback_requested_at = COALESCE(feedback_requested_at, %s)
        WHERE session_id = %s
        """,
        (requested_at, session_id),
    )


def _mark_feedback_failed(session_id: str, error_message: Optional[str]) -> None:
    ready_at = datetime.now(timezone.utc)
    truncated = error_message[:500] if error_message else None
    _update_feedback_status(
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


def _mark_feedback_completed(session_id: str) -> None:
    ready_at = datetime.utcnow()
    _update_feedback_status(
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


def _get_feedback_payload(session_id: str) -> Optional[Dict[str, Any]]:
    try:
        with db_pool.get_connection() as conn:
            if conn:
                with conn.cursor() as cur:
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
                    return {
                        "structured": parsed,
                        "raw": payload_text,
                    }
    except Exception as exc:
        logger.error(f"Error retrieving feedback payload for {session_id}: {exc}")
    return None

@lru_cache(maxsize=config.MAX_CACHE_SIZE)
def get_questions_cached(role: str, industry: str, company: str, difficulty: str, limit: int = 5) -> list:
    """Cached version of get_questions_from_db using LRU cache for static data"""
    try:
        normalized_role = (role or '').strip().lower()
        normalized_industry = (industry or '').strip().lower()
        normalized_company = (company or '').strip().lower()
        normalized_difficulty = (difficulty or '').strip().lower()

        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT question, mandatory_skills, difficulty, question_type
                    FROM interview_questions 
                    WHERE LOWER(TRIM(role)) = %s AND LOWER(TRIM(industry)) = %s AND LOWER(TRIM(company)) = %s AND LOWER(TRIM(difficulty)) = %s
                    ORDER BY id
                    LIMIT %s
                """, (normalized_role, normalized_industry, normalized_company, normalized_difficulty, limit))
                
                results = cur.fetchall()
                return [
                    {
                        'question': r[0],
                        'mandatory_skills': r[1],
                        'difficulty': r[2],
                        'question_type': (r[3] or 'standard').lower()
                    }
                    for r in results
                ]
                
    except Exception as e:
        logger.error(f"Database query error for questions: {e}")
        return []

async def analyze_answer_and_generate_response(question: str, answer: str, mandatory_skills: str, job_role: str) -> Tuple[float, str, str]:
    """Analyze answer and generate acknowledgment + difficulty using Gemini with retry logic"""
    prompt = f"""
            You are evaluating a candidate's response to this interview question for a {job_role} position:
            
            Question: "{question}"
            Candidate's Answer: "{answer}"
            Required Skills: {mandatory_skills}
            
            Based on this response, provide:
            1. Sentiment score (0-10) for confidence/quality
            2. Short acknowledgment (1-2 sentences) as if speaking to the candidate.Dont ask any question or anything here.Just a short acknowledgment.
            3. Difficulty level for the NEXT question
            
            Difficulty level guidelines:
            - easy: If candidate struggled, was unclear, or gave incorrect/incomplete answers
            - medium: If response was adequate but could be more detailed, or correct but basic
            - difficult: If response was excellent with deep knowledge and detailed correct answers
            
            Format your response exactly as:
            SENTIMENT: [0-10]
            ACKNOWLEDGMENT: [your acknowledgment here]
            DIFFICULTY: [easy/medium/difficult]
            
            Example:
            SENTIMENT: 7
            ACKNOWLEDGMENT: Great answer! I can see you have solid experience with this technology.
            DIFFICULTY: medium
            """

    try:
        logger.info("[Gemini] Starting sentiment/difficulty call for job_role=%s", job_role)
        response = await execute_with_retries(
            lambda: genai.GenerativeModel('gemini-3.5-flash-lite').generate_content(prompt),
            max_retries=3,
            base_delay=1.0,
            retry_label="Gemini sentiment analysis",
        )
        logger.info("[Gemini] Completed sentiment/difficulty call for job_role=%s", job_role)
        response_text = response.text.strip()

        # Debug logging
        logger.info(f"Gemini raw response: {response_text[:200]}...")

        # Parse sentiment, acknowledgment, and difficulty
        sentiment_score = DEFAULT_SENTIMENT_SCORE
        acknowledgment = DEFAULT_ACKNOWLEDGMENT
        difficulty = DEFAULT_NEXT_DIFFICULTY
        
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('SENTIMENT:'):
                try:
                    sentiment_text = line.replace('SENTIMENT:', '').strip()
                    sentiment_score = float(sentiment_text.split()[0])
                    sentiment_score = max(0, min(10, sentiment_score))  # Clamp 0-10
                    logger.info(f"Parsed sentiment: {sentiment_score}")
                except Exception as e:
                    logger.warning(f"Failed to parse sentiment: {e}")
            elif line.startswith('ACKNOWLEDGMENT:'):
                acknowledgment = line.replace('ACKNOWLEDGMENT:', '').strip()
                logger.info(f"Parsed acknowledgment: {acknowledgment[:50]}...")
            elif line.startswith('DIFFICULTY:'):
                difficulty = line.replace('DIFFICULTY:', '').strip().lower()
                # Validate difficulty level
                if difficulty not in ["easy", "medium", "difficult"]:
                    difficulty = "medium"
                logger.info(f"Parsed difficulty: {difficulty}")
        
        # More flexible parsing if exact format fails
        if acknowledgment == "Thank you for your response. Let's move on to the next question.":
            logger.warning("Exact format parsing failed, trying alternatives...")
            
            # Try to extract any meaningful text from the response
            response_lines = [line.strip() for line in response_text.split('\n') if line.strip()]
            
            # Look for the longest meaningful line (likely the acknowledgment)
            for line in response_lines:
                # Skip lines that look like sentiment scores
                if not line.lower().startswith('sentiment') and len(line) > 20:
                    # Clean up the line
                    clean_line = line.replace('ACKNOWLEDGMENT:', '').replace('Acknowledgment:', '').strip()
                    if len(clean_line) > 10 and len(clean_line) < 300:
                        acknowledgment = clean_line
                        logger.info(f"Alternative parsing found: {acknowledgment[:50]}...")
                        break
            
            # If still no good acknowledgment, try to use the entire response (cleaned)
            if acknowledgment == DEFAULT_ACKNOWLEDGMENT:
                clean_response = response_text.replace('SENTIMENT:', '').replace('ACKNOWLEDGMENT:', '').strip()
                # Remove any numbers at the start (sentiment scores)
                import re
                clean_response = re.sub(r'^\d+\.?\d*\s*', '', clean_response).strip()
                
                if len(clean_response) > 10 and len(clean_response) < 300:
                    acknowledgment = clean_response
                    logger.info(f"Using cleaned full response: {acknowledgment[:50]}...")
        
        # Fallback if acknowledgment is too long or empty
        if len(acknowledgment) > 300 or len(acknowledgment) < 10:
            acknowledgment = "Thank you for that detailed response. I can see you have relevant experience. Let's continue with the next question."
            logger.warning("Using fallback acknowledgment")
        
        logger.info(f"Sentiment: {sentiment_score}/10, Acknowledgment: {acknowledgment[:50]}..., Difficulty: {difficulty}")
        return sentiment_score, acknowledgment, difficulty
    except Exception as exc:
        logger.error(f"All attempts failed for sentiment analysis: {exc}")
        return 5.0, "Thank you for your response. Let's move on to the next question.", "medium"

async def generate_ai_acknowledgment(question: str, answer: str, mandatory_skills: str, job_role: str) -> str:
    """Generate intelligent AI acknowledgment using Gemini"""
    try:
        prompt = f"""
        You are an AI interviewer conducting a {job_role} interview. 
        
        Question asked: "{question}"
        Candidate's answer: "{answer}"
        Required skills: {mandatory_skills}
        
        Provide a brief, professional acknowledgment (2-3 sentences) that:
        1. Acknowledges their response positively
        2. Shows you understood their answer
        3. Briefly mentions a key point from their response
        
        
        Keep it conversational and encouraging. Don't provide detailed feedback yet.
        
        Example format: "Thank you for sharing that experience with [specific detail]. I can see you have good understanding of [relevant concept]. Let's move to the next question."

        Note: Dont ask any questions to the user in the acknowledgment. You should handle only the acknowledgement part.
        """

        response = await execute_with_retries(
            lambda: genai.GenerativeModel('gemini-3.5-flash-lite').generate_content(prompt),
            retry_label="Gemini acknowledgment generation",
        )
        acknowledgment = response.text.strip()
        
        # Fallback if response is too long or empty
        if len(acknowledgment) > 200 or len(acknowledgment) < 10:
            acknowledgment = "Thank you for that detailed response. I can see you have relevant experience. Let's continue with the next question."
        
        logger.info(f"Generated AI acknowledgment: {acknowledgment[:100]}...")
        return acknowledgment
        
    except Exception as e:
        logger.error(f"Error generating AI acknowledgment: {e}")
        return "Thank you for your response. Let's move on to the next question."

def calculate_next_difficulty(session_id: str, current_sentiment: float) -> str:
    """Calculate next question difficulty based on performance history"""
    try:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                # Get average sentiment score for this session
                cur.execute("""
                    SELECT AVG(sentiment_score) as avg_sentiment, COUNT(*) as question_count
                    FROM interview_data 
                    WHERE session_id = %s AND sentiment_score IS NOT NULL
                """, (session_id,))
                
                result = cur.fetchone()
                avg_sentiment = result[0] if result and result[0] else 5.0
                question_count = result[1] if result else 0
                
                # Dynamic difficulty logic based on performance
                if question_count == 0:
                    return "medium"  # Start with medium
                
                # Weight recent performance more heavily
                weighted_score = (avg_sentiment * 0.7) + (current_sentiment * 0.3)
                
                if weighted_score >= 7.5:
                    return "hard"    # Performing well, increase difficulty
                elif weighted_score >= 5.0:
                    return "medium"  # Average performance, maintain difficulty
                else:
                    return "easy"    # Struggling, reduce difficulty
                    
    except Exception as e:
        logger.error(f"Error calculating difficulty: {e}")
        return "medium"  # Safe default

def get_question_from_db_with_difficulty(
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
    """Get question from database with difficulty preference"""
    try:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                # Check if this session has pre-generated questions first
                if session_id:
                    cur.execute("""
                        SELECT question, mandatory_skills, difficulty, question_type, interview_type, work_experience
                        FROM pre_generated_questions
                        WHERE session_id = %s AND question_number = %s
                        LIMIT 1
                    """, (session_id, question_number))
                    pre_gen_result = cur.fetchone()
                    if pre_gen_result:
                        question_text, mandatory_skills_value, difficulty_value, question_type_value, db_interview_type, db_work_experience = pre_gen_result
                        logger.info(f"[PRE-GENERATED] Using pre-generated question #{question_number} for session {session_id}")
                        _log_question_selection(
                            source="pre_generated",
                            role=job_role,
                            company=company_name,
                            difficulty=difficulty_value or preferred_difficulty,
                            interview_type=db_interview_type or interview_type,
                            work_experience=db_work_experience or work_experience,
                            question_type=question_type_value,
                            question_text=question_text,
                        )
                        return {
                            'question': question_text,
                            'mandatory_skills': mandatory_skills_value or 'Communication, Problem-solving',
                            'difficulty': difficulty_value or preferred_difficulty,
                            'question_type': (question_type_value or 'standard').lower(),
                            'interview_type': (db_interview_type or interview_type or '').lower() if db_interview_type or interview_type else None,
                            'work_experience': (db_work_experience or work_experience or '').lower() if db_work_experience or work_experience else None,
                        }

                # Get already asked questions for this session to avoid repeats
                already_asked = []
                if session_id:
                    cur.execute("""
                        SELECT DISTINCT question, question_type
                        FROM interview_data 
                        WHERE session_id = %s AND question IS NOT NULL AND question != ''
                    """, (session_id,))
                    already_asked_rows = cur.fetchall()
                    already_asked = [row[0] for row in already_asked_rows]
                    if already_asked:
                        logger.info(f"Excluding {len(already_asked)} already asked questions for session {session_id}")

                # Build exclusion clause for already asked questions
                exclusion_clause = ""
                exclusion_params = []
                if already_asked:
                    placeholders = ','.join(['%s'] * len(already_asked))
                    exclusion_clause = f" AND question NOT IN ({placeholders})"
                    exclusion_params = already_asked

                # Try to get question with preferred difficulty first
                difficulty_clause = " AND LOWER(TRIM(difficulty)) = %s"
                
                # Dynamically build the WHERE clause
                normalized_role = (job_role or '').strip().lower()
                normalized_industry = (industry_type or '').strip().lower()
                normalized_company = (company_name or '').strip().lower()
                normalized_interview = (interview_type or '').strip().lower()
                normalized_experience = (work_experience or '').strip().lower()
                normalized_question_type = _normalize_question_type_label(question_type_filter)
                question_type_aliases = _get_question_type_aliases(normalized_question_type)

                filters = ["LOWER(TRIM(role)) = %s"]
                params = [normalized_role]

                if normalized_industry and normalized_industry not in ('any', 'n/a', ''):
                    filters.append("LOWER(TRIM(industry)) = %s")
                    params.append(normalized_industry)

                if normalized_company and normalized_company not in ('any', 'general', ''):
                    filters.append("LOWER(TRIM(company)) = %s")
                    params.append(normalized_company)

                if normalized_interview and normalized_interview not in ('any', 'n/a', ''):
                    filters.append("LOWER(TRIM(interview_type)) = %s")
                    params.append(normalized_interview)

                if normalized_experience and normalized_experience not in ('any', 'n/a', ''):
                    filters.append("LOWER(TRIM(work_experience)) = %s")
                    params.append(normalized_experience)

                if question_type_aliases:
                    filters.append("LOWER(TRIM(question_type)) = ANY(%s)")
                    params.append(question_type_aliases)

                where_clause = " AND ".join(filters)

                # Try to get a question with the preferred difficulty
                logger.info(
                    "[QUESTION QUERY] filters=%s exclusions=%d params=%s difficulty_param=%s",
                    filters,
                    len(exclusion_params),
                    params,
                    preferred_difficulty.lower(),
                )
                query = f"""
                    SELECT question, mandatory_skills, difficulty, question_type, interview_type, work_experience
                    FROM interview_questions
                    WHERE {where_clause} {exclusion_clause} {difficulty_clause}
                    ORDER BY RANDOM()
                    LIMIT 1
                """
                
                effective_params = list(params) + [preferred_difficulty.lower()]
                cur.execute(query, tuple(effective_params + exclusion_params))
                result = cur.fetchone()
                
                # If no match with preferred difficulty, try without difficulty constraint
                if not result:
                    logger.info(f"No {preferred_difficulty} questions found, trying any difficulty")
                    return get_question_from_db(
                        job_role,
                        industry_type,
                        company_name,
                        question_number,
                        session_id,
                        interview_type=normalized_interview,
                        work_experience=normalized_experience,
                        question_type_filter=normalized_question_type,
                    )

                if result:
                    question_text, mandatory_skills_value, difficulty_value, question_type_value, db_interview_type, db_work_experience = result
                    logger.info(f"Selected {preferred_difficulty} difficulty question for session {session_id}")
                    _log_question_selection(
                        source="difficulty_lookup",
                        role=job_role,
                        company=company_name,
                        difficulty=difficulty_value or preferred_difficulty,
                        interview_type=db_interview_type or normalized_interview,
                        work_experience=db_work_experience or normalized_experience,
                        question_type=question_type_value,
                        question_text=question_text,
                    )
                    return {
                        'question': question_text,
                        'mandatory_skills': mandatory_skills_value or 'Communication, Problem-solving',
                        'difficulty': (difficulty_value or preferred_difficulty),
                        'question_type': (question_type_value or 'standard').lower(),
                        'interview_type': (db_interview_type or normalized_interview or '').lower() or normalized_interview,
                        'work_experience': (db_work_experience or normalized_experience or '').lower() or normalized_experience,
                    }
                else:
                    # Fallback to get_question_from_db which has progressive fallback logic
                    logger.info(f"No questions found in get_question_from_db_with_difficulty, falling back to get_question_from_db")
                    return get_question_from_db(
                        job_role,
                        industry_type,
                        company_name,
                        question_number,
                        session_id,
                        interview_type=normalized_interview,
                        work_experience=normalized_experience,
                        question_type_filter=normalized_question_type,
                    )
                    
    except Exception as e:
        logger.error(f"Error getting question with difficulty: {e}")
        return get_question_from_db(
            job_role,
            industry_type,
            company_name,
            question_number,
            session_id,
            interview_type=interview_type,
            work_experience=work_experience,
        )


def get_question_from_db(
    job_role: str,
    industry_type: str,
    company_name: str,
    question_number: int,
    session_id: str = None,
    interview_type: Optional[str] = None,
    work_experience: Optional[str] = None,
    question_type_filter: Optional[str] = None,
) -> Dict[str, str]:
    """Get question from interview_questions table based on role, industry, company and question number"""
    try:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                # Get already asked questions for this session to avoid repeats
                already_asked = []
                if session_id:
                    cur.execute("""
                        SELECT DISTINCT question, question_type
                        FROM interview_data 
                        WHERE session_id = %s AND question IS NOT NULL AND question != ''
                    """, (session_id,))
                    already_asked_rows = cur.fetchall()
                    already_asked = [row[0] for row in already_asked_rows]
                    if already_asked:
                        logger.info(f"Excluding {len(already_asked)} already asked questions for session {session_id}")
                
                # Build exclusion clause for already asked questions
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

                # Dynamically build the WHERE clause with progressive fallback
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
                # Try with all filters first
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

                # Relax interview/work_experience filters if present but too strict
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

                # Fallback: role and industry (if applicable)
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

                # Fallback: role only (with question_type if available)
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
                
                # Final fallback: role only (remove question_type filter)
                if not result and include_question_type_filter:
                    logger.info(f"No questions found with question_type filter, removing question_type constraint")
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
                        role=job_role,
                        company=company_name,
                        difficulty=difficulty_value,
                        interview_type=db_interview_type or normalized_interview,
                        work_experience=db_work_experience or normalized_experience,
                        question_type=question_type_value,
                        question_text=question_text,
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
                    # Fallback if no questions in database
                    _log_question_selection(
                        source="fallback_placeholder",
                        role=job_role,
                        company=company_name,
                        difficulty='medium',
                        interview_type=interview_type,
                        work_experience=work_experience,
                        question_type='standard',
                        question_text=f"Question {question_number}: Tell me about your experience with {job_role} responsibilities.",
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
        # Fallback question
        return {
            'question': f"Question {question_number}: Tell me about your experience with {job_role} responsibilities.",
            'mandatory_skills': 'Communication, Problem-solving',
            'difficulty': 'medium',
            'question_type': 'standard',
            'interview_type': interview_type,
            'work_experience': work_experience,
        }

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."},
    )

# API Endpoints

# Compatibility endpoint for existing React frontend
@app.post("/interview", response_model=InterviewResponse)
async def handle_interview_compatibility(
    session_id: Optional[str] = Form(None),
    job_role: Optional[str] = Form(None),
    industry_type: Optional[str] = Form(None),
    company_name: Optional[str] = Form(None),
    student_name: Optional[str] = Form("Anonymous Student"),  # Default name
    student_email: Optional[str] = Form(None),
    answer: Optional[str] = Form(None),
    question_number: Optional[int] = Form(None),
    audio_file: Optional[UploadFile] = File(None)
):
    """Compatibility endpoint that matches your existing React frontend API"""
    try:
        logger.info(f"Interview endpoint called with session_id={session_id}, job_role={job_role}, student_name={student_name}")
        
        # If no session_id, start new interview
        if not session_id:
            # Create new session with admin tracking
            session_id = create_enhanced_session(
                student_name or "Anonymous Student", 
                job_role or "Software Engineer", 
                industry_type or "Technology", 
                company_name or "Tech Company"
            )
            
            if not session_id:
                raise HTTPException(status_code=500, detail="Failed to create session")
            
            # Get first question from database (no session_id yet for first question)
            first_question_data = get_question_from_db(job_role, industry_type, company_name, 1)
            first_question = first_question_data['question']
            mandatory_skills = first_question_data['mandatory_skills']
            first_question_type = first_question_data.get('question_type', 'standard')
            
            # Store first question in interview_data (use INSERT ... ON CONFLICT)
            try:
                with db_pool.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO interview_data 
                            (session_id, job_role, industry_type, company_name, question_number, 
                             question, answer, analysis_status, difficulty, mandatory_skills, question_type, timestamp)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                            ON CONFLICT (session_id, question_number) 
                            DO UPDATE SET question = EXCLUDED.question,
                                          question_type = EXCLUDED.question_type
                        """, (session_id, job_role, industry_type, company_name, 1, 
                              first_question, "", "pending", first_question_data['difficulty'], mandatory_skills, first_question_type))
                        conn.commit()
                        logger.info(f"Successfully stored first question for session {session_id}")
            except Exception as e:
                logger.error(f"Error storing first question: {e}")
                # Don't fail the entire request if question storage fails
                pass
            
            return {
                "session_id": session_id,
                "response": first_question,
                "response_meta": {
                    "question": first_question,
                    "question_type": first_question_type,
                    "mandatory_skills": mandatory_skills,
                    "difficulty": first_question_data.get('difficulty', 'medium')
                },
                "mandatory_skills": mandatory_skills,
                "question_number": 1,
                "message": "Interview started successfully"
            }
        
        # If session_id exists, handle answer submission and get next question
        else:
            # Handle audio file transcription if provided
            if audio_file and question_number:
                try:
                    # Read audio file content
                    audio_content = await audio_file.read()
                    
                    # Transcribe audio using Gemini
                    # Convert audio to base64 for Gemini
                    import base64
                    audio_base64 = base64.b64encode(audio_content).decode()
                    
                    # Create audio part for Gemini
                    audio_part = {
                        "mime_type": "audio/wav",
                        "data": audio_base64
                    }
                    
                    # Transcribe the audio
                    transcription_response = await execute_with_retries(
                        lambda: genai.GenerativeModel('gemini-3.5-flash-lite').generate_content([
                            "Please transcribe this audio accurately. Only return the transcribed text, nothing else.",
                            audio_part
                        ]),
                        retry_label="Gemini audio transcription",
                    )
                    
                    answer = transcription_response.text.strip()
                    logger.info(f"Audio transcribed for session {session_id}: {answer[:100]}...")
                    
                except Exception as e:
                    logger.error(f"Error transcribing audio: {e}")
                    raise HTTPException(status_code=500, detail=f"Audio transcription failed: {str(e)}")
            
            if answer and question_number:
                # Update current question with answer
                try:
                    with db_pool.get_connection() as conn:
                        if not conn:
                            raise HTTPException(status_code=500, detail="Database connection failed")
                        
                        with conn.cursor() as cur:
                            # Update answer in interview_data
                            cur.execute("""
                                UPDATE interview_data 
                                SET answer = %s, analysis_status = 'pending'
                                WHERE session_id = %s AND question_number = %s
                            """, (answer, session_id, question_number))
                            
                            # Get the current question and session info for context
                            cur.execute("""
                                SELECT question, mandatory_skills, difficulty, job_role, industry_type, company_name, interview_type, work_experience
                                FROM interview_data 
                                WHERE session_id = %s AND question_number = %s
                            """, (session_id, question_number))
                            
                            current_question_data = cur.fetchone()
                            if not current_question_data:
                                raise HTTPException(status_code=404, detail="Question not found")
                            
                            question, mandatory_skills, difficulty, job_role, industry_type, company_name, interview_type, work_experience = current_question_data
                            
                            # Log current session info for debugging
                            logger.info(f"Session {session_id} Q{question_number}: role={job_role}, industry={industry_type}, company={company_name}")
                            sentiment_score, ai_acknowledgment, next_difficulty = await analyze_answer_and_generate_response(
                                question,
                                answer,
                                mandatory_skills,
                                job_role or "Software Engineer"
                            )
                            
                            # Update sentiment score in database
                            cur.execute("""
                                UPDATE interview_data 
                                SET sentiment_score = %s, analysis_status = 'completed'
                                WHERE session_id = %s AND question_number = %s
                            """, (sentiment_score, session_id, question_number))
                            
                            # Check if this was the last question (let's say 5 questions max)
                            if question_number >= 5:
                                conn.commit()
                                return {
                                    "session_id": session_id,
                                    "response": "Thank you for completing the interview! You can now view your feedback.",
                                    "completed": True,
                                    "question_number": question_number
                                }
                            
                            # Use Gemini-determined difficulty for next question
                            logger.info(f"Gemini-determined difficulty for session {session_id}: {next_difficulty} (sentiment: {sentiment_score})")
                            
                            # Get next question from database (exclude already asked questions, use dynamic difficulty)
                            next_question_number = question_number + 1
                            # Map Gemini difficulty to database difficulty
                            db_difficulty = next_difficulty
                            if next_difficulty == "difficult":
                                db_difficulty = "hard"
                            
                            next_question_data = get_question_from_db_with_difficulty(
                                job_role or "Software Engineer", 
                                industry_type or "Technology", 
                                company_name or "General", 
                                next_question_number, 
                                session_id, db_difficulty
                            )
                            next_question = next_question_data['question']
                            next_mandatory_skills = next_question_data['mandatory_skills']
                            next_question_type = next_question_data.get('question_type', 'standard')
                            
                            # Insert next question with proper role/company info (handle conflicts)
                            cur.execute("""
                                INSERT INTO interview_data 
                                (session_id, job_role, industry_type, company_name, interview_type, work_experience, question_number, 
                                 question, answer, analysis_status, difficulty, mandatory_skills, question_type, timestamp)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                                ON CONFLICT (session_id, question_number) 
                                DO UPDATE SET 
                                    question = EXCLUDED.question, 
                                    mandatory_skills = EXCLUDED.mandatory_skills,
                                    difficulty = EXCLUDED.difficulty,
                                    job_role = EXCLUDED.job_role,
                                    industry_type = EXCLUDED.industry_type,
                                    company_name = EXCLUDED.company_name,
                                    interview_type = EXCLUDED.interview_type,
                                    work_experience = EXCLUDED.work_experience,
                                    question_type = EXCLUDED.question_type
                            """, (session_id, job_role or "Software Engineer", 
                                  industry_type or "Technology", 
                                  company_name or "General",
                                  interview_type,
                                  work_experience,
                                  next_question_number, next_question, "", "pending", 
                                  db_difficulty, next_mandatory_skills, next_question_type))
                            
                            conn.commit()
                            
                            return {
                                "session_id": session_id,
                                "response": next_question,
                                "next_question_meta": {
                                    "question": next_question,
                                    "question_type": next_question_type,
                                    "mandatory_skills": next_mandatory_skills,
                                    "difficulty": db_difficulty
                                },
                                "mandatory_skills": next_mandatory_skills,
                                "question_number": next_question_number,
                                "acknowledgment": ai_acknowledgment
                            }
                            
                except Exception as e:
                    logger.error(f"Database error in answer submission: {e}")
                    raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
            
            else:
                raise HTTPException(status_code=400, detail="Missing answer or question_number")
                
    except Exception as e:
        logger.error(f"Error in interview compatibility endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/students/sessions/{session_id}/rating")
async def get_session_rating(session_id: str, student_email: EmailStr = Query(...)):
    """Fetch an existing rating for the given session and student."""
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection unavailable")

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

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching session rating for {session_id}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch session rating") from exc


@app.post("/students/sessions/{session_id}/rating", status_code=201)
async def submit_session_rating(session_id: str, payload: SessionRatingPayload, student_email: EmailStr = Query(...)):
    """Submit or update a rating for the given session."""
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection unavailable")

            with conn.cursor() as cur:
                # Resolve student
                cur.execute(
                    "SELECT student_id FROM students WHERE LOWER(TRIM(email)) = LOWER(TRIM(%s))",
                    (student_email,),
                )
                student_row = cur.fetchone()
                if not student_row:
                    raise HTTPException(status_code=404, detail="Student not found")
                student_id = student_row[0]

                # Ensure session belongs to this student
                cur.execute(
                    "SELECT 1 FROM session_metadata WHERE session_id = %s AND student_id = %s",
                    (session_id, student_id),
                )
                ownership = cur.fetchone()
                if not ownership:
                    raise HTTPException(status_code=403, detail="Session does not belong to this student")

                cur.execute(
                    """
                    INSERT INTO session_ratings (session_id, student_id, rating, comments)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (session_id, student_id)
                    DO UPDATE SET rating = EXCLUDED.rating, comments = EXCLUDED.comments, created_at = CURRENT_TIMESTAMP
                    RETURNING id, created_at
                    """,
                    (session_id, student_id, payload.rating, payload.comments),
                )
                inserted = cur.fetchone()
                conn.commit()

                return {
                    "session_id": session_id,
                    "student_id": student_id,
                    "rating": payload.rating,
                    "comments": payload.comments,
                    "created_at": format_datetime_ist(inserted[1]) if inserted else None,
                }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error submitting session rating for {session_id}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to submit session rating") from exc

def create_enhanced_session(student_name: str, student_email: Optional[str], job_role: str, industry_type: str, company_name: str, interview_type: Optional[str] = None, work_experience: Optional[str] = None) -> Optional[str]:
    """Find/create student, then create a new session_metadata record."""
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                return None

            with conn.cursor() as cur:
                # Find or create student
                student_id = find_student_id(student_name, student_email)
                if not student_id:
                    # This part should ideally use the full registration flow, but for now, create a basic student
                    cur.execute(
                        "INSERT INTO students (name, email) VALUES (%s, %s) RETURNING student_id",
                        (student_name, student_email)
                    )
                    student_id = cur.fetchone()[0]

                # Create a new session
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

@app.get("/interview/active-session")
async def get_active_session(student_email: str):
    """
    Check if the student has an active (in-progress) interview session.
    Returns session details including current question number if active.
    """
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")
            
            with conn.cursor() as cur:
                # Find active session for this student
                cur.execute(
                    """
                    SELECT 
                        sm.session_id,
                        sm.status,
                        sm.interview_type,
                        sm.work_experience,
                        sm.started_at
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
                    return {"has_active_session": False}
                
                session_id, status, interview_type, work_experience, started_at = active_session
                
                # Get current question details
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
                    return {"has_active_session": False}
                
                current_question, question_text, question_type, mandatory_skills, difficulty, job_role, industry_type, company_name = question_data
                
                # Check if current question has been answered
                cur.execute(
                    """
                    SELECT answer FROM interview_data 
                    WHERE session_id = %s AND question_number = %s
                    """,
                    (session_id, current_question)
                )
                answer_row = cur.fetchone()
                current_answer = answer_row[0] if answer_row else None
                has_answered_current = bool(current_answer and current_answer.strip())
                
                # Calculate max questions
                cur.execute(
                    """
                    SELECT difficulty FROM interview_data
                    WHERE session_id = %s AND difficulty IS NOT NULL
                    ORDER BY question_number
                    """,
                    (session_id,)
                )
                difficulty_rows = cur.fetchall()
                difficulty_weights = [DIFFICULTY_WEIGHTS.get((row[0] or "medium").lower(), 2) for row in difficulty_rows]
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
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed checking active session for {student_email}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/interview/{session_id}/terminate")
async def terminate_session(session_id: str, payload: TerminateSessionRequest):
    """Terminate an in-progress interview session (e.g., due to proctoring violations)."""
    reason_text = (payload.reason or "Session terminated by system.").strip()
    truncated_reason = reason_text[:500]

    try:
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")

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
                    raise HTTPException(status_code=404, detail="Session not found")
            conn.commit()

        return {"status": "terminated", "session_id": session_id, "reason": truncated_reason}

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to terminate session {session_id}: {exc}")
        raise HTTPException(status_code=500, detail="Unable to terminate session") from exc


@app.post("/interview/reattempt/check")
async def check_reattempt(payload: ReattemptCheckRequest):
    """Check whether a reattempt confirmation is required without creating a session."""
    try:
        sessions = find_existing_sessions(
            payload.student_name,
            payload.student_email,
            payload.job_role,
            payload.industry_type,
            payload.company_name
        )

        requires_confirmation = len(sessions) > 0

        return {
            "requires_confirmation": requires_confirmation,
            "existing_sessions": sessions,
            "message": "Existing completed interview attempts found." if requires_confirmation else "No completed attempts found."
        }

    except Exception as e:
        logger.error(f"Error checking reattempt requirement: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/interview/start")
async def start_interview(
    request: Request,
    student_name: str = Form(...),
    student_email: Optional[str] = Form(None),
    job_role: str = Form(...),
    industry_type: str = Form(...),
    company_name: str = Form(...),
    interview_type: Optional[str] = Form(None),
    work_experience: Optional[str] = Form(None),
    job_description_id: Optional[int] = Form(None),
    job_description_text: Optional[str] = Form(None),
    job_description_raw_text: Optional[str] = Form(None),
    force_reattempt: bool = Form(False),
    use_resume_questions: bool = Form(False),
    resume_id: Optional[int] = Form(None),
    pre_generated_questions: Optional[str] = Form(None),
):
    """Start new interview with student tracking and reattempt detection
    
    Args:
        is_company_card: If True, only company_name is required to match for reattempt check
    """
    try:
        normalized_interview_type = (interview_type or '').strip()
        normalized_work_experience = (work_experience or '').strip()
        job_description_id_value = job_description_id
        job_description_text_value = (job_description_text or '').strip() or None
        job_description_raw_value = (job_description_raw_text or '').strip() or None

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
            first_payload_context = (pre_generated_question_list[0].get('generation_context') or {})
            if not job_description_text_value:
                excerpt = (first_payload_context.get('job_description_excerpt') or '').strip()
                if excerpt:
                    job_description_text_value = excerpt
            if not job_description_raw_value and job_description_text_value and first_payload_context.get('job_description_used'):
                job_description_raw_value = job_description_text_value

        if not force_reattempt and not resume_generation_enabled:
            # Check if this is a company card request by looking for the source header or param
            is_company_card = (
                request.headers.get('X-Request-Source') == 'company-card' or 
                request.query_params.get('source') == 'company-card' or
                request.query_params.get('is_company_card', '').lower() == 'true'
            )
            
            try:
                existing_sessions = find_existing_sessions(
                    student_name=student_name,
                    student_email=student_email,
                    job_role=job_role,
                    industry_type=industry_type,
                    company_name=company_name,
                    interview_type=normalized_interview_type,
                    work_experience=normalized_work_experience,
                    is_company_card=is_company_card
                )
                logger.info(f"Found {len(existing_sessions)} existing sessions for student {student_email} with is_company_card={is_company_card}")
            except Exception as e:
                logger.error(f"Error finding existing sessions: {e}", exc_info=True)
                existing_sessions = []
            if existing_sessions:
                logger.info(
                    "Existing session(s) detected for %s (%s) with role=%s, industry=%s, company=%s",
                    student_name,
                    student_email,
                    job_role,
                    industry_type,
                    company_name
                )
                return {
                    "requires_confirmation": True,
                    "existing_sessions": existing_sessions,
                    "message": "Existing interview attempts found. Confirm to start a reattempt."
                }

        session_id = create_enhanced_session(student_name, student_email, job_role, industry_type, company_name, normalized_interview_type, normalized_work_experience)
        
        if not session_id:
            raise HTTPException(status_code=500, detail="Failed to create session")
        
        student_id = find_student_id(student_name, student_email)

        if job_description_id_value and student_id and not job_description_raw_value:
            try:
                with db_pool.get_connection() as conn:
                    if conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                """
                                SELECT job_desc
                                FROM job_descriptions
                                WHERE id = %s AND (student_id = %s OR student_id IS NULL)
                                LIMIT 1
                                """,
                                (job_description_id_value, student_id),
                            )
                            jd_lookup = cur.fetchone()
                            if jd_lookup and jd_lookup[0]:
                                fetched_raw = (jd_lookup[0] or '').strip()
                                if fetched_raw:
                                    job_description_raw_value = fetched_raw
                                    if not job_description_text_value:
                                        job_description_text_value = fetched_raw[:1000]
            except Exception as exc:
                logger.warning(f"Unable to fetch job description text for id {job_description_id_value}: {exc}")

        if job_description_text_value and not job_description_raw_value:
            job_description_raw_value = job_description_text_value

        if resume_generation_enabled:
            if not student_id:
                logger.info(
                    "Storing pre-generated questions for session %s without student_id (email=%s)",
                    session_id,
                    student_email,
                )
            store_pre_generated_questions(
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
            _attach_resume_metadata_to_session(session_id, resume_id)

        _attach_job_description_metadata_to_session(
            session_id,
            job_description_id_value,
            job_description_text_value,
            job_description_raw_value,
        )

        # Determine first question source
        payload_mandatory: Union[str, List[str]] = []
        if resume_generation_enabled:
            first_question_payload = pre_generated_question_list[0]
            first_question = first_question_payload.get('question', "Tell me about yourself.")
            payload_mandatory = first_question_payload.get('mandatory_skills') or []
            if isinstance(payload_mandatory, str):
                mandatory_skills = payload_mandatory
            else:
                mandatory_skills = ', '.join(payload_mandatory)
            first_question_difficulty = first_question_payload.get('difficulty', 'medium') or 'medium'
            first_question_type = first_question_payload.get('question_type', 'standard')
            first_question_context = first_question_payload.get('generation_context') or {}
            current_max_questions = max(len(pre_generated_question_list), 1)
        else:
            # Get first question with a guaranteed medium difficulty baseline
            first_question_data = get_question_from_db_with_difficulty(
                job_role,
                industry_type,
                company_name,
                1,
                session_id,
                preferred_difficulty="medium",
                interview_type=normalized_interview_type,
                work_experience=normalized_work_experience,
            )
            first_question = first_question_data['question']
            mandatory_skills = first_question_data['mandatory_skills']
            first_question_difficulty = first_question_data.get('difficulty', 'medium') or 'medium'
            first_question_type = first_question_data.get('question_type', 'standard')
            first_question_context = {}

            first_difficulty_label = (first_question_difficulty or 'medium').lower()
            first_weight = DIFFICULTY_WEIGHTS.get(first_difficulty_label, 2)
            current_max_questions = _determine_max_questions([first_weight], 1)

        # Determine key skills to demonstrate
        key_skills = []
        if resume_generation_enabled:
            if isinstance(payload_mandatory, list):
                key_skills = [skill for skill in payload_mandatory if isinstance(skill, str)][:3]
            elif mandatory_skills:
                key_skills = [skill.strip() for skill in mandatory_skills.split(',') if skill.strip()][:3]
        elif mandatory_skills:
            key_skills = [skill.strip() for skill in mandatory_skills.split(',')[:3]]

        if not key_skills:
            key_skills = ["Problem Solving", "Communication"]

        # Store first question in interview_data
        try:
            with db_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    _ensure_interview_data_columns(cur)
                    insert_params = {
                        "session_id": session_id,
                        "job_role": job_role,
                        "industry_type": industry_type,
                        "company_name": company_name,
                        "interview_type": normalized_interview_type or None,
                        "work_experience": normalized_work_experience or None,
                        "question_number": 1,
                        "question": first_question,
                        "answer": "",
                        "analysis_status": "pending",
                        "difficulty": first_question_difficulty,
                        "mandatory_skills": mandatory_skills,
                        "question_type": first_question_type,
                        "job_description_id": job_description_id_value,
                        "job_description_text": job_description_text_value,
                        "job_desc": job_description_raw_value,
                        "resume_id": resume_id if resume_generation_enabled else None,
                        "generation_context": Json(first_question_context or {}),
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

        return {
            "session_id": session_id,
            "first_question": first_question,
            "first_question_meta": {
                "question": first_question,
                "question_type": first_question_type,
                "mandatory_skills": payload_mandatory if resume_generation_enabled else mandatory_skills,
                "difficulty": first_question_difficulty
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
            "job_description_attached": bool(job_description_id_value or job_description_text_value or job_description_raw_value),
            "message": "Interview started successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

DIFFICULTY_WEIGHTS = {
    "easy": 1,
    "medium": 2,
    "hard": 3,
    "difficult": 3,  # legacy label mapping
}

QUESTION_TYPE_ALIAS_MAP: Dict[str, set[str]] = {
    "standard": {"standard", "general", "default"},
    "behavioral": {"behavioral", "behavioural", "behavioral question"},
    "technical": {"technical", "tech", "technical question"},
    "coding": {"coding", "code", "programming"},
    "system design": {"system design", "design", "architecture", "system-design"},
    "sql": {"sql", "database", "sql query"},
    "speech": {"speech", "speech based", "speech-based", "speech question"},
}


def _get_question_type_aliases(normalized_label: Optional[str]) -> List[str]:
    """Return all acceptable values for a normalized question type label."""
    if not normalized_label:
        return []
    base = normalized_label.strip().lower()
    if not base:
        return []

    if base in QUESTION_TYPE_ALIAS_MAP:
        alias_set = set(value.strip().lower() for value in QUESTION_TYPE_ALIAS_MAP[base] if value)
        alias_set.add(base)
        return sorted(alias_set)

    return [base]

DEFAULT_SENTIMENT_SCORE = 5.0
DEFAULT_ACKNOWLEDGMENT = "Thank you for your response. Let's move on to the next question."
DEFAULT_NEXT_DIFFICULTY = "medium"
ANALYSIS_TIMEOUT_SECONDS = float(os.getenv("ACK_ANALYSIS_TIMEOUT_SECONDS", "12"))

def _translate_difficulty_label(label: str) -> str:
    """Normalize AI difficulty labels to supported values."""
    if not label:
        return "medium"
    normalized = label.lower().strip()
    if normalized == "difficult":
        return "hard"
    if normalized not in ("easy", "medium", "hard"):
        return "medium"
    return normalized


def _normalize_question_type_label(label: Optional[str]) -> Optional[str]:
    """Canonicalize question type labels so DB lookups match flexible inputs."""
    if not label:
        return None
    normalized = label.strip().lower()
    if not normalized:
        return None
    normalized = normalized.replace("-", " ").replace("_", " ")
    normalized = " ".join(part for part in normalized.split() if part)
    for canonical, aliases in QUESTION_TYPE_ALIAS_MAP.items():
        if normalized in aliases or canonical in normalized:
            return canonical
    return normalized


def _compute_running_average(weights: list[int]) -> float:
    if not weights:
        return 0.0
    return sum(weights) / len(weights)


# Concurrent session thresholds for load-based question limiting
CONCURRENT_SESSION_THRESHOLD = 25  # If more than this many active sessions, limit questions
CONCURRENT_SESSION_MAX_QUESTIONS = 5  # Reduced max questions when under high load
CONCURRENT_SESSION_TIMEOUT_MINUTES = 30  # Sessions older than this are considered stale


def _get_active_concurrent_sessions() -> int:
    """
    Count active concurrent sessions that started within the last 30 minutes.
    Sessions older than 30 minutes are considered stale and ignored.
    
    Returns:
        Number of active concurrent sessions within the timeout window.
    """
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                logger.warning("Could not get DB connection for concurrent session check")
                return 0
            
            with conn.cursor() as cur:
                # Query active sessions where started_at is within the last 30 minutes
                # Database stores timestamps in UTC (CURRENT_TIMESTAMP), so we compare with UTC now
                cur.execute(
                    """
                    SELECT COUNT(*) 
                    FROM session_metadata 
                    WHERE LOWER(TRIM(COALESCE(status, ''))) = 'active'
                      AND started_at IS NOT NULL
                      AND started_at > (CURRENT_TIMESTAMP - INTERVAL '%s minutes')
                    """,
                    (CONCURRENT_SESSION_TIMEOUT_MINUTES,)
                )
                result = cur.fetchone()
                count = result[0] if result else 0
                logger.info(f"Active concurrent sessions (within {CONCURRENT_SESSION_TIMEOUT_MINUTES} min): {count}")
                return count
                
    except Exception as e:
        logger.error(f"Error counting concurrent sessions: {e}")
        return 0  # On error, don't limit questions


def _determine_max_questions(weights: list[int], question_number: int, check_load: bool = True) -> int:
    """Determine dynamic interview length based on running average milestones.
    
    Also considers server load - if too many concurrent sessions are active,
    the max questions is limited to reduce server strain.
    
    Args:
        weights: List of difficulty weights for answered questions
        question_number: Current question number in the interview
        check_load: If True, check concurrent sessions and limit if under high load
        
    Returns:
        Maximum number of questions for this interview
    """
    # Check concurrent session load first (only at start or early questions)
    if check_load and question_number <= 2:
        concurrent_sessions = _get_active_concurrent_sessions()
        if concurrent_sessions > CONCURRENT_SESSION_THRESHOLD:
            logger.warning(
                f"High load detected: {concurrent_sessions} concurrent sessions > {CONCURRENT_SESSION_THRESHOLD} threshold. "
                f"Limiting max questions to {CONCURRENT_SESSION_MAX_QUESTIONS}"
            )
            return CONCURRENT_SESSION_MAX_QUESTIONS
    
    # Normal logic based on difficulty running average
    running_avg = _compute_running_average(weights)

    # Default maximum before any milestone is reached
    max_questions = 5

    if question_number >= 4:
        if running_avg < 2:
            max_questions = 5
        elif running_avg < 2.5:
            max_questions = 8
        else:
            max_questions = 10

    return max_questions

def _extract_first_number(text: str) -> Optional[float]:
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def _normalize_video_analysis(raw_value: Optional[Any]) -> Optional[Dict[str, Any]]:
    if raw_value is None:
        return None
    if isinstance(raw_value, dict):
        return raw_value
    if isinstance(raw_value, (bytes, bytearray)):
        raw_value = raw_value.decode("utf-8", errors="ignore")
    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if not raw_value:
            return None
        try:
            parsed = json.loads(raw_value)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, str):
                return {"summary": parsed}
        except Exception:
            return {"summary": raw_value}
    return None


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
    if summary:
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


def _extract_system_design_payload(answer_text: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not answer_text:
        return None, None
    try:
        parsed = json.loads(answer_text)
        if isinstance(parsed, dict):
            summary = parsed.get("summary")
            return parsed, summary
    except Exception:
        pass
    return None, None


async def _process_answer_and_get_next(background_tasks: BackgroundTasks, session_id: str, answer: str):
    """Helper to process answer and manage DB connection for background tasks."""
    """Submit answer and get next question - works with existing interview_data table"""
    try:
        saved_video_path: Optional[str] = None
        video_analysis: Optional[Dict[str, Any]] = None

        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")
            
            with conn.cursor() as cur:
                # Get the current question number
                cur.execute("SELECT MAX(question_number) FROM interview_data WHERE session_id = %s", (session_id,))
                question_number = cur.fetchone()[0]
                if not question_number:
                    raise HTTPException(status_code=404, detail="No active question found for this session.")

                # Update current question with the answer and coding metadata if provided
                update_fields = ["answer = %s", "analysis_status = 'pending'"]
                update_values: list[Any] = [answer]

                if question_type:
                    update_fields.append("question_type = %s")
                    update_values.append(question_type.lower())
                if code is not None:
                    update_fields.append("code_submission = %s")
                    update_values.append(code)

                if response_video is not None:
                    saved_video_path = _save_uploaded_video(session_id, question_number, response_video)

                update_clause = ", ".join(update_fields)
                cur.execute(
                    f"""
                    UPDATE interview_data SET {update_clause}
                    WHERE session_id = %s AND question_number = %s
                    """,
                    (*update_values, session_id, question_number)
                )

                # Get context for analysis
                cur.execute("""
                    SELECT question, mandatory_skills, job_role, industry_type, company_name, question_type, interview_type, work_experience
                    FROM interview_data WHERE session_id = %s AND question_number = %s
                """, (session_id, question_number))
                
                context = cur.fetchone()
                if not context:
                    raise HTTPException(status_code=404, detail="Question context not found.")
                
                question, mandatory_skills, job_role, industry_type, company_name, stored_question_type, interview_type, work_experience = context
                logger.info(
                    "[CONTEXT] question_type=%s interview_type=%s work_experience=%s",
                    stored_question_type,
                    interview_type,
                    work_experience,
                )

                if (not interview_type or not interview_type.strip()) or (not work_experience or not work_experience.strip()):
                    cur.execute(
                        "SELECT interview_type, work_experience FROM session_metadata WHERE session_id = %s",
                        (session_id,),
                    )
                    session_meta = cur.fetchone()
                    if session_meta:
                        session_interview_type, session_work_experience = session_meta
                        if not interview_type or not interview_type.strip():
                            interview_type = session_interview_type
                        if not work_experience or not work_experience.strip():
                            work_experience = session_work_experience

                if not stored_question_type or not str(stored_question_type).strip():
                    stored_question_type = question_type or 'standard'

                normalized_interview = (interview_type or '').strip().lower() or None
                normalized_experience = (work_experience or '').strip().lower() or None

                # Don't preserve question_type - allow next question to be any type
                normalized_question_type = None

                # Analyze the answer
                sentiment_score, acknowledgment, next_difficulty = await _analyze_answer_with_timeout(
                    question,
                    analysis_answer_value,
                    mandatory_skills,
                    job_role
                )

                # Update sentiment score
                cur.execute("""
                    UPDATE interview_data SET sentiment_score = %s, analysis_status = 'completed'
                    WHERE session_id = %s AND question_number = %s
                """, (sentiment_score, session_id, question_number))

                # Track difficulty history for dynamic length control
                cur.execute(
                    """
                        SELECT difficulty
                        FROM interview_data
                        WHERE session_id = %s AND difficulty IS NOT NULL
                        ORDER BY question_number
                    """,
                    (session_id,)
                )
                difficulty_rows = cur.fetchall()
                difficulty_weights = [DIFFICULTY_WEIGHTS.get((row[0] or "medium").lower(), 2) for row in difficulty_rows]

                current_max_questions = _determine_max_questions(difficulty_weights, question_number)

                # Check if interview is finished
                if question_number >= current_max_questions:
                    cur.execute("UPDATE session_metadata SET status = 'completed', completed_at = CURRENT_TIMESTAMP WHERE session_id = %s", (session_id,))
                    conn.commit()

                    # --- FIX: Call the refactored helper to generate feedback and scores ---
                    logger.info(f"Interview {session_id} completed. Generating final feedback and scores in the background.")
                    _mark_feedback_pending(session_id)
                    background_tasks.add_task(_generate_and_process_feedback, session_id)

                    return {
                        "message": "Interview finished. Feedback is being generated.", 
                        "acknowledgment": acknowledgment, 
                        "completed": True,
                        "current_max_questions": current_max_questions
                    }

                # Get next question
                next_question_number = question_number + 1
                normalized_difficulty = _translate_difficulty_label(next_difficulty)
                db_difficulty = normalized_difficulty
                next_question_data = get_question_from_db_with_difficulty(
                    job_role,
                    industry_type,
                    company_name,
                    next_question_number,
                    session_id,
                    db_difficulty,
                    interview_type=normalized_interview,
                    work_experience=normalized_experience,
                    question_type_filter=normalized_question_type,
                )

                # Store next question
                cur.execute("""
                    INSERT INTO interview_data (session_id, job_role, industry_type, company_name, interview_type, work_experience, question_number, question, mandatory_skills, difficulty, question_type, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (session_id, question_number) DO UPDATE SET 
                        question = EXCLUDED.question,
                        mandatory_skills = EXCLUDED.mandatory_skills,
                        difficulty = EXCLUDED.difficulty,
                        interview_type = EXCLUDED.interview_type,
                        work_experience = EXCLUDED.work_experience,
                        question_type = EXCLUDED.question_type
                """, (
                    session_id,
                    job_role,
                    industry_type,
                    company_name,
                    normalized_interview,
                    normalized_experience,
                    next_question_number,
                    next_question_data['question'],
                    next_question_data['mandatory_skills'],
                    db_difficulty,
                    next_question_data.get('question_type', normalized_question_type or 'standard')
                ))

                conn.commit()

                return {
                    "next_question": next_question_data['question'],
                    "next_question_meta": {
                        "question": next_question_data['question'],
                        "question_type": next_question_data.get('question_type', 'standard'),
                        "mandatory_skills": next_question_data['mandatory_skills'],
                        "difficulty": next_question_data.get('difficulty', db_difficulty)
                    },
                    "acknowledgment": acknowledgment,
                    "question_number": next_question_number,
                    "completed": False,
                    "current_max_questions": current_max_questions
                }
            
    except Exception as e:
        logger.error(f"Error submitting answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _save_uploaded_video(session_id: str, question_number: int, upload: UploadFile | None) -> Optional[str]:
    if not upload:
        return None

    safe_name = upload.filename or f"session-{session_id}-q{question_number}.webm"
    safe_name = safe_name.replace("/", "_").replace("\\", "_")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{session_id}-q{question_number}-{timestamp}-{safe_name}"
    target_path = config.INTERVIEW_VIDEO_DIR / filename

    try:
        with target_path.open("wb") as dst:
            shutil.copyfileobj(upload.file, dst)
        upload.file.seek(0)
        return str(target_path)
    except Exception as exc:
        logger.error(f"Failed to save uploaded video for session {session_id}: {exc}")
        return None


def _process_video_analysis_background(session_id: str, question_number: int, video_path: str):
    """Background task to analyze video and update database."""
    try:
        logger.info(f"Starting background video analysis for session {session_id}, question {question_number}")
        
        # Run synchronous video analysis
        analysis = _analyze_video_sentiment_sync(video_path)
        
        # Update database with results
        with db_pool.get_connection() as conn:
            if not conn:
                logger.error(f"Failed to get DB connection for video analysis update (session {session_id})")
                return
            
            with conn.cursor() as cur:
                if analysis:
                    cur.execute(
                        """
                        UPDATE interview_data 
                        SET video_analysis = %s, video_analysis_status = 'completed'
                        WHERE session_id = %s AND question_number = %s
                        """,
                        (json.dumps(analysis), session_id, question_number)
                    )
                else:
                    cur.execute(
                        """
                        UPDATE interview_data 
                        SET video_analysis_status = 'failed'
                        WHERE session_id = %s AND question_number = %s
                        """,
                        (session_id, question_number)
                    )
                conn.commit()
        
        # Clean up video file after analysis
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
                logger.debug(f"Removed video clip after analysis: {video_path}")
            except OSError as cleanup_error:
                logger.warning(f"Failed to remove video clip {video_path}: {cleanup_error}")
        
        logger.info(f"Completed background video analysis for session {session_id}, question {question_number}")
        
    except Exception as exc:
        logger.error(f"Background video analysis failed for session {session_id}, question {question_number}: {exc}")
        # Mark as failed in database
        try:
            with db_pool.get_connection() as conn:
                if conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE interview_data 
                            SET video_analysis_status = 'failed'
                            WHERE session_id = %s AND question_number = %s
                            """,
                            (session_id, question_number)
                        )
                        conn.commit()
        except Exception as db_err:
            logger.error(f"Failed to mark video analysis as failed in DB: {db_err}")


def _analyze_video_sentiment_sync(video_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Synchronous version of video sentiment analysis for background tasks."""
    if not video_path:
        return None

    def _run_gemini_video_call() -> Dict[str, Any]:
        model = genai.GenerativeModel(config.GEMINI_VIDEO_MODEL)
        with open(video_path, "rb") as clip_file:
            response = model.generate_content(
                [
                    (
                        # "You are an expert behavioral interviewer. Analyze this short interview clip and"
                        # " summarize the candidate's performance. Focus on verbal and non-verbal communication,"
                        # " confidence, clarity, posture, eye contact, gestures, stammering, filler words, nervous"
                        # " tics, and emotional cues. Return STRICT JSON with these keys: sentiment (float 0-10),"
                        # " engagement (1-2 sentences describing energy, confidence, clarity, and any hesitation),"
                        # " dominant_expression (1-2 sentences on facial expressions, posture, gestures, non-verbal"
                        # " cues), and verbal_strength (1-2 sentences summarizing answer quality, structure, and reasoning)."
                        # " Use full sentences and return only JSON with no extra commentary."
                        """
                        You are observing a candidate answering an interview question.
                        Analyze only visible behavior and speech delivery in the video.
                        Do NOT assess answer correctness.

                        Evaluate:
                        - Overall sentiment
                        - Engagement and confidence
                        - Non-verbal behavior (posture, eye contact, gestures,nervous facial expressions)
                        - Verbal delivery (clarity, fillers, stammering)

                        Return STRICT JSON only with:
                        {
                        "sentiment": float (0–10),
                        "engagement": string (max 50 words),
                        "dominant_expression": string (max 50 words),
                        "verbal_strength": string (max 30 words)
                        }

                        Be concise. No extra text."""
                    ),
                    {
                        "mime_type": "video/webm",
                        "data": clip_file.read(),
                    },
                ],
                generation_config={"response_mime_type": "application/json"},
            )

        if not response or not response.text:
            raise ValueError("Empty response from Gemini video model")

        try:
            raw = json.loads(response.text)
        except json.JSONDecodeError as decode_error:
            logger.warning(f"Gemini video response not JSON, falling back to heuristic parsing: {decode_error}")
            parsed = {
                "sentiment": raw_score if (raw_score := _extract_numeric(response.text)) is not None else 5.0,
                "engagement": "Engagement details unavailable; Gemini returned non-JSON output.",
                "dominant_expression": "Expression details unavailable; Gemini returned non-JSON output.",
                "verbal_strength": "Verbal delivery details unavailable; Gemini returned non-JSON output.",
                
            }
            return parsed

        return {
            "sentiment": float(raw.get("sentiment", 5.0)),
            "engagement": raw.get("engagement") or raw.get("engagement_summary", "Engagement details unavailable."),
            "dominant_expression": raw.get("dominant_expression") or raw.get("dominant_expression_summary", "Expression details unavailable."),
            "verbal_strength": raw.get("verbal_strength") or raw.get("verbal_summary", "Verbal delivery details unavailable."),
        }

    try:
        # Use synchronous retry wrapper for background tasks
        analysis = _run_gemini_video_call()
        if not isinstance(analysis, dict):
            return None

        analysis.setdefault("sentiment", 5.0)
        analysis.setdefault("engagement", "Engagement details unavailable.")
        analysis.setdefault("dominant_expression", "Expression details unavailable.")
        analysis.setdefault("verbal_strength", "Verbal delivery details unavailable.")
        analysis.setdefault("clip_path", video_path)
        return analysis
    except Exception as exc:
        logger.error(f"Video analysis failed for {video_path}: {exc}")
        return None


@app.post("/interview/{session_id}/upload-video")
async def upload_video_async(
    session_id: str,
    background_tasks: BackgroundTasks,
    question_number: int = Form(...),
    response_video: UploadFile = File(...),
):
    """
    Async video upload endpoint - saves video and triggers background analysis.
    This allows the frontend to upload video without blocking answer submission.
    """
    start_time = time.time()
    logger.info(f"[VIDEO-UPLOAD] Starting async video upload for session {session_id}, question {question_number}")
    
    try:
        # Save video to disk
        saved_video_path = _save_uploaded_video(session_id, question_number, response_video)
        
        if not saved_video_path:
            raise HTTPException(status_code=500, detail="Failed to save video file")
        
        logger.info(f"[VIDEO-UPLOAD] Video saved to {saved_video_path} in {time.time()-start_time:.3f}s")
        
        # Update database with video path and mark as pending analysis
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")
            
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE interview_data 
                    SET video_clip_path = %s, video_analysis_status = 'pending'
                    WHERE session_id = %s AND question_number = %s
                    """,
                    (saved_video_path, session_id, question_number)
                )
                conn.commit()
        
        # Queue background video analysis
        background_tasks.add_task(
            _process_video_analysis_background,
            session_id,
            question_number,
            saved_video_path
        )
        
        logger.info(f"[VIDEO-UPLOAD] Async upload complete for session {session_id}, question {question_number} in {time.time()-start_time:.3f}s")
        
        return {
            "status": "uploaded",
            "message": "Video uploaded successfully. Analysis queued.",
            "session_id": session_id,
            "question_number": question_number
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[VIDEO-UPLOAD] Error uploading video for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Video upload failed: {str(e)}")


@app.post("/interview/{session_id}/answer")
async def submit_answer(
    session_id: str,
    background_tasks: BackgroundTasks,
    answer: str = Form(...),
    question_type: Optional[str] = Form(None),
    code: Optional[str] = Form(None),
    stdin: Optional[str] = Form(None),
    stdout: Optional[str] = Form(None),
    stderr: Optional[str] = Form(None),
    runtime_error: Optional[str] = Form(None),
    execution_success: Optional[str] = Form(None),
    has_run: Optional[str] = Form(None),
    is_final: Optional[str] = Form(None),
    response_video: Optional[UploadFile] = File(None),
    system_design_diagram: Optional[str] = Form(None),
):
    """Submit answer and get next question - works with existing interview_data table"""
    import time
    start_time = time.time()
    logger.info(f"[TIMING] submit_answer started for session={session_id}")
    try:
        # PHASE 1: Save answer and fetch all context data
        context_data = None
        difficulty_weights = []
        current_max_questions = 0
        is_final_question = False
        saved_video_path: Optional[str] = None
        analysis_answer_value = answer
        
        phase1_start = time.time()
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")
            
            logger.info(f"[TIMING] DB connection acquired, elapsed={time.time()-start_time:.3f}s")
            with conn.cursor() as cur:
                # Get the current question number
                cur.execute("SELECT MAX(question_number) FROM interview_data WHERE session_id = %s", (session_id,))
                question_number = cur.fetchone()[0]
                if not question_number:
                    raise HTTPException(status_code=404, detail="No active question found for this session.")

                _ensure_interview_data_columns(cur)

                cur.execute(
                    "SELECT question_type FROM interview_data WHERE session_id = %s AND question_number = %s",
                    (session_id, question_number)
                )
                existing_question_type_row = cur.fetchone()
                existing_question_type = existing_question_type_row[0] if existing_question_type_row else None

                effective_question_type = question_type or existing_question_type
                is_system_design = _is_system_design_question_type(effective_question_type)
                stored_answer_value = answer
                analysis_answer_value = answer

                if is_system_design:
                    stored_answer_value = _prepare_system_design_answer_payload(system_design_diagram, answer)
                    analysis_answer_value = _describe_system_design(stored_answer_value)

                update_fields = ["answer = %s", "analysis_status = 'pending'"]
                update_values: List[Any] = [stored_answer_value]

                if question_type:
                    update_fields.append("question_type = %s")
                    update_values.append(question_type.lower())
                if code is not None:
                    update_fields.append("code_submission = %s")
                    update_values.append(code)
                if stdin is not None:
                    update_fields.append("stdin_input = %s")
                    update_values.append(stdin)
                if stdout is not None:
                    update_fields.append("stdout_output = %s")
                    update_values.append(stdout)
                if stderr is not None:
                    update_fields.append("stderr_output = %s")
                    update_values.append(stderr)
                if runtime_error is not None:
                    update_fields.append("runtime_error = %s")
                    update_values.append(runtime_error)
                
                # Only save video for speech-based questions
                is_speech_question = 'speech' in (effective_question_type or '').lower()
                if response_video is not None and is_speech_question:
                    video_start = time.time()
                    saved_video_path = _save_uploaded_video(session_id, question_number, response_video)
                    logger.info(f"[TIMING] Video save took {time.time()-video_start:.3f}s")
                    if saved_video_path:
                        update_fields.append("video_clip_path = %s")
                        update_values.append(saved_video_path)
                        update_fields.append("video_analysis_status = %s")
                        update_values.append('pending')

                execution_success_bool = _parse_bool(execution_success)
                if execution_success_bool is not None:
                    update_fields.append("execution_success = %s")
                    update_values.append(execution_success_bool)

                has_run_bool = _parse_bool(has_run)
                if has_run_bool is not None:
                    update_fields.append("manual_run = %s")
                    update_values.append(has_run_bool)

                update_clause = ", ".join(update_fields)
                cur.execute(
                    f"""
                    UPDATE interview_data SET {update_clause}
                    WHERE session_id = %s AND question_number = %s
                    """,
                    (*update_values, session_id, question_number)
                )

                # Get context for analysis (needed for both evaluation and metadata lookups)
                cur.execute("""
                    SELECT question, mandatory_skills, job_role, industry_type, company_name, interview_type, work_experience
                    FROM interview_data WHERE session_id = %s AND question_number = %s
                """, (session_id, question_number))
                
                context = cur.fetchone()
                if not context:
                    raise HTTPException(status_code=404, detail="Question context not found.")
                
                question, mandatory_skills, job_role, industry_type, company_name, interview_type, work_experience = context
                logger.info(
                    "[CONTEXT] interview_type=%s work_experience=%s",
                    interview_type,
                    work_experience,
                )

                if (not interview_type or not interview_type.strip()) or (not work_experience or not work_experience.strip()):
                    cur.execute(
                        "SELECT interview_type, work_experience FROM session_metadata WHERE session_id = %s",
                        (session_id,),
                    )
                    session_meta = cur.fetchone()
                    if session_meta:
                        session_interview_type, session_work_experience = session_meta
                        if not interview_type or not interview_type.strip():
                            interview_type = session_interview_type
                        if not work_experience or not work_experience.strip():
                            work_experience = session_work_experience

                normalized_interview = (interview_type or '').strip().lower() or None
                normalized_experience = (work_experience or '').strip().lower() or None
                # Don't preserve question_type - allow next question to be any type
                normalized_question_type = None

                cur.execute(
                    """
                        SELECT difficulty
                        FROM interview_data
                        WHERE session_id = %s AND difficulty IS NOT NULL
                        ORDER BY question_number
                    """,
                    (session_id,)
                )
                difficulty_rows = cur.fetchall()
                difficulty_weights = [DIFFICULTY_WEIGHTS.get((row[0] or "medium").lower(), 2) for row in difficulty_rows]

                current_max_questions = _determine_max_questions(difficulty_weights, question_number)
                is_final_flag = _parse_bool(is_final)
                is_final_question = (is_final_flag is True) or (question_number >= current_max_questions)

                # Store all context data for later phases
                context_data = {
                    'question_number': question_number,
                    'question': question,
                    'mandatory_skills': mandatory_skills,
                    'job_role': job_role,
                    'industry_type': industry_type,
                    'company_name': company_name,
                    'interview_type': normalized_interview,
                    'work_experience': normalized_experience,
                    'question_type': normalized_question_type,
                    'current_max_questions': current_max_questions,
                    'is_final_question': is_final_question
                }
                
                conn.commit()
        # ✅ CONNECTION CLOSED HERE
        logger.info(f"[TIMING] Phase 1 (DB prep) completed, elapsed={time.time()-phase1_start:.3f}s")
        
        # PHASE 2: AI Analysis (NO DB CONNECTION)
        phase2_start = time.time()
        logger.info(f"[TIMING] Starting Gemini analysis, total elapsed={time.time()-start_time:.3f}s")
        sentiment_score, acknowledgment, next_difficulty = await _analyze_answer_with_timeout(
            context_data['question'], analysis_answer_value, context_data['mandatory_skills'], context_data['job_role']
        )

        logger.info(f"[TIMING] Phase 2 (Gemini) completed, elapsed={time.time()-phase2_start:.3f}s")
        
        # Queue video analysis as background task if video was uploaded
        if saved_video_path:
            background_tasks.add_task(
                _process_video_analysis_background,
                session_id,
                context_data['question_number'],
                saved_video_path
            )
        
        # PHASE 3: Save results and get next question
        phase3_start = time.time()
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")
            
            with conn.cursor() as cur:
                # Update sentiment score
                cur.execute(
                    """
                        UPDATE interview_data 
                        SET sentiment_score = %s, analysis_status = 'completed'
                        WHERE session_id = %s AND question_number = %s
                    """,
                    (sentiment_score, session_id, context_data['question_number']),
                )

                if context_data['is_final_question']:
                    logger.info(f"[FINAL-ANSWER] Completed per-question analysis for session {session_id}; queuing feedback job.")
                    cur.execute(
                        "UPDATE session_metadata SET status = 'completed', completed_at = CURRENT_TIMESTAMP WHERE session_id = %s",
                        (session_id,),
                    )
                    conn.commit()

                    _mark_feedback_pending(session_id)
                    background_tasks.add_task(_generate_and_process_feedback, session_id)

                    return {
                        "message": "Interview finished. Feedback is being generated.",
                        "acknowledgment": acknowledgment,
                        "completed": True,
                        "current_max_questions": context_data['current_max_questions']
                    }
                
                conn.commit()
        # ✅ CONNECTION CLOSED HERE
        logger.info(f"[TIMING] Phase 3 (save sentiment) completed, elapsed={time.time()-phase3_start:.3f}s")
        
        # Get next question (this function handles its own connections)
        phase4_start = time.time()
        next_question_data = get_question_from_db_with_difficulty(
            context_data['job_role'],
            context_data['industry_type'],
            context_data['company_name'],
            context_data['question_number'] + 1,
            session_id,
            _translate_difficulty_label(next_difficulty),
            interview_type=context_data['interview_type'],
            work_experience=context_data['work_experience'],
            question_type_filter=context_data['question_type'],
        )
        
        # Save next question
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")
            
            with conn.cursor() as cur:
                next_question_number = context_data['question_number'] + 1
                normalized_difficulty = _translate_difficulty_label(next_difficulty)
                db_difficulty = normalized_difficulty
                
                # Store next question
                cur.execute("""
                    INSERT INTO interview_data (session_id, job_role, industry_type, company_name, interview_type, work_experience, question_number, question, mandatory_skills, difficulty, question_type, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (session_id, question_number) DO UPDATE SET 
                        question = EXCLUDED.question,
                        mandatory_skills = EXCLUDED.mandatory_skills,
                        difficulty = EXCLUDED.difficulty,
                        interview_type = EXCLUDED.interview_type,
                        work_experience = EXCLUDED.work_experience,
                        question_type = EXCLUDED.question_type
                """, (
                    session_id,
                    context_data['job_role'],
                    context_data['industry_type'],
                    context_data['company_name'],
                    context_data['interview_type'],
                    context_data['work_experience'],
                    next_question_number,
                    next_question_data['question'],
                    next_question_data['mandatory_skills'],
                    db_difficulty,
                    next_question_data.get('question_type', 'standard')
                ))

                conn.commit()
        # ✅ CONNECTION CLOSED HERE
        logger.info(f"[TIMING] Phase 4 (next question) completed, elapsed={time.time()-phase4_start:.3f}s")
        logger.info(f"[TIMING] submit_answer TOTAL time: {time.time()-start_time:.3f}s")

        return {
            "next_question": next_question_data['question'],
            "next_question_meta": {
                "question": next_question_data['question'],
                "question_type": next_question_data.get('question_type', 'standard'),
                "mandatory_skills": next_question_data['mandatory_skills'],
                "difficulty": next_question_data.get('difficulty', db_difficulty)
            },
            "acknowledgment": acknowledgment,
            "question_number": next_question_number,
            "completed": False,
            "current_max_questions": context_data['current_max_questions']
        }
            
    except Exception as e:
        logger.error(f"Error submitting answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _analyze_answer_with_timeout(question: str, answer: str, mandatory_skills: str, job_role: str) -> Tuple[float, str, str]:
    try:
        return await asyncio.wait_for(
            analyze_answer_and_generate_response(question, answer, mandatory_skills, job_role),
            timeout=ANALYSIS_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            f"Gemini acknowledgment timed out after {ANALYSIS_TIMEOUT_SECONDS}s; using fallback acknowledgment."
        )
    except Exception as exc:
        logger.error(f"Gemini acknowledgment failed, using fallback: {exc}")
    return DEFAULT_SENTIMENT_SCORE, DEFAULT_ACKNOWLEDGMENT, DEFAULT_NEXT_DIFFICULTY

async def _generate_and_process_feedback(session_id: str) -> dict:
    """Internal helper to generate feedback and update scores."""
    """Generate feedback and automatically extract/update scores"""
    logger.info(f"[FEEDBACK-{session_id}] Entering generate_feedback_and_scores.")
    _mark_feedback_processing(session_id)
    
    # Wait for all video analysis to complete before generating feedback
    # This includes: (1) videos with pending analysis, and (2) speech questions where video hasn't arrived yet
    max_wait_seconds = 300  # 5 minutes max wait
    wait_interval = 2  # Check every 2 seconds
    elapsed = 0
    
    while elapsed < max_wait_seconds:
        try:
            with db_pool.get_connection() as conn:
                if conn:
                    with conn.cursor() as cur:
                        # Count videos still being analyzed
                        cur.execute(
                            """
                                SELECT COUNT(*) 
                                FROM interview_data 
                                WHERE session_id = %s 
                                AND video_analysis_status = 'pending'
                            """,
                            (session_id,)
                        )
                        pending_analysis_count = cur.fetchone()[0]
                        
                        # Count speech questions where video hasn't arrived yet
                        # (question_type contains 'speech' but video_clip_path is NULL)
                        cur.execute(
                            """
                                SELECT COUNT(*) 
                                FROM interview_data 
                                WHERE session_id = %s 
                                AND LOWER(question_type) LIKE '%%speech%%'
                                AND video_clip_path IS NULL
                                AND answer IS NOT NULL
                            """,
                            (session_id,)
                        )
                        awaiting_upload_count = cur.fetchone()[0]
                        
                        total_pending = pending_analysis_count + awaiting_upload_count
                        
                        if total_pending == 0:
                            logger.info(f"[FEEDBACK-{session_id}] All video uploads and analysis completed, proceeding with feedback generation.")
                            break
                        else:
                            logger.info(f"[FEEDBACK-{session_id}] Waiting for videos: {awaiting_upload_count} uploads pending, {pending_analysis_count} analysis pending...")
        except Exception as check_err:
            logger.warning(f"[FEEDBACK-{session_id}] Error checking video analysis status: {check_err}")
        
        await asyncio.sleep(wait_interval)
        elapsed += wait_interval
    
    if elapsed >= max_wait_seconds:
        logger.warning(f"[FEEDBACK-{session_id}] Timeout waiting for video uploads/analysis, proceeding with feedback generation anyway.")
    
    try:
        logger.info(f"[FEEDBACK-{session_id}] Fetching conversation history from database.")
        with db_pool.get_connection() as conn:
            if not conn:
                logger.error(f"[FEEDBACK-{session_id}] Database connection failed.")
                raise HTTPException(status_code=500, detail="Database connection failed")
            
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        id.question_number,
                        id.question,
                        id.answer,
                        id.mandatory_skills,
                        id.job_role,
                        id.company_name,
                        id.industry_type,
                        id.question_type,
                        id.code_submission,
                        id.stdin_input,
                        id.stdout_output,
                        id.stderr_output,
                        id.runtime_error,
                        id.interview_type,
                        id.video_analysis,
                        iq.pre_def_answer
                    FROM interview_data id
                    LEFT JOIN interview_questions iq ON LOWER(TRIM(id.question)) = LOWER(TRIM(iq.question))
                    WHERE id.session_id = %s AND id.answer IS NOT NULL AND id.answer != ''
                    ORDER BY id.question_number
                """, (session_id,))
                
                qa_rows = cur.fetchall()
                
                if not qa_rows:
                    logger.error(f"[FEEDBACK-{session_id}] No answered questions found in database.")
                    raise HTTPException(status_code=404, detail="No answered questions found for this session")
                
                logger.info(f"[FEEDBACK-{session_id}] Found {len(qa_rows)} Q&A pairs.")

                qa_data: List[Dict[str, Any]] = []
                reference_answers: List[Dict[str, Any]] = []
                job_role = "Unknown Role"
                company_name = "Unknown Company"
                industry_type = "Unknown Industry"
                interview_type_value: Optional[str] = None

                for row in qa_rows:
                    (
                        q_number,
                        question_text,
                        answer_text,
                        mandatory_skills_value,
                        job_role_value,
                        company_value,
                        industry_value,
                        question_type_value,
                        code_submission_value,
                        stdin_value,
                        stdout_value,
                        stderr_value,
                        runtime_error_value,
                        interview_type_row,
                        raw_video_analysis,
                        pre_def_answer_value,
                    ) = row

                    if job_role == "Unknown Role" and job_role_value:
                        job_role = job_role_value
                    if company_name == "Unknown Company" and company_value:
                        company_name = company_value
                    if industry_type == "Unknown Industry" and industry_value:
                        industry_type = industry_value
                    if not interview_type_value and interview_type_row:
                        interview_type_value = interview_type_row

                    qa_data.append({
                        "number": q_number,
                        "question": question_text,
                        "answer": answer_text,
                        "mandatory_skills": mandatory_skills_value,
                        "question_type": (question_type_value or "standard").lower(),
                        "code_submission": code_submission_value,
                        "stdin": stdin_value,
                        "stdout": stdout_value,
                        "stderr": stderr_value,
                        "runtime_error": runtime_error_value,
                        "video_analysis": _normalize_video_analysis(raw_video_analysis),
                        "pre_def_answer": pre_def_answer_value,
                    })

                    if pre_def_answer_value:
                        reference_answers.append({
                            "number": q_number,
                            "answer": pre_def_answer_value,
                        })

                unique_skills: List[str] = []
                seen_skills = set()
                for qa in qa_data:
                    raw_skills = qa.get("mandatory_skills") or ""
                    for skill in [s.strip() for s in raw_skills.split(",") if s.strip()]:
                        lowered = skill.lower()
                        if lowered not in seen_skills:
                            seen_skills.add(lowered)
                            unique_skills.append(skill)

                conversation_entries: List[str] = []
                for qa in qa_data:
                    lines: List[str] = [
                        f"Q{qa['number']}: {qa.get('question') or 'Question not available'}",
                        f"Mandatory skills: {qa.get('mandatory_skills') or 'Not specified'}",
                    ]

                    if qa["question_type"] == "coding":
                        code_text = qa.get("code_submission") or qa.get("answer") or "(code submission missing)"
                        manual_input_text = qa.get("stdin") or "(no manual input provided)"
                        stdout_text = qa.get("stdout") or "(no output generated)"
                        stderr_components = [value for value in [qa.get("stderr"), qa.get("runtime_error")] if value]
                        stderr_text = "\n".join(stderr_components) if stderr_components else "(no errors reported)"

                        lines.extend([
                            "Candidate code solution:",
                            code_text,
                            "Manual input provided:",
                            manual_input_text,
                            "Program output (stdout):",
                            stdout_text,
                            "Errors / diagnostics:",
                            stderr_text,
                            "Auto-submitted answer snapshot:",
                            qa.get("answer") or "(no answer summary recorded)",
                        ])
                    elif _is_system_design_question_type(qa["question_type"]):
                        # Pass raw diagram JSON directly to AI for system design questions
                        diagram_json = qa.get("answer") or "{}"
                        lines.extend([
                            "Question Type: SYSTEM DESIGN",
                            "Candidate's System Design Diagram (ReactFlow JSON):",
                            diagram_json,
                            "IMPORTANT: For this system design question, better_example MUST be a JSON object with nodes and edges arrays, NOT text. Example format: {\"nodes\":[{\"id\":\"1\",\"type\":\"default\",\"position\":{\"x\":100,\"y\":100},\"data\":{\"label\":\"Service\",\"description\":\"desc\",\"bgColor\":\"#3b82f6\",\"borderColor\":\"#60a5fa\",\"icon\":\"🔧\"}}],\"edges\":[{\"id\":\"e1\",\"source\":\"1\",\"target\":\"2\",\"label\":\"connects\",\"animated\":true}]}",
                        ])
                    else:
                        lines.append(f"Candidate Answer: {qa.get('answer')}")

                    video_report = qa.get("video_analysis") or {}
                    if isinstance(video_report, dict):
                        video_lines: List[str] = []
                        sentiment_value = video_report.get("sentiment")
                        if sentiment_value is not None:
                            try:
                                video_lines.append(f"Video sentiment score: {float(sentiment_value):.1f}/10")
                            except (TypeError, ValueError):
                                video_lines.append(f"Video sentiment info: {sentiment_value}")

                        engagement_text = video_report.get("engagement")
                        if engagement_text:
                            video_lines.append(f"Video engagement insights: {engagement_text}")

                        expression_text = video_report.get("dominant_expression")
                        if expression_text:
                            video_lines.append(f"Observed demeanor: {expression_text}")

                        if video_lines:
                            lines.append("Video analysis:")
                            lines.extend(video_lines)

                    lines.append('-' * 20)
                    conversation_entries.append("\n".join(lines))

                conversation_text_for_questions = _compose_conversation_excerpt(
                    conversation_entries,
                    MAX_PROMPT_CHARS_QUESTIONS,
                )

                conversation_text_for_competencies = _compose_conversation_excerpt(
                    conversation_entries,
                    MAX_PROMPT_CHARS_COMPETENCIES,
                )

        template_config = _resolve_feedback_template(interview_type_value)

        question_prompt_template_name = template_config["question_template"]
        competency_prompt_template_name = template_config["competency_template"]

        try:
            question_template = jinja_env.get_template(question_prompt_template_name)
            competency_template = jinja_env.get_template(competency_prompt_template_name)
        except Exception as template_error:
            logger.error(
                f"[FEEDBACK-{session_id}] Failed to load templates {question_prompt_template_name} or {competency_prompt_template_name}: {template_error}"
            )
            raise HTTPException(status_code=500, detail="Feedback template unavailable") from template_error

        company_context_query = " ".join(
            [
                f"{qa.get('question') or ''} {qa.get('answer') or ''}".strip()
                for qa in qa_data[:8]
            ]
        ).strip()
        company_context_chunks = await retrieve_company_context(
            company_name,
            company_context_query or f"{job_role} {industry_type} interview evaluation",
        )
        company_context = "\n\n".join(company_context_chunks) if company_context_chunks else ""

        shared_context = dict(
            job_role=job_role,
            company_name=company_name,
            industry_type=industry_type,
            unique_skills=unique_skills,
            core_competencies=template_config["competencies"],
            scoring_guide=SCORING_GUIDE,
            reference_answers=reference_answers,
            company_context=company_context,
        )

        question_prompt = question_template.render(
            **shared_context,
            conversation_text=conversation_text_for_questions,
        )
        competency_prompt = competency_template.render(
            **shared_context,
            conversation_text=conversation_text_for_competencies,
        )

        logger.info(f"[FEEDBACK-{session_id}] Sending request to Gemini for question-wise feedback generation.")
        request_options = {"timeout": 120}
        try:
            question_response = await execute_with_retries(
                lambda: genai.GenerativeModel("gemini-3.5-flash-lite").generate_content(
                    question_prompt, request_options=request_options
                ),
                retry_label=f"Gemini question feedback generation ({session_id})",
            )
            question_raw = question_response.text.strip() if question_response.text else ""
            question_feedback = _parse_feedback_json(question_raw)
            if not question_feedback:
                logger.error(
                    f"[FEEDBACK-{session_id}] Question feedback JSON parsing failed. Raw response:\n{question_raw[:2000]}"
                )
                raise HTTPException(status_code=500, detail="AI returned invalid question-wise analysis")

            if "questions" not in question_feedback or not isinstance(question_feedback.get("questions"), list):
                logger.error(
                    f"[FEEDBACK-{session_id}] Question feedback missing 'questions' array. Raw response:\n{question_raw[:2000]}"
                )
                raise HTTPException(status_code=500, detail="AI question-wise analysis missing questions array")
        except Exception as question_exc:
            logger.error(f"[FEEDBACK-{session_id}] Question feedback generation failed: {question_exc}")
            raise HTTPException(status_code=500, detail="Failed to generate question-wise feedback") from question_exc

        logger.info(f"[FEEDBACK-{session_id}] Sending request to Gemini for competency feedback generation.")
        try:
            competency_response = await execute_with_retries(
                lambda: genai.GenerativeModel("gemini-3.5-flash-lite").generate_content(
                    competency_prompt, request_options=request_options
                ),
                retry_label=f"Gemini competency feedback generation ({session_id})",
            )
            competency_raw = competency_response.text.strip() if competency_response.text else ""
            competency_feedback = _parse_feedback_json(competency_raw)
            if not competency_feedback:
                logger.error(
                    f"[FEEDBACK-{session_id}] Competency feedback JSON parsing failed. Raw response:\n{competency_raw[:2000]}"
                )
                raise HTTPException(status_code=500, detail="AI returned invalid competency analysis")

            if "core_competencies" not in competency_feedback or not isinstance(competency_feedback.get("core_competencies"), list):
                logger.error(
                    f"[FEEDBACK-{session_id}] Competency feedback missing 'core_competencies' array. Raw response:\n{competency_raw[:2000]}"
                )
                raise HTTPException(status_code=500, detail="AI competency analysis missing core competencies array")
        except Exception as competency_exc:
            logger.error(f"[FEEDBACK-{session_id}] Competency feedback generation failed: {competency_exc}")
            raise HTTPException(status_code=500, detail="Failed to generate competency feedback") from competency_exc

        structured_feedback: Optional[Dict[str, Any]] = None
        if question_feedback or competency_feedback:
            structured_feedback = {
                "metadata": {
                    "job_role": job_role,
                    "company_name": company_name,
                    "industry_type": industry_type,
                    "interview_type": interview_type_value,
                },
                "questions": question_feedback.get("questions") if question_feedback else [],
                "mandatory_skill_scores": question_feedback.get("mandatory_skill_scores") if question_feedback else [],
                "core_competencies": competency_feedback.get("core_competencies") if competency_feedback else [],
                "technical_summary": competency_feedback.get("technical_summary") if competency_feedback else None,
                "communication_summary": competency_feedback.get("communication_summary") if competency_feedback else None,
                "attitude_summary": competency_feedback.get("attitude_summary") if competency_feedback else None,
            }

            if competency_feedback and competency_feedback.get("mandatory_skill_scores") and not structured_feedback["mandatory_skill_scores"]:
                structured_feedback["mandatory_skill_scores"] = competency_feedback.get("mandatory_skill_scores")

            video_summary = (question_feedback or {}).get("video_analysis_summary")
            if video_summary:
                structured_feedback["video_analysis_summary"] = video_summary

            answer_lookup: Dict[int, str] = {}
            question_type_lookup: Dict[int, str] = {}
            for qa in qa_data:
                number = qa.get("number")
                if number is None:
                    continue

                candidate_answer = qa.get("answer") or ""
                qa_question_type = (qa.get("question_type") or "").lower()
                if qa_question_type == "coding" or qa_question_type.startswith("coding "):
                    code_snippet = qa.get("code_submission") or ""
                    stdout_text = qa.get("stdout") or ""
                    stderr_text = qa.get("stderr") or qa.get("runtime_error") or ""

                    parts: List[str] = []
                    if code_snippet and code_snippet.strip():
                        parts.append(code_snippet.strip())
                    elif candidate_answer.strip():
                        parts.append(candidate_answer.strip())

                    if stdout_text and stdout_text.strip():
                        parts.append("[Program output]\n" + stdout_text.strip())

                    if stderr_text and stderr_text.strip():
                        parts.append("[Errors / diagnostics]\n" + stderr_text.strip())

                    candidate_answer = "\n\n".join(parts).strip()
                else:
                    candidate_answer = candidate_answer.strip()

                idx = int(number)
                answer_lookup[idx] = candidate_answer
                question_type_lookup[idx] = qa.get("question_type") or None

            for entry in structured_feedback.get("questions", []) or []:
                if not isinstance(entry, dict):
                    continue
                num = entry.get("number")
                if num is None:
                    continue

                idx = int(num)
                original_answer = answer_lookup.get(idx, "")
                if original_answer:
                    entry.setdefault("answer", original_answer)
                    entry["original_answer"] = original_answer

                qt_value = question_type_lookup.get(idx)
                if qt_value:
                    entry.setdefault("question_type", qt_value)
                    normalized_qt = str(qt_value).lower()
                    if normalized_qt == "coding" or normalized_qt.startswith("coding "):
                        entry["is_coding"] = True

        raw_text = json.dumps(
            {
                "question_feedback": question_feedback,
                "competency_feedback": competency_feedback,
            },
            ensure_ascii=False,
        )

        if not structured_feedback:
            logger.warning(f"[FEEDBACK-{session_id}] Structured JSON missing; storing raw text for investigation.")

        stored_payload = json.dumps(structured_feedback, ensure_ascii=False) if structured_feedback else raw_text

        logger.info(f"[FEEDBACK-{session_id}] Storing feedback in database.")
        print(f"[FEEDBACK-{session_id}] Full feedback payload:\n{stored_payload}")
        with db_pool.get_connection() as conn:
            if conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE interview_data 
                        SET detailed_feedback = %s, analysis_status = 'feedback_generated'
                        WHERE session_id = %s AND question_number = (
                            SELECT MAX(question_number) FROM interview_data WHERE session_id = %s
                        )
                    """, (stored_payload, session_id, session_id))
                    conn.commit()
                    logger.info(f"[FEEDBACK-{session_id}] Database update successful.")

        feedback_text = stored_payload
        scores_updated = update_scores_from_feedback(session_id, feedback_text)
        logger.info(f"[FEEDBACK-{session_id}] Score update process completed. Success: {scores_updated}")

        _mark_feedback_completed(session_id)
        
        return {
            "session_id": session_id,
            "feedback": {
                "structured": structured_feedback,
                "raw": raw_text,
            },
            "scores_updated": scores_updated,
            "message": "Feedback generated and scores updated successfully"
        }
            
    except HTTPException as http_exc:
        # When running as a background task we should not re-raise, otherwise Starlette
        # logs "response already started" errors even though the client already has a 200.
        detail = getattr(http_exc, "detail", None)
        message = detail if isinstance(detail, str) else str(detail)
        logger.error(f"[FEEDBACK-{session_id}] Feedback generation failed with HTTPException: {message}")
        _mark_feedback_failed(session_id, message)
        return {
            "session_id": session_id,
            "error": message,
        }
    except Exception as e:
        logger.error(f"[FEEDBACK-{session_id}] An unexpected error occurred in generate_feedback_and_scores: {e}")
        _mark_feedback_failed(session_id, str(e))
        return {
            "session_id": session_id,
            "error": str(e),
        }

@app.post("/interview/{session_id}/generate-feedback")
async def generate_feedback_and_scores(session_id: str):
    """Public endpoint to trigger feedback generation manually if needed."""
    # First, check if feedback is already completed and generated for this session.
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")

            with conn.cursor() as cur:
                cur.execute(
                    """
                        SELECT feedback_status, feedback_generated
                        FROM session_metadata
                        WHERE session_id = %s
                    """,
                    (session_id,),
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Session not found")

                status_value, generated_flag = row

                # If feedback is already completed and generated, avoid regenerating.
                if status_value == "completed" and bool(generated_flag):
                    logger.info(
                        f"[FEEDBACK-{session_id}] Feedback already completed; skipping regeneration request."
                    )
                    return {
                        "status": "completed",
                        "session_id": session_id,
                        "feedback_generated": True,
                    }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[FEEDBACK-{session_id}] Pre-check before regeneration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    _mark_feedback_pending(session_id)
    asyncio.create_task(_generate_and_process_feedback(session_id))
    return {"status": "accepted", "session_id": session_id}

@app.get("/interview/{session_id}/scores")
async def get_session_scores(session_id: str):
    """Get current scores for a session"""
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")
            
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT overall_score, technical_score, communication_score, 
                           attitude_score, status, feedback_generated
                    FROM session_metadata 
                    WHERE session_id = %s
                """, (session_id,))
                
                result = cur.fetchone()
                
                if not result:
                    raise HTTPException(status_code=404, detail="Session not found")
                
                return {
                    "session_id": session_id,
                    "overall_score": float(result[0]) if result[0] else None,
                    "technical_score": float(result[1]) if result[1] else None,
                    "communication_score": float(result[2]) if result[2] else None,
                    "attitude_score": float(result[3]) if result[3] else None,
                    "status": result[4],
                    "feedback_generated": result[5]
                }
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session scores: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feedback-status/{session_id}")
async def get_feedback_status(session_id: str):
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")

            with conn.cursor() as cur:
                cur.execute(
                    """
                        SELECT feedback_status, feedback_error, feedback_generated,
                               feedback_requested_at, feedback_ready_at
                        FROM session_metadata
                        WHERE session_id = %s
                    """,
                    (session_id,),
                )

                result = cur.fetchone()
                if not result:
                    raise HTTPException(status_code=404, detail="Session not found")

                status_value, error_value, generated_flag, requested_at, ready_at = result
                
                # Check for pending video analysis
                cur.execute(
                    """
                        SELECT COUNT(*) 
                        FROM interview_data 
                        WHERE session_id = %s 
                        AND video_analysis_status = 'pending'
                    """,
                    (session_id,)
                )
                pending_videos = cur.fetchone()[0]
                
                # If feedback is processing but videos are still pending, keep status as processing
                if status_value == 'processing' and pending_videos > 0:
                    status_value = 'processing'
                    error_value = f"Processing video analysis ({pending_videos} remaining)"

                def _iso(dt):
                    return dt.isoformat() if isinstance(dt, datetime) else None

                return {
                    "session_id": session_id,
                    "status": status_value or "not_requested",
                    "error": error_value,
                    "feedback_generated": bool(generated_flag),
                    "requested_at": _iso(requested_at),
                    "ready_at": _iso(ready_at),
                    "pending_video_analysis": pending_videos
                }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving feedback status for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Compatibility endpoint for feedback (matches your React frontend)
@app.get("/feedback/{session_id}")
async def get_session_feedback_compatibility(session_id: str):
    logger.info(f"[FEEDBACK-{session_id}] Received request for final feedback.")
    """Get feedback for session - compatibility with existing React frontend"""
    try:
        logger.info(f"Feedback requested for session: {session_id}")
        
        payload = _get_feedback_payload(session_id)

        if not payload:
            logger.warning(f"Feedback for session {session_id} requested but not yet generated.")
            return JSONResponse(
                status_code=202,
                content={
                    "session_id": session_id,
                    "status": "pending",
                    "detail": "Final feedback has not been generated for this session yet. Please try again shortly."
                }
            )

        response_data = {
            "session_id": session_id,
            "feedback": {
                "structured": payload.get("structured"),
                "raw": payload.get("raw"),
            },
            "status": "success"
        }
        
        logger.info(f"Retrieved final feedback for session {session_id}.")
        return response_data
            
    except HTTPException as e:
        logger.error(f"HTTP Error in get_session_feedback_compatibility for {session_id}: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Error getting feedback for session {session_id}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/scores/{session_id}")
async def debug_scores(session_id: str):
    """Debug endpoint to test score extraction"""
    try:
        # Get feedback for this session
        with db_pool.get_connection() as conn:
            if conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT detailed_feedback 
                        FROM interview_data 
                        WHERE session_id = %s AND detailed_feedback IS NOT NULL 
                        ORDER BY question_number DESC 
                        LIMIT 1
                    """, (session_id,))
                    result = cur.fetchone()
                    
                    if result and result[0]:
                        feedback_text = result[0]
                        
                        # Test score extraction
                        scores = extract_scores_from_feedback(feedback_text)
                        
                        # Check session_metadata
                        cur.execute("SELECT * FROM session_metadata WHERE session_id = %s", (session_id,))
                        metadata = cur.fetchone()
                        
                        return {
                            "session_id": session_id,
                            "feedback_length": len(feedback_text),
                            "extracted_scores": scores,
                            "session_metadata_exists": metadata is not None,
                            "session_metadata": dict(zip([desc[0] for desc in cur.description], metadata)) if metadata else None,
                            "debug": "score_extraction_test"
                        }
                    else:
                        return {"error": "No feedback found for this session"}
            else:
                return {"error": "Database connection failed"}
    except Exception as e:
        logger.error(f"Debug scores error: {e}")
        return {"error": str(e)}

@app.post("/debug/force-score-update/{session_id}")
async def force_score_update(session_id: str):
    """Force update scores for a session"""
    try:
        # Get feedback for this session
        with db_pool.get_connection() as conn:
            if conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT detailed_feedback 
                        FROM interview_data 
                        WHERE session_id = %s AND detailed_feedback IS NOT NULL 
                        ORDER BY question_number DESC 
                        LIMIT 1
                    """, (session_id,))
                    result = cur.fetchone()
                    
                    if result and result[0]:
                        feedback_text = result[0]
                        
                        # Force update scores
                        success = update_scores_from_feedback(session_id, feedback_text)
                        
                        return {
                            "session_id": session_id,
                            "success": success,
                            "message": "Score update completed" if success else "Score update failed",
                            "debug": "force_score_update"
                        }
                    else:
                        return {"error": "No feedback found for this session"}
            else:
                return {"error": "Database connection failed"}
    except Exception as e:
        logger.error(f"Force score update error: {e}")
        return {"error": str(e)}

@app.get("/debug/acknowledgment/{session_id}")
async def debug_acknowledgment(session_id: str):
    """Debug endpoint to test acknowledgment generation"""
    try:
        # Get latest answer for this session
        with db_pool.get_connection() as conn:
            if conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT question, answer, job_role, mandatory_skills
                        FROM interview_data 
                        WHERE session_id = %s AND answer IS NOT NULL 
                        ORDER BY question_number DESC 
                        LIMIT 1
                    """, (session_id,))
                    result = cur.fetchone()
                    
                    if result:
                        question, answer, job_role, mandatory_skills = result
                        
                        # Test acknowledgment generation
                        sentiment_score, acknowledgment, difficulty = await analyze_answer_and_generate_response(
                            question, answer, mandatory_skills or "Communication", job_role or "Software Engineer"
                        )
                        
                        return {
                            "session_id": session_id,
                            "question": question,
                            "answer": answer[:100] + "..." if len(answer) > 100 else answer,
                            "sentiment_score": sentiment_score,
                            "acknowledgment": acknowledgment,
                            "next_difficulty": difficulty,
                            "debug": "acknowledgment_test"
                        }
                    else:
                        return {"error": "No answered questions found for this session"}
            else:
                return {"error": "Database connection failed"}
    except Exception as e:
        logger.error(f"Debug acknowledgment error: {e}")
        return {"error": str(e)}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to the Enhanced AI Interview API with Admin Support",
        "version": "2.0",
        "features": [
            "Real-time AI interview processing",
            "Sentiment analysis and dynamic difficulty",
            "Admin dashboard integration",
            "Student progress tracking",
            "Comprehensive feedback generation",
            "Performance caching",
            "Database question management"
        ],
        "endpoints": {
            "interview": "/interview",
            "feedback": "/feedback/{session_id}",
            "health": "/health",
            "cache_clear": "/cache/clear",
            "debug": "/debug/{session_id}"
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

@app.post("/cache/clear")
async def clear_cache():
    """Clear all cache entries"""
    try:
        session_cache.clear()
        questions_cache.clear()
        get_questions_cached.cache_clear()  # Clear LRU cache
        
        logger.info("All caches cleared successfully")
        return {
            "message": "All caches cleared successfully",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")

@app.get("/debug/{session_id}")
async def debug_session(session_id: str):
    """Debug endpoint for session analysis"""
    try:
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                # Get session metadata
                cur.execute("""
                    SELECT session_id, student_name, job_role, company_name, 
                           industry_type, status, started_at
                    FROM session_metadata 
                    WHERE session_id = %s
                """, (session_id,))
                
                session_data = cur.fetchone()
                if not session_data:
                    raise HTTPException(status_code=404, detail="Session not found")
                
                # Get interview data
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
                    ],
                    "cache_info": {
                        "session_cache_size": len(session_cache.cache),
                        "questions_cache_size": len(questions_cache.cache),
                        "lru_cache_info": str(get_questions_cached.cache_info())
                    },
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in debug endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check endpoint"""
    try:
        # Check database connection
        db_connected = False
        try:
            with db_pool.get_connection() as conn:
                if conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                        db_connected = True
        except:
            pass
        
        cache_size = len(session_cache.cache) + len(questions_cache.cache)
        
        return HealthResponse(
            status="healthy" if db_connected else "degraded",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            database_connected=db_connected,
            cache_size=cache_size
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            database_connected=False,
            cache_size=0
        )

# Application startup
@app.on_event("startup")
async def startup_event():
    """Initialize application"""
    logger.info("Starting Enhanced AI Interview API...")
    
    if init_enhanced_db():
        logger.info("Database initialized successfully")
    else:
        logger.error("Failed to initialize database")
    
    logger.info("Application startup completed")
