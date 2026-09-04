import base64
import csv
import io
import json
import logging
import os
import re
from datetime import datetime, timedelta
from itertools import product
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import Depends, HTTPException, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

from app.config.settings import config
from app.modules.admin.repository import AdminRepository
from app.modules.admin.schemas import AdminAuthResponse, AdminLoginRequest, InterviewQuestionPayload
from app.modules.auth.service import hash_password, verify_password
from app.modules.rag.ingestion import chunk_playbook_section, extract_pdf_text

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}


def _create_jwt_token(payload: Dict[str, Any]) -> str:
    expire = datetime.utcnow() + timedelta(minutes=config.ADMIN_JWT_EXPIRES_MINUTES)
    to_encode = {**payload, "exp": expire}
    return jwt.encode(to_encode, config.ADMIN_JWT_SECRET, algorithm=config.ADMIN_JWT_ALGORITHM)


def _decode_jwt_token(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, config.ADMIN_JWT_SECRET, algorithms=[config.ADMIN_JWT_ALGORITHM])
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired admin token") from exc


async def _resolve_admin(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Dict[str, Any]:
    token: Optional[str] = None
    if credentials and credentials.scheme.lower() == "bearer":
        token = credentials.credentials

    if not token:
        raise HTTPException(status_code=401, detail="Missing admin token")

    payload = _decode_jwt_token(token)
    admin_id = payload.get("sub")
    if not admin_id:
        raise HTTPException(status_code=401, detail="Invalid admin token payload")

    row = AdminRepository.get_admin_by_id(admin_id)
    if not row:
        raise HTTPException(status_code=401, detail="Admin account not found")
    if row["is_active"] is False:
        raise HTTPException(status_code=403, detail="Admin account is disabled")

    return {
        "admin_id": row["admin_id"],
        "email": row["email"],
        "display_name": row["display_name"],
    }


async def verify_admin_token(admin: Dict[str, Any] = Depends(_resolve_admin)) -> Dict[str, Any]:
    return admin


def _normalize_text(value: Optional[str], preserve_newlines: bool = False) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    if not preserve_newlines:
        cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or None


def _normalize_question_payload(payload: InterviewQuestionPayload) -> Dict[str, Any]:
    difficulty = _normalize_text(payload.difficulty)
    if difficulty:
        lower_diff = difficulty.lower()
        if lower_diff not in ALLOWED_DIFFICULTIES:
            raise HTTPException(status_code=400, detail="Difficulty must be one of easy, medium, or hard")
        difficulty = lower_diff

    record = {
        "industry": _normalize_text(payload.industry),
        "company": _normalize_text(payload.company),
        "role": _normalize_text(payload.role),
        "question": _normalize_text(payload.question, preserve_newlines=True),
        "mandatory_skills": _normalize_text(payload.mandatory_skills),
        "pre_def_answer": _normalize_text(payload.pre_def_answer, preserve_newlines=True),
        "difficulty": difficulty,
        "question_type": (_normalize_text(payload.question_type) or None),
        "interview_type": _normalize_text(payload.interview_type),
        "work_experience": _normalize_text(payload.work_experience),
    }

    if not record["question"]:
        raise HTTPException(status_code=400, detail="Question text is required")
    return record


def _slugify_company_name(name: str) -> str:
    """Lowercase, alphanumeric-only slug matching the existing public/logos/*.png convention."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


async def _save_company_logo_temp_local(company_name: str, logo: Optional[UploadFile]) -> Optional[str]:
    """TEMPORARY LOCAL-ONLY SOLUTION.

    Saves an admin-uploaded company logo directly into the frontend's
    public/logos/ folder so it's servable as a static asset immediately,
    without adding a static-file route to this backend. This only works
    because the frontend and backend both run from source on the same
    machine during local development.

    Local disk writes do not survive on serverless hosting. BEFORE
    DEPLOYING to Vercel, replace this with real cloud storage (Vercel
    Blob / S3 / Cloudinary) and point logo_url at that instead.
    """
    if logo is None or not logo.filename:
        return None
    if logo.content_type not in {"image/png", "application/octet-stream"}:
        logger.warning("Rejected non-PNG logo upload for %s (content_type=%s)", company_name, logo.content_type)
        return None

    try:
        slug = _slugify_company_name(company_name)
        if not slug:
            return None

        logos_dir = config.FRONTEND_LOGOS_DIR
        os.makedirs(logos_dir, exist_ok=True)
        file_path = logos_dir / f"{slug}.png"

        content = await logo.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)

        return f"/logos/{slug}.png"
    except Exception as exc:
        logger.warning("Failed to save company logo for %s, falling back to no logo: %s", company_name, exc)
        return None


def _normalize_company_payload(
    name: str,
    industry: Optional[str],
    difficulty_tag: Optional[str],
    work_experience_tag: Optional[str],
) -> Dict[str, Any]:
    normalized_name = _normalize_text(name)
    if not normalized_name:
        raise HTTPException(status_code=400, detail="Company name is required")

    return {
        "name": normalized_name,
        "industry": _normalize_text(industry),
        "difficulty_tag": _normalize_text(difficulty_tag),
        "work_experience_tag": _normalize_text(work_experience_tag),
    }


async def create_company_service(
    name: str,
    industry: Optional[str],
    difficulty_tag: Optional[str],
    work_experience_tag: Optional[str],
    logo: Optional[UploadFile],
) -> Dict[str, Any]:
    record = _normalize_company_payload(name, industry, difficulty_tag, work_experience_tag)
    record["logo_url"] = None
    created = AdminRepository.insert_company(record)

    if logo is not None:
        logo_url = await _save_company_logo_temp_local(created["name"], logo)
        if logo_url:
            created = AdminRepository.update_company_logo(created["id"], logo_url)

    return created


async def upload_company_playbook_service(company_id: int, playbook_file: UploadFile) -> Dict[str, Any]:
    company = AdminRepository.get_company_by_id(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")

    is_pdf_extension = (playbook_file.filename or "").lower().endswith(".pdf")
    if playbook_file.content_type not in {"application/pdf", "application/octet-stream"} and not is_pdf_extension:
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    try:
        text = extract_pdf_text(playbook_file.file)
    except Exception as exc:
        logger.error("Failed to read uploaded playbook PDF: %s", exc)
        raise HTTPException(status_code=400, detail="Unable to read uploaded PDF") from exc

    chunks = chunk_playbook_section(text)
    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="No text could be extracted from this PDF (it may be scanned/image-only, or contain no recognized section headings). Nothing was ingested.",
        )

    chunks_ingested = await AdminRepository.replace_company_playbook_chunks(company["name"], chunks)

    return {
        "message": "Playbook uploaded successfully",
        "company_id": company["id"],
        "company_name": company["name"],
        "chunks_ingested": chunks_ingested,
    }


def admin_login_service(payload: AdminLoginRequest) -> AdminAuthResponse:
    email = payload.email.strip()
    password = payload.password

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required")

    admin_row = AdminRepository.get_admin_by_email(email)
    if not admin_row:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if admin_row["is_active"] is False:
        raise HTTPException(status_code=403, detail="Admin account is disabled")

    if not verify_password(password, admin_row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    stored_hash = admin_row["password_hash"]
    if not stored_hash or not str(stored_hash).startswith("pbkdf2_sha256$"):
        AdminRepository.update_admin_password_hash(admin_row["admin_id"], hash_password(password))

    token_payload = {
        "sub": str(admin_row["admin_id"]),
        "email": admin_row["email"],
    }
    access_token = _create_jwt_token(token_payload)

    return AdminAuthResponse(
        access_token=access_token,
        admin_id=admin_row["admin_id"],
        email=admin_row["email"],
        display_name=admin_row["display_name"],
    )


async def bulk_upload_interview_questions_service(
    industries: str,
    companies: str,
    roles: str,
    interview_types: str,
    work_experiences: str,
    difficulties: str,
    question_types: str,
    questions_file: UploadFile,
) -> Dict[str, Any]:
    try:
        industries_list = json.loads(industries) if industries else []
        companies_list = json.loads(companies) if companies else []
        roles_list = json.loads(roles) if roles else []
        interview_types_list = json.loads(interview_types) if interview_types else []
        work_experiences_list = json.loads(work_experiences) if work_experiences else []
        difficulties_list = json.loads(difficulties) if difficulties else []
        question_types_list = json.loads(question_types) if question_types else []
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse category arrays: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid category data format") from exc

    if not industries_list:
        raise HTTPException(status_code=400, detail="At least one industry is required")
    if not companies_list:
        raise HTTPException(status_code=400, detail="At least one company is required")
    if not roles_list:
        raise HTTPException(status_code=400, detail="At least one role is required")

    if questions_file.content_type not in {"text/csv", "application/vnd.ms-excel", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    try:
        raw_bytes = await questions_file.read()
        decoded = raw_bytes.decode("utf-8", errors="ignore")
    except Exception as exc:
        logger.error("Failed to read uploaded CSV: %s", exc)
        raise HTTPException(status_code=400, detail="Unable to read uploaded file") from exc

    csv_reader = csv.DictReader(io.StringIO(decoded))
    if not csv_reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV file must include headers")

    fieldname_map = {f.strip().lower(): f for f in csv_reader.fieldnames}
    question_column = fieldname_map.get("question")
    skills_column = fieldname_map.get("mandatory_skills")
    answer_column = fieldname_map.get("predefined_answer")
    interview_type_column = fieldname_map.get("interview_type")
    difficulty_column = fieldname_map.get("difficulty")
    question_type_column = fieldname_map.get("question_type")

    if not question_column:
        raise HTTPException(status_code=400, detail="CSV file must contain a 'question' column")
    if not skills_column:
        raise HTTPException(status_code=400, detail="CSV file must contain a 'mandatory_skills' column")
    if not interview_type_column:
        raise HTTPException(status_code=400, detail="CSV file must contain an 'interview_type' column")
    if not difficulty_column:
        raise HTTPException(status_code=400, detail="CSV file must contain a 'difficulty' column")
    if not question_type_column:
        raise HTTPException(status_code=400, detail="CSV file must contain a 'question_type' column")

    valid_question_types = {"Coding (Python)", "Coding (SQL)", "Speech Based"}

    csv_questions = []
    skipped: List[int] = []
    row_number = 1

    for row in csv_reader:
        row_number += 1
        question_text = row.get(question_column, "").strip()
        mandatory_skills = row.get(skills_column, "").strip() if skills_column else ""
        predefined_answer = row.get(answer_column, "").strip() if answer_column else ""
        csv_interview_type = row.get(interview_type_column, "").strip() if interview_type_column else ""
        csv_difficulty = row.get(difficulty_column, "").strip() if difficulty_column else ""
        csv_question_type = row.get(question_type_column, "").strip() if question_type_column else ""

        if not question_text or not mandatory_skills or not csv_interview_type or not csv_difficulty or not csv_question_type:
            skipped.append(row_number)
            continue

        if csv_question_type not in valid_question_types:
            skipped.append(row_number)
            continue

        csv_questions.append({
            "question": question_text,
            "mandatory_skills": mandatory_skills,
            "pre_def_answer": predefined_answer or None,
            "interview_type": csv_interview_type,
            "difficulty": csv_difficulty,
            "question_type": csv_question_type,
        })

    if not csv_questions:
        raise HTTPException(status_code=400, detail="No valid questions found in CSV")

    work_experiences_for_product = work_experiences_list if work_experiences_list else [None]
    category_combinations = list(product(industries_list, companies_list, roles_list, work_experiences_for_product))

    records: List[Dict[str, Any]] = []
    for q in csv_questions:
        for industry, company, role, work_exp in category_combinations:
            try:
                effective_industry = None if industry == "No specific industry" else industry
                payload = InterviewQuestionPayload(
                    industry=effective_industry,
                    company=company,
                    role=role,
                    question=q["question"],
                    mandatory_skills=q["mandatory_skills"],
                    pre_def_answer=q["pre_def_answer"],
                    difficulty=q["difficulty"],
                    question_type=q["question_type"],
                    interview_type=q["interview_type"],
                    work_experience=work_exp,
                )
                normalized = _normalize_question_payload(payload)
                records.append(normalized)
            except HTTPException:
                pass

    if not records:
        raise HTTPException(status_code=400, detail="No valid question records could be created")

    inserted_ids = AdminRepository.insert_interview_questions(records)
    return {
        "message": "Bulk upload completed",
        "inserted": len(inserted_ids),
        "skipped_rows": skipped,
        "category_combinations": len(category_combinations),
        "questions_in_csv": len(csv_questions),
    }


def export_sessions_service(export_format: str) -> Dict[str, Any]:
    rows = AdminRepository.get_export_sessions_repo()
    columns = [
        "session_id", "student_name", "job_role", "company_name",
        "status", "overall_score", "technical_score", "communication_score",
        "attitude_score", "started_at", "completed_at", "duration_minutes"
    ]

    if export_format == "csv":
        df = pd.DataFrame(rows, columns=columns)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        csv_b64 = base64.b64encode(csv_content.encode()).decode()

        return {
            "format": "csv",
            "filename": f"interview_sessions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "data": csv_b64,
            "record_count": len(rows),
        }
    else:
        data = []
        for row in rows:
            record = {}
            for i, col in enumerate(columns):
                value = row[i]
                if isinstance(value, datetime):
                    value = value.isoformat()
                record[col] = value
            data.append(record)

        return {
            "format": "json",
            "filename": f"interview_sessions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "data": data,
            "record_count": len(rows),
        }
