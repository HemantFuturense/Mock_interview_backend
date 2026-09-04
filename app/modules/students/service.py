import csv
import io
import json
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import docx
import pdfplumber
from fastapi import BackgroundTasks, HTTPException, UploadFile

from app.core.database import db_pool
from app.core.logger import logger
from app.modules.ai.client import _generate_content_with_fallback, execute_with_retries
from app.modules.auth.service import generate_temp_password, hash_password, send_credentials_email
from app.modules.rag.service import retrieve_company_context
from app.modules.students.repository import (
    ensure_job_descriptions_table,
    find_student_id,
    resolve_program_id_by_name,
    resolve_ubp_id,
)
from app.modules.students.schemas import (
    ResumeQuestionRequest,
    StudentDuplicateInfo,
    StudentEmailWarning,
    StudentImportError,
    StudentImportResult,
)


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


async def parse_resume_background(resume_id: int, file_path: str) -> None:
    try:
        raw_text = extract_text_from_resume(file_path)
        structured_data = structure_resume_text(raw_text)
        parsed_data = {"raw_text": raw_text, "structured_sections": structured_data}

        with db_pool.get_connection() as conn:
            if conn:
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


async def parse_job_description_background(job_description_id: int, file_path: str) -> None:
    try:
        raw_text = extract_text_from_resume(file_path)
        parsed_data = {"raw_text": raw_text}

        with db_pool.get_connection() as conn:
            if conn:
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


def get_student_resume_service(student_id: int) -> Dict[str, Any]:
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection unavailable")
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, filename, file_path, file_size, upload_date,
                           parsed_data, is_active, created_at, updated_at
                    FROM student_resumes
                    WHERE student_id = %s AND is_active = true
                    ORDER BY created_at DESC LIMIT 1
                """,
                    (student_id,),
                )

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
                    "has_resume": True,
                }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get student resume: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve resume")


async def upload_student_resume_service(
    file: UploadFile, student_id: int, background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    if not file.filename.lower().endswith((".pdf", ".doc", ".docx")):
        raise HTTPException(status_code=400, detail="Only PDF, DOC, and DOCX files are allowed")

    try:
        file_extension = file.filename.split(".")[-1].lower()
        unique_filename = f"{student_id}_{int(time.time())}.{file_extension}"
        file_path = f"uploads/resumes/{unique_filename}"

        os.makedirs("uploads/resumes", exist_ok=True)

        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)

        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection unavailable")
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO student_resumes
                    (student_id, filename, file_path, file_size)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """,
                    (student_id, file.filename, file_path, len(content)),
                )

                resume_id = cur.fetchone()[0]
                conn.commit()

        background_tasks.add_task(parse_resume_background, resume_id, file_path)

        return {"message": "Resume uploaded successfully", "resume_id": resume_id, "status": "processing"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resume upload failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload resume")


async def upload_job_description_service(
    file: UploadFile, student_id: int, background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    if not file.filename.lower().endswith((".pdf", ".doc", ".docx", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF, DOC, DOCX, or TXT files are allowed")

    try:
        file_extension = file.filename.split(".")[-1].lower()
        unique_filename = f"{student_id}_{int(time.time())}_jd.{file_extension}"
        file_path = f"uploads/job_descriptions/{unique_filename}"

        os.makedirs("uploads/job_descriptions", exist_ok=True)

        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)

        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection unavailable")
            with conn.cursor() as cur:
                ensure_job_descriptions_table(cur)
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
                    (student_id, file.filename, file_path, len(content)),
                )
                job_description_id = cur.fetchone()[0]
                conn.commit()

        background_tasks.add_task(parse_job_description_background, job_description_id, file_path)

        return {
            "message": "Job description uploaded successfully",
            "job_description_id": job_description_id,
            "status": "processing",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job description upload failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload job description")


def get_student_job_description_service(student_id: int) -> Dict[str, Any]:
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection unavailable")
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get student job description: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve job description")


def get_student_resume_interview_sessions_service(student_id: int) -> Dict[str, Any]:
    try:
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection unavailable")
            with conn.cursor() as cur:
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

                    result.append(
                        {
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
                                    "mandatory_skills": q[4],
                                }
                                for q in questions
                            ],
                        }
                    )

                return {"sessions": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get resume interview sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve resume interview sessions")


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
    company_context_chunks = await retrieve_company_context(
        company,
        f"{effective_role} interview questions, evaluation style, and hiring signals",
    )

    prompt_sections = [
        f"You are an AI interview coach helping candidates prepare for a {effective_role} position at {company}.",
        f"Use the candidate's resume data, job description, and context below to craft {question_count} personalized interview questions.",
        "",
        "Resume JSON:",
        json.dumps(resume_data, ensure_ascii=False),
    ]

    if jd_excerpt:
        prompt_sections.extend(["", "Job Description Text:", jd_excerpt])

    if jd_json_snippet:
        prompt_sections.extend(["", "Job Description JSON:", jd_json_snippet])

    if company_context_chunks:
        prompt_sections.extend(
            [
                "",
                "Company Interview Context (retrieved from the internal knowledge base):",
                "Use this context only when it is relevant to the target role. Do not invent facts beyond it.",
                "\n\n".join(company_context_chunks),
            ]
        )

    prompt_sections.extend(
        [
            "",
            "Context:",
            f"- Target Role: {effective_role}",
            f"- Company: {company}",
            f"- Interview Type: {interview_type or 'Not specified'}",
            f"- Work Experience Level: {work_experience or 'Not specified'}",
            "",
            "Critical Alignment Instructions:",
            "- Treat the candidate's resume and the provided job description as the PRIMARY source of truth for question generation.",
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
            "- SQL questions MUST be framed around realistic datasets, tables, and business problems inferred from the candidate's past experience and the job description domain.",
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
            '    * question_type (types available: "Coding (Python)", "Coding (SQL)", "Speech Based", "System Design")',
            "    * question text (clear and concise, but context-aware)",
            "    * difficulty (Easy/Medium/Hard)",
            "    * mandatory_skills (array of at least minimum of two and maximum of 4 skills tied directly to resume and job description keywords)",
            "- Respond ONLY in strict JSON using this schema:",
            "{",
            '    "questions": [',
            "        {",
            '            "question_type": "...",',
            '            "question": "...",',
            '            "difficulty": "Easy|Medium|Hard",',
            '            "mandatory_skills": ["skill1", "skill2", ...]',
            "        }",
            "    ]",
            "}",
        ]
    )

    prompt = "\n".join(prompt_sections)

    async def _call_gemini() -> List[Dict[str, Any]]:
        response = await _generate_content_with_fallback(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.35,
                "max_output_tokens": 3000,
            },
            retry_label="Resume question generation",
        )
        return _extract_question_list_from_response((response.text or "").strip())

    questions_payload = await execute_with_retries(
        _call_gemini,
        max_retries=2,
        base_delay=1.5,
        retry_label="Resume questions generation",
    )

    if not questions_payload:
        return []

    normalized_questions: List[Dict[str, Any]] = []
    for item in questions_payload:
        if not isinstance(item, dict):
            continue
        question_text = (item.get("question") or "").strip()
        if not question_text:
            continue
        question_type = item.get("question_type") or "General"
        difficulty = (item.get("difficulty") or "Medium").title()
        mandatory_skills = item.get("mandatory_skills") or []
        if isinstance(mandatory_skills, str):
            mandatory_skills = [skill.strip() for skill in mandatory_skills.split(",") if skill.strip()]
        if not isinstance(mandatory_skills, list):
            mandatory_skills = []
        if len(mandatory_skills) < 2:
            mandatory_skills.extend([effective_role, "Communication"])

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
                    "rag_context_used": bool(company_context_chunks),
                    "rag_chunk_count": len(company_context_chunks),
                },
            }
        )

    return normalized_questions[:question_count]


async def generate_resume_based_questions_service(request: ResumeQuestionRequest) -> Dict[str, Any]:
    try:
        resume_data: Optional[dict] = None
        job_description_data: Optional[dict] = None
        job_description_text: Optional[str] = (request.job_description_text or "").strip() or None
        job_description_id_used: Optional[int] = request.job_description_id

        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection unavailable")
            with conn.cursor() as cur:
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
            raise HTTPException(
                status_code=400, detail="Provide either a job role or a job description to tailor questions"
            )

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


async def import_students_from_csv_service(
    file: UploadFile,
    program_id: Optional[int],
    ubp_id: Optional[int],
    university_name: Optional[str],
    program_name: Optional[str],
    batch_label: Optional[str],
) -> StudentImportResult:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    raw_bytes = await file.read()
    try:
        decoded = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="CSV must be UTF-8 encoded")

    reader = csv.DictReader(io.StringIO(decoded))
    fieldnames = {(field or "").strip().lower() for field in (reader.fieldnames or [])}
    required_fields = {"name", "email"}

    if not required_fields.issubset(fieldnames):
        raise HTTPException(status_code=400, detail="CSV must include columns: name, email")

    total_rows = 0
    imported = 0
    email_sent = 0
    duplicates_ignored: List[StudentDuplicateInfo] = []
    errors: List[StudentImportError] = []
    email_warnings: List[StudentEmailWarning] = []

    with db_pool.get_connection() as conn:
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection unavailable")

        with conn.cursor() as cur:
            for row_index, row in enumerate(reader, start=2):
                total_rows += 1
                normalized_row = {(key or "").strip().lower(): (value or "") for key, value in row.items()}
                name = normalized_row.get("name", "").strip()
                email = normalized_row.get("email", "").strip().lower()
                program_from_csv = normalized_row.get("program_name", "").strip()

                try:
                    if not name or not email:
                        raise ValueError("Missing required fields")

                    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
                        raise ValueError("Invalid email format")

                    resolved_program_id = ubp_id or program_id
                    if not resolved_program_id and (university_name and program_name and batch_label):
                        resolved_program_id = resolve_ubp_id(
                            (university_name or "").strip(),
                            (program_name or "").strip(),
                            (batch_label or "").strip(),
                        )
                    if not resolved_program_id and program_from_csv:
                        resolved_program_id = resolve_program_id_by_name(program_from_csv)
                    if not resolved_program_id:
                        raise ValueError("Program context not provided or not found (select University/Program/Batch)")

                    cur.execute(
                        """
                        SELECT student_id
                        FROM students
                        WHERE LOWER(TRIM(email)) = LOWER(TRIM(%s))
                        """,
                        (email,),
                    )
                    existing_student = cur.fetchone()

                    if existing_student:
                        duplicates_ignored.append(StudentDuplicateInfo(row=row_index, email=email))
                        continue

                    temp_password = generate_temp_password()
                    hashed_password = hash_password(temp_password)

                    cur.execute(
                        """
                        INSERT INTO students (name, email, program_id, password, last_active)
                        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                        RETURNING student_id
                        """,
                        (name, email, resolved_program_id, hashed_password),
                    )

                    imported += 1
                    try:
                        send_credentials_email(name, email, temp_password)
                        email_sent += 1
                    except Exception as email_exc:
                        logger.warning(
                            "Student %s (%s) created but credential email failed to send: %s",
                            name, email, email_exc,
                        )
                        email_warnings.append(
                            StudentEmailWarning(
                                row=row_index,
                                email=email,
                                note="Student created, credential email could not be sent - check SMTP config",
                            )
                        )

                except Exception as exc:
                    errors.append(StudentImportError(row=row_index, email=email or None, error=str(exc)))

            conn.commit()

    return StudentImportResult(
        total_rows=total_rows,
        imported=imported,
        email_sent=email_sent,
        duplicates_ignored=duplicates_ignored,
        errors=errors,
        email_warnings=email_warnings,
    )
