from typing import Any, Dict, List, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import EmailStr

from app.modules.admin.service import verify_admin_token
from app.modules.auth.service import verify_student_token
from app.modules.students.repository import (
    fetch_all_programs,
    fetch_program_info,
    get_student_profile_repo,
    list_batches_repo,
    list_programs_by_university_repo,
    list_universities_repo,
    register_student_repo,
    resolve_ubp_id,
)
from app.modules.students.schemas import (
    ResumeQuestionRequest,
    StudentImportResult,
    StudentRegistration,
    StudentRegistrationResponse,
)
from app.modules.students.service import (
    generate_resume_based_questions_service,
    get_student_job_description_service,
    get_student_resume_interview_sessions_service,
    get_student_resume_service,
    import_students_from_csv_service,
    upload_job_description_service,
    upload_student_resume_service,
)

router = APIRouter(tags=["Students & Profiles"])


@router.post("/students/register", response_model=StudentRegistrationResponse)
async def register_student(payload: StudentRegistration) -> Any:
    """Register or update a student and issue a JWT for immediate authentication."""
    return register_student_repo(payload)


@router.get("/students/profile/{email}")
async def get_student_profile(email: EmailStr, student: Dict[str, Any] = Depends(verify_student_token)) -> Any:
    """Fetch a student's complete profile by email."""
    if student["email"].strip().lower() != email.strip().lower():
        raise HTTPException(status_code=403, detail="Cannot access another student's profile")
    return get_student_profile_repo(email)


@router.get("/students/{student_id}/resume")
async def get_student_resume(student_id: int, student: Dict[str, Any] = Depends(verify_student_token)) -> Any:
    """Check if student has uploaded a resume and return parsed data"""
    if student["student_id"] != student_id:
        raise HTTPException(status_code=403, detail="Cannot access another student's resume")
    return get_student_resume_service(student_id)


@router.post("/students/upload-resume")
async def upload_student_resume(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    student_id: int = Form(...),
    student: Dict[str, Any] = Depends(verify_student_token),
) -> Any:
    """Upload and parse a student's resume (PDF/DOCX support)"""
    if student["student_id"] != student_id:
        raise HTTPException(status_code=403, detail="Cannot upload a resume for another student")
    return await upload_student_resume_service(file, student_id, background_tasks)


@router.post("/students/upload-job-description")
async def upload_job_description(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    student_id: int = Form(...),
    student: Dict[str, Any] = Depends(verify_student_token),
) -> Any:
    """Upload and parse a job description for targeted interview prep."""
    if student["student_id"] != student_id:
        raise HTTPException(status_code=403, detail="Cannot upload a job description for another student")
    return await upload_job_description_service(file, student_id, background_tasks)


@router.get("/students/{student_id}/job-description")
async def get_student_job_description(student_id: int, student: Dict[str, Any] = Depends(verify_student_token)) -> Any:
    """Retrieve the latest active job description for a student."""
    if student["student_id"] != student_id:
        raise HTTPException(status_code=403, detail="Cannot access another student's job description")
    return get_student_job_description_service(student_id)


@router.get("/students/{student_id}/resume-interview-sessions")
async def get_student_resume_interview_sessions(
    student_id: int, student: Dict[str, Any] = Depends(verify_student_token)
) -> Any:
    """Get all resume-based interview sessions for a student with question details."""
    if student["student_id"] != student_id:
        raise HTTPException(status_code=403, detail="Cannot access another student's sessions")
    return get_student_resume_interview_sessions_service(student_id)


@router.post("/api/generate-resume-questions")
async def generate_resume_based_questions(
    request: ResumeQuestionRequest, student: Dict[str, Any] = Depends(verify_student_token)
) -> Any:
    """Generate resume-driven and job description-tailored interview questions"""
    if student["student_id"] != request.student_id:
        raise HTTPException(status_code=403, detail="Cannot generate questions for another student")
    return await generate_resume_based_questions_service(request)


@router.post(
    "/mentors/students/import",
    response_model=StudentImportResult,
    dependencies=[Depends(verify_admin_token)],
)
async def import_students_from_csv(
    file: UploadFile = File(...),
    program_id: Optional[int] = Form(None),
    ubp_id: Optional[int] = Form(None),
    university_name: Optional[str] = Form(None),
    program_name: Optional[str] = Form(None),
    batch_label: Optional[str] = Form(None),
) -> Any:
    """Bulk register students using a CSV upload, attaching them to a program context."""
    return await import_students_from_csv_service(
        file, program_id, ubp_id, university_name, program_name, batch_label
    )


# UBP & Program Endpoints
@router.get("/ubp/universities")
async def list_universities() -> List[str]:
    return list_universities_repo()


@router.get("/ubp/programs")
async def list_programs_by_university(university: str = Query(..., alias="university_name")) -> List[str]:
    return list_programs_by_university_repo(university)


@router.get("/ubp/batches")
async def list_batches(
    university: str = Query(..., alias="university_name"), program: str = Query(..., alias="program_name")
) -> List[str]:
    return list_batches_repo(university, program)


@router.get("/ubp/resolve")
async def resolve_ubp(
    university: str = Query(..., alias="university_name"),
    program: str = Query(..., alias="program_name"),
    batch: str = Query(..., alias="batch_label"),
) -> Dict[str, Any]:
    ubp_id = resolve_ubp_id(university.strip(), program.strip(), batch.strip())
    if not ubp_id:
        raise HTTPException(status_code=404, detail="UBP combination not found")
    return {"ubp_id": ubp_id}


@router.get("/programs")
async def list_programs() -> List[Dict[str, Any]]:
    return fetch_all_programs()


@router.get("/programs/{program_id}/job_roles")
async def list_program_job_roles(program_id: int) -> Dict[str, Any]:
    program_info = fetch_program_info(program_id)
    if not program_info:
        raise HTTPException(status_code=404, detail="Program not found")
    return program_info
