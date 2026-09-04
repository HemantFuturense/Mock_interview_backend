from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from app.modules.admin.repository import AdminRepository
from app.modules.admin.schemas import (
    AdminAuthResponse,
    AdminLoginRequest,
    DashboardStats,
    InterviewQuestionPayload,
    LeaderboardEntry,
    ProgramRoleMappingRequest,
    SessionDetail,
)
from app.modules.admin.service import (
    _normalize_question_payload,
    admin_login_service,
    bulk_upload_interview_questions_service,
    create_company_service,
    export_sessions_service,
    upload_company_playbook_service,
    verify_admin_token,
)
from app.modules.auth.service import verify_student_token

router = APIRouter(tags=["Admin & Institutional Oversight"])


@router.get("/admin/health")
async def admin_health_check():
    """Admin health check confirming DB status."""
    connected = AdminRepository.check_admin_db_health_repo()
    if not connected:
        return {"status": "unhealthy", "database": "disconnected"}
    return {"status": "healthy", "database": "connected", "timestamp": datetime.now().isoformat()}


@router.post("/admin/auth/login", response_model=AdminAuthResponse, tags=["Admin Auth"])
async def admin_login(payload: AdminLoginRequest):
    """Authenticate admin using credentials stored in admin_users table."""
    return admin_login_service(payload)


@router.get("/admin/auth/me", tags=["Admin Auth"])
async def admin_me(admin: Dict[str, Any] = Depends(verify_admin_token)):
    """Return currently authenticated admin details."""
    return {
        "admin_id": admin["admin_id"],
        "email": admin["email"],
        "display_name": admin.get("display_name"),
    }


@router.post("/admin/auth/logout", tags=["Admin Auth"])
async def admin_logout(_: Dict[str, Any] = Depends(verify_admin_token)):
    """Stateless logout placeholder for clients to clear stored credentials."""
    return {"message": "Logged out"}


@router.post("/admin/interview-questions", dependencies=[Depends(verify_admin_token)], tags=["Admin Question Bank"])
async def create_interview_question(payload: InterviewQuestionPayload) -> Dict[str, Any]:
    """Create a single interview question record."""
    record = _normalize_question_payload(payload)
    inserted_ids = AdminRepository.insert_interview_questions([record])
    return {
        "message": "Interview question saved successfully",
        "question_id": inserted_ids[0] if inserted_ids else None,
    }


@router.post("/admin/interview-questions/bulk-upload", dependencies=[Depends(verify_admin_token)], tags=["Admin Question Bank"])
async def bulk_upload_interview_questions(
    industries: str = Form("[]"),
    companies: str = Form("[]"),
    roles: str = Form("[]"),
    interview_types: str = Form("[]"),
    work_experiences: str = Form("[]"),
    difficulties: str = Form("[]"),
    question_types: str = Form("[]"),
    questions_file: UploadFile = File(...),
) -> Dict[str, Any]:
    """Bulk upload interview questions via CSV with multi-category support."""
    return await bulk_upload_interview_questions_service(
        industries=industries,
        companies=companies,
        roles=roles,
        interview_types=interview_types,
        work_experiences=work_experiences,
        difficulties=difficulties,
        question_types=question_types,
        questions_file=questions_file,
    )


@router.post("/admin/companies", dependencies=[Depends(verify_admin_token)], tags=["Admin Companies"])
async def create_company(
    name: str = Form(...),
    industry: Optional[str] = Form(None),
    difficulty_tag: Optional[str] = Form(None),
    work_experience_tag: Optional[str] = Form(None),
    logo: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    """Create a company row; the trending-companies list and playbook uploads both key off it."""
    return await create_company_service(name, industry, difficulty_tag, work_experience_tag, logo)


@router.get("/admin/companies", dependencies=[Depends(verify_admin_token)], tags=["Admin Companies"])
async def list_companies() -> List[Dict[str, Any]]:
    return AdminRepository.list_companies()


@router.get("/admin/companies/playbook-chunk-counts", dependencies=[Depends(verify_admin_token)], tags=["Admin Companies"])
async def get_playbook_chunk_counts() -> List[Dict[str, Any]]:
    return AdminRepository.get_playbook_chunk_counts()


@router.post("/admin/companies/{company_id}/upload-playbook", dependencies=[Depends(verify_admin_token)], tags=["Admin Companies"])
async def upload_company_playbook(company_id: int, playbook_file: UploadFile = File(...)) -> Dict[str, Any]:
    return await upload_company_playbook_service(company_id, playbook_file)


@router.get("/admin/dashboard", response_model=DashboardStats, dependencies=[Depends(verify_admin_token)], tags=["Admin Dashboard"])
async def get_dashboard_stats() -> DashboardStats:
    """Get main dashboard statistics."""
    try:
        data = AdminRepository.get_dashboard_stats_repo()
        return DashboardStats(**data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/admin/students", dependencies=[Depends(verify_admin_token)], tags=["Admin Student Oversight"])
async def get_students(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    name: Optional[str] = Query(None),
    email: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    university_name: Optional[str] = Query(None),
    program_name: Optional[str] = Query(None),
    batch_label: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """Get paginated list of students with optional name and UBP filters."""
    try:
        return AdminRepository.get_students_paginated_repo(
            page=page,
            limit=limit,
            name=name,
            email=email,
            status=status,
            university_name=university_name,
            program_name=program_name,
            batch_label=batch_label,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/admin/sessions", dependencies=[Depends(verify_admin_token)], tags=["Admin Session Oversight"])
async def get_sessions(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    student_name: Optional[str] = Query(None),
    student_email: Optional[str] = Query(None),
    job_role: Optional[str] = Query(None, description="Filter by interview role"),
    company_name: Optional[str] = Query(None, description="Filter by company"),
    min_score: Optional[float] = Query(None, ge=0, description="Minimum overall score"),
    max_score: Optional[float] = Query(None, ge=0, description="Maximum overall score"),
    university_name: Optional[str] = Query(None, description="Filter by university name"),
    program_name: Optional[str] = Query(None, description="Filter by program name"),
    batch_label: Optional[str] = Query(None, description="Filter by batch label"),
) -> Dict[str, Any]:
    """Get paginated list of interview sessions with rich filters."""
    if min_score is not None and max_score is not None and min_score > max_score:
        raise HTTPException(status_code=400, detail="min_score cannot be greater than max_score")
    try:
        return AdminRepository.get_sessions_paginated_repo(
            page=page,
            limit=limit,
            status=status,
            student_name=student_name,
            student_email=student_email,
            job_role=job_role,
            company_name=company_name,
            min_score=min_score,
            max_score=max_score,
            university_name=university_name,
            program_name=program_name,
            batch_label=batch_label,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/admin/sessions/{session_id}", dependencies=[Depends(verify_admin_token)], tags=["Admin Session Oversight"])
async def get_session_details(session_id: str) -> Dict[str, Any]:
    """Get detailed session information including Q&A evaluations."""
    res = AdminRepository.get_session_details_repo(session_id)
    if not res:
        raise HTTPException(status_code=404, detail="Session not found")
    return res


@router.get("/admin/job-roles", dependencies=[Depends(verify_admin_token)], tags=["Admin Options"])
async def get_job_roles() -> List[str]:
    """Get distinct list of all job roles in the system."""
    try:
        return AdminRepository.get_job_roles_repo()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/admin/industry-types", dependencies=[Depends(verify_admin_token)], tags=["Admin Options"])
async def get_industry_types() -> List[str]:
    """Get distinct list of all industry types in the system."""
    try:
        return AdminRepository.get_industry_types_repo()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/admin/interview-types", dependencies=[Depends(verify_admin_token)], tags=["Admin Options"])
async def get_interview_types() -> List[str]:
    """Get distinct list of interview types from interview questions."""
    try:
        return AdminRepository.get_interview_types_repo()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/admin/work-experience-levels", dependencies=[Depends(verify_admin_token)], tags=["Admin Options"])
async def get_work_experience_levels() -> List[str]:
    """Get distinct list of work experience levels from interview questions."""
    try:
        return AdminRepository.get_work_experience_levels_repo()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/admin/programs/map-roles", dependencies=[Depends(verify_admin_token)], tags=["Admin Cohort Mapping"])
async def map_program_roles(request: ProgramRoleMappingRequest) -> Dict[str, Any]:
    """Map one or more job roles and a work experience band to a program."""
    university = request.university_name.strip()
    program = request.program_name.strip()
    batch = request.batch_label.strip()
    work_exp = request.work_experience.strip()
    roles = [r.strip() for r in request.job_roles if r and r.strip()]

    if not (university and program and batch and work_exp and roles):
        raise HTTPException(
            status_code=422,
            detail="university_name, program_name, batch_label, work_experience and at least one job_role are required",
        )

    ubp_id = AdminRepository.resolve_ubp_id_for_admin(university, program, batch)
    if not ubp_id:
        raise HTTPException(
            status_code=422,
            detail="Unknown university / program / batch combination in university_batch_program",
        )

    try:
        inserted, skipped = AdminRepository.map_program_roles_repo(program, work_exp, roles)
        return {
            "message": "Program roles mapped successfully",
            "inserted": inserted,
            "skipped": skipped,
            "program_name": program,
            "work_experience": work_exp,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to map roles to program") from exc


@router.get("/admin/question-types", dependencies=[Depends(verify_admin_token)], tags=["Admin Options"])
async def get_question_types() -> List[str]:
    """Get distinct list of question types from interview questions."""
    try:
        return AdminRepository.get_question_types_repo()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/students/sessions/by_email/{email}", tags=["Student Sessions"])
async def get_sessions_by_email(
    email: str,
    exclude_resume: bool = Query(
        False,
        description="Exclude resume-driven sessions (question_generation_type='resume') from results",
    ),
    student: Dict[str, Any] = Depends(verify_student_token),
) -> List[SessionDetail]:
    """Get all interview sessions for a student by their email."""
    if student["email"].strip().lower() != email.strip().lower():
        raise HTTPException(status_code=403, detail="Cannot access another student's sessions")
    try:
        data = AdminRepository.get_sessions_by_email_repo(email, exclude_resume)
        return [SessionDetail(**item) for item in data]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/admin/company-names", dependencies=[Depends(verify_admin_token)], tags=["Admin Options"])
async def get_company_names() -> List[str]:
    """Get distinct list of all company names in the system."""
    try:
        return AdminRepository.get_company_names_repo()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/admin/bulk-upload-options", dependencies=[Depends(verify_admin_token)], tags=["Admin Options"])
async def get_bulk_upload_options() -> Dict[str, List[str]]:
    """Return distinct options for bulk upload multi-select fields without filtering."""
    try:
        return AdminRepository.get_bulk_upload_options_repo()
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to fetch bulk upload options") from exc


@router.get("/admin/interview-options/companies", dependencies=[Depends(verify_admin_token)], tags=["Admin Options"])
async def get_admin_companies_by_industry(industry: Optional[str] = Query(None)) -> List[str]:
    """Return distinct companies for a given industry for the Admin UI."""
    try:
        return AdminRepository.get_admin_companies_by_industry_repo(industry)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to fetch admin companies") from exc


@router.get("/admin/interview-options/interview-types", dependencies=[Depends(verify_admin_token)], tags=["Admin Options"])
async def get_admin_interview_types_for_selection(
    industry: Optional[str] = Query(None),
    company: str = Query(...),
) -> List[str]:
    """Return distinct interview types for the selected industry/company for Admin."""
    try:
        return AdminRepository.get_admin_interview_types_for_selection_repo(industry, company)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to fetch admin interview types") from exc


@router.get("/admin/interview-options/work-experience", dependencies=[Depends(verify_admin_token)], tags=["Admin Options"])
async def get_admin_work_experience_for_selection(
    industry: Optional[str] = Query(None),
    company: str = Query(...),
    interview_type: str = Query(...),
) -> List[str]:
    """Return distinct work experience levels for the selected context for Admin."""
    try:
        return AdminRepository.get_admin_work_experience_for_selection_repo(industry, company, interview_type)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to fetch admin work experience levels") from exc


@router.get("/admin/interview-options/job-roles", dependencies=[Depends(verify_admin_token)], tags=["Admin Options"])
async def get_admin_job_roles_for_selection(
    industry: Optional[str] = Query(None),
    company: str = Query(...),
    interview_type: str = Query(...),
    work_experience: str = Query(...),
) -> List[str]:
    """Return distinct job roles for the selected context for Admin."""
    try:
        return AdminRepository.get_admin_job_roles_for_selection_repo(industry, company, interview_type, work_experience)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to fetch admin job roles") from exc


@router.get("/admin/analytics/performance", dependencies=[Depends(verify_admin_token)], tags=["Admin Analytics"])
async def get_performance_analytics(days: int = Query(30, ge=1, le=365)) -> Dict[str, Any]:
    """Get performance analytics including daily trends and score distribution."""
    try:
        return AdminRepository.get_performance_analytics_repo(days)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/admin/analytics/insights", dependencies=[Depends(verify_admin_token)], tags=["Admin Analytics"])
async def get_admin_insights() -> Dict[str, Any]:
    """Collate deeper admin insights spanning programs, engagement, and reattempts."""
    try:
        return AdminRepository.get_admin_insights_repo()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/admin/analytics/ubp-performance", dependencies=[Depends(verify_admin_token)], tags=["Admin Analytics"])
async def get_ubp_performance() -> Dict[str, Any]:
    """Cohort-level performance metrics grouped by university / program / batch (UBP)."""
    try:
        return AdminRepository.get_ubp_performance_repo()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/admin/analytics/retention", dependencies=[Depends(verify_admin_token)], tags=["Admin Analytics"])
async def get_retention_analytics() -> Dict[str, Any]:
    """Funnel and engagement metrics such as 7/30 day retention and time between sessions."""
    try:
        return AdminRepository.get_retention_analytics_repo()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/admin/student/{student_id}/analytics", dependencies=[Depends(verify_admin_token)], tags=["Admin Student Oversight"])
async def get_student_analytics(student_id: int) -> Dict[str, Any]:
    """Get comprehensive analytics for a specific student."""
    res = AdminRepository.get_student_analytics_repo(student_id)
    if not res:
        raise HTTPException(status_code=404, detail="Student not found")
    return res


@router.get("/admin/analytics/leaderboard", response_model=List[LeaderboardEntry], dependencies=[Depends(verify_admin_token)], tags=["Admin Leaderboard"])
async def get_analytics_leaderboard() -> List[LeaderboardEntry]:
    """Get top students for the analytics leaderboard."""
    try:
        data = AdminRepository.get_leaderboard_repo(role=None, company=None, program=None, university=None, batch=None)
        return [LeaderboardEntry(**item) for item in data[:20]]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/admin/session/{session_id}/detailed", dependencies=[Depends(verify_admin_token)], tags=["Admin Session Oversight"])
async def get_detailed_session_analytics(session_id: str) -> Dict[str, Any]:
    """Get detailed analytics for a specific session including question progression."""
    res = AdminRepository.get_detailed_session_analytics_repo(session_id)
    if not res:
        raise HTTPException(status_code=404, detail="Session not found")
    return res


@router.get("/admin/analytics/comparative", dependencies=[Depends(verify_admin_token)], tags=["Admin Analytics"])
async def get_comparative_analytics() -> Dict[str, Any]:
    """Get comparative analytics across roles and companies."""
    try:
        return AdminRepository.get_comparative_analytics_repo()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/admin/filter-options", dependencies=[Depends(verify_admin_token)], tags=["Admin Leaderboard"])
async def get_filter_options():
    """Get unique roles, companies, universities, programs, and batches for filtering leaderboards."""
    try:
        return AdminRepository.get_filter_options_repo()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/admin/filter-options/roles", dependencies=[Depends(verify_admin_token)], tags=["Admin Leaderboard"])
async def get_role_filter_options(
    university: Optional[str] = Query(None, description="Filter by university name"),
    program: Optional[str] = Query(None, description="Filter by program name"),
    batch: Optional[str] = Query(None, description="Filter by batch label"),
) -> List[str]:
    """Get distinct job roles for filters, optionally scoped by university / program / batch."""
    try:
        return AdminRepository.get_role_filter_options_repo(university, program, batch)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/admin/leaderboard", response_model=List[LeaderboardEntry], dependencies=[Depends(verify_admin_token)], tags=["Admin Leaderboard"])
async def get_leaderboard(
    role: Optional[str] = Query(None, description="Filter by job role"),
    company: Optional[str] = Query(None, description="Filter by company name"),
    program: Optional[str] = Query(None, description="Filter by program name"),
    university: Optional[str] = Query(None, description="Filter by university name"),
    batch: Optional[str] = Query(None, description="Filter by batch label"),
) -> List[LeaderboardEntry]:
    """Get top 20 students ranked by average score with optional cohort/role filtering."""
    try:
        data = AdminRepository.get_leaderboard_repo(role, company, program, university, batch)
        return [LeaderboardEntry(**item) for item in data[:20]]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/admin/export/sessions", dependencies=[Depends(verify_admin_token)], tags=["Admin Export"])
async def export_sessions(format: str = Query("csv", pattern="^(csv|json)$")) -> Dict[str, Any]:
    """Export session data in CSV or JSON format."""
    try:
        return export_sessions_service(format)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
