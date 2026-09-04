from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query

from app.modules.auth.service import verify_student_token
from app.modules.students.dashboard_extras_service import (
    get_student_interview_history_service,
    get_student_performance_summary_service,
)

router = APIRouter(tags=["Students Dashboard Extras"])


@router.get("/students/{student_id}/performance-summary")
async def get_student_performance_summary(
    student_id: int, student: Dict[str, Any] = Depends(verify_student_token)
) -> Any:
    """Return aggregated performance statistics for the student's dashboard."""
    if student["student_id"] != student_id:
        raise HTTPException(status_code=403, detail="Cannot access another student's performance summary")
    return get_student_performance_summary_service(student_id)


@router.get("/students/{student_id}/interview-history")
async def get_student_interview_history(
    student_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
    student: Dict[str, Any] = Depends(verify_student_token),
) -> Any:
    """Return paginated interview history for the student's dashboard."""
    if student["student_id"] != student_id:
        raise HTTPException(status_code=403, detail="Cannot access another student's interview history")
    return get_student_interview_history_service(student_id, page=page, page_size=page_size)
