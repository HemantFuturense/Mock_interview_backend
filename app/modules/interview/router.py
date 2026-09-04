import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import EmailStr

from app.core.logger import logger
from app.modules.auth.service import verify_student_token
from app.utils.datetime_utils import format_datetime_ist
from app.modules.ai.grading import analyze_answer_and_generate_response, extract_scores_from_feedback, update_scores_from_feedback
from app.modules.ai.video import process_video_analysis_background
from app.modules.interview.repository import InterviewRepository
from app.modules.interview.schemas import (
    InterviewResponse,
    ReattemptCheckRequest,
    SessionRatingPayload,
    TerminateSessionRequest,
)
from app.modules.interview.service import (
    _save_uploaded_video,
    generate_and_process_feedback_background,
    generate_and_process_feedback_service,
    start_compatibility_interview_service,
    start_interview_service,
    submit_answer_service,
)

router = APIRouter()


@router.post("/interview", response_model=InterviewResponse, tags=["Interview Compatibility"])
async def handle_interview_compatibility(
    session_id: Optional[str] = Form(None),
    job_role: Optional[str] = Form(None),
    industry_type: Optional[str] = Form(None),
    company_name: Optional[str] = Form(None),
    interview_type: Optional[str] = Form(None),
    work_experience: Optional[str] = Form(None),
):
    """Compatibility endpoint for existing React frontend."""
    return await start_compatibility_interview_service(
        session_id=session_id,
        job_role=job_role,
        industry_type=industry_type,
        company_name=company_name,
        interview_type=interview_type,
        work_experience=work_experience,
    )


@router.post("/interview/start", tags=["Interview Session"])
async def start_interview(
    request: Request,
    student_name: str = Form(...),
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
    student: Dict[str, Any] = Depends(verify_student_token),
):
    """Start new interview with student tracking and reattempt detection."""
    try:
        return await start_interview_service(
            request=request,
            student_name=student_name,
            student_email=student["email"],
            job_role=job_role,
            industry_type=industry_type,
            company_name=company_name,
            interview_type=interview_type,
            work_experience=work_experience,
            job_description_id=job_description_id,
            job_description_text=job_description_text,
            job_description_raw_text=job_description_raw_text,
            force_reattempt=force_reattempt,
            use_resume_questions=use_resume_questions,
            resume_id=resume_id,
            pre_generated_questions=pre_generated_questions,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error starting interview: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/interview/reattempt/check", tags=["Interview Session"])
async def check_reattempt(
    payload: ReattemptCheckRequest, student: Dict[str, Any] = Depends(verify_student_token)
):
    """Check whether a reattempt confirmation is required without creating a session."""
    if payload.student_email and payload.student_email.strip().lower() != student["email"].strip().lower():
        raise HTTPException(status_code=403, detail="Cannot check reattempt status for another student")
    try:
        sessions = InterviewRepository.find_existing_sessions(
            payload.student_name,
            payload.student_email,
            payload.job_role,
            payload.industry_type,
            payload.company_name,
        )
        requires_confirmation = len(sessions) > 0
        return {
            "requires_confirmation": requires_confirmation,
            "existing_sessions": sessions,
            "message": "Existing completed interview attempts found." if requires_confirmation else "No completed attempts found.",
        }
    except Exception as exc:
        logger.error(f"Error checking reattempt requirement: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/interview/{session_id}/terminate", tags=["Interview Session"])
async def terminate_session(
    session_id: str,
    payload: TerminateSessionRequest,
    student: Dict[str, Any] = Depends(verify_student_token),
):
    """Terminate an in-progress interview session (e.g., due to proctoring violations)."""
    if not InterviewRepository.verify_session_belongs_to_student(session_id, student["student_id"]):
        raise HTTPException(status_code=403, detail="Cannot terminate another student's session")
    reason_text = (payload.reason or "Session terminated by system.").strip()
    try:
        success = InterviewRepository.terminate_session_repo(session_id, reason_text)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "terminated", "session_id": session_id, "reason": reason_text[:500]}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to terminate session {session_id}: {exc}")
        raise HTTPException(status_code=500, detail="Unable to terminate session") from exc


@router.get("/interview/active-session", tags=["Interview Session"])
async def get_active_session(student_email: str, student: Dict[str, Any] = Depends(verify_student_token)):
    """Check if the student has an active (in-progress) interview session."""
    if student["email"].strip().lower() != student_email.strip().lower():
        raise HTTPException(status_code=403, detail="Cannot access another student's session status")
    try:
        res = InterviewRepository.get_active_session_repo(student_email)
        if not res:
            return {"has_active_session": False}
        return res
    except Exception as exc:
        logger.error(f"Failed checking active session for {student_email}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/students/sessions/{session_id}/rating", tags=["Session Rating"])
async def get_session_rating(
    session_id: str,
    student_email: EmailStr = Query(...),
    student: Dict[str, Any] = Depends(verify_student_token),
):
    """Fetch an existing rating for the given session and student."""
    if student["email"].strip().lower() != student_email.strip().lower():
        raise HTTPException(status_code=403, detail="Cannot access another student's rating")
    try:
        return InterviewRepository.get_session_rating_repo(session_id, str(student_email))
    except Exception as exc:
        logger.error(f"Error fetching session rating for {session_id}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch session rating") from exc


@router.post("/students/sessions/{session_id}/rating", status_code=201, tags=["Session Rating"])
async def submit_session_rating(
    session_id: str,
    payload: SessionRatingPayload,
    student_email: EmailStr = Query(...),
    student: Dict[str, Any] = Depends(verify_student_token),
):
    """Submit or update a rating for the given session."""
    if student["email"].strip().lower() != student_email.strip().lower():
        raise HTTPException(status_code=403, detail="Cannot submit a rating as another student")
    try:
        student_id, created_at_str = InterviewRepository.submit_session_rating_repo(
            session_id, str(student_email), payload.rating, payload.comments
        )
        return {
            "session_id": session_id,
            "student_id": student_id,
            "rating": payload.rating,
            "comments": payload.comments,
            "created_at": created_at_str,
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Student not found")
    except PermissionError:
        raise HTTPException(status_code=403, detail="Session does not belong to this student")
    except Exception as exc:
        logger.error(f"Error submitting session rating for {session_id}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to submit session rating") from exc


@router.post("/interview/{session_id}/answer", tags=["Answer Evaluation"])
async def submit_answer(
    background_tasks: BackgroundTasks,
    session_id: str,
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
    student: Dict[str, Any] = Depends(verify_student_token),
):
    """Process candidate response and determine the next question dynamically."""
    try:
        return await submit_answer_service(
            background_tasks=background_tasks,
            session_id=session_id,
            student_id=student["student_id"],
            answer=answer,
            question_type=question_type,
            code=code,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            runtime_error=runtime_error,
            execution_success=execution_success,
            has_run=has_run,
            is_final=is_final,
            response_video=response_video,
            system_design_diagram=system_design_diagram,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error submitting answer for session {session_id}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to submit answer")


@router.post("/interview/{session_id}/upload-video", tags=["Video Analysis"])
async def upload_video(
    background_tasks: BackgroundTasks,
    session_id: str,
    question_number: int = Form(...),
    video: UploadFile = Form(...),
    student: Dict[str, Any] = Depends(verify_student_token),
):
    """Endpoint specifically for uploading recorded video chunks per question."""
    if not InterviewRepository.verify_session_belongs_to_student(session_id, student["student_id"]):
        raise HTTPException(status_code=403, detail="Cannot upload a video into another student's session")
    saved_video_path = _save_uploaded_video(session_id, question_number, video)
    if not saved_video_path:
        raise HTTPException(status_code=500, detail="Failed to save video file")

    InterviewRepository.update_video_upload_path(session_id, question_number, saved_video_path)
    background_tasks.add_task(process_video_analysis_background, session_id, question_number, saved_video_path)
    return {"status": "success", "message": "Video uploaded and queued for analysis", "file_path": saved_video_path}


@router.post("/interview/{session_id}/generate-feedback", tags=["Feedback & Grading"])
async def generate_feedback(session_id: str):
    """Trigger background feedback generation and rubric scoring."""
    try:
        return await generate_and_process_feedback_service(session_id)
    except Exception as exc:
        logger.error(f"Error initiating feedback generation: {exc}")
        raise HTTPException(status_code=500, detail="Failed to initiate feedback generation")


@router.get("/interview/{session_id}/scores", tags=["Feedback & Grading"])
async def get_session_scores(session_id: str):
    """Retrieve session overall score and detailed rubric evaluations."""
    res = InterviewRepository.get_session_scores(session_id)
    if not res:
        raise HTTPException(status_code=404, detail="Session or scores not found.")
    overall_score, rubric_scores, status, completed_at = res
    if overall_score is None:
        raise HTTPException(status_code=404, detail="Scores have not been calculated yet.")
    return {
        "session_id": session_id,
        "overall_score": float(overall_score) if overall_score is not None else None,
        "rubric_scores": rubric_scores or {},
        "status": status,
        "completed_at": format_datetime_ist(completed_at) if completed_at else None,
    }


@router.get("/feedback-status/{session_id}", tags=["Feedback & Grading"])
async def get_feedback_status(session_id: str):
    """Poll feedback generation state for the specified interview session."""
    res = InterviewRepository.get_feedback_status_data(session_id)
    if not res:
        raise HTTPException(status_code=404, detail="Session not found")
    status, error, req_at, rdy_at, generated = res
    if not status:
        status = "completed" if generated else "not_requested"
    return {
        "session_id": session_id,
        "status": status,
        "error": error,
        "requested_at": format_datetime_ist(req_at) if req_at else None,
        "ready_at": format_datetime_ist(rdy_at) if rdy_at else None,
        "generated": bool(generated),
    }


@router.get("/feedback/{session_id}", tags=["Feedback & Grading"])
async def get_feedback(session_id: str):
    """Get interview feedback by session_id."""
    res = InterviewRepository.get_feedback_payload(session_id)
    if not res:
        return {"session_id": session_id, "feedback": "Feedback generation in progress or failed.", "structured_feedback": None}
    return {
        "session_id": session_id,
        "feedback": res["raw"],
        "structured_feedback": res["structured"],
    }


# Options & Metadata endpoints
@router.get("/interview-options/industries", tags=["Options & Metadata"])
async def get_interview_industries():
    try:
        return InterviewRepository.list_interview_industries()
    except Exception as exc:
        logger.error(f"Error loading industry options: {exc}")
        return []


@router.get("/interview-options/companies", tags=["Options & Metadata"])
async def get_interview_companies(industry: Optional[str] = Query(None)):
    try:
        return InterviewRepository.list_companies_by_industry(industry)
    except Exception as exc:
        logger.error(f"Error loading company options for industry '{industry}': {exc}")
        return []


@router.get("/companies/trending", tags=["Options & Metadata"])
async def get_trending_companies():
    try:
        return InterviewRepository.list_trending_companies()
    except Exception as exc:
        logger.error(f"Error loading trending companies: {exc}")
        return []


@router.get("/interview-options/interview-types", tags=["Options & Metadata"])
async def get_interview_types(industry: Optional[str] = Query(None), company: Optional[str] = Query(None)):
    try:
        return InterviewRepository.list_interview_types(industry, company)
    except Exception as exc:
        logger.error(f"Error loading interview type options: {exc}")
        return []


@router.get("/interview-options/work-experience", tags=["Options & Metadata"])
async def get_work_experience_levels(
    industry: Optional[str] = Query(None),
    company: Optional[str] = Query(None),
    interview_type: Optional[str] = Query(None),
):
    try:
        return InterviewRepository.list_work_experience_levels(industry, company, interview_type)
    except Exception as exc:
        logger.error(f"Error loading work experience levels: {exc}")
        return []


@router.get("/interview-options/job-roles", tags=["Options & Metadata"])
async def get_job_roles(
    industry: Optional[str] = Query(None),
    company: Optional[str] = Query(None),
    interview_type: Optional[str] = Query(None),
    work_experience: Optional[str] = Query(None),
    program_name: Optional[str] = Query(None),
):
    try:
        return InterviewRepository.list_job_roles_for_selection(industry, company, interview_type, work_experience, program_name)
    except Exception as exc:
        logger.error(f"Error loading job role options: {exc}")
        return []


@router.get("/job-roles/by-work-experience", tags=["Options & Metadata"])
async def get_job_roles_by_experience(work_experience: Optional[str] = Query(None), program_name: Optional[str] = Query(None)):
    try:
        return InterviewRepository.list_job_roles_by_work_experience(work_experience, program_name)
    except Exception as exc:
        logger.error(f"Error loading job roles by experience '{work_experience}': {exc}")
        return []


@router.get("/metadata/interview-types", tags=["Options & Metadata"])
async def get_metadata_interview_types():
    try:
        return InterviewRepository.fetch_distinct_question_metadata("interview_type")
    except Exception as exc:
        logger.error(f"Error reading interview_type metadata: {exc}")
        return []


@router.get("/metadata/work-experience-levels", tags=["Options & Metadata"])
async def get_metadata_work_experience():
    try:
        return InterviewRepository.fetch_distinct_question_metadata("work_experience")
    except Exception as exc:
        logger.error(f"Error reading work_experience metadata: {exc}")
        return []


@router.get("/metadata/question-types", tags=["Options & Metadata"])
async def get_metadata_question_types():
    try:
        return InterviewRepository.fetch_distinct_question_metadata("question_type")
    except Exception as exc:
        logger.error(f"Error reading question_type metadata: {exc}")
        return []


# Debug endpoints
@router.get("/debug/scores/{session_id}", tags=["Debug"])
async def debug_scores(session_id: str):
    """Debug endpoint to test score extraction."""
    try:
        feedback_text, meta_dict = InterviewRepository.get_latest_feedback_for_debug(session_id)
        if not feedback_text:
            return {"error": "No feedback found for this session"}
        scores = extract_scores_from_feedback(feedback_text)
        return {
            "session_id": session_id,
            "feedback_length": len(feedback_text),
            "extracted_scores": scores,
            "session_metadata_exists": meta_dict is not None,
            "session_metadata": meta_dict,
            "debug": "score_extraction_test",
        }
    except Exception as exc:
        logger.error(f"Debug scores error: {exc}")
        return {"error": str(exc)}


@router.post("/debug/force-score-update/{session_id}", tags=["Debug"])
async def force_score_update(session_id: str):
    """Force update scores for a session."""
    try:
        feedback_text, _ = InterviewRepository.get_latest_feedback_for_debug(session_id)
        if not feedback_text:
            return {"error": "No feedback found for this session"}
        success = update_scores_from_feedback(session_id, feedback_text)
        return {
            "session_id": session_id,
            "success": success,
            "message": "Score update completed" if success else "Score update failed",
            "debug": "force_score_update",
        }
    except Exception as exc:
        logger.error(f"Force score update error: {exc}")
        return {"error": str(exc)}


@router.get("/debug/acknowledgment/{session_id}", tags=["Debug"])
async def debug_acknowledgment(session_id: str):
    """Debug endpoint to test acknowledgment generation."""
    try:
        answer_row = InterviewRepository.get_latest_answer_for_debug(session_id)
        if not answer_row:
            return {"error": "No answered questions found for this session"}
        question, answer, job_role, mandatory_skills = answer_row
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
            "debug": "acknowledgment_test",
        }
    except Exception as exc:
        logger.error(f"Debug acknowledgment error: {exc}")
        return {"error": str(exc)}


@router.get("/debug/{session_id}", tags=["Debug"])
async def debug_session(session_id: str):
    """Debug endpoint for session analysis."""
    try:
        from app.core.cache import questions_cache, session_cache
        from app.modules.interview.service import get_questions_cached
        res = InterviewRepository.get_debug_session_data(session_id)
        res["cache_info"] = {
            "session_cache_size": len(session_cache.cache),
            "questions_cache_size": len(questions_cache.cache),
            "lru_cache_info": str(get_questions_cached.cache_info()),
        }
        res["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        return res
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error in debug endpoint: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
