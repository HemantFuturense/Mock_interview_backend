from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, conint


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


class ReattemptCheckRequest(BaseModel):
    student_name: str
    student_email: Optional[str] = None
    job_role: str
    industry_type: str
    company_name: str


class SessionRatingPayload(BaseModel):
    rating: conint(ge=1, le=5)  # type: ignore[valid-type]
    comments: Optional[str] = None


class StartInterviewPayload(BaseModel):
    session_id: Optional[str] = None
    student_name: Optional[str] = None
    student_email: Optional[str] = None
    student_phone: Optional[str] = None
    job_role: Optional[str] = None
    industry_type: Optional[str] = None
    company_name: Optional[str] = None
    interview_type: Optional[str] = None
    work_experience: Optional[str] = None
    job_description_id: Optional[int] = None
    job_description_text: Optional[str] = None
    job_description_raw_text: Optional[str] = None
    job_desc: Optional[str] = None
    force_reattempt: Optional[bool] = False
    use_resume_questions: Optional[bool] = False
    resume_id: Optional[int] = None
    pre_generated_questions: Optional[Union[str, List[Dict[str, Any]]]] = None
    program_name: Optional[str] = None
