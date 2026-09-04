from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class AdminLoginRequest(BaseModel):
    email: str
    password: str


class AdminAuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    admin_id: int
    email: str
    display_name: Optional[str] = None


class DashboardStats(BaseModel):
    total_students: int
    total_sessions: int
    active_sessions: int
    completed_sessions: int
    avg_score: float
    today_sessions: int


class StudentSummary(BaseModel):
    student_id: int
    name: str
    email: str
    total_sessions: int
    avg_score: Optional[float]
    last_session: Optional[str]
    status: str
    program_name: Optional[str] = None
    university_name: Optional[str] = None
    batch_label: Optional[str] = None


class SessionDetail(BaseModel):
    session_id: str
    student_id: Optional[int] = None
    student_name: str
    job_role: str
    company_name: str
    interview_type: Optional[str]
    work_experience: Optional[str]
    status: str
    overall_score: Optional[float]
    technical_score: Optional[float]
    communication_score: Optional[float]
    attitude_score: Optional[float]
    rubric_scores: Optional[Dict[str, Any]] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_minutes: Optional[int] = None


class ProgramRoleMappingRequest(BaseModel):
    """Request payload for mapping job roles to a program with work experience.

    This operates on the canonical program_name in the programs table and
    validates that the (university, program, batch) combination exists in
    university_batch_program.
    """
    university_name: str
    program_name: str
    batch_label: str
    work_experience: str
    job_roles: List[str]


class QuestionAnalysis(BaseModel):
    question_id: int
    question_text: str
    avg_score: float
    total_responses: int
    difficulty: str


class InterviewQuestionPayload(BaseModel):
    industry: Optional[str] = None
    company: Optional[str] = None
    role: Optional[str] = None
    question: str
    mandatory_skills: Optional[str] = None
    pre_def_answer: Optional[str] = None
    difficulty: Optional[str] = None
    question_type: Optional[str] = None
    interview_type: Optional[str] = None
    work_experience: Optional[str] = None


class LeaderboardEntry(BaseModel):
    rank: int
    student_id: int
    student_name: str
    avg_score: float
    total_sessions: int
