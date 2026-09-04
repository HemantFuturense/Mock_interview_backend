from typing import Any, Dict, List, Optional
from pydantic import BaseModel, EmailStr, Field


class StudentRegistration(BaseModel):
    student_name: str
    student_email: EmailStr
    password: Optional[str] = None
    program_id: Optional[int] = None
    university_name: Optional[str] = None
    program_name: Optional[str] = None
    batch_label: Optional[str] = None


class StudentRegistrationResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    student_id: int
    name: str
    email: str
    program_id: Optional[int] = None
    program_name: Optional[str] = None
    batch_label: Optional[str] = None
    university_name: Optional[str] = None
    job_roles: List[str] = Field(default_factory=list)
    temporary_password: Optional[str] = None


class ResumeQuestionRequest(BaseModel):
    student_id: int
    company_name: str
    job_role: Optional[str] = None
    interview_type: Optional[str] = None
    work_experience: Optional[str] = None
    resume_id: Optional[int] = None
    job_description_id: Optional[int] = None
    job_description_text: Optional[str] = None


class StudentDuplicateInfo(BaseModel):
    row: int
    email: str


class StudentImportError(BaseModel):
    row: int
    email: Optional[str] = None
    error: str


class StudentEmailWarning(BaseModel):
    row: int
    email: str
    note: str


class StudentImportResult(BaseModel):
    total_rows: int
    imported: int
    email_sent: int
    duplicates_ignored: List[StudentDuplicateInfo]
    errors: List[StudentImportError]
    email_warnings: List[StudentEmailWarning] = []
