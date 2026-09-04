from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field


class StudentLoginRequest(BaseModel):
    email: EmailStr
    password: str
    program_id: Optional[int] = None


class StudentAuthResponse(BaseModel):
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


class StudentMeResponse(BaseModel):
    student_id: int
    name: str
    email: str
    program_id: Optional[int] = None
    program_name: Optional[str] = None
    batch_label: Optional[str] = None
    university_name: Optional[str] = None
    job_roles: List[str] = Field(default_factory=list)


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    email: EmailStr
    token: str
    new_password: str
