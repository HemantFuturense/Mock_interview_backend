from typing import Any
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.logger import logger
from app.modules.auth.schemas import (
    PasswordResetConfirm,
    PasswordResetRequest,
    StudentAuthResponse,
    StudentLoginRequest,
    StudentMeResponse,
)
from app.modules.auth.service import (
    create_password_reset_token,
    login_student_service,
    reset_password_service,
    verify_student_token,
)

router = APIRouter(prefix="/students", tags=["Authentication & Password"])
security = HTTPBearer(auto_error=False)


@router.post("/login", response_model=StudentAuthResponse)
async def login_student_endpoint(payload: StudentLoginRequest) -> Any:
    """Authenticate a student and return a signed JWT token."""
    return login_student_service(payload)


@router.get("/me", response_model=StudentMeResponse)
async def get_current_student_profile(student: dict = Depends(verify_student_token)) -> Any:
    """Return the authenticated student's profile from the JWT payload."""
    return StudentMeResponse(**student)


@router.post("/password/forgot")
async def request_password_reset_endpoint(payload: PasswordResetRequest) -> Any:
    try:
        token_info = create_password_reset_token(payload.email.strip())
        logger.info(
            "Password reset requested for %s, token expires at %s",
            payload.email,
            token_info["expires_at"],
        )
        return {
            "message": "Password reset instructions have been emailed if the account exists.",
            "expires_at": token_info["expires_at"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error generating password reset token: {exc}")
        raise HTTPException(status_code=500, detail="Failed to generate password reset token")


@router.post("/password/reset")
async def reset_password_endpoint(payload: PasswordResetConfirm) -> Any:
    return reset_password_service(payload)
