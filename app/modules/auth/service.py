import base64
import hashlib
import hmac
import secrets
import smtplib
import ssl
import textwrap
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Any, Dict, Optional
from urllib.parse import quote_plus

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.config.settings import config
from app.core.database import db_pool
from app.core.logger import logger
from app.modules.auth.schemas import PasswordResetConfirm, StudentAuthResponse, StudentLoginRequest, StudentMeResponse
from app.utils.datetime_utils import IST_TZ

security = HTTPBearer(auto_error=False)


def generate_temp_password(length: int = 10) -> str:
    token = secrets.token_urlsafe(max(8, length))
    return token[:length]


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    iterations = 200_000
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    salt_b64 = base64.b64encode(salt).decode("ascii")
    derived_b64 = base64.b64encode(derived).decode("ascii")
    return f"pbkdf2_sha256${iterations}${salt_b64}${derived_b64}"


def verify_password(password: str, stored_password: Optional[str]) -> bool:
    if not password or not stored_password:
        return False

    if stored_password.startswith("pbkdf2_sha256$"):
        try:
            _, iterations_str, salt_b64, derived_b64 = stored_password.split("$", 3)
            iterations = int(iterations_str)
            salt = base64.b64decode(salt_b64.encode("ascii"))
            expected = base64.b64decode(derived_b64.encode("ascii"))
            derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
            return hmac.compare_digest(derived, expected)
        except Exception:
            return False

    return hmac.compare_digest(password, stored_password)


def create_student_jwt_token(payload: Dict[str, Any]) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=config.STUDENT_JWT_EXPIRES_MINUTES)
    to_encode = {**payload, "exp": expire}
    return jwt.encode(to_encode, config.STUDENT_JWT_SECRET, algorithm=config.STUDENT_JWT_ALGORITHM)


def _decode_student_jwt_token(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, config.STUDENT_JWT_SECRET, algorithms=[config.STUDENT_JWT_ALGORITHM])
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired student token") from exc


async def verify_student_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict[str, Any]:
    token: Optional[str] = None
    if credentials and credentials.scheme.lower() == "bearer":
        token = credentials.credentials

    if not token:
        raise HTTPException(status_code=401, detail="Missing student token")

    payload = _decode_student_jwt_token(token)
    student_id = payload.get("sub")
    if not student_id:
        raise HTTPException(status_code=401, detail="Invalid student token payload")

    with db_pool.get_connection() as conn:
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection unavailable")

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT student_id, name, email, program_id
                FROM students
                WHERE student_id = %s
                """,
                (student_id,),
            )
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=401, detail="Student account not found")

    student_id, student_name, student_email, current_program_id = row
    return {
        "student_id": student_id,
        "name": student_name,
        "email": student_email,
        "program_id": current_program_id,
    }


def send_credentials_email(name: str, email: str, temp_password: str) -> None:
    smtp_cfg = config.SMTP_SETTINGS
    sender = config.EMAIL_SENDER

    if not (smtp_cfg.get("host") and smtp_cfg.get("port") and sender):
        logger.error("SMTP configuration incomplete; credential email cannot be sent")
        raise HTTPException(status_code=500, detail="Email service not configured")

    login_url = config.PASSWORD_RESET_URL_BASE or "/login"
    reset_hint = "If you wish to set your own password, visit the login page and choose 'Forgot password?'."

    subject = "Your AI Mock Interview login credentials"
    plain_body = textwrap.dedent(
        f"""
        Hello {name or 'there'},

        You have been registered for the AI Mock Interview platform.

        Email: {email}
        Temporary password: {temp_password}
        Login here: {login_url}

        {reset_hint}
        After logging in, please reset your password for security.

        Thank you
        """
    ).strip()

    html_body = f"""
    <html>
      <body style="margin:0;padding:0;background:#f5f5ff;font-family:'Segoe UI',Arial,sans-serif;color:#1f1b3a;">
        <table role="presentation" cellpadding="0" cellspacing="0" width="100%">
          <tr>
            <td align="center" style="padding:32px 16px;">
              <table width="480" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:16px;box-shadow:0 18px 48px rgba(31,27,58,0.12);overflow:hidden;">
                <tr>
                  <td style="background:linear-gradient(135deg,#6c63ff,#5d40f5);padding:28px 32px;color:#ffffff;">
                    <h1 style="margin:0;font-size:24px;font-weight:700;">Welcome to AI Mock Interview Platform</h1>
                    <p style="margin:8px 0 0;font-size:14px;opacity:0.9;">Your interview practice space is ready.</p>
                  </td>
                </tr>
                <tr>
                  <td style="padding:32px;">
                    <p style="margin:0 0 16px;font-size:16px;">Hi {name or 'there'},</p>
                    <p style="margin:0 0 16px;font-size:15px;line-height:1.6;">
                      You're now registered for the AI Mock Interview platform. Use the credentials below to sign in and start practising.
                    </p>
                    <table role="presentation" cellpadding="0" cellspacing="0" style="width:100%;margin:0 0 10px;border-collapse:separate;border-spacing:0 4px;">
                      <tr>
                        <td style="padding:8px 12px;background:#f5f3ff;font-weight:600;border-radius:10px 0 0 10px;width:160px;">Email</td>
                        <td style="padding:8px 12px;border:1px solid #ece8ff;border-left:none;border-radius:0 10px 10px 0;font-size:15px;">{email}</td>
                      </tr>
                      <tr>
                        <td style="padding:8px 12px;background:#f5f3ff;font-weight:600;border-radius:10px 0 0 10px;white-space:nowrap;">Password</td>
                        <td style="padding:8px 12px;border:1px solid #ece8ff;border-left:none;border-radius:0 10px 10px 0;font-size:15px;letter-spacing:0.6px;">{temp_password}</td>
                      </tr>
                    </table>
                    <p style="margin:0 0 20px;font-size:15px;line-height:1.6;">Click below to launch the app.</p>
                    <p style="margin:0 0 28px;text-align:center;">
                      <a href="{login_url}" style="display:inline-block;background:#6c63ff;color:#ffffff;text-decoration:none;padding:12px 24px;border-radius:999px;font-weight:600;font-size:15px;">Go to login</a>
                    </p>
                    <p style="margin:0 0 16px;font-size:15px;line-height:1.6;">{reset_hint}</p>
                    <p style="margin:24px 0 0;font-size:15px;line-height:1.6;">Thank you</p>
                  </td>
                </tr>
              </table>
              <p style="margin:24px 0 0;font-size:12px;color:#7a7794;">If the button doesn't work, copy and paste this link into your browser:<br/><span style="word-break:break-all;">{login_url}</span></p>
            </td>
          </tr>
        </table>
      </body>
    </html>
    """

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = email
    message.set_content(plain_body)
    message.add_alternative(html_body, subtype="html")

    try:
        if smtp_cfg.get("use_tls", True):
            context = ssl.create_default_context()
            with smtplib.SMTP(smtp_cfg["host"], smtp_cfg["port"]) as server:
                server.starttls(context=context)
                if smtp_cfg.get("username") and smtp_cfg.get("password"):
                    server.login(smtp_cfg["username"], smtp_cfg["password"])
                server.send_message(message)
        else:
            with smtplib.SMTP(smtp_cfg["host"], smtp_cfg["port"]) as server:
                if smtp_cfg.get("username") and smtp_cfg.get("password"):
                    server.login(smtp_cfg["username"], smtp_cfg["password"])
                server.send_message(message)
    except Exception as exc:
        logger.error(f"Failed to send credentials email to {email}: {exc}")
        raise HTTPException(status_code=500, detail="Unable to send credential email") from exc


def send_password_reset_email(email: str, token: str, expires_at: datetime) -> None:
    smtp_cfg = config.SMTP_SETTINGS
    sender = config.EMAIL_SENDER

    if not (smtp_cfg.get("host") and smtp_cfg.get("port") and sender):
        logger.error("SMTP configuration incomplete; password reset email cannot be sent")
        raise HTTPException(status_code=500, detail="Password reset email service not configured")

    base_url = config.PASSWORD_RESET_URL_BASE
    if not base_url:
        base_url = "/login"

    reset_url = base_url.rstrip("/")
    reset_url = f"{reset_url}?email={quote_plus(email)}&token={quote_plus(token)}"

    token_inline = token
    token_html = token

    subject = "Password reset instructions"
    expires_dt = expires_at
    if expires_dt.tzinfo is None:
        expires_dt = expires_dt.replace(tzinfo=timezone.utc)
    expires_ist = expires_dt.astimezone(IST_TZ)
    expires_str = expires_ist.strftime("%Y-%m-%d %H:%M:%S IST")
    plain_body = textwrap.dedent(
        f"""
        Hello,

        You recently requested to reset your password for the AI Mock Interview platform.

        Your one-time reset token is: {token_inline}
        This token expires at {expires_str}.

        Copy the token above and enter it on the password reset screen,
        or open this link in your browser: {reset_url}

        If you did not request this change, please ignore this email.

        Thank you.
        """
    ).strip()

    html_body = f"""
    <html>
      <body style="margin:0;padding:0;background:#f5f5ff;font-family:'Segoe UI',Arial,sans-serif;color:#1f1b3a;">
        <table role="presentation" cellpadding="0" cellspacing="0" width="100%">
          <tr>
            <td align="center" style="padding:32px 16px;">
              <table width="480" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:16px;box-shadow:0 18px 48px rgba(31,27,58,0.12);overflow:hidden;">
                <tr>
                  <td style="background:linear-gradient(135deg,#6c63ff,#5d40f5);padding:28px 32px;color:#ffffff;">
                    <h1 style="margin:0;font-size:24px;font-weight:700;">Reset your AI Mock Interview password</h1>
                    <p style="margin:8px 0 0;font-size:14px;opacity:0.9;">Use the secure token below to finish resetting.</p>
                  </td>
                </tr>
                <tr>
                  <td style="padding:32px;">
                    <p style="margin:0 0 16px;font-size:16px;">Hello,</p>
                    <p style="margin:0 0 16px;font-size:15px;line-height:1.6;">We received a request to reset the password for your AI Mock Interview account. Use the details below to continue.</p>
                    <table role="presentation" cellpadding="0" cellspacing="0" style="width:100%;margin:0 0 20px;border-collapse:separate;border-spacing:0 6px;">
                      <tr>
                        <td style="padding:8px 12px;background:#f5f3ff;font-weight:600;border-radius:10px 0 0 10px;white-space:nowrap;">Reset token</td>
                        <td style="padding:6px 10px;border:1px solid #ece8ff;border-left:none;border-radius:0 10px 10px 0;font-size:11px;font-family:'Roboto Mono','Courier New',Courier,monospace;letter-spacing:0;white-space:nowrap;">
                          <span style="display:inline-block;white-space:nowrap;word-break:keep-all;overflow-wrap:normal;line-height:1;">{token_html}</span>
                        </td>
                      </tr>
                      <tr>
                        <td style="padding:8px 12px;background:#f5f3ff;font-weight:600;border-radius:10px 0 0 10px;white-space:nowrap;">Expires</td>
                        <td style="padding:8px 12px;border:1px solid #ece8ff;border-left:none;border-radius:0 10px 10px 0;font-size:14px;white-space:nowrap;">{expires_str}</td>
                      </tr>
                    </table>
                    <p style="margin:0 0 20px;font-size:15px;line-height:1.6;">Copy the token above and paste it into the password reset screen, or click the button below to jump straight there.</p>
                    <p style="margin:0 0 28px;text-align:center;">
                      <a href="{reset_url}" style="display:inline-block;background:#6c63ff;color:#ffffff;text-decoration:none;padding:12px 24px;border-radius:999px;font-weight:600;font-size:15px;">Open reset page</a>
                    </p>
                    <p style="margin:24px 0 0;font-size:15px;line-height:1.6;">Thank you</p>
                  </td>
                </tr>
              </table>
              <p style="margin:24px 0 0;font-size:12px;color:#7a7794;">If the button doesn't work, copy and paste this link into your browser:<br/><span style="word-break:break-all;">{reset_url}</span></p>
            </td>
          </tr>
        </table>
      </body>
    </html>
    """

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = email
    message.set_content(plain_body)
    message.add_alternative(html_body, subtype="html")

    try:
        if smtp_cfg.get("use_tls", True):
            context = ssl.create_default_context()
            with smtplib.SMTP(smtp_cfg["host"], smtp_cfg["port"]) as server:
                server.starttls(context=context)
                if smtp_cfg.get("username") and smtp_cfg.get("password"):
                    server.login(smtp_cfg["username"], smtp_cfg["password"])
                server.send_message(message)
        else:
            with smtplib.SMTP(smtp_cfg["host"], smtp_cfg["port"]) as server:
                if smtp_cfg.get("username") and smtp_cfg.get("password"):
                    server.login(smtp_cfg["username"], smtp_cfg["password"])
                server.send_message(message)
    except Exception as exc:
        logger.error(f"Failed to send password reset email to {email}: {exc}")
        raise HTTPException(status_code=500, detail="Unable to send password reset email") from exc


def create_password_reset_token(email: str) -> Dict[str, Any]:
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

    with db_pool.get_connection() as conn:
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection unavailable")

        with conn.cursor() as cur:
            cur.execute(
                "SELECT student_id FROM students WHERE LOWER(TRIM(email)) = LOWER(TRIM(%s))",
                (email.strip(),),
            )
            student_row = cur.fetchone()
            if not student_row:
                raise HTTPException(status_code=404, detail="No account found for this email")

            cur.execute(
                """
                INSERT INTO password_reset_tokens (email, token, expires_at)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (email.strip(), token, expires_at),
            )
            conn.commit()

    email_sent = True
    try:
        send_password_reset_email(email.strip(), token, expires_at)
    except Exception as email_exc:
        email_sent = False
        logger.warning(
            "Password reset token created for %s but reset email failed to send: %s",
            email.strip(), email_exc,
        )

    return {"token": token, "expires_at": expires_at.isoformat(), "email_sent": email_sent}


def validate_password_reset_token(email: str, token: str) -> int:
    with db_pool.get_connection() as conn:
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection unavailable")

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, expires_at, used
                FROM password_reset_tokens
                WHERE LOWER(TRIM(email)) = LOWER(TRIM(%s))
                  AND token = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (email, token),
            )
            row = cur.fetchone()

            if not row:
                raise HTTPException(status_code=400, detail="Invalid or expired token")

            token_id, expires_at, used = row
            if used:
                raise HTTPException(status_code=400, detail="This reset link has already been used")

            now_utc = datetime.now(timezone.utc)

            if not isinstance(expires_at, datetime):
                raise HTTPException(status_code=500, detail="Invalid token expiry timestamp")

            if expires_at.tzinfo is None:
                expires_utc = expires_at.replace(tzinfo=timezone.utc)
                if expires_utc < now_utc:
                    try:
                        expires_ist = expires_at.replace(tzinfo=IST_TZ).astimezone(timezone.utc)
                    except Exception:
                        expires_ist = expires_utc
                    if expires_ist >= now_utc:
                        expires_utc = expires_ist
                expires_at_utc = expires_utc
            else:
                expires_at_utc = expires_at.astimezone(timezone.utc)

            if expires_at_utc < now_utc:
                raise HTTPException(status_code=400, detail="This reset link has expired")

            return token_id


def mark_token_used(token_id: int) -> None:
    with db_pool.get_connection() as conn:
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection unavailable")

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE password_reset_tokens
                SET used = TRUE, used_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (token_id,),
            )
            conn.commit()


def update_student_password(email: str, new_password: str) -> None:
    hashed_password = hash_password(new_password)
    with db_pool.get_connection() as conn:
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection unavailable")

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE students
                SET password = %s, last_active = CURRENT_TIMESTAMP
                WHERE LOWER(TRIM(email)) = LOWER(TRIM(%s))
                RETURNING student_id
                """,
                (hashed_password, email),
            )
            result = cur.fetchone()
            if not result:
                raise HTTPException(status_code=404, detail="No account found for this email")
            conn.commit()


def login_student_service(payload: StudentLoginRequest) -> StudentAuthResponse:
    from app.modules.students.repository import fetch_program_info

    try:
        with db_pool.get_connection() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection unavailable")

            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT student_id, name, email, program_id, password
                    FROM students
                    WHERE LOWER(TRIM(email)) = LOWER(TRIM(%s))
                    """,
                    (payload.email,),
                )
                row = cur.fetchone()

                if not row:
                    raise HTTPException(status_code=401, detail="Invalid email or password")

                student_id, student_name, student_email, current_program_id, stored_password = row

                if not verify_password(payload.password, stored_password):
                    raise HTTPException(status_code=401, detail="Invalid email or password")

                if not stored_password or not str(stored_password).startswith("pbkdf2_sha256$"):
                    hashed_password = hash_password(payload.password)
                    cur.execute(
                        "UPDATE students SET password = %s WHERE student_id = %s",
                        (hashed_password, student_id),
                    )

                final_program_id = current_program_id

                if payload.program_id and payload.program_id != current_program_id:
                    cur.execute(
                        "UPDATE students SET program_id = %s WHERE student_id = %s",
                        (payload.program_id, student_id),
                    )
                    final_program_id = payload.program_id

                conn.commit()

                program_details = {}
                if final_program_id:
                    program_info = fetch_program_info(final_program_id)
                    if program_info:
                        program_details = program_info

                token_payload = {
                    "sub": str(student_id),
                    "email": student_email,
                }
                access_token = create_student_jwt_token(token_payload)

                return StudentAuthResponse(
                    access_token=access_token,
                    student_id=student_id,
                    name=student_name,
                    email=student_email,
                    program_id=final_program_id,
                    program_name=program_details.get("program_name"),
                    batch_label=program_details.get("batch_label"),
                    university_name=program_details.get("university_name"),
                    job_roles=program_details.get("job_roles", []),
                )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error logging in student {payload.email}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to log in student") from exc


def reset_password_service(payload: PasswordResetConfirm) -> Dict[str, Any]:
    try:
        if len(payload.new_password.strip()) < 6:
            raise HTTPException(status_code=422, detail="Password must be at least 6 characters long")

        token_id = validate_password_reset_token(payload.email.strip(), payload.token.strip())
        update_student_password(payload.email.strip(), payload.new_password.strip())
        mark_token_used(token_id)

        return {"message": "Password updated successfully"}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error resetting password for {payload.email}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to reset password")
