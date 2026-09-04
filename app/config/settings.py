import os
from pathlib import Path
from dotenv import load_dotenv

# Always load the backend-local environment file.  Depending on the terminal's
# working directory made the previous call load a different .env (or none at
# all), which is particularly confusing when launching Uvicorn from VS Code.
BACKEND_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_ROOT / ".env")


class Config:
    """Centralized application configuration loaded from environment variables."""

    # Database  
    DB_CONFIG = {
        "dbname": os.getenv("DB_NAME", "ai_mock_interviews"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
    }

    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    # Google AI Studio issues both legacy standard keys (AIza...) and the
    # newer authorization keys (AQ...). Both authenticate Gemini API calls.
    if not (GEMINI_API_KEY.startswith("AIza") or GEMINI_API_KEY.startswith("AQ.")):
        raise ValueError(
            "GEMINI_API_KEY does not look like a Google AI Studio key. "
            "Use a standard key (AIza...) or authorization key (AQ...) from Google AI Studio."
        )

    # Email / SMTP Settings
    SMTP_SETTINGS = {
        "host": os.getenv("SMTP_HOST"),
        "port": int(os.getenv("SMTP_PORT", "587")),
        "username": os.getenv("SMTP_USERNAME"),
        "password": os.getenv("SMTP_PASSWORD"),
        "use_tls": os.getenv("SMTP_USE_TLS", "true").lower() == "true",
    }
    EMAIL_SENDER = os.getenv("EMAIL_SENDER")
    PASSWORD_RESET_URL_BASE = os.getenv("PASSWORD_RESET_URL_BASE")

    # Caching
    CACHE_TTL = int(os.getenv("CACHE_TTL", "900"))  # seconds
    MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "128"))

    # Database Connection Pool
    MIN_CONNECTIONS = int(os.getenv("MIN_DB_CONNECTIONS", "5"))
    MAX_CONNECTIONS = int(os.getenv("MAX_DB_CONNECTIONS", "50"))

    # Piston Code Runner configuration
    PISTON_BASE_URL = os.getenv("PISTON_BASE_URL", "https://emkc.org/api/v2/piston")

    # Deployed frontend origin(s) to allow via CORS, comma-separated (e.g. Vercel URL(s)).
    # Local dev origins are always allowed regardless of this setting.
    FRONTEND_URLS = [url.strip() for url in os.getenv("FRONTEND_URLS", "").split(",") if url.strip()]

    # Media storage (resolve two levels up from app/config/settings.py to reach root)
    MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", Path(__file__).resolve().parent.parent.parent / "media"))
    INTERVIEW_VIDEO_DIR = MEDIA_ROOT / "interview_videos"

    # Gemini models
    GEMINI_VIDEO_MODEL = os.getenv("GEMINI_VIDEO_MODEL", "gemini-2.5-flash-lite")
    GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

    # Admin JWT Settings
    ADMIN_JWT_SECRET = os.getenv("ADMIN_JWT_SECRET")
    if not ADMIN_JWT_SECRET:
        raise ValueError("ADMIN_JWT_SECRET environment variable not set")
    ADMIN_JWT_ALGORITHM = os.getenv("ADMIN_JWT_ALGORITHM", "HS256")
    ADMIN_JWT_EXPIRES_MINUTES = int(os.getenv("ADMIN_JWT_EXPIRES", "1440"))

    # Student JWT Settings
    STUDENT_JWT_SECRET = os.getenv("STUDENT_JWT_SECRET")
    if not STUDENT_JWT_SECRET:
        raise ValueError("STUDENT_JWT_SECRET environment variable not set")
    STUDENT_JWT_ALGORITHM = os.getenv("STUDENT_JWT_ALGORITHM", "HS256")
    STUDENT_JWT_EXPIRES_MINUTES = int(os.getenv("STUDENT_JWT_EXPIRES", "1440"))

    # TEMPORARY LOCAL-ONLY: path to the frontend's public/logos folder, used by
    # the admin "add company" flow to save uploaded logos as static frontend
    # assets. Only viable pre-deploy, when both apps run from source on the
    # same machine. See app/modules/admin/service.py for the full warning.
    FRONTEND_LOGOS_DIR = Path(
        os.getenv(
            "FRONTEND_LOGOS_DIR",
            str(
                BACKEND_ROOT.parents[2]
                / "ai_mock_interview_v5-main"
                / "ai_mock_interview_v5-main"
                / "public"
                / "logos"
            ),
        )
    )


# Create global config instance
config = Config()
settings = config
# Media directories are created lazily by the code paths that write into them
# (see interview/service.py:_save_uploaded_video) rather than here at import
# time, so a read-only MEDIA_ROOT doesn't crash app startup.
