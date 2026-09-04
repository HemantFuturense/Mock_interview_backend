import logging
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import config
from app.core.database import db_pool
from app.modules.auth.router import router as auth_router
from app.modules.students.router import router as students_router
from app.modules.students.dashboard_extras_router import router as dashboard_extras_router
from app.modules.sandbox.router import router as sandbox_router
from app.modules.interview.router import router as interview_router
from app.modules.admin.router import router as admin_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app.main")

app = FastAPI(
    title="AI Mock Interview & Readiness Platform (Modular Monolith)",
    version="2.0.0-modular",
    description="Enterprise-grade modular monolith backend combining student readiness, hybrid code execution, AI grading, and institutional oversight.",
)

app.add_middleware(
    CORSMiddleware,
    # Local dev origins are always allowed; deployed frontend origin(s) come from
    # FRONTEND_URLS in .env (comma-separated) so this doesn't need a code change per deploy.
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", *config.FRONTEND_URLS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include feature module routers
app.include_router(auth_router)
app.include_router(students_router)
app.include_router(dashboard_extras_router)
app.include_router(sandbox_router)
app.include_router(interview_router)
app.include_router(admin_router)


@app.get("/", tags=["Root"])
async def root():
    """Application root endpoint listing available APIs and modular architecture status."""
    return {
        "app": "AI Mock Interview & Readiness Platform",
        "architecture": "Modular Monolith",
        "version": "2.0.0-modular",
        "status": "online",
        "modules": {
            "auth": "Student authentication and password recovery",
            "students": "Student profiles, resume parsing, and job description processing",
            "sandbox": "Hybrid code execution (Local, Judge0, Piston)",
            "interview": "Live AI interview sessions, audio/video analysis, and grading",
            "admin": "Institutional analytics, dashboard oversight, and question management",
        },
    }


@app.get("/health", tags=["Root"])
async def health_check():
    """Global system health check including database connection state."""
    db_ok = False
    try:
        with db_pool.get_connection() as conn:
            if conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    db_ok = True
    except Exception as exc:
        logger.error("Health check DB failure: %s", exc)

    return {
        "status": "healthy" if db_ok else "degraded",
        "database": "connected" if db_ok else "disconnected",
        "timestamp": datetime.now().isoformat(),
    }
