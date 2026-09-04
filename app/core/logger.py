import logging
from typing import Optional

# Configure root application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("app")


def safe_print(msg: str) -> None:
    """Print message safely without raising UnicodeEncodeError on Windows/console."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def log_question_selection(
    source: str,
    role: Optional[str],
    company: Optional[str],
    difficulty: Optional[str],
    interview_type: Optional[str],
    work_experience: Optional[str],
    question_type: Optional[str],
    question_text: Optional[str],
) -> None:
    """Emit a concise log/print describing the question that was fetched."""
    summary = (question_text or "").replace("\n", " ").strip()
    if len(summary) > 140:
        summary = summary[:137] + "..."
    message = (
        f"[QUESTION FETCH][{source}] role={role or '-'} company={company or '-'} "
        f"difficulty={difficulty or '-'} interview_type={interview_type or '-'} "
        f"work_experience={work_experience or '-'} question_type={question_type or '-'} "
        f"question=\"{summary or 'N/A'}\""
    )
    logger.info(message)
    safe_print(message)
