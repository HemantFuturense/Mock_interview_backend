from typing import Any, Dict, List, Optional

from app.core.logger import logger
from app.modules.students.repository import (
    get_student_interview_history_repo,
    get_student_performance_summary_repo,
)


def get_student_performance_summary_service(student_id: int) -> Dict[str, Any]:
    try:
        summary = get_student_performance_summary_repo(student_id)
        if not summary:
            return {
                "avg_score": 0.0,
                "total_sessions": 0,
                "hours_practiced": 0.0,
                "streak": 0,
                "sessions": 0,
            }

        completion_dates = summary.get("completion_dates", [])
        streak = 0
        if completion_dates:
            current_date = None
            for completion_date in sorted(completion_dates, reverse=True):
                if current_date is None:
                    streak = 1
                    current_date = completion_date
                    continue
                expected = current_date - __import__('datetime').timedelta(days=1)
                if completion_date == expected:
                    streak += 1
                    current_date = completion_date
                else:
                    break

        return {
            "avg_score": float(summary.get("avg_score") or 0.0),
            "total_sessions": int(summary.get("total_sessions") or 0),
            "hours_practiced": float(summary.get("hours_practiced") or 0.0),
            "streak": streak,
            "sessions": int(summary.get("total_sessions") or 0),
        }
    except Exception as exc:
        logger.error(f"Failed to build performance summary for student {student_id}: {exc}")
        raise


def get_student_interview_history_service(student_id: int, page: int = 1, page_size: int = 10) -> Dict[str, Any]:
    try:
        page = max(1, page)
        page_size = max(1, page_size)
        return get_student_interview_history_repo(student_id, page, page_size)
    except Exception as exc:
        logger.error(f"Failed to build interview history for student {student_id}: {exc}")
        raise
