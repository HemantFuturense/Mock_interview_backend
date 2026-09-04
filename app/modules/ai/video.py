import asyncio
import json
import os
from typing import Any, Dict, Optional

from app.config.constants import FALLBACK_GEMINI_MODELS
from app.config.settings import config
from app.core.database import db_pool
from app.core.logger import logger
from app.modules.ai.client import generate_content_with_fallback
from app.utils.text_utils import extract_first_number


def analyze_video_sentiment_sync(video_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Synchronous version of video sentiment analysis using Gemini video model."""
    if not video_path or not os.path.exists(video_path):
        return None

    def _run_gemini_video_call() -> Dict[str, Any]:
        with open(video_path, "rb") as clip_file:
            video_bytes = clip_file.read()

        response = asyncio.run(
            generate_content_with_fallback(
                [
                    (
                        """
                        You are observing a candidate answering an interview question.
                        Analyze only visible behavior and speech delivery in the video.
                        Do NOT assess answer correctness.

                        Evaluate:
                        - Overall sentiment
                        - Engagement and confidence
                        - Non-verbal behavior (posture, eye contact, gestures,nervous facial expressions)
                        - Verbal delivery (clarity, fillers, stammering)

                        Return STRICT JSON only with:
                        {
                        "sentiment": float (0–10),
                        "engagement": string (max 50 words),
                        "dominant_expression": string (max 50 words),
                        "verbal_strength": string (max 30 words)
                        }

                        Be concise. No extra text."""
                    ),
                    {
                        "mime_type": "video/webm",
                        "data": video_bytes,
                    },
                ],
                generation_config={"response_mime_type": "application/json"},
                retry_label="Video sentiment analysis",
                models=[config.GEMINI_VIDEO_MODEL, *FALLBACK_GEMINI_MODELS],
            )
        )

        if not response or not response.text:
            raise ValueError("Empty response from Gemini video model")

        try:
            raw = json.loads(response.text)
        except json.JSONDecodeError as decode_error:
            logger.warning(f"Gemini video response not JSON, falling back to heuristic parsing: {decode_error}")
            raw_score = extract_first_number(response.text)
            parsed = {
                "sentiment": raw_score if raw_score is not None else 5.0,
                "engagement": "Engagement details unavailable; Gemini returned non-JSON output.",
                "dominant_expression": "Expression details unavailable; Gemini returned non-JSON output.",
                "verbal_strength": "Verbal delivery details unavailable; Gemini returned non-JSON output.",
            }
            return parsed

        return {
            "sentiment": float(raw.get("sentiment", 5.0)),
            "engagement": raw.get("engagement") or raw.get("engagement_summary", "Engagement details unavailable."),
            "dominant_expression": raw.get("dominant_expression")
            or raw.get("dominant_expression_summary", "Expression details unavailable."),
            "verbal_strength": raw.get("verbal_strength") or raw.get("verbal_summary", "Verbal delivery details unavailable."),
        }

    try:
        analysis = _run_gemini_video_call()
        if not isinstance(analysis, dict):
            return None

        analysis.setdefault("sentiment", 5.0)
        analysis.setdefault("engagement", "Engagement details unavailable.")
        analysis.setdefault("dominant_expression", "Expression details unavailable.")
        analysis.setdefault("verbal_strength", "Verbal delivery details unavailable.")
        analysis.setdefault("clip_path", video_path)
        return analysis
    except Exception as exc:
        logger.error(f"Synchronous video analysis failed for clip {video_path}: {exc}")
        return None


def process_video_analysis_background(session_id: str, question_number: int, video_path: str) -> None:
    """Background task to analyze video and update database."""
    try:
        logger.info(f"Starting background video analysis for session {session_id}, question {question_number}")

        # Run synchronous video analysis
        analysis = analyze_video_sentiment_sync(video_path)

        # Update database with results
        with db_pool.get_connection() as conn:
            if not conn:
                logger.error(f"Failed to get DB connection for video analysis update (session {session_id})")
                return

            with conn.cursor() as cur:
                if analysis:
                    cur.execute(
                        """
                        UPDATE interview_data 
                        SET video_analysis = %s, video_analysis_status = 'completed'
                        WHERE session_id = %s AND question_number = %s
                        """,
                        (json.dumps(analysis), session_id, question_number),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE interview_data 
                        SET video_analysis_status = 'failed'
                        WHERE session_id = %s AND question_number = %s
                        """,
                        (session_id, question_number),
                    )
                conn.commit()

        # Clean up video file after analysis
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
                logger.debug(f"Removed video clip after analysis: {video_path}")
            except OSError as cleanup_error:
                logger.warning(f"Failed to remove video clip {video_path}: {cleanup_error}")

        logger.info(f"Completed background video analysis for session {session_id}, question {question_number}")

    except Exception as exc:
        logger.error(f"Background video analysis failed for session {session_id}, question {question_number}: {exc}")
        # Mark as failed in database
        try:
            with db_pool.get_connection() as conn:
                if conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE interview_data 
                            SET video_analysis_status = 'failed'
                            WHERE session_id = %s AND question_number = %s
                            """,
                            (session_id, question_number),
                        )
                        conn.commit()
        except Exception as db_err:
            logger.error(f"Failed to mark video analysis as failed in DB: {db_err}")
