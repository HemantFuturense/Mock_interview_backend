import re
from typing import Tuple
from app.config.constants import DEFAULT_ACKNOWLEDGMENT, DEFAULT_NEXT_DIFFICULTY, DEFAULT_SENTIMENT_SCORE
from app.core.logger import logger
from app.modules.ai.client import generate_content_with_fallback


async def analyze_answer_and_generate_response(
    question: str, answer: str, mandatory_skills: str, job_role: str
) -> Tuple[float, str, str]:
    """Analyze answer and generate acknowledgment + difficulty using Gemini with retry logic."""
    prompt = f"""
            You are evaluating a candidate's response to this interview question for a {job_role} position:
            
            Question: "{question}"
            Candidate's Answer: "{answer}"
            Required Skills: {mandatory_skills}
            
            Based on this response, provide:
            1. Sentiment score (0-10) for confidence/quality
            2. Short acknowledgment (1-2 sentences) as if speaking to the candidate.Dont ask any question or anything here.Just a short acknowledgment.
            3. Difficulty level for the NEXT question
            
            Difficulty level guidelines:
            - easy: If candidate struggled, was unclear, or gave incorrect/incomplete answers
            - medium: If response was adequate but could be more detailed, or correct but basic
            - difficult: If response was excellent with deep knowledge and detailed correct answers
            
            Format your response exactly as:
            SENTIMENT: [0-10]
            ACKNOWLEDGMENT: [your acknowledgment here]
            DIFFICULTY: [easy/medium/difficult]
            
            Example:
            SENTIMENT: 7
            ACKNOWLEDGMENT: Great answer! I can see you have solid experience with this technology.
            DIFFICULTY: medium
            """

    try:
        logger.info("[Gemini] Starting sentiment/difficulty call for job_role=%s", job_role)
        response = await generate_content_with_fallback(
            prompt,
            retry_label="Gemini sentiment analysis",
        )
        logger.info("[Gemini] Completed sentiment/difficulty call for job_role=%s", job_role)
        response_text = response.text.strip() if response and response.text else ""

        # Debug logging
        logger.info(f"Gemini raw response: {response_text[:200]}...")

        # Parse sentiment, acknowledgment, and difficulty
        sentiment_score = DEFAULT_SENTIMENT_SCORE
        acknowledgment = DEFAULT_ACKNOWLEDGMENT
        difficulty = DEFAULT_NEXT_DIFFICULTY

        lines = response_text.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("SENTIMENT:"):
                try:
                    sentiment_text = line.replace("SENTIMENT:", "").strip()
                    sentiment_score = float(sentiment_text.split()[0])
                    sentiment_score = max(0.0, min(10.0, sentiment_score))  # Clamp 0-10
                    logger.info(f"Parsed sentiment: {sentiment_score}")
                except Exception as e:
                    logger.warning(f"Failed to parse sentiment: {e}")
            elif line.startswith("ACKNOWLEDGMENT:"):
                acknowledgment = line.replace("ACKNOWLEDGMENT:", "").strip()
                logger.info(f"Parsed acknowledgment: {acknowledgment[:50]}...")
            elif line.startswith("DIFFICULTY:"):
                difficulty = line.replace("DIFFICULTY:", "").strip().lower()
                # Validate difficulty level
                if difficulty not in ["easy", "medium", "difficult"]:
                    difficulty = "medium"
                logger.info(f"Parsed difficulty: {difficulty}")

        # More flexible parsing if exact format fails
        if acknowledgment == DEFAULT_ACKNOWLEDGMENT:
            logger.warning("Exact format parsing failed, trying alternatives...")

            # Try to extract any meaningful text from the response
            response_lines = [line.strip() for line in response_text.split("\n") if line.strip()]

            # Look for the longest meaningful line (likely the acknowledgment)
            for line in response_lines:
                # Skip lines that look like sentiment scores
                if not line.lower().startswith("sentiment") and len(line) > 20:
                    clean_line = line.replace("ACKNOWLEDGMENT:", "").replace("Acknowledgment:", "").strip()
                    if 10 < len(clean_line) < 300:
                        acknowledgment = clean_line
                        logger.info(f"Alternative parsing found: {acknowledgment[:50]}...")
                        break

            # If still no good acknowledgment, try to use the entire response (cleaned)
            if acknowledgment == DEFAULT_ACKNOWLEDGMENT:
                clean_response = response_text.replace("SENTIMENT:", "").replace("ACKNOWLEDGMENT:", "").strip()
                clean_response = re.sub(r"^\d+\.?\d*\s*", "", clean_response).strip()

                if 10 < len(clean_response) < 300:
                    acknowledgment = clean_response
                    logger.info(f"Using cleaned full response: {acknowledgment[:50]}...")

        # Fallback if acknowledgment is too long or empty
        if len(acknowledgment) > 300 or len(acknowledgment) < 10:
            acknowledgment = (
                "Thank you for that detailed response. I can see you have relevant experience. "
                "Let's continue with the next question."
            )
            logger.warning("Using fallback acknowledgment")

        logger.info(
            f"Sentiment: {sentiment_score}/10, Acknowledgment: {acknowledgment[:50]}..., Difficulty: {difficulty}"
        )
        return sentiment_score, acknowledgment, difficulty
    except Exception as exc:
        logger.error(f"All attempts failed for sentiment analysis: {exc}")
        return DEFAULT_SENTIMENT_SCORE, DEFAULT_ACKNOWLEDGMENT, DEFAULT_NEXT_DIFFICULTY


async def generate_ai_acknowledgment(
    question: str, answer: str, mandatory_skills: str, job_role: str
) -> str:
    """Generate intelligent AI acknowledgment using Gemini."""
    try:
        prompt = f"""
        You are an AI interviewer conducting a {job_role} interview. 
        
        Question asked: "{question}"
        Candidate's answer: "{answer}"
        Required skills: {mandatory_skills}
        
        Provide a brief, professional acknowledgment (2-3 sentences) that:
        1. Acknowledges their response positively
        2. Shows you understood their answer
        3. Briefly mentions a key point from their response
        
        Keep it conversational and encouraging. Don't provide detailed feedback yet.
        
        Example format: "Thank you for sharing that experience with [specific detail]. I can see you have good understanding of [relevant concept]. Let's move to the next question."

        Note: Dont ask any questions to the user in the acknowledgment. You should handle only the acknowledgement part.
        """

        response = await generate_content_with_fallback(
            prompt,
            retry_label="Gemini acknowledgment generation",
        )
        acknowledgment = response.text.strip() if response and response.text else ""

        # Fallback if response is too long or empty
        if len(acknowledgment) > 200 or len(acknowledgment) < 10:
            acknowledgment = (
                "Thank you for that detailed response. I can see you have relevant experience. "
                "Let's continue with the next question."
            )

        logger.info(f"Generated AI acknowledgment: {acknowledgment[:100]}...")
        return acknowledgment

    except Exception as e:
        logger.error(f"Error generating AI acknowledgment: {e}")
        return DEFAULT_ACKNOWLEDGMENT


def extract_scores_from_feedback(feedback_text: str) -> dict:
    """Derive overall score and rubric breakdown from structured feedback."""
    from app.utils.text_utils import _parse_feedback_json
    try:
        parsed = _parse_feedback_json(feedback_text)
        if parsed and "core_competencies" in parsed:
            competencies = parsed.get("core_competencies", [])
            if not competencies:
                logger.warning("core_competencies array is empty")
                return {"overall_score": 0.0}

            total_weighted_score = 0.0
            total_weight = 0.0
            for comp in competencies:
                if isinstance(comp, dict):
                    score = float(comp.get("score", 0.0) or 0.0)
                    weight = float(comp.get("weight", 0.0) or 0.0)
                    total_weighted_score += score * weight
                    total_weight += weight

            if total_weight > 0:
                overall_score = total_weighted_score / total_weight
                logger.info(
                    f"Weighted score extraction: Overall={overall_score:.2f} (from {len(competencies)} competencies)"
                )
                rubric_entries = []
                for comp in competencies:
                    if not isinstance(comp, dict):
                        continue
                    rubric_entries.append({
                        "name": comp.get("name"),
                        "score": comp.get("score"),
                    })
                interview_type = (
                    parsed.get("metadata", {}).get("interview_type")
                    or parsed.get("metadata", {}).get("interviewType")
                    or "unknown"
                )
                return {
                    "overall_score": overall_score,
                    "rubric_scores": {
                        "interview_type": interview_type,
                        "rubric": rubric_entries,
                    },
                }

            logger.warning("Total weight is zero, cannot calculate weighted average")
            return {"overall_score": 0.0}

        summary_scores = []
        if parsed:
            for key in ("technical_summary", "communication_summary", "attitude_summary"):
                score_value = parsed.get(key, {}).get("score") if isinstance(parsed.get(key), dict) else None
                if score_value is not None:
                    try:
                        summary_scores.append(float(score_value))
                    except (TypeError, ValueError):
                        logger.warning(f"Unable to parse {key} score from feedback payload")

            if summary_scores:
                overall_score = sum(summary_scores) / len(summary_scores)
                logger.info(
                    f"Legacy structured score extraction produced overall score {overall_score:.2f}"
                )
                return {"overall_score": overall_score}

        section_patterns = (
            r"##\s*`?TECHNICAL_SKILLS_SUMMARY`?.*?###\s*Score:\s*`?([0-9]+(?:\.[0-9]+)?)/5`?",
            r"##\s*`?COMMUNICATION_STAR_RESPONSE`?.*?###\s*Score:\s*`?([0-9]+(?:\.[0-9]+)?)/5`?",
            r"##\s*`?ATTITUDE_INTERVIEW_READINESS`?.*?###\s*Score:\s*`?([0-9]+(?:\.[0-9]+)?)/5`?",
        )

        markdown_scores = []
        for pattern in section_patterns:
            match = re.search(pattern, feedback_text, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    markdown_scores.append(float(match.group(1)))
                except (TypeError, ValueError):
                    logger.warning("Failed to parse markdown score while extracting overall score")

        if markdown_scores:
            overall_score = sum(markdown_scores) / len(markdown_scores)
            logger.info(
                f"Markdown score extraction (legacy) produced overall score {overall_score:.2f}"
            )
            return {"overall_score": overall_score}

        logger.warning("No score signals detected in feedback; defaulting overall score to 0.0")
        return {"overall_score": 0.0}

    except Exception as e:
        logger.error(f"Error extracting scores from feedback: {e}")
        return {"overall_score": 0.0}


def update_session_scores(session_id: str, overall_score: float, rubric_scores: dict | None = None) -> bool:
    """Persist the overall score for a session, along with rubric breakdown if provided."""
    from datetime import datetime
    from psycopg2.extras import Json
    from app.core.database import db_pool

    try:
        with db_pool.get_connection() as conn:
            if not conn:
                return False

            with conn.cursor() as cur:
                cur.execute(
                    "SELECT started_at FROM session_metadata WHERE session_id = %s",
                    (session_id,),
                )
                started_at = cur.fetchone()
                duration_minutes = None
                if started_at and started_at[0]:
                    duration_seconds = (datetime.now() - started_at[0]).total_seconds()
                    duration_minutes = int(duration_seconds / 60)

                cur.execute(
                    """
                        UPDATE session_metadata
                        SET overall_score = %s,
                            rubric_scores = %s,
                            technical_score = NULL,
                            communication_score = NULL,
                            attitude_score = NULL,
                            completed_at = CURRENT_TIMESTAMP,
                            status = 'completed',
                            feedback_generated = TRUE,
                            duration_minutes = %s
                        WHERE session_id = %s
                    """,
                    (
                        overall_score,
                        Json(rubric_scores) if rubric_scores is not None else None,
                        duration_minutes,
                        session_id,
                    ),
                )

                conn.commit()
                logger.info(
                    f"Updated overall score for session {session_id}: Overall={overall_score:.2f}"
                )
                return True

    except Exception as e:
        logger.error(f"Error updating session scores: {e}")
        return False


def update_scores_from_feedback(session_id: str, feedback_text: str) -> bool:
    """Extract the overall score from feedback and persist it."""
    from app.core.database import db_pool

    try:
        logger.info(f"Starting score extraction for session {session_id}")

        scores = extract_scores_from_feedback(feedback_text)
        overall_score = scores.get("overall_score")
        if overall_score is None:
            logger.error("Overall score missing after extraction; aborting DB update")
            return False

        with db_pool.get_connection() as conn:
            if conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT session_id FROM session_metadata WHERE session_id = %s",
                        (session_id,),
                    )
                    exists = cur.fetchone()

                    if not exists:
                        logger.warning(
                            f"Session {session_id} not found in session_metadata table, creating it..."
                        )
                        cur.execute(
                            """
                                INSERT INTO session_metadata (session_id, status, created_at)
                                VALUES (%s, 'active', CURRENT_TIMESTAMP)
                                ON CONFLICT (session_id) DO NOTHING
                            """,
                            (session_id,),
                        )
                        conn.commit()

        rubric_scores = scores.get("rubric_scores")

        success = update_session_scores(session_id, float(overall_score), rubric_scores)
        if success:
            logger.info(
                f"Successfully updated overall score from feedback for session {session_id}"
            )
        else:
            logger.error(f"Failed to update overall score for session {session_id}")

        return success

    except Exception as e:
        logger.error(f"Error updating scores from feedback: {e}")
        return False

