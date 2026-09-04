import os
from typing import Dict, List, Optional, Set

# Gemini Fallback Models for rate limiting / quota protection.
# NOTE: gemini-2.5-flash, gemini-2.0-flash, gemini-2.0-flash-lite, and gemini-2.5-flash-lite
# all 404 ("no longer available to new users") against this project's API key as of 2026-08-14 --
# verified directly against the Gemini API, not just assumed. Keeping them in this list wastes
# a full retry+backoff cycle per dead model on every single Gemini call in the app before it
# ever reaches a working model. Only list models confirmed to actually respond.
FALLBACK_GEMINI_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
]

# Supported Built-in & Piston Runtimes
DEFAULT_RUNTIMES = [
    {"language": "python", "version": "3.10.0", "aliases": ["py", "python3"]},
    {"language": "javascript", "version": "18.15.0", "aliases": ["js", "node"]},
    {"language": "sql", "version": "3.36.0", "aliases": ["sqlite", "sqlite3"]},
    {"language": "java", "version": "17.0.0", "aliases": []},
    {"language": "cpp", "version": "10.2.0", "aliases": ["c++"]},
    {"language": "c", "version": "10.2.0", "aliases": []},
    {"language": "go", "version": "1.18.0", "aliases": ["golang"]},
    {"language": "rust", "version": "1.60.0", "aliases": []},
]

# Judge0 CE Language IDs
JUDGE0_LANG_MAP = {
    "python": 71, "py": 71, "py3": 71, "python3": 71,
    "javascript": 63, "js": 63, "node": 63, "nodejs": 63,
    "typescript": 74, "ts": 74,
    "java": 62,
    "c": 50,
    "cpp": 54, "c++": 54, "cplusplus": 54,
    "go": 60, "golang": 60,
    "rust": 73,
    "ruby": 72,
    "php": 68,
    "swift": 83,
    "kotlin": 78,
    "sql": 82, "sqlite": 82,
}

# Difficulty Weights for Running Average Calculation
DIFFICULTY_WEIGHTS = {
    "easy": 1,
    "medium": 2,
    "hard": 3,
    "difficult": 3,  # legacy label mapping
}

# Question Type Aliases for flexible database lookups
QUESTION_TYPE_ALIAS_MAP: Dict[str, Set[str]] = {
    "standard": {"standard", "general", "default"},
    "behavioral": {"behavioral", "behavioural", "behavioral question"},
    "technical": {"technical", "tech", "technical question"},
    "coding": {"coding", "code", "programming"},
    "system design": {"system design", "design", "architecture", "system-design"},
    "sql": {"sql", "database", "sql query"},
    "speech": {"speech", "speech based", "speech-based", "speech question"},
}

# Default Analysis & Scoring Constants
DEFAULT_SENTIMENT_SCORE = 5.0
DEFAULT_ACKNOWLEDGMENT = "Thank you for your response. Let's move on to the next question."
DEFAULT_NEXT_DIFFICULTY = "medium"
ANALYSIS_TIMEOUT_SECONDS = float(os.getenv("ACK_ANALYSIS_TIMEOUT_SECONDS", "12"))

# Concurrent session thresholds for load-based question limiting
CONCURRENT_SESSION_THRESHOLD = 25  # If more than this many active sessions, limit questions
CONCURRENT_SESSION_MAX_QUESTIONS = 5  # Reduced max questions when under high load
CONCURRENT_SESSION_TIMEOUT_MINUTES = 30  # Sessions older than this are considered stale

# Scoring Guide and Competencies
SCORING_GUIDE = [
    {"level": "Excellent", "range": "85-100", "description": "Highly ready; recommend strongly."},
    {"level": "Good", "range": "70-84", "description": "Ready with minor gaps."},
    {"level": "Average", "range": "55-69", "description": "Needs guided support before job readiness."},
    {"level": "Below Average", "range": "0-54", "description": "Requires significant improvement."},
]

TECHNICAL_COMPETENCIES = [
    {
        "name": "Problem Solving & Logical Thinking",
        "weight": 30,
        "description": "Ability to analyse problems, structure approaches, and consider efficiency trade-offs.",
    },
    {
        "name": "Technical Concepts & Domain Knowledge",
        "weight": 25,
        "description": "Depth of fundamentals across the relevant technical stack and correctness of reasoning.",
    },
    {
        "name": "Application & Project Readiness",
        "weight": 20,
        "description": "Real-world experience, project articulation, and tool proficiency.",
    },
    {
        "name": "Communication & STAR Response",
        "weight": 15,
        "description": "Clarity and structure while explaining using STAR/PEEL frameworks.",
    },
    {
        "name": "Aptitude & Interview Readiness",
        "weight": 10,
        "description": "Overall composure, analytical reasoning, and industry awareness.",
    },
]

HR_COMPETENCIES = [
    {
        "name": "Career Motivation & Goal Alignment",
        "weight": 25,
        "description": "Understanding of career choices and alignment with company goals.",
    },
    {
        "name": "Cultural Fit & Value Alignment",
        "weight": 25,
        "description": "Alignment with organisational values, adaptability, and integrity.",
    },
    {
        "name": "Communication & Presentation",
        "weight": 20,
        "description": "Professional communication, confidence, and listening skills.",
    },
    {
        "name": "Emotional Intelligence & Interpersonal Skills",
        "weight": 15,
        "description": "Empathy, collaboration, and conflict management.",
    },
    {
        "name": "Learning Agility & Growth Orientation",
        "weight": 15,
        "description": "Curiosity, openness to feedback, and continuous learning.",
    },
]

BEHAVIORAL_COMPETENCIES = [
    {
        "name": "Ownership & Accountability",
        "weight": 25,
        "description": "Taking responsibility, initiative, and learning from mistakes.",
    },
    {
        "name": "Teamwork & Collaboration",
        "weight": 20,
        "description": "Working well with diverse teams, contributing ideas, and respecting others.",
    },
    {
        "name": "Problem-Solving in Real Situations",
        "weight": 20,
        "description": "Handling real-world challenges with creativity, resilience, and structure.",
    },
    {
        "name": "Adaptability & Resilience",
        "weight": 20,
        "description": "Responding positively to change, setbacks, and ambiguity.",
    },
    {
        "name": "Ethical Judgment & Professionalism",
        "weight": 15,
        "description": "Acting with integrity, honesty, and professionalism in decisions.",
    },
]

MAX_PROMPT_CHARS_QUESTIONS = 48000
MAX_PROMPT_CHARS_COMPETENCIES = 64000



def get_question_type_aliases(normalized_label: Optional[str]) -> List[str]:
    """Return all acceptable values for a normalized question type label."""
    if not normalized_label:
        return []
    base = normalized_label.strip().lower()
    if not base:
        return []

    if base in QUESTION_TYPE_ALIAS_MAP:
        alias_set = set(value.strip().lower() for value in QUESTION_TYPE_ALIAS_MAP[base] if value)
        alias_set.add(base)
        return sorted(alias_set)

    return [base]


def translate_difficulty_label(label: str) -> str:
    """Normalize AI difficulty labels to supported values."""
    if not label:
        return "medium"
    normalized = label.lower().strip()
    if normalized == "difficult":
        return "hard"
    if normalized not in ("easy", "medium", "hard"):
        return "medium"
    return normalized


def normalize_question_type_label(label: Optional[str]) -> Optional[str]:
    """Canonicalize question type labels so DB lookups match flexible inputs."""
    if not label:
        return None
    normalized = label.strip().lower()
    if not normalized:
        return None
    normalized = normalized.replace("-", " ").replace("_", " ")
    normalized = " ".join(part for part in normalized.split() if part)
    for canonical, aliases in QUESTION_TYPE_ALIAS_MAP.items():
        if normalized in aliases or canonical in normalized:
            return canonical
    return normalized
