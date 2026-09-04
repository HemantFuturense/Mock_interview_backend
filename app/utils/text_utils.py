import json
import re
from typing import Any, Dict, Optional
from app.core.logger import logger


def parse_bool(value: Optional[Any]) -> Optional[bool]:
    """Safely parse boolean values from strings, numbers, or booleans."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def clean_json_text(raw_text: str) -> str:
    """Clean markdown code block wrappers (```json ... ```) around JSON text."""
    if not raw_text:
        return ""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    return cleaned


def parse_feedback_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """Parse structured feedback JSON with robust recovery mechanisms."""
    cleaned = clean_json_text(raw_text)
    if not cleaned:
        return None

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse structured feedback JSON: {exc}")

        # Fallback: try trimming any trailing non-JSON content (e.g., extra commentary)
        last_brace = cleaned.rfind("}")
        if last_brace != -1 and last_brace < len(cleaned) - 1:
            trimmed = cleaned[: last_brace + 1]
            try:
                parsed = json.loads(trimmed)
                logger.info("Recovered structured feedback JSON after trimming trailing content.")
                return parsed
            except json.JSONDecodeError:
                logger.error("Fallback JSON trimming also failed; giving up on structured feedback.")

        return None


def extract_first_number(text: str) -> Optional[float]:
    """Extract the first floating point or integer number from a string."""
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def normalize_video_analysis(raw_value: Optional[Any]) -> Optional[Dict[str, Any]]:
    """Normalize video analysis payloads from JSON strings, dictionaries, or raw text."""
    if raw_value is None:
        return None
    if isinstance(raw_value, dict):
        return raw_value
    if isinstance(raw_value, (bytes, bytearray)):
        raw_value = raw_value.decode("utf-8", errors="ignore")
    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if not raw_value:
            return None
        try:
            parsed = json.loads(raw_value)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, str):
                return {"summary": parsed}
        except Exception:
            return {"summary": raw_value}
    return None


# Aliases for internal usage across modules
_clean_json_text = clean_json_text
_parse_feedback_json = parse_feedback_json
