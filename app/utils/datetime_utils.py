from datetime import date, datetime, timezone
from typing import Optional, Union
from zoneinfo import ZoneInfo

IST_TZ = ZoneInfo("Asia/Kolkata")


def ensure_datetime(value: Optional[Union[datetime, date]]) -> Optional[datetime]:
    """Ensure date/datetime input is normalized as a datetime object."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.combine(value, datetime.min.time())


def to_ist_datetime(
    value: Optional[Union[datetime, date]], assume_tz: timezone = timezone.utc
) -> Optional[datetime]:
    """Convert a UTC or naive datetime to Indian Standard Time (IST)."""
    dt_value = ensure_datetime(value)
    if not dt_value:
        return None
    if dt_value.tzinfo is None:
        dt_value = dt_value.replace(tzinfo=assume_tz)
    return dt_value.astimezone(IST_TZ)


def format_datetime_ist(
    value: Optional[Union[datetime, date]], assume_tz: timezone = timezone.utc
) -> Optional[str]:
    """Format a datetime or date as an ISO-8601 string in Indian Standard Time (IST)."""
    ist_dt = to_ist_datetime(value, assume_tz)
    return ist_dt.isoformat() if ist_dt else None


# Aliases for internal usage across modules
_ensure_datetime = ensure_datetime
_to_ist_datetime = to_ist_datetime
