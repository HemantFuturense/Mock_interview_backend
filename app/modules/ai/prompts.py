from pathlib import Path
from typing import Any
from jinja2 import Environment, FileSystemLoader, select_autoescape
from app.core.logger import logger

# Template root resolved pointing to root templates/feedback directory
# Path(__file__).resolve().parent.parent.parent.parent points to workspace root
TEMPLATE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "templates" / "feedback"

jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(disabled_extensions=("j2",), default_for_string=False, default=True),
    trim_blocks=True,
    lstrip_blocks=True,
)


def get_template(template_name: str) -> Any:
    """Safely retrieve a Jinja2 template by name."""
    try:
        return jinja_env.get_template(template_name)
    except Exception as exc:
        logger.error(f"Failed to load Jinja template '{template_name}': {exc}")
        raise


def render_template(template_name: str, **context: Any) -> str:
    """Render a Jinja2 template to string given a context dictionary."""
    template = get_template(template_name)
    return template.render(**context)
