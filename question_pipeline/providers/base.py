import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..models import ResearchOutput, ValidationResult


def extract_json_object(text: str) -> Any:
    """
    Robust JSON extraction shared by every LLM provider's generate_json().

    LLM responses occasionally aren't clean JSON: they may be wrapped in
    markdown code fences, or -- the failure actually observed in a real
    pilot run ("Extra data: line 81 column 1") -- contain a complete, valid
    JSON value followed by trailing extra content the model appended after
    it. This tries progressively more tolerant strategies, in order:

      1. Parse as-is.
      2. Strip a leading/trailing ```json / ``` code fence, then parse.
      3. Use json.JSONDecoder().raw_decode() to parse only the FIRST
         complete JSON value in the text and ignore anything after it --
         this is what actually recovers from "Extra data" errors, since
         plain json.loads() refuses to parse when trailing content exists.

    This only extracts a JSON value that is already present in the text
    verbatim -- it never invents, repairs, or alters field content. If none
    of the three strategies succeed, the original json.JSONDecodeError is
    raised so callers can apply their own bounded retry/failure handling.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Last resort: parse only the first complete JSON value, discarding
    # whatever trailing content follows it.
    return json.JSONDecoder().raw_decode(cleaned)[0]


class LLMResult:
    """Wrapper for LLM output and usage metadata."""
    def __init__(
        self,
        text: str,
        data: Optional[Any] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = "",
        provider: str = ""
    ):
        self.text = text
        self.data = data
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.model = model
        self.provider = provider

class ResearchProvider(ABC):
    """Abstract interface for web research on job roles and companies."""
    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @abstractmethod
    async def research_role(self, role: str) -> ResearchOutput:
        """Perform comprehensive web research for a job role."""
        pass

    @abstractmethod
    async def research_company(self, company: str) -> ResearchOutput:
        """Perform web research for a company's technology stack and engineering practices."""
        pass

class LLMProvider(ABC):
    """Abstract interface for LLM synthesis, question generation, and validation."""
    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @abstractmethod
    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        task_name: str = "generation"
    ) -> LLMResult:
        """Generate structured JSON response adhering to prompt requirements."""
        pass

    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        task_name: str = "text"
    ) -> LLMResult:
        """Generate plain text output."""
        pass

class EmbeddingProvider(ABC):
    """Abstract interface for generating dense vector embeddings."""
    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @abstractmethod
    async def embed_texts(self, texts: List[str], task_name: str = "rag_embedding") -> List[List[float]]:
        """Compute embeddings for a list of document chunks or questions."""
        pass

    @abstractmethod
    async def embed_query(self, text: str, task_name: str = "query_embedding") -> List[float]:
        """Compute embedding for a single search query."""
        pass

class ValidationProvider(ABC):
    """Abstract interface for multi-tier question quality validation."""
    @abstractmethod
    async def validate_question(self, question_data: Dict[str, Any], role_context: str) -> ValidationResult:
        """Evaluate question quality, rubric alignment, role invariance, and depth."""
        pass
