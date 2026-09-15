import os
from pathlib import Path
from dotenv import load_dotenv

from typing import Optional

# Ensure environment variables are loaded
load_dotenv()
# Also check parent .env if needed
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

def is_valid_key(key: Optional[str]) -> bool:
    """Return True if an API key is present and is not a dummy/placeholder value."""
    if not key or not isinstance(key, str):
        return False
    cleaned = key.strip().lower()
    if len(cleaned) < 8:
        return False
    placeholders = ("dummy", "replace_with_real_key", "placeholder", "your_", "todo", "none")
    return not any(p in cleaned for p in placeholders)


class PricingNotConfiguredError(RuntimeError):
    """Raised when cost tracking is asked to price a provider/model pair
    that has no entry in PipelineConfig.PRICING. This is deliberate:
    fabricating a $0.00 cost for a real, paid API call would silently
    misrepresent actual spend. Fix by adding a verified entry (from the
    provider's official pricing page) to PipelineConfig.PRICING -- never a
    guessed number."""
    pass


class PipelineConfig:
    # Directory paths
    BASE_DIR: Path = Path(__file__).resolve().parent
    # QUESTION_PIPELINE_DATA_DIR lets the test suite redirect all pipeline
    # data (state, cost log, vector store, question bank, validation
    # attempts) to an isolated temp directory instead of the real
    # question_pipeline/data/ folder. Unset in production, so production
    # behavior is exactly as before -- this is opt-in isolation for tests
    # only (set by question_pipeline/tests/__init__.py before any other
    # pipeline module is imported).
    DATA_DIR: Path = Path(os.getenv("QUESTION_PIPELINE_DATA_DIR", str(BASE_DIR / "data")))
    KNOWLEDGE_DIR: Path = DATA_DIR / "knowledge_base"
    STATE_FILE: Path = DATA_DIR / "state.json"
    COST_LOG_FILE: Path = DATA_DIR / "cost_log.json"
    VECTOR_STORE_FILE: Path = DATA_DIR / "vector_store.json"
    QUESTION_BANK_FILE: Path = DATA_DIR / "question_bank.json"
    PILOT_REPORT_FILE: Path = DATA_DIR / "pilot_quality_report.json"

    # Testing & Mode
    MOCK_MODE: bool = os.getenv("MOCK_MODE", "false").lower() in ("true", "1", "yes")

    # Providers
    RESEARCH_PROVIDER: str = os.getenv("RESEARCH_PROVIDER", "perplexity").lower()
    RESEARCH_MODEL: str = os.getenv("RESEARCH_MODEL", "sonar")

    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")

    # Configurable embedding provider and model
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")

    # Multi-tier Validation Providers
    VALIDATION_PROVIDER: str = os.getenv("VALIDATION_PROVIDER", "gemini").lower()
    FALLBACK_LLM_PROVIDER: str = os.getenv("FALLBACK_LLM_PROVIDER", "openai").lower()
    FALLBACK_LLM_MODEL: str = os.getenv("FALLBACK_LLM_MODEL", "gpt-4o-mini")
    ESCALATION_PROVIDER: str = os.getenv("ESCALATION_PROVIDER", "anthropic").lower()
    ESCALATION_MODEL: str = os.getenv("ESCALATION_MODEL", "claude-3-5-haiku-20241022")

    # API Keys
    # NOTE: never log, print, or otherwise surface these values -- only pass
    # them to the provider clients that need them.
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    PERPLEXITY_API_KEY: str = os.getenv("PERPLEXITY_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    # Real research stack: Tavily (search/discovery) + Firecrawl (extraction).
    # Same env var names already used by the sibling web_research module, so
    # both read identical values from the one .env file -- no duplication.
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    FIRECRAWL_API_KEY: str = os.getenv("FIRECRAWL_API_KEY", "")

    # Configurable Rate Limits (Adjustment #3)
    # Default is 15 RPM for Gemini free-tier
    GEMINI_RPM_LIMIT: int = int(os.getenv("GEMINI_RPM_LIMIT", "15"))
    GEMINI_TPM_LIMIT: int = int(os.getenv("GEMINI_TPM_LIMIT", "1000000"))
    PERPLEXITY_RPM_LIMIT: int = int(os.getenv("PERPLEXITY_RPM_LIMIT", "20"))
    OPENAI_RPM_LIMIT: int = int(os.getenv("OPENAI_RPM_LIMIT", "60"))
    ANTHROPIC_RPM_LIMIT: int = int(os.getenv("ANTHROPIC_RPM_LIMIT", "50"))

    # General provider request interval (seconds) based on RPM
    REQUEST_DELAY_SECONDS: float = max(60.0 / GEMINI_RPM_LIMIT, 0.5)

    # Cost table per 1M tokens (USD). Every entry here MUST be a real,
    # verified rate from the provider's official pricing page -- never a
    # guess. cost_tracker.py's record_usage() (via config.get_pricing())
    # raises PricingNotConfiguredError for any real (non-mock) call whose
    # provider/model isn't listed here, rather than silently recording a
    # fabricated $0.00.
    PRICING = {
        "perplexity": {
            "sonar": {"input": 1.00, "output": 1.00, "request": 0.005},
            "sonar-pro": {"input": 3.00, "output": 15.00, "request": 0.005},
        },
        "gemini": {
            "gemini-2.5-flash-lite": {"input": 0.075, "output": 0.30},
            "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
            # Verified 2026-09-13 against the official Gemini Developer API
            # pricing page (ai.google.dev/gemini-api/docs/pricing), standard
            # (non-batch, non-priority) tier -- this is the tier the
            # pipeline's synchronous generateContent calls actually use.
            # Input covers text/image/video ($0.50/1M for audio input,
            # not used by this pipeline).
            "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
            # Corrected 2026-09-13 (Phase 5B follow-up): the prior $0.02 value
            # was wrong by 10x. Re-verified against the official Gemini
            # Developer API pricing page (ai.google.dev/gemini-api/docs/pricing)
            # -- standard tier, text input, no output-token cost for embeddings.
            "gemini-embedding-2": {"input": 0.20, "output": 0.0},
            "embedding-001": {"input": 0.02, "output": 0.0},
            # MockLLMProvider (providers/mock_provider.py) always reports
            # model="mock-model" regardless of which provider it's standing
            # in for -- explicitly, honestly free (MOCK_MODE never reaches a
            # real API), not a fallback default. See the matching entries
            # under "openai"/"anthropic" below.
            "mock-model": {"input": 0.0, "output": 0.0},
        },
        "openai": {
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-5.4-nano": {"input": 0.10, "output": 0.40},
            "mock-model": {"input": 0.0, "output": 0.0},
        },
        "anthropic": {
            "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
            "mock-model": {"input": 0.0, "output": 0.0},
        },
        "local": {
            "tfidf": {"input": 0.0, "output": 0.0},
        }
    }

    def get_pricing(self, provider: str, model: str) -> dict:
        """Centralized, model-name-based pricing lookup used by
        cost_tracker.py. Deliberately mode-independent (does NOT branch on
        MOCK_MODE): a missing provider/model entry always raises
        PricingNotConfiguredError instead of silently returning $0.00 --
        this is what prevents a real, paid call from being recorded as free
        just because PRICING wasn't updated for a newly-configured model
        (see config.LLM_MODEL / config.EMBEDDING_MODEL / etc.), and it keeps
        this function usable for verifying pricing MATH independent of
        whichever mode the process happens to be running in. Mock/test
        model names (e.g. "mock-model") are priced explicitly at $0.00
        above -- genuinely free, not a fabricated default."""
        provider_key = (provider or "").lower()
        model_key = (model or "").lower().replace("models/", "")
        provider_pricing = self.PRICING.get(provider_key)
        if provider_pricing is None:
            raise PricingNotConfiguredError(
                f"No pricing table configured for provider '{provider}'. "
                f"Add a verified entry to PipelineConfig.PRICING before this provider can be cost-tracked."
            )
        pricing_info = provider_pricing.get(model_key)
        if pricing_info is None:
            raise PricingNotConfiguredError(
                f"No pricing configured for {provider}/{model} (looked up as '{model_key}'). "
                f"Add a verified entry (from the provider's official pricing page) to "
                f"PipelineConfig.PRICING['{provider_key}'] -- a real API call must never be recorded as $0.00."
            )
        return pricing_info

    # Deduplication Thresholds
    SEMANTIC_SIMILARITY_THRESHOLD: float = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.86"))
    NORMALIZED_OVERLAP_THRESHOLD: float = float(os.getenv("NORMALIZED_OVERLAP_THRESHOLD", "0.85"))

    # Generation Batch Size (configurable)
    BATCH_SIZE: int = int(os.getenv("PIPELINE_BATCH_SIZE", "4"))

    # Question Schema Settings
    EXPERIENCE_BANDS = ["0-1", "1-2", "3-5", "5-8", "8+"]
    PARADIGMS = ["EXECUTION", "IMPLEMENTATION", "ARCHITECTURE", "STRATEGY", "DOMAIN_OWNERSHIP"]
    SCOPES = ["UNIVERSAL", "DOMAIN", "COMPANY"]

    # Single source of truth for experience-band -> paradigm mapping, shared by
    # generator_service (what it asks the LLM to generate) and validator_service
    # (what it checks the LLM actually produced). Kept in one place so the two
    # can never silently drift out of sync and cause false paradigm rejections.
    PARADIGM_BY_EXPERIENCE_BAND = {
        "0-1": "EXECUTION",
        "1-2": "IMPLEMENTATION",
        "3-5": "ARCHITECTURE",
        "5-8": "STRATEGY",
        "8+": "DOMAIN_OWNERSHIP"
    }

    # Calibrated difficulty bounds per experience band. Deliberately NOT a
    # single global formula -- these bounds are only used as a sanity envelope
    # in the quality gate; role-specific ceilings/plateaus (e.g. Product
    # Manager topping out lower than Machine Learning Engineer) are judged by
    # the LLM rubric, not hardcoded here.
    EXPERIENCE_DIFFICULTY_BOUNDS = {
        "0-1": (1, 6),
        "1-2": (3, 7),
        "3-5": (5, 9),
        "5-8": (6, 10),
        "8+": (7, 10)
    }

    # Quality gate thresholds (Phase 2 validation ladder)
    VALIDATION_PASS_SCORE: float = float(os.getenv("VALIDATION_PASS_SCORE", "0.70"))
    VALIDATION_UNCERTAIN_LOW: float = float(os.getenv("VALIDATION_UNCERTAIN_LOW", "0.60"))
    VALIDATION_UNCERTAIN_HIGH: float = float(os.getenv("VALIDATION_UNCERTAIN_HIGH", "0.75"))
    # Max allowed gap between claimed overall `difficulty` and the average of
    # the 7 difficulty dimensions before it is treated as an internally
    # inconsistent (and therefore untrustworthy) self-rating.
    DIFFICULTY_DIMENSION_CONSISTENCY_TOLERANCE: float = 4.0

    # Research Cadence (Days)
    ROLE_RESEARCH_CADENCE_DAYS: int = 30
    COMPANY_RESEARCH_CADENCE_DAYS: int = 14

    # 6 Pilot Roles
    PILOT_ROLES = [
        "Machine Learning Engineer",
        "Data Engineer",
        "MLOps Engineer (Cloud Deployment)",
        "Computer Vision Engineer",
        "Product Manager (Tech)",
        "Senior RTL / Logic Design Engineer"
    ]
    PILOT_TARGET_QUESTIONS_PER_ROLE: int = 20

    # Full 59-role production taxonomy. This is the complete set the
    # production rollout orchestrator (question_pipeline/orchestrator.py)
    # can be pointed at -- PILOT_ROLES above remains the original 6-role
    # subset and is unaffected by this list's existence.
    PRODUCTION_ROLES = [
        "Cloud AI Integration Engineer",
        "MLOps Engineer (Cloud Deployment)",
        "Cloud Data Engineer (Mid-level)",
        "Cloud Data & AI Engineer",
        "Verification Engineer (Functional / UVM)",
        "AI / Software Engineer (AI Developer)",
        "ASIC Implementation Engineer",
        "Applied NLP Engineer (LLM Fine-tuning)",
        "Cloud AI Solutions Engineer",
        "AI Consultant",
        "Data Integration Engineer",
        "ETL Developer",
        "FPGA Design Engineer",
        "Conversational AI Engineer",
        "DevOps Engineer – AI/ML Workloads",
        "Technology Consultant / IT Strategy Consultant",
        "Prompt Engineer / Prompt Designer",
        "Generative AI Engineer (Entry)",
        "MLOps / AIOps Associate Engineer",
        "Big Data Engineer (Spark/Hadoop)",
        "Data Engineer",
        "Prompt Engineer",
        "AI Integration Engineer",
        "Physical Design Engineer",
        "Project Manager (Technology)",
        "Product Manager (Tech)",
        "Data Pipeline Engineer",
        "Data Platform Engineer",
        "AIOps Engineer",
        "Senior Verification Engineer",
        "Digital Transformation Analyst",
        "MLOps / AI Ops Engineer",
        "Product Analyst",
        "Computer Vision Engineer",
        "Junior Data Engineer",
        "SQL Developer / Data Analyst (ETL-heavy)",
        "Mixed-Signal Design Engineer",
        "Business Analyst (Tech)",
        "Big Data Developer",
        "AI Product Engineer",
        "NLP Engineer",
        "Senior RTL / Logic Design Engineer",
        "AI Chatbot Developer",
        "AI Platform Support Engineer",
        "MLOps / LLMOps Engineer",
        "SoC Integration Engineer",
        "GenAI Solutions Engineer",
        "LLM Application Developer",
        "Cloud Data Engineer (Junior)",
        "Technical Program Coordinator",
        "Technology Operations Manager",
        "Generative AI Engineer",
        "VLSI Design Engineer (RTL / Front-End)",
        "Technology Consultant (Associate)",
        "Solutions Consultant",
        "Machine Learning Engineer",
        "Data Scientist",
        "Cloud AI Engineer (Entry)",
        "DFT (Design for Test) Engineer",
    ]

    # The 24 companies used as contextual metadata (question_companies),
    # never as a primary identity and never invented beyond evidence found
    # during real research -- see classify_scope() in pipeline_runner.py.
    COMPANIES = [
        "Oracle", "Cognizant", "Adobe", "Flipkart", "Accenture", "PwC", "Tesla",
        "Alibaba Cloud", "eBay", "EY", "Spotify", "Apple", "Google", "Deloitte",
        "Tech Mahindra", "Nvidia", "Microsoft", "Netflix", "Amazon", "Walmart",
        "TCS", "OpenAI", "Meta", "Broadcom",
    ]

    # When a knowledge refresh detects a MEANINGFUL change for a role that
    # already has an established bank, top up with a small batch rather than
    # regenerating the full pilot-sized bank from scratch.
    REFRESH_TOP_UP_QUESTIONS_PER_ROLE: int = int(os.getenv("REFRESH_TOP_UP_QUESTIONS_PER_ROLE", "8"))

    # Minimum cosine similarity for a company knowledge chunk to be
    # considered relevant enough to surface during question generation
    # (rag_service.retrieve_company_context). This is the mechanism that
    # keeps company context evidence-driven instead of injected: a company
    # whose research doesn't actually relate to the role/band/paradigm being
    # generated simply won't clear this bar. Conservative default; tune via
    # env var once real embedding similarity distributions are observed at
    # scale.
    COMPANY_CONTEXT_MIN_SIMILARITY: float = float(os.getenv("COMPANY_CONTEXT_MIN_SIMILARITY", "0.35"))

config = PipelineConfig()
# Ensure directories exist
config.DATA_DIR.mkdir(parents=True, exist_ok=True)
config.KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
