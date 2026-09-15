from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

class QuestionObject(BaseModel):
    """Structured question object matching user specifications and project conventions."""
    question: str
    role: str
    experience_band: str = Field(..., description="'0-1', '1-2', '3-5', '5-8', '8+'")
    difficulty: int = Field(..., ge=1, le=10)

    # 7 Complexity / Depth Dimensions
    technical_depth: int = Field(..., ge=1, le=10)
    problem_complexity: int = Field(..., ge=1, le=10)
    architecture_complexity: int = Field(..., ge=1, le=10)
    troubleshooting: int = Field(..., ge=1, le=10)
    business_complexity: int = Field(..., ge=1, le=10)
    decision_making: int = Field(..., ge=1, le=10)
    leadership_ownership: int = Field(..., ge=1, le=10)

    question_type: str = Field(default="technical")
    paradigm: str = Field(..., description="EXECUTION, IMPLEMENTATION, ARCHITECTURE, STRATEGY, DOMAIN_OWNERSHIP")

    mandatory_skills: List[str] = Field(default_factory=list)
    scope: str = Field(..., description="UNIVERSAL, DOMAIN, COMPANY")
    domains: List[str] = Field(default_factory=list)
    applicable_companies: List[str] = Field(default_factory=list)

    source_references: List[str] = Field(default_factory=list)
    knowledge_version: str = Field(default="v1.0")
    generated_at: str = Field(default_factory=utc_now_iso)

class ResearchOutput(BaseModel):
    """Web research result from Perplexity or configured provider."""
    role: str
    company: Optional[str] = None
    technologies: List[str] = Field(default_factory=list)
    engineering_practices: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)
    interview_topics: List[str] = Field(default_factory=list)
    trends: List[str] = Field(default_factory=list)
    raw_summary: str = ""
    source_references: List[str] = Field(default_factory=list)
    source_hash: str = ""
    knowledge_version: str = "v1.0"
    actual_provider: str = Field(..., description="Actual provider used (Adjustment #2)")
    researched_at: str = Field(default_factory=utc_now_iso)

class KnowledgeChunk(BaseModel):
    """Chunk of synthesized knowledge indexed in the vector store."""
    chunk_id: str
    role: str
    company: Optional[str] = None
    topic: str
    text: str
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    source_reference: Optional[str] = None
    knowledge_version: str = "v1.0"
    created_at: str = Field(default_factory=utc_now_iso)

class VectorStoreData(BaseModel):
    """Container for vector store embeddings versioned by embedding model and provider."""
    embedding_provider: str
    embedding_model: str
    chunks: List[KnowledgeChunk] = Field(default_factory=list)
    updated_at: str = Field(default_factory=utc_now_iso)

class ValidationResult(BaseModel):
    """Output from multi-tier validation ladder."""
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    tier_used: str = Field(..., description="primary_gemini, fallback_gpt, escalation_claude")
    reasoning: str = ""
    uncertain: bool = False
    timestamp: str = Field(default_factory=utc_now_iso)

class QualityGateResult(BaseModel):
    """
    Structured, auditable output of the Phase 2 quality gate -- NOT just a
    single score. This is the in-process representation; `to_attempt_record()`
    converts it into the exact row shape that migrations/0001's
    `question_validation_attempts` table expects.

    FAIL-CLOSED CONTRACT: `approved` must NEVER be True when `critical_failure`
    is True. This is enforced in code here (ValidatorService always computes
    `approved = (not critical_failure) and ...`) AND, once migration 0001 is
    applied, redundantly enforced by the database via chk_qva_fail_closed.
    """
    approved: bool
    quality_score: float = Field(ge=0.0, le=1.0)
    critical_failure: bool = False
    rejection_reasons: List[str] = Field(default_factory=list)
    check_results: Dict[str, bool] = Field(default_factory=dict)
    validation_tier: str = Field(..., description="primary_gemini, fallback_gpt, escalation_claude")
    validation_provider: str
    validation_model: str
    reasoning: str = ""
    uncertain: bool = False
    validated_at: str = Field(default_factory=utc_now_iso)

    def to_attempt_record(
        self,
        role: str,
        candidate_question_text: str,
        question_id: Optional[int] = None,
    ) -> "ValidationAttemptRecord":
        return ValidationAttemptRecord(
            question_id=question_id,
            role=role,
            candidate_question_text=candidate_question_text,
            approved=self.approved,
            quality_score=self.quality_score,
            critical_failure=self.critical_failure,
            rejection_reasons=self.rejection_reasons,
            validation_tier=self.validation_tier,
            validation_provider=self.validation_provider,
            validation_model=self.validation_model,
            validated_at=self.validated_at,
        )

class ValidationAttemptRecord(BaseModel):
    """
    Exact mirror of a `question_validation_attempts` row (see
    migrations/0001_question_bank_and_knowledge_layer.sql). One instance per
    validation attempt, approved or rejected. `id` is None until a store
    assigns one (PostgresValidationAttemptStore on INSERT ... RETURNING id;
    InMemoryValidationAttemptStore on append).
    """
    id: Optional[int] = None
    question_id: Optional[int] = None
    role: str
    candidate_question_text: str
    approved: bool
    quality_score: float = Field(ge=0.0, le=1.0)
    critical_failure: bool = False
    rejection_reasons: List[str] = Field(default_factory=list)
    validation_tier: str
    validation_provider: str
    validation_model: str
    validated_at: str = Field(default_factory=utc_now_iso)

class CostRecord(BaseModel):
    """Individual API cost and token tracking record."""
    provider: str
    model: str
    task: str
    input_tokens: int
    output_tokens: int
    estimated_cost: float
    timestamp: str = Field(default_factory=utc_now_iso)
    status: str = "SUCCESS"

class RolePipelineState(BaseModel):
    """Resumable state machine record for a specific role or company."""
    role: str
    status: str = Field(default="PENDING", description="PENDING, RUNNING, COMPLETED, WAITING_FOR_QUOTA, BLOCKED, FAILED, RETRY, SKIPPED")
    current_batch: int = 0
    total_batches: int = 0
    target_questions: int = 20
    accepted_questions_count: int = 0
    last_researched_at: Optional[str] = None
    next_research_at: Optional[str] = None
    knowledge_version: Optional[str] = None
    source_hash: Optional[str] = None
    reason: Optional[str] = None
    updated_at: str = Field(default_factory=utc_now_iso)

class PilotQualityReport(BaseModel):
    """Comprehensive pilot audit report covering all requested metrics."""
    total_generated: int
    accepted: int
    rejected: int
    duplicates: int
    validation_failures: int
    scope_distribution: Dict[str, int]
    difficulty_distribution: Dict[str, int]
    experience_distribution: Dict[str, int]
    role_correctness: Dict[str, bool]
    api_usage: Dict[str, Any]
    estimated_api_cost: float
    cost_breakdown_by_provider: Dict[str, float]
    cost_breakdown_by_task: Dict[str, float]
    failures_or_errors: List[str]
    generated_at: str = Field(default_factory=utc_now_iso)
