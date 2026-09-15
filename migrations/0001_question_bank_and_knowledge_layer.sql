-- ============================================================================
-- Migration 0001: Reusable Question Bank + Knowledge Layer  (REVISION 3)
-- ============================================================================
-- Purpose:
--   Adds the two new layers described in the question-bank architecture
--   proposal: (1) a Knowledge layer that formalizes research provenance and
--   embeddings, and (2) a reusable Question Bank layer where ONE question
--   row can be associated with many companies/domains/skills instead of
--   being physically duplicated per company.
--
-- Scope of this migration: PURELY ADDITIVE.
--   - Zero ALTER/DROP statements against any existing table.
--   - interview_questions, pre_generated_questions, interview_data,
--     session_metadata, students, student_resumes, job_descriptions,
--     programs, program_job_roles, university_batch_program,
--     session_ratings, admin_users, password_reset_tokens: UNTOUCHED.
--   - Every CREATE is IF NOT EXISTS, so this file is safe to re-run.
--   - Wrapped in a single transaction: it applies fully or not at all.
--
-- Master-data check (requirement 1 of revision 2):
--   Inspected all 13 existing tables across all 3 schemas in this database
--   (public, information_schema, pg_catalog). There is NO canonical
--   `companies`, `domains`, or interview-taxonomy `roles` table anywhere.
--   `programs.job_role` / `program_job_roles.job_role` are free text and
--   serve a different purpose (which role a student's enrollment program
--   targets), not a master list for the question bank's role taxonomy.
--   Conclusion: company_name / domain_name / role stay as free TEXT below.
--   No duplicate master tables were created. If a real companies/domains
--   master table is introduced later, question_companies.company_name /
--   question_domains.domain_name can be swapped for FKs without touching
--   `questions` itself.
--
-- Embedding dimension:
--   Verified live against the configured Gemini embedding model
--   (models/gemini-embedding-2) on 2026-09-11: output is 3072-dim float.
--   pgvector's indexable `vector` type caps out at 2000 dimensions, so
--   embedding columns use `halfvec(3072)` (half-precision, pgvector >=0.7,
--   confirmed available: this DB has pgvector 0.8.1) which supports HNSW
--   indexing up to 4000 dimensions. If a future embedding model is
--   truncated (e.g. via output_dimensionality=768) or swapped, add a new
--   column/table version rather than resizing this one in place, since
--   changing a vector column's dimension invalidates existing rows.
--
-- Changes since revision 1 (see inline comments at each site for detail):
--   1. No master-table changes needed (see note above).
--   2. questions uniqueness left as UNIQUE(role, question_text_normalized);
--      documented (not enforced) that experience/seniority variants are
--      expected to be separate rows because their question_text differs.
--   3. knowledge_versions SPLIT into knowledge_state (current, mutable
--      pointer) + knowledge_version_history (append-only ledger), so a
--      refresh updates the pointer without destroying prior versions.
--   4. question_sources: added partial unique indexes so the same question
--      cannot be linked twice to the same chunk/document/raw reference.
--   5. research_sources: split single `provider` into discovery_provider
--      + extraction_provider. research_documents.provider renamed to
--      research_provider. research_chunks gained embedding_provider and
--      synthesis_provider so discovery/extraction/research/synthesis are
--      four distinguishable facts instead of one collapsed field.
--   6. Added missing NOT NULL (research_chunks.research_document_id),
--      a self-reference guard on questions.superseded_by_question_id, and
--      embedding/embedding_model/embedding_provider consistency CHECKs.
--
-- Changes since revision 2 (quality-gate lock-in, "QUALITY MUST NEVER BE
-- COMPROMISED"):
--   7. Added `question_validation_attempts`: an append-only audit log of
--      EVERY quality-validation attempt, approved and rejected alike, with
--      structured fields (approved, quality_score, critical_failure,
--      rejection_reasons, validation_tier/provider/model, validated_at) --
--      this is the "smallest additive change" for the structured validation
--      output requirement. This is the 13th new table.
--   8. FAIL-CLOSED IS ENFORCED AT THE DATABASE LEVEL, not just in
--      application logic: chk_qva_fail_closed makes it IMPOSSIBLE to insert
--      a row with critical_failure = TRUE and approved = TRUE at the same
--      time -- a high quality_score literally cannot override a critical
--      failure, because the database itself will reject that row.
--      chk_qva_approved_has_question makes it impossible to mark a row
--      approved without it pointing at a real `questions.id` -- i.e. an
--      "approved" record that never actually produced a question is invalid
--      data, not just a logic bug waiting to happen.
--   9. `questions` itself is unchanged by this revision: it was already
--      designed to hold only what the pipeline chooses to insert, and the
--      pipeline only inserts after validation passes (see pipeline_runner.py
--      today: validate -> dedup -> append -> save, never insert-then-check).
--      question_validation_attempts is what makes that guarantee auditable
--      after the fact, for both the winners and the rejects.
--   10. No data migration of any kind is included in this file. Legacy
--      `interview_questions` rows are NOT copied, scored, or referenced.
--      A future legacy-question migration would run those rows through this
--      same validation ladder and this same question_validation_attempts
--      table -- it is not a separate quality standard.
-- ============================================================================

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- LAYER 1: KNOWLEDGE
-- (formalizes what question_pipeline/data/*.json currently holds as flat files)
-- ============================================================================

-- CHANGE (req. 5): a single `provider` column can't say "Tavily found this
-- URL, Firecrawl extracted it" -- it can only say one thing. Split into two
-- independent, nullable columns so each provider's actual responsibility is
-- recorded. For a single-shot provider (e.g. Perplexity, which discovers
-- its own sources internally), set discovery_provider and leave
-- extraction_provider NULL -- we genuinely don't control/observe a separate
-- extraction step there, so NULL is the honest value, not a guess.
CREATE TABLE IF NOT EXISTS research_sources (
    id                   SERIAL PRIMARY KEY,
    url                  TEXT NOT NULL,
    title                TEXT,
    discovery_provider   TEXT,               -- e.g. 'tavily', 'perplexity' -- who found/ranked this URL
    extraction_provider  TEXT,               -- e.g. 'firecrawl' -- who pulled clean content from it (NULL if only a search snippet was used)
    content_hash         TEXT,
    retrieved_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMENT ON TABLE research_sources IS 'One row per web source (URL) consumed by the research pipeline. discovery_provider and extraction_provider are tracked separately since they are frequently different services.';

CREATE INDEX IF NOT EXISTS idx_research_sources_url ON research_sources (url);

-- CHANGE (req. 5): `provider` renamed to `research_provider` to make clear
-- this is specifically the research/retrieval stage (e.g. 'perplexity', or
-- 'tavily_firecrawl' when built from the multi-source web_research path),
-- NOT the synthesis stage -- synthesis provider lives on research_chunks,
-- since that is where synthesis (Gemini) actually produces chunk text.
CREATE TABLE IF NOT EXISTS research_documents (
    id                      SERIAL PRIMARY KEY,
    role                    TEXT NOT NULL,
    company                 TEXT,                 -- NULL for role-level research
    research_provider       TEXT NOT NULL,        -- e.g. 'perplexity' (ResearchOutput.actual_provider)
    knowledge_version       TEXT NOT NULL DEFAULT 'v1.0',
    source_hash             TEXT,
    raw_summary             TEXT,
    technologies            JSONB DEFAULT '[]'::jsonb,
    engineering_practices   JSONB DEFAULT '[]'::jsonb,
    responsibilities        JSONB DEFAULT '[]'::jsonb,
    interview_topics        JSONB DEFAULT '[]'::jsonb,
    trends                  JSONB DEFAULT '[]'::jsonb,
    researched_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMENT ON TABLE research_documents IS 'One row per synthesized research run for a role (and optional company/domain) -- maps to ResearchOutput. Insert-only by convention: a refresh creates a NEW row rather than overwriting, so history is never lost.';

CREATE INDEX IF NOT EXISTS idx_research_documents_role ON research_documents (role);
CREATE INDEX IF NOT EXISTS idx_research_documents_role_company ON research_documents (role, company);

CREATE TABLE IF NOT EXISTS research_document_sources (
    research_document_id   INTEGER NOT NULL REFERENCES research_documents(id) ON DELETE CASCADE,
    research_source_id     INTEGER NOT NULL REFERENCES research_sources(id) ON DELETE CASCADE,
    PRIMARY KEY (research_document_id, research_source_id)
);
COMMENT ON TABLE research_document_sources IS 'Many-to-many: which sources fed which research document.';

-- CHANGE (req. 5, 6): research_document_id is now NOT NULL -- every chunk
-- the pipeline produces comes from synthesize(research: ResearchOutput), so
-- an orphan chunk would indicate a bug, not a valid state. Added
-- embedding_provider alongside embedding_model (mirrors rag_service.py's
-- own VectorStoreData, which invalidates its store when EITHER changes) and
-- synthesis_provider (e.g. 'gemini') since synthesis happens at chunk
-- creation, not at the document level. Added a CHECK so embedding /
-- embedding_model / embedding_provider are always filled in together or not
-- at all (no half-written embedding state).
CREATE TABLE IF NOT EXISTS research_chunks (
    id                    SERIAL PRIMARY KEY,
    chunk_id              TEXT UNIQUE NOT NULL,     -- natural key from the pipeline (KnowledgeChunk.chunk_id)
    research_document_id  INTEGER NOT NULL REFERENCES research_documents(id) ON DELETE CASCADE,
    role                  TEXT NOT NULL,
    company               TEXT,
    topic                 TEXT,
    text                  TEXT NOT NULL,
    synthesis_provider    TEXT,                     -- e.g. 'gemini' -- who turned the research document into this chunk's text
    embedding_provider    TEXT,
    embedding_model       TEXT,
    embedding             halfvec(3072),
    knowledge_version     TEXT NOT NULL DEFAULT 'v1.0',
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT chk_research_chunks_embedding_complete CHECK (
        (embedding IS NULL AND embedding_model IS NULL AND embedding_provider IS NULL)
        OR
        (embedding IS NOT NULL AND embedding_model IS NOT NULL AND embedding_provider IS NOT NULL)
    )
);
COMMENT ON TABLE research_chunks IS 'Embedded knowledge chunks for RAG retrieval -- replaces question_pipeline/data/vector_store.json.';

CREATE INDEX IF NOT EXISTS idx_research_chunks_role ON research_chunks (role);
CREATE INDEX IF NOT EXISTS idx_research_chunks_embedding_hnsw
    ON research_chunks USING hnsw (embedding halfvec_cosine_ops);

-- CHANGE (req. 3): the original single `knowledge_versions` table (one row
-- per role+company, updated in place on every refresh) would silently lose
-- the previous status/reason/source_hash on each refresh. Split into:
--   - knowledge_state: current pointer, one row per (role, company),
--     UPDATED in place -- this is what the pipeline checks before deciding
--     whether to skip/refresh (mirrors state_manager.py's RolePipelineState).
--   - knowledge_version_history: append-only ledger, a NEW row every time
--     a role/company's knowledge_version or status changes. Never updated
--     or deleted, so "what did we know before this refresh" is always
--     answerable.
CREATE TABLE IF NOT EXISTS knowledge_state (
    id                          SERIAL PRIMARY KEY,
    role                        TEXT NOT NULL,
    company                     TEXT,                     -- NULL = role-level cadence, set = company/domain cadence
    current_knowledge_version   TEXT,
    current_research_document_id INTEGER REFERENCES research_documents(id) ON DELETE SET NULL,
    source_hash                 TEXT,
    status                      TEXT NOT NULL DEFAULT 'PENDING'
                                CHECK (status IN ('PENDING','RUNNING','COMPLETED','WAITING_FOR_QUOTA','BLOCKED','FAILED','SKIPPED')),
    reason                      TEXT,
    last_researched_at          TIMESTAMPTZ,
    next_research_at            TIMESTAMPTZ,
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMENT ON TABLE knowledge_state IS 'CURRENT research cadence/status per (role, company) -- replaces question_pipeline/data/state.json. Mutable: updated in place on each refresh.';

CREATE UNIQUE INDEX IF NOT EXISTS uq_knowledge_state_role_company
    ON knowledge_state (role, COALESCE(company, ''));

CREATE TABLE IF NOT EXISTS knowledge_version_history (
    id                      SERIAL PRIMARY KEY,
    role                    TEXT NOT NULL,
    company                 TEXT,
    knowledge_version       TEXT,
    research_document_id    INTEGER REFERENCES research_documents(id) ON DELETE SET NULL,
    source_hash             TEXT,
    status                  TEXT NOT NULL
                            CHECK (status IN ('PENDING','RUNNING','COMPLETED','WAITING_FOR_QUOTA','BLOCKED','FAILED','SKIPPED')),
    reason                  TEXT,
    recorded_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMENT ON TABLE knowledge_version_history IS 'APPEND-ONLY ledger of every knowledge_state transition ever recorded for a (role, company). Never updated or deleted -- this is how historical research versions survive a refresh.';

CREATE INDEX IF NOT EXISTS idx_knowledge_version_history_role_company
    ON knowledge_version_history (role, company, recorded_at DESC);

-- ============================================================================
-- LAYER 2: REUSABLE QUESTION BANK
-- ============================================================================

-- CHANGE (req. 2): uniqueness intentionally stays UNIQUE(role, question_text_normalized).
-- Rationale documented via COMMENT below rather than changed: a "0-1 years"
-- and an "8+ years" question for the same role are expected to have
-- MATERIALLY DIFFERENT question_text (different depth/expected answer), so
-- they naturally get different normalized text and different rows without
-- needing experience_band in the key. Deliberately NOT adding
-- experience_band to the unique key: if the generator ever produces
-- byte-for-byte identical question_text for two different experience bands,
-- that is a genuine generation defect (the question isn't actually
-- differentiated by depth), and the constraint is supposed to catch and
-- reject that, not paper over it by allowing two "identical" rows to coexist.
CREATE TABLE IF NOT EXISTS questions (
    id                          SERIAL PRIMARY KEY,
    role                        TEXT NOT NULL,
    question_text               TEXT NOT NULL,
    question_text_normalized    TEXT GENERATED ALWAYS AS (
                                    lower(regexp_replace(trim(question_text), '\s+', ' ', 'g'))
                                 ) STORED,

    question_type               TEXT NOT NULL DEFAULT 'technical',
    experience_band              TEXT NOT NULL CHECK (experience_band IN ('0-1','1-2','3-5','5-8','8+')),

    difficulty                  SMALLINT NOT NULL CHECK (difficulty BETWEEN 1 AND 10),
    technical_depth              SMALLINT NOT NULL CHECK (technical_depth BETWEEN 1 AND 10),
    problem_complexity           SMALLINT NOT NULL CHECK (problem_complexity BETWEEN 1 AND 10),
    architecture_complexity      SMALLINT NOT NULL CHECK (architecture_complexity BETWEEN 1 AND 10),
    troubleshooting              SMALLINT NOT NULL CHECK (troubleshooting BETWEEN 1 AND 10),
    business_complexity          SMALLINT NOT NULL CHECK (business_complexity BETWEEN 1 AND 10),
    decision_making              SMALLINT NOT NULL CHECK (decision_making BETWEEN 1 AND 10),
    leadership_ownership         SMALLINT NOT NULL CHECK (leadership_ownership BETWEEN 1 AND 10),

    paradigm                    TEXT NOT NULL CHECK (paradigm IN ('EXECUTION','IMPLEMENTATION','ARCHITECTURE','STRATEGY','DOMAIN_OWNERSHIP')),
    scope                       TEXT NOT NULL CHECK (scope IN ('UNIVERSAL','DOMAIN','COMPANY')),

    knowledge_version            TEXT NOT NULL DEFAULT 'v1.0',
    is_active                   BOOLEAN NOT NULL DEFAULT TRUE,
    superseded_by_question_id    INTEGER REFERENCES questions(id) ON DELETE SET NULL,

    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT uq_questions_role_text UNIQUE (role, question_text_normalized),
    -- CHANGE (req. 6): a question can never supersede itself.
    CONSTRAINT chk_questions_no_self_supersede CHECK (superseded_by_question_id IS DISTINCT FROM id)
);
COMMENT ON TABLE questions IS 'Canonical reusable interview questions. ONE row per question regardless of how many companies/domains it applies to. Uniqueness is (role, normalized question_text) ONLY -- experience/seniority variants get separate rows naturally because their text differs in depth; experience_band is deliberately excluded from the unique key (see migration header).';

CREATE INDEX IF NOT EXISTS idx_questions_role ON questions (role);
CREATE INDEX IF NOT EXISTS idx_questions_role_scope ON questions (role, scope);
CREATE INDEX IF NOT EXISTS idx_questions_active ON questions (is_active);
CREATE INDEX IF NOT EXISTS idx_questions_superseded_by ON questions (superseded_by_question_id);

CREATE TABLE IF NOT EXISTS question_companies (
    question_id     INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    company_name    TEXT NOT NULL,
    PRIMARY KEY (question_id, company_name)
);
COMMENT ON TABLE question_companies IS 'Many-to-many: which companies a question applies to. Does NOT duplicate the question row. company_name is free text (no canonical companies table exists in this database today -- see migration header).';
CREATE INDEX IF NOT EXISTS idx_question_companies_company ON question_companies (company_name);

CREATE TABLE IF NOT EXISTS question_domains (
    question_id     INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    domain_name     TEXT NOT NULL,
    PRIMARY KEY (question_id, domain_name)
);
COMMENT ON TABLE question_domains IS 'Many-to-many: business/technology domains a question applies to (e.g. E-commerce, FinTech). domain_name is free text (no canonical domains table exists today).';
CREATE INDEX IF NOT EXISTS idx_question_domains_domain ON question_domains (domain_name);

CREATE TABLE IF NOT EXISTS question_skills (
    question_id     INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    skill_name      TEXT NOT NULL,
    PRIMARY KEY (question_id, skill_name)
);
COMMENT ON TABLE question_skills IS 'Many-to-many: mandatory/associated skills for a question.';
CREATE INDEX IF NOT EXISTS idx_question_skills_skill ON question_skills (skill_name);

-- CHANGE (req. 4): added partial unique indexes so a question cannot receive
-- two provenance rows pointing at the same chunk, the same document, or the
-- same raw citation string. Partial (WHERE ... IS NOT NULL) because a given
-- row only ever populates ONE of the three provenance pathways, and plain
-- UNIQUE would not de-duplicate NULLs (Postgres treats each NULL as
-- distinct). Also added a CHECK so a provenance row can't be created empty.
CREATE TABLE IF NOT EXISTS question_sources (
    id                      SERIAL PRIMARY KEY,
    question_id             INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    research_chunk_id       INTEGER REFERENCES research_chunks(id) ON DELETE SET NULL,
    research_document_id    INTEGER REFERENCES research_documents(id) ON DELETE SET NULL,
    source_reference        TEXT,                 -- raw citation/URL string when no chunk/document linkage exists yet
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT chk_question_sources_not_empty CHECK (
        research_chunk_id IS NOT NULL OR research_document_id IS NOT NULL OR source_reference IS NOT NULL
    )
);
COMMENT ON TABLE question_sources IS 'Provenance: which research chunk/document/raw source justifies a generated question.';
CREATE INDEX IF NOT EXISTS idx_question_sources_question ON question_sources (question_id);

CREATE UNIQUE INDEX IF NOT EXISTS uq_question_sources_chunk
    ON question_sources (question_id, research_chunk_id) WHERE research_chunk_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_question_sources_document
    ON question_sources (question_id, research_document_id) WHERE research_document_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_question_sources_reference
    ON question_sources (question_id, source_reference)
    WHERE source_reference IS NOT NULL AND research_chunk_id IS NULL AND research_document_id IS NULL;

-- CHANGE (req. 5): added embedding_provider alongside embedding_model, same
-- rationale as research_chunks -- a model name alone is ambiguous about
-- which provider produced it.
CREATE TABLE IF NOT EXISTS question_embeddings (
    question_id      INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    embedding_provider TEXT NOT NULL,
    embedding_model  TEXT NOT NULL,
    embedding        halfvec(3072) NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (question_id, embedding_model)
);
COMMENT ON TABLE question_embeddings IS 'Semantic-dedup embeddings per question, keyed by embedding model so re-embedding with a new model does not destroy history.';

CREATE INDEX IF NOT EXISTS idx_question_embeddings_hnsw
    ON question_embeddings USING hnsw (embedding halfvec_cosine_ops);

-- NEW (revision 3, req. 8): structured, auditable output of the quality
-- validation ladder (Gemini primary -> OpenAI fallback -> Claude escalation).
-- One row per validation ATTEMPT, not per question -- a rejected candidate
-- still gets logged here (with question_id NULL, since it never becomes a
-- questions row) so rejection patterns are visible later, and an approved
-- candidate's row is the permanent record of which tier approved it, at
-- what score, and when.
--
-- The two CHECK constraints below are the literal database-level
-- enforcement of "a high overall score must NOT override a critical
-- failure": it is not possible to store a row where critical_failure and
-- approved are both TRUE, and not possible to store an approved row that
-- doesn't point at a real question.
CREATE TABLE IF NOT EXISTS question_validation_attempts (
    id                       SERIAL PRIMARY KEY,
    question_id              INTEGER REFERENCES questions(id) ON DELETE SET NULL,
    role                     TEXT NOT NULL,
    candidate_question_text  TEXT NOT NULL,        -- kept even on rejection, for audit -- this text is NOT the same row as questions.question_text
    approved                 BOOLEAN NOT NULL,
    quality_score            NUMERIC(4,3) CHECK (quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 1)),
    critical_failure         BOOLEAN NOT NULL DEFAULT FALSE,
    rejection_reasons        JSONB NOT NULL DEFAULT '[]'::jsonb,   -- structured list of reason codes/strings, e.g. ["wrong_role","hallucinated_technology"]
    validation_tier          TEXT NOT NULL CHECK (validation_tier IN ('primary_gemini','fallback_gpt','escalation_claude')),
    validation_provider      TEXT NOT NULL,
    validation_model         TEXT NOT NULL,
    validated_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at               TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT chk_qva_fail_closed CHECK (NOT (critical_failure = TRUE AND approved = TRUE)),
    CONSTRAINT chk_qva_approved_has_question CHECK (NOT (approved = TRUE AND question_id IS NULL))
);
COMMENT ON TABLE question_validation_attempts IS 'Append-only audit log of every quality-validation attempt (approved AND rejected). Enforces fail-closed at the data level: a critical_failure row can never be approved=TRUE, and an approved=TRUE row must reference a real questions.id.';

CREATE INDEX IF NOT EXISTS idx_qva_question_id ON question_validation_attempts (question_id);
CREATE INDEX IF NOT EXISTS idx_qva_role ON question_validation_attempts (role);
CREATE INDEX IF NOT EXISTS idx_qva_approved ON question_validation_attempts (approved);
CREATE INDEX IF NOT EXISTS idx_qva_critical_failure ON question_validation_attempts (critical_failure) WHERE critical_failure = TRUE;

COMMIT;
