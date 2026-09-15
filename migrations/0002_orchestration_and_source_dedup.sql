-- ============================================================================
-- Migration 0002: Source URL idempotency + rollout orchestration job table
-- ============================================================================
-- Purpose:
--   1. Fixes the production concern from the first real DB-backed run:
--      research_sources had no URL-level dedup, so repeated loader/research
--      runs could accumulate duplicate rows for the same source.
--   2. Adds `pipeline_jobs`, the minimal dedicated table the 59-role/24-
--      company production orchestrator (question_pipeline/orchestrator.py)
--      uses to track per-(role, company, job_type) checkpoint/resume state,
--      per the instruction to add a new table rather than abuse
--      knowledge_state (which represents research freshness, not job
--      execution progress -- a different concern).
--
-- Scope: PURELY ADDITIVE, and only ever touches tables created by migration
-- 0001 (research_sources) or entirely new tables. Zero statements reference
-- any of the 13 original legacy application tables.
-- ============================================================================

BEGIN;

-- ----------------------------------------------------------------------------
-- 1. research_sources: normalized-URL idempotency
-- ----------------------------------------------------------------------------
-- Two research runs (or two loader runs against the same local file) can
-- observe the exact same source with trivial formatting differences
-- (trailing slash, http vs https, a #fragment). `url_normalized` collapses
-- those before comparison; the unique index makes re-inserting an
-- already-known source a safe no-op via ON CONFLICT DO NOTHING in the
-- loader, rather than accumulating duplicate rows. Deliberately NOT
-- stripping query strings -- some real URLs are only distinguished by them,
-- and collapsing those could wrongly merge two different pages.
ALTER TABLE research_sources
    ADD COLUMN IF NOT EXISTS url_normalized TEXT
    GENERATED ALWAYS AS (
        lower(
            regexp_replace(
                regexp_replace(
                    regexp_replace(trim(url), '#.*$', ''),
                    '/+$', ''
                ),
                '^https?://', ''
            )
        )
    ) STORED;

-- Data cleanup: runs before this fix existed (the real ML Engineer pilots)
-- already inserted duplicate research_sources rows for the same normalized
-- URL, since nothing prevented it at the time. The unique index below can't
-- be created until those are collapsed. For each duplicate group, keep the
-- earliest row (lowest id) as canonical, repoint any
-- research_document_sources links from the duplicates to it (dropping a
-- repoint only when the canonical row already has that exact link, which
-- would otherwise violate research_document_sources' own primary key), then
-- remove the now-redundant duplicate source rows. No provenance is lost --
-- every research_document that referenced a duplicate still references the
-- (now-canonical) source afterward.
DO $$
DECLARE
    dup RECORD;
BEGIN
    FOR dup IN
        SELECT url_normalized, MIN(id) AS keep_id, array_agg(id) AS all_ids
        FROM research_sources
        GROUP BY url_normalized
        HAVING COUNT(*) > 1
    LOOP
        UPDATE research_document_sources rds
        SET research_source_id = dup.keep_id
        WHERE rds.research_source_id = ANY(dup.all_ids)
          AND rds.research_source_id <> dup.keep_id
          AND NOT EXISTS (
              SELECT 1 FROM research_document_sources rds2
              WHERE rds2.research_document_id = rds.research_document_id
                AND rds2.research_source_id = dup.keep_id
          );

        DELETE FROM research_document_sources
        WHERE research_source_id = ANY(dup.all_ids)
          AND research_source_id <> dup.keep_id;

        DELETE FROM research_sources
        WHERE id = ANY(dup.all_ids) AND id <> dup.keep_id;
    END LOOP;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS uq_research_sources_url_normalized
    ON research_sources (url_normalized);

COMMENT ON COLUMN research_sources.url_normalized IS 'Generated: lowercased URL with scheme, trailing slash, and #fragment stripped. Query strings are preserved. Used for idempotent source upsert.';

-- ----------------------------------------------------------------------------
-- 2. pipeline_jobs: production rollout checkpoint/resume state
-- ----------------------------------------------------------------------------
-- One row per (role, company, job_type) UNIT, upserted in place as that
-- unit progresses through its lifecycle -- this is a current-state pointer,
-- not a history log (knowledge_version_history and
-- question_validation_attempts already provide history for their own
-- concerns). company = NULL means a role-level job; job_type distinguishes
-- a role-level research refresh, a company-level research refresh, and the
-- question-generation pass for a role.
CREATE TABLE IF NOT EXISTS pipeline_jobs (
    id              SERIAL PRIMARY KEY,
    role            TEXT NOT NULL,
    company         TEXT,
    job_type        TEXT NOT NULL CHECK (job_type IN ('ROLE_RESEARCH', 'COMPANY_RESEARCH', 'ROLE_GENERATION')),
    status          TEXT NOT NULL DEFAULT 'PENDING' CHECK (status IN (
                        'PENDING', 'RESEARCHING', 'RESEARCHED', 'SYNTHESIZING', 'KNOWLEDGE_UPDATED',
                        'GENERATING', 'VALIDATING', 'LOADING', 'COMPLETED', 'SKIPPED_NO_CHANGE',
                        'WAITING_FOR_QUOTA', 'BLOCKED', 'FAILED'
                    )),
    reason          TEXT,
    metrics         JSONB NOT NULL DEFAULT '{}'::jsonb,
    attempt_count   INTEGER NOT NULL DEFAULT 0,
    started_at      TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ
);
COMMENT ON TABLE pipeline_jobs IS 'Checkpoint/resume state for the production rollout orchestrator. One current-state row per (role, company, job_type) unit; safe to upsert repeatedly. A FAILED/WAITING_FOR_QUOTA/BLOCKED row for one unit never affects any other unit''s row.';

CREATE UNIQUE INDEX IF NOT EXISTS uq_pipeline_jobs_unit
    ON pipeline_jobs (role, COALESCE(company, ''), job_type);

CREATE INDEX IF NOT EXISTS idx_pipeline_jobs_status ON pipeline_jobs (status);
CREATE INDEX IF NOT EXISTS idx_pipeline_jobs_role ON pipeline_jobs (role);

COMMIT;
