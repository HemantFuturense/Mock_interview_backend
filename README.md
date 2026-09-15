# Interview Question Pipeline

An automated pipeline that keeps a role- and company-aware interview question bank
continuously fresh. It researches the live web for a given job role (or company),
synthesizes that research into a versioned knowledge base, generates interview
questions grounded in that evidence via RAG, deduplicates and validates each
question through a multi-model fail-closed ladder, and loads the survivors into
PostgreSQL — resuming exactly where it left off if a provider quota runs out
mid-run.

The pipeline is driven one "tick" at a time by a scheduler, so a long rollout
across the full 59-role / 24-company universe is just the same command invoked
repeatedly. Database state alone decides what is due on each tick; nothing is
hardcoded per invocation.

---

## Where the code lives

```
.
├── app/                      # the modular FastAPI app
│   ├── config/               #   settings + constants
│   ├── core/                 #   db pool, cache, logging
│   └── modules/              #   auth / students / sandbox / interview / admin
├── question_pipeline/        # the question pipeline (this document)
│   ├── rollout_cli.py        #   entry point
│   ├── orchestrator.py       #   state machine + batch runners
│   ├── *_service.py          #   research / synthesis / rag / generation / dedup / validation
│   ├── providers/            #   per-vendor adapters behind four ABCs
│   ├── data/                 #   local working state (gitignored, regenerated)
│   ├── docs/                 #   line-by-line implementation walkthrough
│   └── tests/
├── web_research/             # on-demand research router, mounted at /api/research
├── migrations/               # 0001 question bank + knowledge layer, 0002 orchestration
└── main.py                   # entrypoint shim -> app.main
```

`question_pipeline` and `web_research` are **top-level packages**, not members of
`app.modules`. That is not stylistic: `question_pipeline` imports
`web_research.clients` directly (see
`question_pipeline/providers/tavily_firecrawl_provider.py`), so both have to be
importable from the repository root.

Every bare filename in this document (`config.py`, `orchestrator.py`, `data/…`)
is relative to `question_pipeline/`. All commands below are run from the
repository root.

---

## Table of contents

- [The two axes: roles and companies](#the-two-axes-roles-and-companies)
- [End-to-end flow](#end-to-end-flow)
- [Stage 1 — Freshness and cadence check](#stage-1--freshness-and-cadence-check)
- [Stage 2 — Web research](#stage-2--web-research)
- [Stage 3 — Change detection](#stage-3--change-detection)
- [Stage 4 — Synthesis](#stage-4--synthesis)
- [Stage 5 — RAG indexing and retrieval](#stage-5--rag-indexing-and-retrieval)
- [Stage 6 — Question generation](#stage-6--question-generation)
- [Stage 7 — Deduplication](#stage-7--deduplication)
- [Stage 8 — Validation](#stage-8--validation)
- [Stage 9 — Database load](#stage-9--database-load)
- [The state machine and checkpoint recovery](#the-state-machine-and-checkpoint-recovery)
- [Quota, blocking and retry handling](#quota-blocking-and-retry-handling)
- [Cost tracking](#cost-tracking)
- [Provider abstraction](#provider-abstraction)
- [Configuration](#configuration)
- [Running the pipeline](#running-the-pipeline)
- [Database schema](#database-schema)
- [Local data files](#local-data-files)
- [Tests](#tests)
- [File map](#file-map)

---

## The two axes: roles and companies

The pipeline processes two independent kinds of work unit:

| Axis | Unit | Cadence | Produces |
|---|---|---|---|
| **Role** | e.g. `Machine Learning Engineer` | every 30 days | role knowledge + generated questions |
| **Company** | e.g. `Google` | every 14 days | company knowledge chunks used as extra generation context |

Within a single invocation **companies are always processed before roles**. This
ordering is fixed, not configurable: company research feeds the RAG store that
role generation then retrieves from, so running it the other way round would
generate questions against stale company context.

Each unit's progress is tracked as a row in the `pipeline_jobs` table, keyed by
role (and company, where applicable). That row is the pipeline's only source of
truth about what has been done.

---

## End-to-end flow

```
                         ┌──────────────────────────┐
                         │   rollout_cli.py main()  │
                         │  (--once / explicit sel.)│
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │    determine_targets()   │
                         │ (59 roles / 24 companies)│
                         └────────────┬─────────────┘
                                      │
                    ┌─────────────────┴──────────────────┐
                    ▼                                    ▼
        ┌───────────────────────┐            ┌───────────────────────┐
        │  run_company_batch()  │            │      run_batch()      │
        └───────────┬───────────┘            └───────────┬───────────┘
                    ▼                                    ▼
      ┌───────────────────────────┐        ┌──────────────────────────┐
      │ process_company_research()│        │      process_role()      │
      │     A/B/C/D dispatch      │        │     A/B/C/D dispatch     │
      └──────────────┬────────────┘        └──────────────┬───────────┘
                     │                                    │
          ┌──────────┴──────────┐            ┌────────────┴───────────┐
          ▼                     ▼            ▼                        ▼
   [research fresh]     [research stale]  [research fresh]    [research stale]
          │                     │            │                        │
          │                     ▼            │                        ▼
   resume downstream    research_service   resume generation    research_service
   (no new research)    .execute_*()       (no new research)    .execute_research()
          │                     │            │                        │
          │                     ▼            │                        ▼
          │             synthesis_service    │              synthesis_service
          │                     │            │                        │
          │                     ▼            │                        ▼
          │             rag_service          │              rag_service
          │             .index_chunks()      │              .index_chunks()
          │                     │            │                        │
          └──────────┬──────────┘            └───────────┬────────────┘
                     ▼                                   ▼
        ┌────────────────────────────────────────────────────────────────┐
        │                     _generate_and_load()                       │
        │  for each experience band (0-1, 1-2, 3-5, 5-8, 8+):             │
        │    generator_service.generate_batch_for_band()                  │
        │        (role RAG context + company RAG context if relevant)     │
        │    deduplicator_service.check_duplicate()                       │
        │        (exact → Jaccard → cosine)                               │
        │    validator_service.validate_question()                        │
        │        (deterministic → Gemini → OpenAI → Claude, fail-closed)  │
        │  db_loader.load_role()                                          │
        │        → questions / embeddings / knowledge tables              │
        └───────────────────────────┬────────────────────────────────────┘
                                    ▼
                          pipeline_jobs = COMPLETED
```

---

## Stage 1 — Freshness and cadence check

`freshness.py` → `KnowledgeFreshnessTracker`

Before any paid API call, the tracker decides whether this unit even needs
research. It compares the unit's `last_researched_at` in `knowledge_state`
against the cadence for its axis (30 days for roles, 14 for companies).

- **Within cadence** → research is skipped entirely. The orchestrator jumps
  straight to resuming whatever downstream work is still incomplete.
- **Past cadence** → research proceeds to Stage 2.

This is the pipeline's first and cheapest cost control: a role researched last
week costs nothing to re-tick.

---

## Stage 2 — Web research

`research_service.py` → `execute_research()` / `execute_company_research()`
`providers/tavily_firecrawl_provider.py` → `_research()`

Research is a two-step discovery-then-extraction process:

1. **Discovery (Tavily).** A search query built from the role or company name
   returns ranked candidate URLs with snippets.
2. **Extraction (Firecrawl).** The top candidates are scraped to clean Markdown
   for full-text content rather than relying on search snippets.

If Firecrawl extraction fails or returns nothing usable for a URL, the provider
falls back to the Tavily snippet for that source rather than dropping it. The
result is a research dossier: raw text plus the list of sources that produced it,
each recorded with a normalized URL so the same page discovered under different
query strings is not counted twice.

Every source is persisted to `research_sources` / `research_documents` /
`research_document_sources`, which is what makes a generated question traceable
back to the pages it came from.

---

## Stage 3 — Change detection

`freshness.py`

After research returns, the tracker hashes the set of source content. If the hash
is **identical** to the previous cycle's, the web has not meaningfully changed for
this unit, and there is no reason to spend generation and validation money
re-deriving the same questions.

In that case the unit is marked `SKIPPED_NO_CHANGE` — but only if downstream work
was *already* complete from a prior cycle. If research is unchanged yet generation
never finished, the pipeline still resumes generation. (Getting this distinction
wrong was one of the two real production bugs this code has been fixed for; see
[the state machine](#the-state-machine-and-checkpoint-recovery).)

---

## Stage 4 — Synthesis

`synthesis_service.py` → `synthesize()`

The raw dossier is passed to an LLM that condenses it into structured knowledge
chunks — discrete, self-contained statements about what the role or company
actually requires, each carrying its provenance.

On success the unit's `knowledge_version` is bumped and the bump is recorded in
`knowledge_version_history`, so the knowledge that produced any given question
batch can be reconstructed later.

---

## Stage 5 — RAG indexing and retrieval

`rag_service.py` → `RAGService`

**Indexing** (`index_chunks`) embeds each synthesized chunk and writes it to the
vector store, and into `research_chunks` with an HNSW index for fast similarity
search.

**Retrieval** happens at generation time and comes in two flavors:

- `retrieve_context(role, band, paradigm, top_k=3)` — role knowledge for the
  current generation query.
- `retrieve_company_context(role, band, paradigm, companies=None, top_k=1)` —
  company knowledge, gated by `COMPANY_CONTEXT_MIN_SIMILARITY` (0.35).

Company context is **evidence-driven, not injected**. Company chunks are ranked by
the same cosine similarity as role chunks, and only surface if they clear the
relevance threshold. A consulting firm's audit practices will simply score too low
against a Machine Learning Engineer query and never reach the prompt. This is
enforced structurally by the threshold rather than by a hardcoded allow-list of
which companies "go with" which roles.

---

## Stage 6 — Question generation

`generator_service.py` → `generate_batch_for_band()`

Generation runs **per experience band**, and the band determines the cognitive
paradigm the question must test:

| Experience band | Paradigm | Difficulty envelope |
|---|---|---|
| `0-1` | `EXECUTION` | 1–6 |
| `1-2` | `IMPLEMENTATION` | 3–7 |
| `3-5` | `ARCHITECTURE` | 5–9 |
| `5-8` | `STRATEGY` | 6–10 |
| `8+` | `DOMAIN_OWNERSHIP` | 7–10 |

`PARADIGM_BY_EXPERIENCE_BAND` in `config.py` is the single source of truth for
this mapping, shared by the generator (what it asks for) and the validator (what
it checks was produced), so the two cannot silently drift apart and cause false
paradigm rejections.

The difficulty bounds are a **sanity envelope only**. They are deliberately not a
global formula — role-specific ceilings (a Product Manager question topping out
lower than an ML Engineer one) are judged by the LLM rubric, not hardcoded.

Each question is also assigned a **scope** — `UNIVERSAL`, `DOMAIN`, or `COMPANY` —
classified by `classify_scope()`, which controls how broadly it can later be
served.

---

## Stage 7 — Deduplication

`deduplicator_service.py` → `check_duplicate()`

Three tiers, cheapest first, short-circuiting on the first match:

1. **Exact match** — normalized text equality. Free.
2. **Normalized Jaccard** — token-overlap similarity against
   `NORMALIZED_OVERLAP_THRESHOLD` (0.85). Free.
3. **Semantic cosine** — embedding similarity against
   `SEMANTIC_SIMILARITY_THRESHOLD` (0.86). The only tier that costs an API call,
   and it is only reached when tiers 1 and 2 found nothing.

---

## Stage 8 — Validation

`validator_service.py` → `validate_question()`

A **fail-closed** ladder. Deterministic checks run first and can reject a question
outright without spending a single LLM call. Surviving questions then climb:

- **Tier 1 — Gemini (primary).** Most questions resolve here.
- **Tier 2 — OpenAI (fallback).** Entered when Tier 1 fails for a non-quota
  reason, or flags the question as needing escalation.
- **Tier 3 — Claude (escalation).** Entered on persistent disagreement between
  the first two tiers.

Fail-closed means exactly one thing: **a question passes only if every tier that
ran agrees it should pass.** There is no majority vote and no override. A single
tier's rejection is final.

This guarantee is enforced twice — in `_merge()` in application code, and by the
`chk_qva_fail_closed` CHECK constraint in migration `0001`. The database
constraint means even a future application bug cannot insert a question that
bypassed the rule. Every attempt, passed or rejected, is recorded in
`question_validation_attempts`.

Scores are read against three thresholds: `VALIDATION_PASS_SCORE` (0.70) and the
uncertain band between `VALIDATION_UNCERTAIN_LOW` (0.60) and
`VALIDATION_UNCERTAIN_HIGH` (0.75). A claimed overall difficulty that diverges
from the average of its 7 difficulty dimensions by more than
`DIFFICULTY_DIMENSION_CONSISTENCY_TOLERANCE` (4.0) is treated as an internally
inconsistent — and therefore untrustworthy — self-rating.

---

## Stage 9 — Database load

`db_loader.py` → `QuestionBankLoader.load_role()`

The single write boundary to PostgreSQL. Validated questions are inserted into
`questions`, their vectors into `question_embeddings`, and their relationships
into the `question_companies` / `question_domains` / `question_skills` /
`question_sources` join tables. Superseded questions are linked via
`superseded_by_question_id` rather than deleted, so question history survives.

Once the load succeeds, `pipeline_jobs.status` becomes `COMPLETED` for this unit
and this cycle.

---

## The state machine and checkpoint recovery

This is the part of the pipeline whose correctness everything else depends on.

### States

| Status | Meaning |
|---|---|
| `PENDING` | Never processed, or explicitly reset |
| `RESEARCHING` | Research call in flight (transient) |
| `RESEARCHED` | Research completed this cycle |
| `SYNTHESIZING` | Synthesis call in flight (transient) |
| `KNOWLEDGE_UPDATED` | Synthesis + RAG indexing done, `knowledge_version` bumped |
| `GENERATING` | Generation in flight (transient) |
| `VALIDATING` | Validation ladder in flight (transient) |
| `LOADING` | DB load in flight (transient) |
| `COMPLETED` | Full pipeline finished for this unit this cycle |
| `SKIPPED_NO_CHANGE` | Research ran, source hash unchanged, downstream already complete |
| `WAITING_FOR_QUOTA` | Hit a provider quota; safe to retry next tick, no data lost |
| `BLOCKED` | Provider hard-blocked (bad/missing key) — needs a human fix |
| `FAILED` | A non-quota, non-blocked exception occurred |

### A/B/C/D dispatch

`process_role()` and `process_company_research()` each dispatch on two independent
questions — *is research fresh?* and *is downstream work complete?* — giving four
cases:

| | Downstream complete | Downstream incomplete |
|---|---|---|
| **Research fresh** | A: nothing to do | B: resume generation only, no new research |
| **Research stale** | C: research → if unchanged, `SKIPPED_NO_CHANGE` | D: full run |

The crucial cases are **B** and **C**. Case B is why an interrupted run does not
re-pay for research it already did. Case C is why unchanged research does not
re-pay for generation — but *only* when downstream really was finished, which is
the distinction that case B protects. Both production bugs fixed in this code were
misdispatches between these cells: one for roles, and the same bug mirrored for
companies.

---

## Quota, blocking and retry handling

`governor.py` → `RateLimitGovernor` and the pipeline's exception types

Every provider call goes through the governor, which paces requests against the
configured RPM limits and classifies failures into three kinds:

```
                    ┌────────────────────────┐
                    │    provider API call   │
                    └───────────┬────────────┘
                                ▼
                    ┌────────────────────────┐
                    │  governor classifies   │
                    │      the failure       │
                    └───────────┬────────────┘
             ┌──────────────────┼──────────────────────┐
             ▼                  ▼                      ▼
   ┌─────────────────┐ ┌──────────────────┐  ┌─────────────────────┐
   │  429 / quota    │ │ missing/invalid  │  │   unrecoverable     │
   │   exhausted     │ │ key / hard block │  │     exception       │
   └────────┬────────┘ └────────┬─────────┘  └──────────┬──────────┘
            ▼                   ▼                       ▼
  QuotaExhaustedException  ProviderBlockedException   (propagates)
            ▼                   ▼                       ▼
  _load_partial() persists   pipeline_jobs =       pipeline_jobs =
  already-validated work       BLOCKED               FAILED
            ▼
    pipeline_jobs =
    WAITING_FOR_QUOTA
            ▼
    batch runner stops cleanly (no crash);
    every other pending unit left untouched
    for the next tick
```

The behavior that matters operationally: **a quota exhaustion never loses work.**
`_load_partial()` writes out every question that already passed validation before
the quota hit, the unit is parked at `WAITING_FOR_QUOTA`, the batch runner exits
cleanly, and the next scheduler tick picks up from there. Other pending units in
the same batch are left untouched rather than being dragged down by one unit's
quota failure.

---

## Cost tracking

`cost_tracker.py` → `CostTracker`, priced through `config.get_pricing()`

Every provider call records its token usage and dollar cost to `data/cost_log.json`.
Pricing comes from `PipelineConfig.PRICING`, a table of rates per 1M tokens taken
from each provider's official pricing page.

Pricing is **fail-closed**: a provider/model pair with no `PRICING` entry raises
`PricingNotConfiguredError` rather than recording $0.00. Silently pricing a real,
paid API call at zero would misrepresent actual spend, so a newly configured model
must have a verified rate added before it can run. Mock models are priced at
$0.00 explicitly — genuinely free, not a fabricated default.

---

## Provider abstraction

`providers/base.py` defines four ABCs — `ResearchProvider`, `LLMProvider`,
`EmbeddingProvider`, `ValidationProvider` — and `providers/factory.py` resolves
each from config at runtime.

| Provider | Role in the pipeline |
|---|---|
| `tavily_firecrawl_provider.py` | Real web research (discovery + extraction) |
| `perplexity_provider.py` | Alternative research provider |
| `gemini_provider.py` | Primary generation, embeddings, Tier 1 validation |
| `openai_provider.py` | Tier 2 validation fallback |
| `claude_provider.py` | Tier 3 validation escalation |
| `local_provider.py` | TF-IDF embeddings, no API cost |
| `mock_provider.py` | `MOCK_MODE` stand-in; never reaches a real API |

Because everything is behind the factory, swapping a model or provider is a config
change, not a code change.

---

## Configuration

All settings live in `PipelineConfig` in `config.py` and are environment-driven.
See `.env.example` for the full list of variables. The ones that change
behavior most:

| Variable | Default | Effect |
|---|---|---|
| `MOCK_MODE` | `false` | Routes every provider through the mock; no real calls, no spend |
| `RESEARCH_PROVIDER` / `RESEARCH_MODEL` | `perplexity` / `sonar` | Research backend |
| `LLM_PROVIDER` / `LLM_MODEL` | `gemini` / `gemini-2.5-flash-lite` | Generation + synthesis |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` | `gemini` / `gemini-embedding-2` | Vector embeddings |
| `VALIDATION_PROVIDER` | `gemini` | Validation Tier 1 |
| `FALLBACK_LLM_PROVIDER` / `FALLBACK_LLM_MODEL` | `openai` / `gpt-4o-mini` | Validation Tier 2 |
| `ESCALATION_PROVIDER` / `ESCALATION_MODEL` | `anthropic` / `claude-3-5-haiku-20241022` | Validation Tier 3 |
| `GEMINI_RPM_LIMIT` | `15` | Request pacing (free-tier default) |
| `PIPELINE_BATCH_SIZE` | `4` | Questions generated per band batch |
| `SEMANTIC_SIMILARITY_THRESHOLD` | `0.86` | Cosine dedup cutoff |
| `NORMALIZED_OVERLAP_THRESHOLD` | `0.85` | Jaccard dedup cutoff |
| `COMPANY_CONTEXT_MIN_SIMILARITY` | `0.35` | Relevance bar a company chunk must clear to reach the prompt |
| `VALIDATION_PASS_SCORE` | `0.70` | Validation pass bar |
| `QUESTION_PIPELINE_DATA_DIR` | unset | Test-only: redirects all local data to a temp dir |

API keys (`GEMINI_API_KEY`, `TAVILY_API_KEY`, `FIRECRAWL_API_KEY`,
`PERPLEXITY_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) are read from the
environment and must never be logged or printed. `is_valid_key()` rejects
placeholder values (`your_`, `dummy`, `placeholder`, `todo`, …) so a half-filled
`.env` cannot be mistaken for real credentials.

---

## Running the pipeline

`rollout_cli.py` is a thin argument parser — it holds no pipeline logic, only
target resolution and pre-flight safety checks, so the CLI, a cron endpoint, and
the test harness all drive the identical orchestration code.

```bash
# What is pending right now? Reads pipeline_jobs and exits. No work, no calls.
python -m question_pipeline.rollout_cli --status

# Preview pending work across the full universe. No API calls, no DB writes.
python -m question_pipeline.rollout_cli --dry-run

# One scheduler tick: DB state decides what is due.
python -m question_pipeline.rollout_cli --once --batch-size 2

# Explicit targets
python -m question_pipeline.rollout_cli --pilot --batch-size 1
python -m question_pipeline.rollout_cli --roles "Data Engineer,NLP Engineer"
python -m question_pipeline.rollout_cli --companies "Google,Amazon"

# Full universe, explicitly requested
python -m question_pipeline.rollout_cli --all-roles --all-companies
```

### Flags

| Flag | Effect |
|---|---|
| `--status` | Print `pipeline_jobs` status and exit |
| `--dry-run` | Preview pending work; no API calls, no DB writes |
| `--once` | Scheduler-tick mode; targets the full universe, DB state decides what runs |
| `--pilot` | Target the 6 pilot roles |
| `--roles` | Comma-separated roles (must be in the 59-role production set) |
| `--all-roles` | Target all 59 roles — must be explicit |
| `--companies` | Comma-separated companies (must be in the 24-company set) |
| `--all-companies` | Target all 24 companies — must be explicit |
| `--batch-size` | Max roles **and** max companies this invocation (default 1, applied per axis) |
| `--resume` | Label for a resuming batch run |

### Safety rails

- A bare invocation with **no** selector and no `--once`/`--dry-run` exits with an
  error rather than defaulting to all 59 roles. Targeting everything is only
  reachable through `--once`, `--dry-run`, or the explicit `--all-*` flags.
- A real (non-`--dry-run`) batch **refuses to start** if `MOCK_MODE=true`, or if
  the Tavily / Firecrawl / Gemini keys are missing or placeholders. This is what
  prevents an accidental real spend when credentials are not actually configured.
- `--batch-size` caps each axis independently, so a tick can never run away across
  the whole universe in one process.

### Scheduling

A production scheduler — cron, a Vercel Cron job, a shell loop — invokes the same
unchanging command on an interval:

```bash
python -m question_pipeline.rollout_cli --once --batch-size 2
```

Nothing about the command changes between ticks. All progression lives in
`pipeline_jobs`.

---

## Database schema

Applied in order from `migrations/`.

**`0001_question_bank_and_knowledge_layer.sql`**

| Table | Holds |
|---|---|
| `research_sources` | Every URL ever used, normalized for dedup |
| `research_documents` | Research dossiers, keyed by role (+ company) |
| `research_document_sources` | Which sources produced which dossier |
| `research_chunks` | Synthesized knowledge chunks + embeddings (HNSW indexed) |
| `knowledge_state` | Per-unit `last_researched_at`, source hash, `knowledge_version` |
| `knowledge_version_history` | Audit trail of every version bump |
| `questions` | The question bank itself, with `is_active` and supersession links |
| `question_companies` / `question_domains` / `question_skills` | Tagging join tables |
| `question_sources` | Question → source provenance |
| `question_embeddings` | Question vectors (HNSW indexed) for semantic dedup |
| `question_validation_attempts` | Every validation attempt, with the `chk_qva_fail_closed` constraint |

**`0002_orchestration_and_source_dedup.sql`**

| Table / column | Holds |
|---|---|
| `pipeline_jobs` | Per-unit status — the state machine's backing store |
| `research_sources.url_normalized` | Canonical URL form, so one page is one source |

---

## Local data files

`data/` holds the pipeline's local working state. All paths are relative to
`DATA_DIR`, which `QUESTION_PIPELINE_DATA_DIR` redirects to a temp directory under
test so the suite never touches real data.

**The directory is gitignored and ships empty.** Everything in it is regenerated
by running the pipeline, and none of it belongs in version control: the knowledge
base holds scraped third-party page text, the vector store holds embeddings
derived from it, and the cost log records real API spend. A fresh clone starts
with an empty `data/` and populates it on the first run.

| File | Contents |
|---|---|
| `state.json` | Local run state |
| `cost_log.json` | Per-call token usage and dollar cost |
| `vector_store.json` | Embedded knowledge chunks for RAG retrieval |
| `question_bank.json` | Generated questions in local form |
| `validation_attempts.json` | Local validation attempt log |
| `knowledge_base/*.json` | Per-role and per-company research + synthesized knowledge |
| `pilot_quality_report.json` | Pilot-run quality summary |

---

## Tests

```bash
python -m pytest question_pipeline/tests -v
```

`tests/__init__.py` sets `QUESTION_PIPELINE_DATA_DIR` to an isolated temp
directory **before any other pipeline module is imported**, so the suite cannot
read or write real pipeline data. Coverage centers on the parts of the pipeline
where a regression would be expensive or silent:

| Test | Guards |
|---|---|
| `test_state_machine_and_quota_recovery.py` | A/B/C/D dispatch and quota resume |
| `test_orchestrator.py` | Orchestration and batch runners |
| `test_quality_gate.py` | Validation thresholds and fail-closed merge |
| `test_fallback_and_json_recovery.py` | Tier fallback and malformed LLM JSON |
| `test_cost_tracker_pricing.py` | Fail-closed pricing lookups |
| `test_data_isolation.py` | That tests cannot touch real data |
| `test_knowledge_state_authority.py` | `knowledge_state` as freshness authority |
| `test_company_rag_context.py` | Threshold-gated company retrieval |
| `test_db_loader.py` | The DB write boundary |
| `test_tavily_firecrawl_provider.py` | Discovery/extraction, fallback, key redaction |
| `test_scope_and_generation_fixes.py` | Scope classification and generation |
| `test_rollout_cli_scheduler.py` | Target resolution and safety rails |
| `test_pipeline.py` | End-to-end run under mock providers |

---

## File map

| Responsibility | File | Entry point |
|---|---|---|
| CLI entry point | `rollout_cli.py` | `main()`, `determine_targets()` |
| Production orchestration | `orchestrator.py` | `RolloutOrchestrator` |
| Job status store | `orchestrator.py` | `PipelineJobStore` |
| Role state machine | `orchestrator.py` | `process_role()` |
| Company state machine | `orchestrator.py` | `process_company_research()` |
| Generate → validate → load | `orchestrator.py` | `_generate_and_load()`, `_load_partial()` |
| Batch runners | `orchestrator.py` | `run_batch()`, `run_company_batch()` |
| Legacy single-role CLI | `pipeline_runner.py` | `run_role_pipeline()` |
| Scope classification | `pipeline_runner.py` | `classify_scope()` |
| Research orchestration | `research_service.py` | `execute_research()`, `execute_company_research()` |
| Freshness / change detection | `freshness.py` | `KnowledgeFreshnessTracker` |
| Synthesis | `synthesis_service.py` | `synthesize()` |
| RAG indexing & retrieval | `rag_service.py` | `RAGService` |
| Question generation | `generator_service.py` | `generate_batch_for_band()` |
| Deduplication | `deduplicator_service.py` | `check_duplicate()` |
| Validation ladder | `validator_service.py` | `validate_question()` |
| Validation persistence | `validation_store.py` | — |
| Rate limiting / quota classification | `governor.py` | `RateLimitGovernor` |
| DB write boundary | `db_loader.py` | `QuestionBankLoader.load_role()` |
| Local state persistence | `state_manager.py` | — |
| Cost tracking | `cost_tracker.py` | `CostTracker` |
| Configuration & pricing | `config.py` | `PipelineConfig`, `get_pricing()` |
| Provider ABCs | `providers/base.py` | the four ABCs |
| Provider factory | `providers/factory.py` | `get_*_provider()` |
| Pydantic models | `models.py` | — |
| Schema | `migrations/*.sql` | — |

For a line-by-line walkthrough of the implementation — real file paths, function
names and line numbers — see
[`question_pipeline/docs/pipeline_logic_explanation.md`](question_pipeline/docs/pipeline_logic_explanation.md).
