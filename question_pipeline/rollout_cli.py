"""
Production rollout CLI for the 59-role question pipeline. Thin argument
parser over question_pipeline.orchestrator.RolloutOrchestrator -- all actual
work is delegated there (and, beneath that, to the existing
research/synthesis/generation/validation/loader components).

Examples:
    # Preview what a batch would do, no API calls, no DB writes:
    python -m question_pipeline.rollout_cli --dry-run --pilot

    # Process a small, controlled batch of the 6 pilot roles:
    python -m question_pipeline.rollout_cli --pilot --batch-size 2

    # Process a small batch of explicitly named roles:
    python -m question_pipeline.rollout_cli --roles "Data Scientist,Prompt Engineer" --batch-size 2

    # Resume: identical to a normal batch run -- get_pending_roles() is
    # already DB-state-aware, so already-completed/still-fresh roles are
    # skipped automatically. --resume exists as an explicit, readable
    # invocation for "continue where we left off".
    python -m question_pipeline.rollout_cli --pilot --resume --batch-size 3

    # Show current status without doing any work:
    python -m question_pipeline.rollout_cli --status

    # Refresh company knowledge (feeds the generator's RAG company context --
    # see rag_service.retrieve_company_context()) for specific companies:
    python -m question_pipeline.rollout_cli --companies "Stripe,Netflix" --batch-size 2

    # Roles and companies can be combined in one invocation; each runs as
    # its own independent job type (ROLE_GENERATION vs COMPANY_RESEARCH).
    python -m question_pipeline.rollout_cli --pilot --companies "Stripe" --batch-size 1

    # SCHEDULER TICK: no --roles/--companies given at all, with --once or
    # --dry-run, automatically considers the FULL 59-role + 24-company
    # universe, looks up DB state for each, and processes/previews up to
    # --batch-size pending roles AND up to --batch-size pending companies.
    # This is the command meant to be invoked repeatedly (by hand, by a
    # shell loop, or by cron/Task Scheduler later) to simulate a real
    # scheduler locally -- each invocation is one bounded "tick":
    python -m question_pipeline.rollout_cli --dry-run --batch-size 5
    python -m question_pipeline.rollout_cli --once --batch-size 5

Outside of --once/--dry-run with no selector, --all-roles/--all-companies
must still be passed explicitly (never implied) before the full 59-role/
24-company config sets can be targeted by a one-off manual invocation, and
--batch-size always caps how many are processed in one invocation.
"""
import os
import sys
import argparse
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"), override=True)

from question_pipeline.config import config, is_valid_key  # noqa: E402
from question_pipeline.orchestrator import RolloutOrchestrator  # noqa: E402


def get_db_config():
    return {
        "dbname": os.getenv("DB_NAME"), "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"), "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"), "connect_timeout": 10,
    }


def resolve_roles(args) -> list:
    if args.roles:
        requested = [r.strip() for r in args.roles.split(",") if r.strip()]
        unknown = [r for r in requested if r not in config.PRODUCTION_ROLES]
        if unknown:
            print(f"ERROR: unknown role(s) not in the 59-role production set: {unknown}")
            sys.exit(1)
        return requested
    if args.pilot:
        return list(config.PILOT_ROLES)
    if args.all_roles:
        return list(config.PRODUCTION_ROLES)
    print("ERROR: specify one of --pilot, --roles \"A,B\", or --all-roles.")
    sys.exit(1)


def resolve_companies(args) -> list:
    if args.companies:
        requested = [c.strip() for c in args.companies.split(",") if c.strip()]
        unknown = [c for c in requested if c not in config.COMPANIES]
        if unknown:
            print(f"ERROR: unknown company/companies not in the 24-company set: {unknown}")
            sys.exit(1)
        return requested
    if args.all_companies:
        return list(config.COMPANIES)
    return []


def determine_targets(args):
    """Resolves which roles and companies this invocation targets.

    Returns (roles, companies, used_default_universe). Each of roles/
    companies is [] when that axis wasn't requested at all.

    Scheduler defaulting: --once and --dry-run exist specifically to be run
    unattended and repeatedly (a cron-like "tick" -- see the module
    docstring), so when NEITHER axis has an explicit selector, they default
    to the FULL 59-role + 24-company universe rather than forcing
    --all-roles/--all-companies to be typed on every tick. A bare
    invocation with neither --once nor --dry-run still requires an explicit
    selector, preserving the original safety rail against accidentally
    targeting all 59 roles by omission in a one-off manual run.
    """
    role_selection_requested = bool(args.roles or args.pilot or args.all_roles)
    company_selection_requested = bool(args.companies or args.all_companies)

    if not role_selection_requested and not company_selection_requested and (args.once or args.dry_run):
        return list(config.PRODUCTION_ROLES), list(config.COMPANIES), True

    roles = resolve_roles(args) if role_selection_requested else []
    companies = resolve_companies(args) if company_selection_requested else []
    return roles, companies, False


def print_status(orchestrator: RolloutOrchestrator, roles):
    rows = orchestrator.status(roles)
    if not rows:
        print("No pipeline_jobs recorded yet for the requested role(s).")
        return
    print(f"{'ROLE':<45} {'JOB TYPE':<18} {'STATUS':<18} {'ATTEMPTS':<9} REASON")
    print("-" * 120)
    for r in rows:
        role_label = r["role"] if not r["company"] else f"{r['role']} [{r['company']}]"
        print(f"{role_label[:44]:<45} {r['job_type']:<18} {r['status']:<18} {r['attempt_count']:<9} {(r['reason'] or '')[:40]}")


def main():
    parser = argparse.ArgumentParser(description="Production rollout orchestrator for the question_pipeline question bank.")
    parser.add_argument("--pilot", action="store_true", help="Target the 6 pilot roles (config.PILOT_ROLES).")
    parser.add_argument("--roles", type=str, help="Comma-separated explicit role list (must be in the 59-role production set).")
    parser.add_argument("--all-roles", action="store_true", help="Target the full 59-role production set. Must be explicit.")
    parser.add_argument("--companies", type=str, help="Comma-separated explicit company list (must be in the 24-company set) to run COMPANY_RESEARCH for.")
    parser.add_argument("--all-companies", action="store_true", help="Target the full 24-company set for COMPANY_RESEARCH. Must be explicit.")
    parser.add_argument("--batch-size", type=int, default=1, help="Max roles AND max companies to process in this invocation (default: 1; applied independently to each).")
    parser.add_argument("--once", action="store_true", help="Scheduler-tick mode: process one bounded batch of pending work then exit (no daemon/loop). With no --roles/--companies given, defaults to the full 59-role + 24-company universe. Meant to be invoked repeatedly (by hand, a shell loop, or an external scheduler later) to simulate a real scheduler locally.")
    parser.add_argument("--resume", action="store_true", help="Explicit label for a resuming batch run (behavior is identical to a normal batch run -- already-completed/fresh roles and companies are always skipped automatically, so resuming after an interruption never duplicates work).")
    parser.add_argument("--dry-run", action="store_true", help="Preview pending work; makes NO API calls and NO DB writes.")
    parser.add_argument("--status", action="store_true", help="Print current pipeline_jobs status and exit; does no work.")
    args = parser.parse_args()

    db_config = get_db_config()
    orchestrator = RolloutOrchestrator(db_config)

    if args.status:
        roles, companies, _ = determine_targets(args)
        print_status(orchestrator, (roles + companies) or None)
        return

    roles, companies, auto_universe = determine_targets(args)
    if auto_universe:
        print(f"[SCHEDULER] No explicit --roles/--companies given with --once/--dry-run: defaulting to the "
              f"full scheduler universe ({len(roles)} roles, {len(companies)} companies) -- DB state alone "
              f"decides what's actually due.")

    if not roles and not companies:
        print("ERROR: specify at least one of --pilot, --roles \"A,B\", --all-roles, --companies \"A,B\", "
              "--all-companies -- or use --once/--dry-run alone for the full scheduler universe.")
        sys.exit(1)

    if not args.dry_run:
        if config.MOCK_MODE:
            print("ERROR: MOCK_MODE is true. A real rollout batch requires MOCK_MODE=false.")
            sys.exit(1)
        if not (is_valid_key(config.TAVILY_API_KEY) and is_valid_key(config.FIRECRAWL_API_KEY) and is_valid_key(config.GEMINI_API_KEY)):
            print("ERROR: required API keys (Tavily/Firecrawl/Gemini) are missing or placeholder.")
            sys.exit(1)

    all_results = []

    if companies:
        pending_companies = orchestrator.get_pending_companies(companies)
        print(f"Requested companies: {len(companies)}. Pending (due for research or never completed): {len(pending_companies)}.")
        if args.dry_run:
            print(f"[DRY-RUN] Would process up to {min(args.batch_size, len(pending_companies))} of these pending companies:")
            for c in pending_companies[:args.batch_size]:
                print(f"  - {c}")
            if len(pending_companies) > args.batch_size:
                print(f"  ... and {len(pending_companies) - args.batch_size} more remain pending for a future batch.")
        all_results.extend(asyncio.run(orchestrator.run_company_batch(companies, args.batch_size, dry_run=args.dry_run)))

    if roles:
        pending = orchestrator.get_pending_roles(roles)
        print(f"Requested roles: {len(roles)}. Pending (due for research or never completed): {len(pending)}.")
        if args.dry_run:
            print(f"[DRY-RUN] Would process up to {min(args.batch_size, len(pending))} of these pending roles:")
            for r in pending[:args.batch_size]:
                print(f"  - {r}")
            if len(pending) > args.batch_size:
                print(f"  ... and {len(pending) - args.batch_size} more remain pending for a future batch.")
        all_results.extend(asyncio.run(orchestrator.run_batch(roles, args.batch_size, dry_run=args.dry_run)))

    print("\n=== BATCH RESULT ===")
    for r in all_results:
        label = r.role if not r.company else f"{r.role} [{r.company}]"
        print(f"  {label}: {r.status}" + (f" ({r.reason})" if r.reason else "") + (f" {r.metrics}" if r.metrics else ""))


if __name__ == "__main__":
    main()
