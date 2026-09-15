"""
Tests for rollout_cli.py's scheduler-tick target resolution
(determine_targets()) -- added for the local-automation task so that
--once/--dry-run with no explicit --roles/--companies default to the full
59-role/24-company universe (a real "scheduler tick" over everything DB
state says is due), while a bare invocation with neither flag still
requires an explicit selector (preserving the original safety rail against
accidentally targeting all 59 roles by omission in a one-off manual run).

Pure argument-resolution logic: no DB connection, no network call, no
mocks needed -- config.PRODUCTION_ROLES/config.COMPANIES are static lists.
"""
import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("MOCK_MODE", "true")

from question_pipeline.config import config
from question_pipeline import rollout_cli


def make_args(**overrides):
    base = dict(
        roles=None, pilot=False, all_roles=False,
        companies=None, all_companies=False,
        once=False, dry_run=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestDetermineTargets(unittest.TestCase):
    def test_once_with_no_selector_defaults_to_full_universe(self):
        roles, companies, auto = rollout_cli.determine_targets(make_args(once=True))
        self.assertTrue(auto)
        self.assertEqual(roles, config.PRODUCTION_ROLES)
        self.assertEqual(companies, config.COMPANIES)

    def test_dry_run_with_no_selector_defaults_to_full_universe(self):
        roles, companies, auto = rollout_cli.determine_targets(make_args(dry_run=True))
        self.assertTrue(auto)
        self.assertEqual(len(roles), 59)
        self.assertEqual(len(companies), 24)

    def test_bare_invocation_with_no_selector_does_not_default(self):
        """Without --once or --dry-run, omitting every selector must NOT
        silently target all 59 roles -- main() is expected to print its
        usage error and exit in this case."""
        roles, companies, auto = rollout_cli.determine_targets(make_args())
        self.assertFalse(auto)
        self.assertEqual(roles, [])
        self.assertEqual(companies, [])

    def test_explicit_roles_selector_is_not_expanded_to_companies(self):
        roles, companies, auto = rollout_cli.determine_targets(make_args(once=True, roles="Data Scientist"))
        self.assertFalse(auto)
        self.assertEqual(roles, ["Data Scientist"])
        self.assertEqual(companies, [], "narrowing explicitly to one role must not silently pull in all companies")

    def test_explicit_companies_selector_is_not_expanded_to_roles(self):
        roles, companies, auto = rollout_cli.determine_targets(make_args(dry_run=True, companies="Netflix"))
        self.assertFalse(auto)
        self.assertEqual(companies, ["Netflix"])
        self.assertEqual(roles, [], "narrowing explicitly to one company must not silently pull in all roles")

    def test_pilot_flag_resolves_to_pilot_roles_not_full_universe(self):
        roles, companies, auto = rollout_cli.determine_targets(make_args(pilot=True))
        self.assertFalse(auto)
        self.assertEqual(roles, config.PILOT_ROLES)
        self.assertEqual(companies, [])

    def test_all_roles_and_all_companies_explicit_flags_still_work(self):
        roles, companies, auto = rollout_cli.determine_targets(make_args(all_roles=True, all_companies=True))
        self.assertFalse(auto, "explicit --all-roles/--all-companies is not the 'auto-defaulted' scheduler path")
        self.assertEqual(len(roles), 59)
        self.assertEqual(len(companies), 24)

    def test_once_combined_with_explicit_pilot_does_not_trigger_auto_universe(self):
        roles, companies, auto = rollout_cli.determine_targets(make_args(once=True, pilot=True))
        self.assertFalse(auto)
        self.assertEqual(roles, config.PILOT_ROLES)


if __name__ == "__main__":
    unittest.main()
