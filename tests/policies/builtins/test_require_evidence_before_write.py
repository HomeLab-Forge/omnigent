"""Tests for the ``require_evidence_before_write`` builtin policy.

Covers:

- A matching write with no evidence takes the action; a non-matching one does not.
- Loading a named skill, or using a named tool, satisfies the guard for the session.
- Evidence persists across later writes rather than being demanded per file.
- Globs match both a full path and a bare filename.
- An empty ``paths`` list disables the policy.
- Writes outside ``write_tools`` are untouched.
- The reason names the primary-source route, since a denial the model cannot
  act on is a dead end.
"""

from __future__ import annotations

from omnigent.policies.builtins.safety import require_evidence_before_write

PATHS = ["**/compose.yaml", "**/blueprints/*.yaml", "*.caddy"]


def _policy(**kw):
    kw.setdefault("paths", PATHS)
    kw.setdefault("evidence_skills", ["research"])
    return require_evidence_before_write(**kw)


def _event(tool, args, state=None):
    return {
        "type": "tool_call",
        "data": {"name": tool, "arguments": args},
        "session_state": state or {},
    }


def _apply(result, state):
    merged = dict(state)
    for update in result.get("state_updates", []):
        merged[update["key"]] = update["value"]
    return merged


def test_an_unevidenced_write_to_a_contract_file_is_denied() -> None:
    result = _policy()(_event("sys_os_edit", {"path": "/ws/ops/stacks/productivity/compose.yaml"}))

    assert result["result"] == "DENY"
    assert "research" in result["reason"], "the denial must name the way out"


def test_a_write_elsewhere_is_untouched() -> None:
    policy = _policy()

    assert (
        policy(_event("sys_os_edit", {"path": "/ws/docs/setup/secrets.md"}))["result"] == "ALLOW"
    )


def test_loading_the_named_skill_satisfies_the_guard() -> None:
    policy = _policy()
    state = _apply(policy(_event("load_skill", {"name": "research"})), {})

    assert (
        policy(_event("sys_os_edit", {"path": "/ws/ops/compose.yaml"}, state))["result"] == "ALLOW"
    )


def test_an_unrelated_skill_does_not_satisfy_it() -> None:
    policy = _policy()
    state = _apply(policy(_event("load_skill", {"name": "contribute"})), {})

    assert (
        policy(_event("sys_os_edit", {"path": "/ws/ops/compose.yaml"}, state))["result"] == "DENY"
    )


def test_a_named_tool_also_counts_as_evidence() -> None:
    policy = _policy(evidence_tools=["oracle__fetch"])
    state = _apply(policy(_event("oracle__fetch", {"ref": "https://upstream/docs"})), {})

    assert policy(_event("sys_os_write", {"path": "/ws/a.caddy"}, state))["result"] == "ALLOW"


def test_evidence_is_not_demanded_per_file() -> None:
    policy = _policy()
    state = _apply(policy(_event("load_skill", {"name": "research"})), {})

    for path in ("/ws/ops/compose.yaml", "/ws/ops/stacks/auth/blueprints/vw-oidc.yaml"):
        assert policy(_event("sys_os_edit", {"path": path}, state))["result"] == "ALLOW"


def test_a_bare_filename_glob_matches_a_full_path() -> None:
    policy = _policy(paths=["compose.yaml"])

    assert policy(_event("sys_os_edit", {"path": "/deep/nested/compose.yaml"}))["result"] == "DENY"


def test_no_paths_disables_the_policy() -> None:
    policy = _policy(paths=[])

    assert policy(_event("sys_os_edit", {"path": "/ws/ops/compose.yaml"}))["result"] == "ALLOW"


def test_a_non_write_tool_is_untouched() -> None:
    policy = _policy()

    assert policy(_event("sys_os_read", {"path": "/ws/ops/compose.yaml"}))["result"] == "ALLOW"


def test_the_real_session_would_have_been_stopped() -> None:
    """Session c5723032 edited compose.yaml having never loaded ``research``."""
    policy = _policy()
    state = {}
    for tool, args in (
        ("load_skill", {"name": "contribute"}),
        ("sys_os_read", {"path": "/ws/ops/stacks/productivity/compose.yaml"}),
        ("sys_os_shell", {"command": "git ls-files"}),
    ):
        state = _apply(policy(_event(tool, args, state)), state)

    first_write = policy(
        _event("sys_os_edit", {"path": "/ws/ops/stacks/productivity/compose.yaml"}, state)
    )
    assert first_write["result"] == "DENY"
