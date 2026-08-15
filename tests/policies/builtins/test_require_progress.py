"""Tests for the ``require_progress`` builtin policy.

Covers:

- Calls inside the budget pass; the call past it takes the action.
- A progress call resets the budget, and is itself never denied.
- Exempt tools neither spend nor reset.
- A new user turn restores a full budget.
- An empty ``progress_tools`` list disables the policy rather than
  bricking the agent (nothing could ever reset the counter).
- The denial reason names the tools that count as progress, because a
  guard the model cannot act on is just a dead end.
- A replay of the traffic from session ``5bf24a41``, which made 79 calls
  and zero writes, fires where it should.
"""

from __future__ import annotations

from omnigent.policies.builtins.safety import require_progress


def _progress_policy(**kw):
    kw.setdefault("progress_tools", ["sys_os_write", "sys_os_edit"])
    return require_progress(**kw)


def _call(policy, tool, state=None):
    return policy({"type": "tool_call", "data": {"name": tool}, "session_state": state or {}})


def _state_after(result, previous=None):
    state = dict(previous or {})
    for update in result.get("state_updates", []):
        state[update["key"]] = update["value"]
    return state


def _spend(policy, tool, n, state=None):
    state = state or {}
    result = {}
    for _ in range(n):
        result = _call(policy, tool, state)
        state = _state_after(result, state)
    return result, state


def test_reading_is_allowed_inside_the_budget() -> None:
    result, _ = _spend(_progress_policy(budget=5), "sys_os_read", 5)

    assert result["result"] == "ALLOW"


def test_reading_past_the_budget_is_denied() -> None:
    result, _ = _spend(_progress_policy(budget=5), "sys_os_read", 6)

    assert result["result"] == "DENY"
    assert "nothing has been written yet" in result["reason"]
    assert "sys_os_edit" in result["reason"], "the reason must name the way out"


def test_a_write_restores_the_budget() -> None:
    policy = _progress_policy(budget=5)
    _, state = _spend(policy, "sys_os_read", 5)

    wrote = _call(policy, "sys_os_write", state)
    state = _state_after(wrote, state)
    after, _ = _spend(policy, "sys_os_read", 5, state)

    assert wrote["result"] == "ALLOW"
    assert after["result"] == "ALLOW", "a real change must buy a fresh budget"


def test_a_progress_tool_is_never_denied() -> None:
    policy = _progress_policy(budget=2)
    _, state = _spend(policy, "sys_os_read", 20)

    assert _call(policy, "sys_os_edit", state)["result"] == "ALLOW"


def test_exempt_tools_neither_spend_nor_reset() -> None:
    policy = _progress_policy(budget=3, exempt_tools=["sys_read_inbox"])
    _, state = _spend(policy, "sys_os_read", 3)

    exempt = _call(policy, "sys_read_inbox", state)
    assert exempt["result"] == "ALLOW"
    assert not exempt.get("state_updates"), "an exempt call must not touch the counter"
    assert _call(policy, "sys_os_read", state)["result"] == "DENY"


def test_a_new_user_turn_restores_the_budget() -> None:
    policy = _progress_policy(budget=2)
    _, state = _spend(policy, "sys_os_read", 5)

    reset = policy({"type": "request", "data": {}, "session_state": state})
    state = _state_after(reset, state)

    assert _call(policy, "sys_os_read", state)["result"] == "ALLOW"


def test_no_progress_tools_disables_the_policy() -> None:
    policy = _progress_policy(budget=1, progress_tools=[])
    result, _ = _spend(policy, "sys_os_read", 50)

    assert result["result"] == "ALLOW", "an empty progress list must not brick the agent"


def test_the_real_session_would_have_fired() -> None:
    """Session 5bf24a41: 79 calls, 31 reads, 21 shell, 17 searches, zero writes."""
    policy = _progress_policy(budget=25)
    state = {}
    denied_at = None
    for i, tool in enumerate(["sys_os_read", "sys_os_shell", "oracle__search"] * 27, start=1):
        result = _call(policy, tool, state)
        state = _state_after(result, state)
        if result["result"] == "DENY":
            denied_at = i
            break

    assert denied_at == 26, f"expected the 26th call to be denied, got {denied_at}"
