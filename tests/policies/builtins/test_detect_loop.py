"""Tests for the ``detect_loop`` builtin policy.

Covers:

- Basic loop detection when a tool call repeats ≥ threshold times.
- Distinct calls within the window do not trigger.
- Sliding window eviction: old entries drop off and no longer count.
- Different arguments produce different hashes (no false positives).
- Non-tool_call phases pass through.
- State updates are always emitted (the window advances on every call).
- Custom window/threshold parameters.
- Target-keyed detection via ``ignore_arg_keys`` / ``normalize_uri_args``.
- The guard latch denying the rest of the turn once tripped.
"""

from __future__ import annotations

from omnigent.policies.builtins.safety import (
    _GUARD_LATCH_ARMED_KEY,
    _GUARD_LATCH_KEY,
    _LOOP_STATE_KEY,
    _args_hash,
    _normalize_uri,
    _normalized_arguments,
    detect_loop,
)
from tests.policies.builtins.helpers import tool_call_event as tc


def _state_with_hashes(hashes: list[str]) -> dict:
    return {_LOOP_STATE_KEY: list(hashes)}


# ── Basic detection ─────────────────────────────────────────────────────────


def test_detect_loop_triggers_on_repeated_calls() -> None:
    """Three identical calls in a row trigger ASK."""
    policy = detect_loop(window=10, threshold=3)
    h = _args_hash("sys_os_shell", {"command": "ls"})

    result = policy(tc("sys_os_shell", {"command": "ls"}, _state_with_hashes([h, h])))
    assert result["result"] == "ASK"
    assert "Loop guard" in result["reason"]
    assert "sys_os_shell" in result["reason"]


def test_detect_loop_allows_below_threshold() -> None:
    """Two identical calls (below threshold=3) are allowed."""
    policy = detect_loop(window=10, threshold=3)
    h = _args_hash("sys_os_shell", {"command": "ls"})

    result = policy(tc("sys_os_shell", {"command": "ls"}, _state_with_hashes([h])))
    assert result["result"] == "ALLOW"


def test_detect_loop_first_call_allows() -> None:
    """The very first call (empty state) is allowed."""
    policy = detect_loop(window=10, threshold=3)
    result = policy(tc("sys_os_shell", {"command": "ls"}))
    assert result["result"] == "ALLOW"


# ── Different args ──────────────────────────────────────────────────────────


def test_detect_loop_different_args_no_trigger() -> None:
    """Same tool with different arguments does not trigger."""
    policy = detect_loop(window=10, threshold=3)
    h1 = _args_hash("sys_os_shell", {"command": "ls"})
    h2 = _args_hash("sys_os_shell", {"command": "pwd"})

    result = policy(tc("sys_os_shell", {"command": "cat foo"}, _state_with_hashes([h1, h2])))
    assert result["result"] == "ALLOW"


def test_detect_loop_different_tools_no_trigger() -> None:
    """Different tools with same arguments do not trigger."""
    policy = detect_loop(window=10, threshold=3)
    h1 = _args_hash("Read", {"path": "/tmp/f"})
    h2 = _args_hash("Write", {"path": "/tmp/f"})

    result = policy(tc("Edit", {"path": "/tmp/f"}, _state_with_hashes([h1, h2])))
    assert result["result"] == "ALLOW"


# ── Sliding window ──────────────────────────────────────────────────────────


def test_detect_loop_window_eviction() -> None:
    """Old entries outside the window no longer count.

    With window=4 and threshold=3, two old matching hashes plus
    two intervening different calls means only one match remains
    in the window when the third identical call arrives.
    """
    policy = detect_loop(window=4, threshold=3)
    h_target = _args_hash("Bash", {"command": "fail"})
    h_other = _args_hash("Read", {"path": "x"})

    # History: [target, target, other, other] — window=4 keeps all four.
    # The current call adds a third target, but the window trims to last 4:
    # [target, other, other, target] → only 2 matches, below threshold=3.
    state = _state_with_hashes([h_target, h_target, h_other, h_other])
    result = policy(tc("Bash", {"command": "fail"}, state))
    assert result["result"] == "ALLOW"


def test_detect_loop_window_keeps_recent() -> None:
    """Matches within the window still trigger.

    With window=4, threshold=3: history is [other, target, target],
    current call is target → window is [other, target, target, target]
    → 3 matches → ASK.
    """
    policy = detect_loop(window=4, threshold=3)
    h_target = _args_hash("Bash", {"command": "fail"})
    h_other = _args_hash("Read", {"path": "x"})

    state = _state_with_hashes([h_other, h_target, h_target])
    result = policy(tc("Bash", {"command": "fail"}, state))
    assert result["result"] == "ASK"


# ── State updates ───────────────────────────────────────────────────────────


def test_detect_loop_emits_state_updates_on_allow() -> None:
    """ALLOW results still carry state_updates to advance the window."""
    policy = detect_loop(window=10, threshold=3)
    result = policy(tc("web_search", {"query": "hello"}))
    assert result["result"] == "ALLOW"
    updates = result.get("state_updates", [])
    assert len(updates) == 1
    assert updates[0]["key"] == _LOOP_STATE_KEY
    assert updates[0]["action"] == "set"


def test_detect_loop_emits_state_updates_on_ask() -> None:
    """ASK results carry state_updates too (the window still advances)."""
    policy = detect_loop(window=10, threshold=3)
    h = _args_hash("Bash", {"command": "fail"})
    state = _state_with_hashes([h, h])
    result = policy(tc("Bash", {"command": "fail"}, state))
    assert result["result"] == "ASK"
    updates = result.get("state_updates", [])
    assert len(updates) == 1
    assert updates[0]["key"] == _LOOP_STATE_KEY


def test_detect_loop_window_trimmed_in_state() -> None:
    """The state_updates value is trimmed to the window size."""
    policy = detect_loop(window=3, threshold=3)
    h = _args_hash("Bash", {"command": "x"})
    # Pre-fill with 5 entries — more than the window.
    state = _state_with_hashes([h, h, h, h, h])
    result = policy(tc("Bash", {"command": "x"}, state))
    updates = result.get("state_updates", [])
    stored = updates[0]["value"]
    assert len(stored) == 3


# ── Phase filtering ─────────────────────────────────────────────────────────


def test_detect_loop_ignores_non_tool_call_phase() -> None:
    """Non-tool_call phases pass through with ALLOW."""
    policy = detect_loop()
    result = policy(
        {
            "type": "response",
            "target": None,
            "data": "some response",
            "context": {"actor": {}, "usage": {}},
            "session_state": {},
        }
    )
    assert result["result"] == "ALLOW"


def test_detect_loop_ignores_non_dict_data() -> None:
    """tool_call with non-dict data passes through."""
    policy = detect_loop()
    result = policy(
        {
            "type": "tool_call",
            "target": None,
            "data": "not a dict",
            "context": {"actor": {}, "usage": {}},
            "session_state": {},
        }
    )
    assert result["result"] == "ALLOW"


# ── Custom parameters ──────────────────────────────────────────────────────


def test_detect_loop_custom_threshold() -> None:
    """Higher threshold requires more repeats."""
    policy = detect_loop(window=20, threshold=5)
    h = _args_hash("Bash", {"command": "x"})

    # 4 prior calls + current = 5 → triggers at threshold=5.
    state = _state_with_hashes([h, h, h, h])
    result = policy(tc("Bash", {"command": "x"}, state))
    assert result["result"] == "ASK"

    # 3 prior + current = 4 → below threshold=5.
    state = _state_with_hashes([h, h, h])
    result = policy(tc("Bash", {"command": "x"}, state))
    assert result["result"] == "ALLOW"


def test_detect_loop_can_deny_without_user_prompt() -> None:
    policy = detect_loop(window=5, threshold=2, action="DENY")
    h = _args_hash("sys_os_read", {"path": "README.md"})

    result = policy(
        tc(
            "sys_os_read",
            {"path": "README.md"},
            _state_with_hashes([h]),
        )
    )

    assert result["result"] == "DENY"
    assert "Summarize what you have established" in result["reason"]
    assert "ask the user one question" in result["reason"]


def test_detect_loop_exempts_bounded_poll_tool() -> None:
    policy = detect_loop(
        window=5,
        threshold=2,
        action="DENY",
        exempt_tools=["sys_read_inbox"],
    )
    h = _args_hash("sys_read_inbox", {})

    result = policy(
        tc(
            "sys_read_inbox",
            {},
            _state_with_hashes([h, h, h]),
        )
    )

    assert result["result"] == "ALLOW"
    assert "state_updates" not in result


def test_detect_loop_resets_on_user_request() -> None:
    policy = detect_loop(reset_on_request=True)

    result = policy(
        {
            "type": "request",
            "data": {"user_content": "continue"},
            "session_state": _state_with_hashes(["old"]),
        }
    )

    assert result["state_updates"] == [{"key": _LOOP_STATE_KEY, "action": "set", "value": []}]


# ── Args hash determinism ──────────────────────────────────────────────────


def test_args_hash_deterministic() -> None:
    """Same inputs produce the same hash."""
    h1 = _args_hash("tool", {"a": 1, "b": "x"})
    h2 = _args_hash("tool", {"b": "x", "a": 1})
    assert h1 == h2


def test_args_hash_different_tools() -> None:
    """Different tool names produce different hashes."""
    h1 = _args_hash("tool_a", {"x": 1})
    h2 = _args_hash("tool_b", {"x": 1})
    assert h1 != h2


def test_args_hash_different_args() -> None:
    """Different arguments produce different hashes."""
    h1 = _args_hash("tool", {"x": 1})
    h2 = _args_hash("tool", {"x": 2})
    assert h1 != h2


# ── Target-keyed detection ─────────────────────────────────────────────────
#
# The refs below are verbatim from session f5d06617, where an agent that did
# not know Oracle's ref grammar guessed at it. Every guess was a distinct
# argument tuple, so the raw-argument hash never saw a repeat.


def test_normalize_uri_collapses_scheme_and_escaping() -> None:
    """One target reached through three spellings keys the same."""
    target = "hub.docker.com/r/vaultwarden/server"
    assert _normalize_uri("https://hub.docker.com/r/vaultwarden/server") == target
    assert _normalize_uri("web://https://hub.docker.com/r/vaultwarden/server") == target
    escaped = "web://https%3A%2F%2Fhub.docker.com%2Fr%2Fvaultwarden%2Fserver/"
    assert _normalize_uri(escaped) == target


def test_normalized_arguments_drops_ignored_keys() -> None:
    """A backend selector does not make a call distinct."""
    args = {"ref": "repo://vaultwarden/vaultwarden@main/README.md", "source": "code"}
    projected = _normalized_arguments(args, ignore_keys=frozenset({"source"}), normalize_uris=True)
    assert projected == {"ref": "vaultwarden/vaultwarden@main/readme.md"}


def test_normalized_arguments_passthrough_when_unconfigured() -> None:
    """With neither option set the raw arguments are hashed unchanged."""
    args = {"ref": "web://x", "source": "web"}
    assert _normalized_arguments(args, ignore_keys=frozenset(), normalize_uris=False) is args


def test_detect_loop_keys_on_target_not_arguments() -> None:
    """Same ref under a varied scheme and source counts as a repeat."""
    policy = detect_loop(
        window=12,
        threshold=3,
        action="DENY",
        ignore_arg_keys=["source"],
        normalize_uri_args=True,
    )
    h = _args_hash("oracle__fetch", {"ref": "vaultwarden/vaultwarden@main/readme.md"})

    result = policy(
        tc(
            "oracle__fetch",
            {"ref": "web://https://vaultwarden/vaultwarden@main/README.md", "source": "web"},
            _state_with_hashes([h, h]),
        )
    )
    assert result["result"] == "DENY"
    assert "the same target" in result["reason"]


def test_detect_loop_raw_arguments_miss_the_same_sequence() -> None:
    """Without the options the same three calls do not register a repeat."""
    policy = detect_loop(window=12, threshold=3, action="DENY")
    h = _args_hash("oracle__fetch", {"ref": "vaultwarden/vaultwarden@main/readme.md"})

    result = policy(
        tc(
            "oracle__fetch",
            {"ref": "web://https://vaultwarden/vaultwarden@main/README.md", "source": "web"},
            _state_with_hashes([h, h]),
        )
    )
    assert result["result"] == "ALLOW"


# ── Guard latch ────────────────────────────────────────────────────────────


def test_latch_closes_on_trip() -> None:
    """Tripping stores the reason so later calls read the same instruction."""
    policy = detect_loop(window=10, threshold=3, action="DENY", latch=True)
    h = _args_hash("sys_os_shell", {"command": "ls"})

    result = policy(tc("sys_os_shell", {"command": "ls"}, _state_with_hashes([h, h])))
    latch = [u for u in result["state_updates"] if u["key"] == _GUARD_LATCH_KEY]
    assert latch and latch[0]["value"] == result["reason"]


def test_latched_turn_denies_an_unrelated_tool() -> None:
    """Once armed, a different tool on a different target is denied too."""
    policy = detect_loop(window=10, threshold=3, action="DENY", latch=True)
    state = {_GUARD_LATCH_KEY: "Loop guard: earlier trip.", _GUARD_LATCH_ARMED_KEY: True}

    result = policy(tc("sys_os_read", {"path": "/some/other/file"}, state))
    assert result["result"] == "DENY"
    assert result["reason"] == "Loop guard: earlier trip."


def test_latch_does_not_deny_the_batch_it_tripped_on() -> None:
    """A model emits several calls from one response.

    The calls after the one that tripped the guard were committed before any
    result came back, so denying them punishes a decision it could not have
    revised. They run; the next batch is refused.
    """
    policy = detect_loop(window=10, threshold=3, action="DENY", latch=True)
    state = {_GUARD_LATCH_KEY: "Loop guard: earlier trip."}

    assert (
        policy(tc("sys_os_read", {"path": "/committed/before/the/trip"}, state))["result"]
        == "ALLOW"
    )


def test_a_round_trip_arms_the_latch() -> None:
    """``llm_request`` is where the model has demonstrably seen the denial."""
    policy = detect_loop(window=10, threshold=3, action="DENY", latch=True)
    state = {_GUARD_LATCH_KEY: "Loop guard: earlier trip."}

    armed = policy({"type": "llm_request", "data": {}, "session_state": state})
    assert {"key": _GUARD_LATCH_ARMED_KEY, "action": "set", "value": True} in armed[
        "state_updates"
    ]


def test_a_round_trip_without_a_latch_arms_nothing() -> None:
    policy = detect_loop(window=10, threshold=3, action="DENY", latch=True)
    result = policy({"type": "llm_request", "data": {}, "session_state": {}})
    assert result.get("state_updates", []) == []


def test_trip_stores_the_latch_unarmed() -> None:
    policy = detect_loop(window=10, threshold=3, action="DENY", latch=True)
    h = _args_hash("sys_os_shell", {"command": "ls"})

    result = policy(tc("sys_os_shell", {"command": "ls"}, _state_with_hashes([h, h])))
    keys = {update["key"] for update in result["state_updates"]}
    assert _GUARD_LATCH_KEY in keys
    assert _GUARD_LATCH_ARMED_KEY not in keys


def test_latch_releases_on_a_new_user_turn() -> None:
    """A fresh instruction reopens the latch."""
    policy = detect_loop(window=10, threshold=3, action="DENY", latch=True)
    event = {"type": "request", "data": {}, "session_state": {_GUARD_LATCH_KEY: "tripped"}}
    result = policy(event)
    assert {"key": _GUARD_LATCH_KEY, "action": "set", "value": ""} in result["state_updates"]


def test_latch_off_by_default() -> None:
    """An unlatched guard denies one call and writes no latch."""
    policy = detect_loop(window=10, threshold=3, action="DENY")
    h = _args_hash("sys_os_shell", {"command": "ls"})

    result = policy(tc("sys_os_shell", {"command": "ls"}, _state_with_hashes([h, h])))
    assert all(u["key"] != _GUARD_LATCH_KEY for u in result["state_updates"])
