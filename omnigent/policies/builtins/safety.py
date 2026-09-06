"""Built-in safety policies for common guardrails.

Provides ready-to-use policy callables that admins and users
can attach to sessions via the CRUD API without writing custom
Python. Each callable follows the :class:`PolicyEvent` →
:class:`PolicyResponse` contract.
"""

from __future__ import annotations

import fnmatch as _fnmatch
import hashlib as _hashlib
import json as _json
import re as _re
from typing import Literal
from urllib.parse import unquote as _unquote

from omnigent.policies.schema import (
    PolicyCallable,
    PolicyEvent,
    PolicyResponse,
    request_attachments,
    request_user_text,
)

_ALLOW: PolicyResponse = {"result": "ALLOW"}

_SYS_OS_TOOLS = frozenset({"sys_os_read", "sys_os_write", "sys_os_edit", "sys_os_shell"})

# Claude Code and Codex native tool names surfaced via the PreToolUse /
# PostToolUse hook contract (see ``omnigent.native_policy_hook``).
# These bypass Omnigent' ``sys_os_*`` MCP tools and execute directly
# inside the CLI subprocess.
_NATIVE_OS_TOOLS = frozenset({"Bash", "Read", "Write", "Edit", "Glob", "Grep"})

# Cursor SDK native tool names surfaced via the preToolUse hook
# (see ``omnigent.inner.cursor_policy_hook``). Cursor uses ``Shell``
# for its terminal tool (not ``Bash``). ``Read`` / ``Write`` / ``Edit``
# are already in ``_NATIVE_OS_TOOLS`` above.
_CURSOR_NATIVE_OS_TOOLS = frozenset({"Shell"})

# Pi native tool names (lowercase), surfaced via the pi ``tool_call``
# extension hook (see ``omnigent.inner.pi_executor._gate_native_tool``).
# Pi runs these in-process and routes them through the same TOOL_CALL
# policy verdict — but under its own names, distinct from the
# Claude/Codex-cased ``_NATIVE_OS_TOOLS``. Pi uses the same argument keys
# as the Omnigent ``sys_os_*`` tools (``path`` / ``command``), so the
# previews below resolve without a Pi-specific arg branch.
_PI_NATIVE_OS_TOOLS = frozenset({"read", "bash", "write", "edit"})

# Hermes Agent tool names surfaced via the ``pre_tool_call`` shell hook
# (see ``omnigent.inner.hermes_policy_hook``). Hermes uses its own naming
# convention for file/shell operations.
_HERMES_OS_TOOLS = frozenset(
    {"terminal", "execute_code", "read_file", "write_file", "search_files"}
)

# Goose native tool names. Goose namespaces its built-in "developer" extension
# tools as ``developer__<tool>``; ``shell`` is the terminal tool and
# ``write`` / ``edit`` / ``text_editor`` / ``read_image`` / ``tree`` are the file
# tools (names vary slightly by Goose version, so cover both the split write/edit
# and the unified text_editor spellings).
_GOOSE_NATIVE_OS_TOOLS = frozenset(
    {
        "developer__shell",
        "developer__write",
        "developer__edit",
        "developer__text_editor",
        "developer__read_image",
        "developer__tree",
    }
)

# opencode-native permission CATEGORIES (the ``permission`` field of a
# ``permission.asked`` event, mapped to the policy-event tool name by the SSE
# forwarder — live-verified against 1.17.7). opencode collapses write/edit/patch
# into ``edit``; ``bash`` is its shell tool (``ShellID.ToolID``). ``bash`` /
# ``read`` / ``edit`` overlap the lowercase pi set above, but list them
# explicitly so opencode coverage does not silently depend on that set, and add
# the file-search categories (``grep`` / ``glob``) pi lacks.
_OPENCODE_NATIVE_OS_TOOLS = frozenset({"bash", "edit", "read", "grep", "glob"})

# Codex in-process harness tool names surfaced as observational
# ``ToolCallRequest`` events. The codex app-server executor translates
# ``commandExecution`` items to a ``ToolCallRequest(name="shell")`` and
# ``fileChange`` items to ``ToolCallRequest(name="apply_patch")``. Only
# the shell tool needs to be in the OS gate here; apply_patch carries no
# ``command`` argument, so the shell-preview branch below is not reached.
_CODEX_IN_PROCESS_OS_TOOLS = frozenset({"shell"})


# ── Rate limiting ────────────────────────────────────────────────────────────


def max_tool_calls_per_session(limit: int = 100) -> PolicyCallable:
    """Factory: deny after *limit* total tool calls in the session.

    Uses ``event["session_state"]`` to persist the counter across
    turns. Returns ``state_updates`` to increment the count on
    each tool call.

    :param limit: Maximum tool calls allowed across the entire
        session. Defaults to ``100``.
    :returns: A policy callable that DENYs after the limit.
    """

    def evaluate(event: PolicyEvent) -> PolicyResponse:
        """Evaluate the session-wide rate limit.

        :param event: Policy event dict.
        :returns: DENY if over limit, ALLOW otherwise.
        """
        if event.get("type") != "tool_call":
            return _ALLOW
        state = event.get("session_state") or {}
        count_value = state.get("_policy_tool_call_count", 0)
        count = int(count_value) if isinstance(count_value, int | float | str) else 0
        if count >= limit:
            return {
                "result": "DENY",
                "reason": f"Exceeded {limit} tool calls this session",
            }
        return {
            "result": "ALLOW",
            "state_updates": [
                {"key": "_policy_tool_call_count", "action": "increment", "value": 1},
            ],
        }

    return evaluate


_TURN_TOOL_COUNT_KEY = "_policy_turn_tool_call_count"
_TURN_TOOL_FAILURE_KEY = "_policy_turn_tool_failure_count"


def _state_count(value: object) -> int:
    try:
        return int(value) if isinstance(value, int | float | str) else 0
    except (TypeError, ValueError):
        return 0


def _tool_result_failed(data: object) -> bool:
    if isinstance(data, dict) and "result" in data:
        data = data["result"]
    if isinstance(data, str):
        try:
            data = _json.loads(data)
        except (TypeError, ValueError):
            return bool(_re.match(r"^\s*(?:error|fatal)\s*:", data, _re.IGNORECASE))
    if not isinstance(data, dict):
        return False
    exit_code = data.get("exit_code")
    return bool(
        data.get("isError") is True
        or data.get("is_error") is True
        or data.get("success") is False
        or data.get("error") not in (None, False, "", {}, [])
        or (isinstance(exit_code, int | float) and exit_code != 0)
        or str(data.get("status", "")).lower() in {"error", "failed", "failure", "cancelled"}
        or str(data.get("outcome", "")).lower() in {"error", "failed", "failure", "cancelled"}
    )


def tool_budget_per_turn(
    max_calls: int = 12,
    max_failures: int = 2,
) -> PolicyCallable:
    """Factory: bound tool calls and consecutive failed results within one turn."""
    max_calls = max(1, max_calls)
    max_failures = max(1, max_failures)

    def evaluate(event: PolicyEvent) -> PolicyResponse:
        event_type = event.get("type")
        if event_type == "request":
            return {
                "result": "ALLOW",
                "state_updates": [
                    {"key": _TURN_TOOL_COUNT_KEY, "action": "set", "value": 0},
                    {"key": _TURN_TOOL_FAILURE_KEY, "action": "set", "value": 0},
                ],
            }
        state = event.get("session_state") or {}
        calls = _state_count(state.get(_TURN_TOOL_COUNT_KEY, 0))
        failures = _state_count(state.get(_TURN_TOOL_FAILURE_KEY, 0))
        if event_type == "tool_call":
            if failures >= max_failures:
                return {
                    "result": "DENY",
                    "reason": (
                        f"Stopped after {failures} consecutive failed tool calls "
                        "in this turn. "
                        "Read the errors and report the blocker."
                    ),
                }
            if calls >= max_calls:
                return {
                    "result": "DENY",
                    "reason": (
                        f"Exceeded the {max_calls}-tool budget for this turn. "
                        "Checkpoint progress and continue in a new turn."
                    ),
                }
            return {
                "result": "ALLOW",
                "state_updates": [
                    {"key": _TURN_TOOL_COUNT_KEY, "action": "increment", "value": 1},
                ],
            }
        if event_type == "tool_result":
            if _tool_result_failed(event.get("data")):
                return {
                    "result": "ALLOW",
                    "state_updates": [
                        {"key": _TURN_TOOL_FAILURE_KEY, "action": "increment", "value": 1},
                    ],
                }
            if failures:
                return {
                    "result": "ALLOW",
                    "state_updates": [
                        {"key": _TURN_TOOL_FAILURE_KEY, "action": "set", "value": 0},
                    ],
                }
        return _ALLOW

    return evaluate


_LOOP_STATE_KEY = "_policy_loop_recent_hashes"
_PROGRESS_STATE_KEY = "_policy_calls_since_progress"
_EVIDENCE_STATE_KEY = "_policy_evidence_seen"


def require_evidence_before_write(
    paths: list[str] | None = None,
    evidence_skills: list[str] | None = None,
    evidence_tools: list[str] | None = None,
    write_tools: list[str] | None = None,
    action: Literal["ASK", "DENY"] = "DENY",
) -> PolicyCallable:
    """Factory: refuse to author a file the session has no evidence for.

    Some files state a contract the agent cannot derive from the repository:
    a third-party service's environment variables, an image tag, an upstream
    API's field names. Reading more of your own code never establishes them,
    so an agent that only reads locally will write them from memory, fluently
    and wrongly. Observed: a password-vault service authored with four
    invented environment variables, a database configured through fields that
    do not exist, and a comment asserting a setting that was never set.

    The mechanism is deliberately empty of policy. WHICH files carry a
    contract, and WHAT counts as having checked, are the agent author's
    judgement and arrive as parameters. This keeps the repository-specific
    part in the spec, where it can change without a release, and keeps the
    rule itself short enough that the model meets it once, at the moment it
    matters, rather than as one more line in a prompt it read ten thousand
    tokens ago.

    Evidence is session-scoped and never expires: having checked upstream
    once, the agent may keep writing. The guard exists to prevent authoring
    blind, not to demand a lookup per file.

    :param paths: Glob patterns, matched against the write's ``path``
        argument, naming files that state an external contract, e.g.
        ``["**/compose.yaml", "**/blueprints/*.yaml"]``. Empty disables the
        policy.
    :param evidence_skills: Skill names whose loading counts as evidence,
        e.g. ``["research"]``.
    :param evidence_tools: Tool names whose use counts as evidence, e.g. a
        documentation fetch.
    :param write_tools: Tools treated as authoring. Defaults to
        ``["sys_os_write", "sys_os_edit"]``.
    :param action: ``"ASK"`` or ``"DENY"`` when evidence is missing.
    :returns: A policy callable that gates authorship on evidence.
    """
    patterns = tuple(paths or ())
    skills = frozenset(evidence_skills or ())
    tools = frozenset(evidence_tools or ())
    writers = frozenset(write_tools or ["sys_os_write", "sys_os_edit"])
    normalized_action = action.upper() if action.upper() in {"ASK", "DENY"} else "DENY"

    def _matches(path: str) -> bool:
        # Match the whole path and the bare name, so a caller may write either
        # "**/compose.yaml" or "compose.yaml" and mean the same thing.
        tail = path.rsplit("/", 1)[-1]
        return any(
            _fnmatch.fnmatch(path, pattern) or _fnmatch.fnmatch(tail, pattern)
            for pattern in patterns
        )

    def evaluate(event: PolicyEvent) -> PolicyResponse:
        """Evaluate whether this write has the evidence behind it.

        :param event: Policy event dict.
        :returns: The configured action for an unevidenced write, else ALLOW.
        """
        if event.get("type") != "tool_call" or not patterns:
            return _ALLOW
        data = event.get("data")
        if not isinstance(data, dict):
            return _ALLOW

        tool_name = data.get("name", "")
        arguments = data.get("arguments")
        arguments = arguments if isinstance(arguments, dict) else {}

        state = event.get("session_state") or {}
        raw_seen = state.get(_EVIDENCE_STATE_KEY)
        seen = (
            [item for item in raw_seen if isinstance(item, str)]
            if isinstance(raw_seen, list)
            else []
        )

        # Record evidence. load_skill is checked by the skill it names, every
        # other tool by its own name.
        found = None
        if tool_name == "load_skill":
            skill = arguments.get("name")
            if isinstance(skill, str) and skill in skills:
                found = f"skill:{skill}"
        elif tool_name in tools:
            found = f"tool:{tool_name}"
        if found is not None and found not in seen:
            return {
                "result": "ALLOW",
                "state_updates": [
                    {"key": _EVIDENCE_STATE_KEY, "action": "set", "value": [*seen, found]},
                ],
            }

        if tool_name not in writers or seen:
            return _ALLOW
        path = arguments.get("path")
        if not isinstance(path, str) or not _matches(path):
            return _ALLOW

        wanted = sorted({*(f"the {name} skill" for name in skills), *tools})
        return {
            "result": normalized_action,
            "reason": (
                f"Evidence guard: {path} states a contract owned by something "
                "outside this repository — environment variable names, image "
                "tags, endpoints — and nothing in this session has checked what "
                "that contract actually is. Reading more of our own code cannot "
                "establish it. Check the primary source first"
                + (f" using {' or '.join(wanted)}" if wanted else "")
                + ", then write. Names copied from memory are the failure this "
                "guard exists to catch."
            ),
        }

    return evaluate


def require_progress(
    budget: int = 25,
    progress_tools: list[str] | None = None,
    action: Literal["ASK", "DENY"] = "DENY",
    exempt_tools: list[str] | None = None,
    reset_on_request: bool = True,
) -> PolicyCallable:
    """Factory: stop an agent that gathers evidence and never acts.

    Counts tool calls since the last one that CHANGED something, and
    denies further calls once the budget is spent. A call naming any
    tool in *progress_tools* resets the counter to zero.

    This is a different failure from :func:`detect_loop`, which needs
    identical arguments to fire. An agent reading a hundred different
    files makes no repeated call and no progress either: observed at 79
    calls across 31 reads, 21 shell commands and 17 searches with zero
    writes, having announced four separate times that it now understood
    the code and would begin. Nothing in the stack could see that, because
    every individual call was reasonable and none of them repeated.

    The denial is not a stop. Its reason names the budget, the fact that
    nothing has changed yet, and the tools that count as progress, so the
    next decision is to make the smallest real change rather than to
    gather more. Deny is preferred over ask for exactly that: an ask can
    be answered with more reading.

    :param budget: Calls allowed since the last progress call before the
        action fires. Defaults to ``25``. Clamped to a minimum of ``1``.
    :param progress_tools: Tool names that count as progress and reset
        the counter, e.g. ``["sys_os_write", "sys_os_edit"]``. A call to
        one of these is always allowed. Empty means nothing ever resets,
        which is a misconfiguration, so an empty list disables the policy.
    :param action: ``"ASK"`` or ``"DENY"`` once the budget is spent.
    :param exempt_tools: Tool names that neither count against the budget
        nor reset it, such as a bounded poll.
    :param reset_on_request: Clear the counter on each user turn, so a
        fresh instruction starts with a full budget.
    :returns: A policy callable that forces a phase transition.
    """
    budget = max(1, budget)
    normalized_action = action.upper() if action.upper() in {"ASK", "DENY"} else "DENY"
    progress = frozenset(progress_tools or [])
    exempt = frozenset(exempt_tools or [])

    def evaluate(event: PolicyEvent) -> PolicyResponse:
        """Evaluate whether this call is allowed given the progress budget.

        :param event: Policy event dict.
        :returns: The configured action once the budget is spent, else ALLOW.
        """
        event_type = event.get("type")
        if event_type == "request" and reset_on_request:
            return {
                "result": "ALLOW",
                "state_updates": [
                    {"key": _PROGRESS_STATE_KEY, "action": "set", "value": 0},
                ],
            }
        if event_type != "tool_call" or not progress:
            return _ALLOW
        data = event.get("data")
        if not isinstance(data, dict):
            return _ALLOW

        tool_name = data.get("name", "")
        if tool_name in exempt:
            return _ALLOW
        if tool_name in progress:
            return {
                "result": "ALLOW",
                "state_updates": [
                    {"key": _PROGRESS_STATE_KEY, "action": "set", "value": 0},
                ],
            }

        state = event.get("session_state") or {}
        raw = state.get(_PROGRESS_STATE_KEY)
        spent = raw + 1 if isinstance(raw, int) and raw >= 0 else 1

        if spent > budget:
            return {
                "result": normalized_action,
                "reason": (
                    f"Progress guard: {spent - 1} tool calls since anything last "
                    f"changed, against a budget of {budget}, and nothing has been "
                    "written yet. You have enough to start. Make the smallest "
                    "change that moves the task forward using one of: "
                    f"{', '.join(sorted(progress))}. An incomplete first change "
                    "that can be reviewed is worth more than more evidence. If "
                    "the task genuinely cannot be started, say what is missing "
                    "in an incomplete-stop handoff instead of reading further."
                ),
                "state_updates": [
                    {"key": _PROGRESS_STATE_KEY, "action": "set", "value": spent},
                ],
            }

        return {
            "result": "ALLOW",
            "state_updates": [
                {"key": _PROGRESS_STATE_KEY, "action": "set", "value": spent},
            ],
        }

    return evaluate


def _args_hash(tool_name: str, arguments: object) -> str:
    """Deterministic hash of (tool_name, arguments).

    :param tool_name: Tool being called.
    :param arguments: Arguments dict (or any JSON-serializable value).
    :returns: SHA-256 hex digest identifying this (tool, args) pair.
    """
    blob = _json.dumps({"t": tool_name, "a": arguments}, sort_keys=True, default=str)
    return _hashlib.sha256(blob.encode()).hexdigest()


# ── Guard latch ──────────────────────────────────────────────────────────────
#
# A DENY suppresses one tool result. It does not end the turn, and an agent
# under pressure reads the denial as a failed call and tries the next variant.
# Session f5d06617 is the shape: the thrashing guard fired after five
# consecutive errors with "End this approach and provide an incomplete-stop
# handoff", and the agent made sixteen more tool calls.
#
# The latch closes that. Once a guard trips it, every later tool call in the
# turn is denied with the same instruction, so continuing costs a round trip
# and returns nothing. Summarising and asking is then the only move left, and
# the instruction describes the situation instead of requesting cooperation.
#
# ``detect_loop`` is the enforcer because it is the policy that sees
# ``tool_call``. ``detect_thrashing`` fires on ``tool_result`` and can only set
# the latch, so latching it without also enabling ``detect_loop`` sets a flag
# nothing reads.
#
# The latch arms one round-trip after it closes. A model emits a whole batch of
# tool calls from a single response, so the calls after the one that tripped the
# guard were already committed before any result came back — denying them
# punishes a decision the model had no chance to revise. Session d24accf4 shows
# the shape: one assistant message at 12:34:42, then three fetches, the second
# suppressed and the third denied, none of which it could have reconsidered.
#
# ``llm_request`` fires once per round-trip, so it is the point where the model
# has demonstrably seen everything so far. Arming there means the batch in
# flight finishes and the next one is refused.
_GUARD_LATCH_KEY = "_policy_guard_latch"
_GUARD_LATCH_ARMED_KEY = "_policy_guard_latch_armed"

#: A guard verdict names the behaviour, never the count or the threshold behind
#: it. Told "2 times in the last 12 calls", a model starts working the budget —
#: how many it has left, whether a different spelling resets the window —
#: instead of why the call did not work. The repetition is the signal it needs;
#: the arithmetic is ours. Applies to every reason built here and in
#: ``policies/builtins/context.py``.
#:
#: Opens a STOP. The executor adapter matches this prefix to end the turn
#: behind a final handoff (``_is_terminal_tool_guard_reason``), so it belongs
#: only on a verdict that really is the end of the road.
_GUARD_STOP_PREFIX = "Loop guard:"

#: Opens a STEER: one call denied, the turn carries on. It must NOT contain the
#: stop prefix. Both verdicts used to open with "Loop guard:", and the adapter
#: cannot see a policy's intent — only that string — so every correction ended
#: the turn it was issued to rescue, and ``corrections_before_latch`` bought
#: nothing.
_GUARD_STEER_PREFIX = "Loop steer:"

_GUARD_DO_NEXT = (
    "do_next: stop calling tools. Summarize what you have established so far, "
    "list what is blocking you, and ask the user one question."
)

#: Said on a steer rather than a stop. The repeat is evidence the call itself is
#: wrong, not that the task is finished — a guard that only ever ends the turn
#: makes the human the recovery mechanism.
_GUARD_DO_DIFFERENTLY = (
    "do_next: the turn is still yours — this denied one call, not the turn. "
    "This exact call will keep failing and rephrasing its arguments will not "
    "change that. Name in one line what you were trying to establish, then "
    "reach that fact another way. If there is no other way, say what is "
    "blocking you and stop calling tools."
)
#: Corrections issued this turn, so the second detection can stop instead of
#: correcting again.
_LOOP_CORRECTION_KEY = "_policy_loop_corrections"


def _latched_reason(state: object, *, armed_only: bool = True) -> str:
    """Read the latched guard reason.

    :param state: The event's ``session_state`` mapping.
    :param armed_only: When true, a latch that has not yet survived a
        round-trip reads as open, so the batch it tripped on finishes.
    :returns: The stored reason, or ``""`` when the latch is open.
    """
    if not isinstance(state, dict):
        return ""
    value = state.get(_GUARD_LATCH_KEY)
    reason = value if isinstance(value, str) else ""
    if reason and armed_only and not state.get(_GUARD_LATCH_ARMED_KEY):
        return ""
    return reason


def _latch_update(reason: str) -> dict[str, object]:
    """Build the state update that closes the latch, unarmed."""
    return {"key": _GUARD_LATCH_KEY, "action": "set", "value": reason}


def _latch_arm() -> dict[str, object]:
    """Build the state update that makes a closed latch start denying."""
    return {"key": _GUARD_LATCH_ARMED_KEY, "action": "set", "value": True}


def _latch_release() -> list[dict[str, object]]:
    """Build the state updates that open the latch on a new user turn."""
    return [
        {"key": _GUARD_LATCH_KEY, "action": "set", "value": ""},
        {"key": _GUARD_LATCH_ARMED_KEY, "action": "set", "value": False},
    ]


_URI_SCHEME_RE = _re.compile(r"^[a-z][a-z0-9+.-]*://")


def _normalize_uri(value: str) -> str:
    """Reduce one target expressed through different ref schemes to one key.

    An agent that does not know a ref grammar guesses at it, and every guess
    is a distinct argument tuple to a hash keyed on the raw arguments. In
    session f5d06617 ``hub.docker.com/r/vaultwarden/server`` was fetched as a
    bare URL twice and as ``web://…`` once, and
    ``vaultwarden/vaultwarden@main/README.md`` was fetched under ``repo://``
    three times with the ``source`` argument varied. Nine calls, two targets,
    no repeat the loop guard could see.

    Percent-decoding runs first so ``web://https%3A%2F%2Fhost`` reaches the
    same key as ``https://host``, then schemes are stripped repeatedly because
    a wrapped ref carries two.

    :param value: A raw string argument value.
    :returns: The target with scheme, escaping, trailing slash, and case
        removed.
    """
    text = _unquote(value.strip())
    for _ in range(3):
        stripped = _URI_SCHEME_RE.sub("", text, count=1)
        if stripped == text:
            break
        text = stripped
    return text.rstrip("/").lower()


def _normalized_arguments(
    arguments: object,
    *,
    ignore_keys: frozenset[str],
    normalize_uris: bool,
) -> object:
    """Project a tool-call argument mapping down to its loop-detection key.

    :param arguments: The raw ``arguments`` value from the tool-call event.
    :param ignore_keys: Argument names dropped before hashing, for fields that
        select a backend rather than a target (Oracle's ``source``).
    :param normalize_uris: Whether string values are reduced by
        :func:`_normalize_uri`.
    :returns: The mapping to hash, or *arguments* unchanged when neither
        option is configured or the value is not a mapping.
    """
    if not isinstance(arguments, dict) or not (ignore_keys or normalize_uris):
        return arguments
    projected: dict[str, object] = {}
    for key, value in arguments.items():
        if key in ignore_keys:
            continue
        projected[key] = (
            _normalize_uri(value) if normalize_uris and isinstance(value, str) else value
        )
    return projected


def detect_loop(
    window: int = 10,
    threshold: int = 3,
    action: Literal["ASK", "DENY"] = "ASK",
    exempt_tools: list[str] | None = None,
    reset_on_request: bool = True,
    ignore_arg_keys: list[str] | None = None,
    normalize_uri_args: bool = False,
    latch: bool = False,
    corrections_before_latch: int = 0,
) -> PolicyCallable:
    """Factory: detect repeated tool calls against the same target.

    Tracks recent tool-call hashes in ``session_state`` as a
    bounded list of SHA-256 hex digests keyed by
    ``_policy_loop_recent_hashes``.  When the same hash
    appears *threshold* times within the last *window* calls,
    returns the configured action so the loop can end with a handoff.

    This catches the #1 token-waste pattern — an agent retrying
    the same failing tool call — which
    ``max_tool_calls_per_session`` cannot detect because it only
    counts total calls.

    By default the hash covers the raw arguments, so a retry counts only
    when it is byte-identical. *ignore_arg_keys* and *normalize_uri_args*
    widen it to the call's target, which is what catches an agent guessing
    at a ref grammar rather than repeating one ref.

    :param window: Number of recent calls to consider.
        Defaults to ``10``. Clamped to a minimum of ``1``.
    :param threshold: How many times a call must repeat within
        *window* to trigger. Defaults to ``3``. Clamped to a
        minimum of ``1``.
    :param action: ``"ASK"`` or ``"DENY"`` when a loop is detected.
    :param exempt_tools: Tool names allowed to repeat, such as a bounded
        async-result poll.
    :param reset_on_request: Clear loop history for each user turn.
    :param ignore_arg_keys: Argument names dropped before hashing. Use for
        fields that pick a backend rather than a target, so varying one does
        not read as a new call — Oracle's ``source`` is the case this exists
        for.
    :param normalize_uri_args: Reduce string arguments to their target before
        hashing: percent-decode, strip ``scheme://`` prefixes, drop a trailing
        slash, lowercase. Collapses ``web://https%3A%2F%2Fhost/p``,
        ``web://https://host/p`` and ``https://host/p`` to one key.
    :param latch: Once this guard or ``detect_thrashing`` trips, deny every
        remaining tool call in the turn with the same instruction, instead of
        denying one result and letting the agent try the next variant.
    :param corrections_before_latch: How many detections are answered with a
        correction — deny this call, say why, and clear the window so a
        different approach is not flagged by the stale hashes — before the
        latch closes. ``0`` latches on the first detection. Above ``0`` the
        agent gets that many chances to route around a wrong call on its own,
        which is the difference between a guard that recovers a turn and one
        that only ends it.
    :returns: A policy callable that detects repeated calls on one target.
    """
    window = max(1, window)
    threshold = max(1, threshold)
    normalized_action = action.upper() if action.upper() in {"ASK", "DENY"} else "ASK"
    exempt = frozenset(exempt_tools or [])
    ignore_keys = frozenset(ignore_arg_keys or [])

    def evaluate(event: PolicyEvent) -> PolicyResponse:
        """Evaluate whether the current tool call is a repeated loop.

        :param event: Policy event dict.
        :returns: ASK if loop detected, ALLOW otherwise.
        """
        event_type = event.get("type")
        if event_type == "request" and reset_on_request:
            return {
                "result": "ALLOW",
                "state_updates": [
                    {"key": _LOOP_STATE_KEY, "action": "set", "value": []},
                    *(
                        [{"key": _LOOP_CORRECTION_KEY, "action": "set", "value": 0}]
                        if corrections_before_latch > 0
                        else []
                    ),
                    *(_latch_release() if latch else []),
                ],
            }
        # One round-trip has passed since the guard tripped, so the batch it
        # tripped on has finished and the model has seen the denial.
        if latch and event_type == "llm_request":
            if _latched_reason(event.get("session_state"), armed_only=False):
                return {"result": "ALLOW", "state_updates": [_latch_arm()]}
            return _ALLOW
        if event_type != "tool_call":
            return _ALLOW
        data = event.get("data")
        if not isinstance(data, dict):
            return _ALLOW

        tool_name = data.get("name", "")
        if tool_name in exempt:
            return _ALLOW

        # A guard already tripped this turn. Nothing the agent calls now can
        # make progress, so say so rather than denying one call at a time.
        if latch:
            latched = _latched_reason(event.get("session_state"))
            if latched:
                return {"result": normalized_action, "reason": latched}

        arguments = _normalized_arguments(
            data.get("arguments", {}),
            ignore_keys=ignore_keys,
            normalize_uris=normalize_uri_args,
        )
        h = _args_hash(tool_name, arguments)

        state = event.get("session_state") or {}
        recent_value = state.get(_LOOP_STATE_KEY)
        recent = (
            [item for item in recent_value if isinstance(item, str)]
            if isinstance(recent_value, list)
            else []
        )

        recent.append(h)
        # Trim to sliding window.
        recent = recent[-window:]

        count = recent.count(h)
        if count >= threshold:
            repeated = (
                "against the same target"
                if (ignore_keys or normalize_uri_args)
                else "with identical arguments"
            )
            corrections = state.get(_LOOP_CORRECTION_KEY)
            issued = corrections if isinstance(corrections, int) else 0
            correcting = issued < corrections_before_latch
            reason = (
                f"{_GUARD_STEER_PREFIX if correcting else _GUARD_STOP_PREFIX} "
                f"tool '{tool_name}' has already been called {repeated} in this "
                "turn and it did not advance anything. "
                f"{_GUARD_DO_DIFFERENTLY if correcting else _GUARD_DO_NEXT}"
            )
            return {
                "result": normalized_action,
                "reason": reason,
                "state_updates": [
                    # A correction clears the window: the next call should be a
                    # different approach, and the hashes behind this trip would
                    # otherwise flag it on arrival.
                    {
                        "key": _LOOP_STATE_KEY,
                        "action": "set",
                        "value": [] if correcting else recent,
                    },
                    *(
                        [{"key": _LOOP_CORRECTION_KEY, "action": "set", "value": issued + 1}]
                        if corrections_before_latch > 0
                        else []
                    ),
                    *([_latch_update(reason)] if latch and not correcting else []),
                ],
            }

        return {
            "result": "ALLOW",
            "state_updates": [
                {"key": _LOOP_STATE_KEY, "action": "set", "value": recent},
            ],
        }

    return evaluate


# ── OS tool approval ────────────────────────────────────────────────────────


def ask_on_os_tools(event: PolicyEvent) -> PolicyResponse:
    """ASK for user approval before any file or shell tool call.

    Covers six tool-name families:

    - **Omnigent built-in OS tools** (``sys_os_read``,
      ``sys_os_write``, ``sys_os_edit``, ``sys_os_shell``).
    - **Claude Code native tools** (``Bash``, ``Read``, ``Write``,
      ``Edit``, ``Glob``, ``Grep``) — surfaced via the
      ``PreToolUse`` hook contract.
    - **Codex native tools** — uses the same ``PreToolUse`` hook
      contract with the same tool names (e.g. ``Bash``).
    - **Cursor SDK native tools** (``Shell``) — surfaced via the
      ``preToolUse`` hook (see ``cursor_policy_hook.py``). Cursor
      uses ``Shell`` instead of ``Bash`` for its terminal tool.
    - **Pi native tools** (``read``, ``bash``, ``write``, ``edit``)
      — surfaced via the pi ``tool_call`` extension hook. Lowercase
      and distinct from the Claude/Codex casing.
    - **Hermes Agent tools** (``terminal``, ``execute_code``,
      ``read_file``, ``write_file``, ``search_files``) — surfaced
      via the ``pre_tool_call`` shell hook.
    - **opencode native tools** (``bash``, ``edit``, ``read``,
      ``grep``, ``glob``) — opencode's permission CATEGORIES, surfaced
      via the SSE forwarder's ``permission.asked`` → policy-evaluate
      path. opencode collapses write/edit/patch into ``edit``.

    Returns ASK so the user sees an approval prompt before the tool
    executes.

    :param event: Policy event dict.
    :returns: ASK if a file/shell tool is being called, ALLOW
        otherwise.
    """
    if event.get("type") != "tool_call":
        return _ALLOW
    data = event.get("data")
    if not isinstance(data, dict):
        return _ALLOW
    tool = data.get("name", "")
    _all_os_tools = (
        _SYS_OS_TOOLS
        | _NATIVE_OS_TOOLS
        | _CURSOR_NATIVE_OS_TOOLS
        | _PI_NATIVE_OS_TOOLS
        | _HERMES_OS_TOOLS
        | _GOOSE_NATIVE_OS_TOOLS
        | _OPENCODE_NATIVE_OS_TOOLS
        | _CODEX_IN_PROCESS_OS_TOOLS
    )
    if tool in _all_os_tools:
        args = data.get("arguments", {})
        # Build a short preview of what the tool is doing.
        if tool in (
            "sys_os_shell",
            "Bash",
            "bash",
            "Shell",
            "terminal",
            "developer__shell",
            "shell",
        ):
            preview = args.get("command", "") if isinstance(args, dict) else ""
        elif tool in ("Grep", "Glob", "search_files", "grep", "glob"):
            preview = args.get("pattern", "") if isinstance(args, dict) else ""
        elif tool == "execute_code":
            preview = args.get("code", "")[:80] if isinstance(args, dict) else ""
        else:
            # Omnigent tools use ``path``; Claude native tools use ``file_path``.
            preview = (
                (args.get("path") or args.get("file_path", "")) if isinstance(args, dict) else ""
            )
        return {
            "result": "ASK",
            "reason": f"Agent wants to call {tool}({preview!r}). Approve?",
        }
    return _ALLOW


# ── Policy tool approval ───────────────────────────────────────────────────


def ask_on_add_policy(event: PolicyEvent) -> PolicyResponse:
    """ASK for user approval before ``sys_add_policy`` executes.

    Agents must not silently install new policies on a session.
    This callable is injected unconditionally by the builder so
    every ``sys_add_policy`` call parks for approval — the user
    sees what the agent wants to add and can approve or deny.

    :param event: Policy event dict.
    :returns: ASK if ``sys_add_policy`` is being called, ALLOW
        otherwise.
    """
    if event.get("type") != "tool_call":
        return _ALLOW
    data = event.get("data")
    if not isinstance(data, dict):
        return _ALLOW
    if data.get("name") != "sys_add_policy":
        return _ALLOW
    args = data.get("arguments")
    if isinstance(args, dict):
        policy_name = args.get("name", "")
        handler = args.get("handler", "")
        preview = f"{policy_name} ({handler})" if handler else policy_name
    else:
        preview = ""
    return {
        "result": "ASK",
        "reason": f"Agent wants to add policy: {preview}. Approve?",
    }


# ── Skill blocking ──────────────────────────────────────────────────────────

# Omnigent runner tools that load skills in non-native (SDK) harnesses.
_SKILL_TOOLS = frozenset({"load_skill", "read_skill_file"})

# Claude Code's native ``Skill`` tool, fired via ``PreToolUse`` hook and
# evaluated server-side at ``POST /v1/sessions/{id}/policies/evaluate``.
# The tool takes a ``skill`` argument with the skill name.
_NATIVE_SKILL_TOOL = "Skill"


def block_skills(blocked: list[str]) -> PolicyCallable:
    """Factory: deny skill loading for specific skill names.

    Intercepts three loading paths:

    1. **AP runner tools** — ``load_skill`` and ``read_skill_file`` tool
       calls dispatched by non-native (SDK) harnesses.
    2. **Native ``Skill`` tool** — Claude Code's built-in ``Skill`` tool,
       intercepted via the ``PreToolUse`` command hook which POSTs to
       the Omnigent server's ``/policies/evaluate`` endpoint. This is how
       ``block_skills`` enforces on native Claude Code and Codex
       harnesses — there is no ``load_skill`` runner tool in native
       mode.
    3. **Slash commands** — ``/skill-name`` commands submitted by the
       user (or UI). The Omnigent server evaluates these at the ``request``
       phase as synthetic ``"/<name> <args>"`` text via
       ``_build_skill_slash_command_policy_body``.

    Matching is case-insensitive.

    :param blocked: Skill names to block, e.g.
        ``["code-review", "deploy"]``.
    :returns: A policy callable that DENYs blocked skill loads.
    """
    blocked_lower = frozenset(name.lower() for name in blocked)

    def evaluate(event: PolicyEvent) -> PolicyResponse:
        """Evaluate whether the skill load should be blocked.

        :param event: Policy event dict.
        :returns: DENY if the skill name is blocked, ALLOW otherwise.
        """
        event_type = event.get("type")

        # ── Path 1 & 2: tool_call interception ────────────────────────
        if event_type == "tool_call":
            data = event.get("data")
            if not isinstance(data, dict):
                return _ALLOW
            tool = data.get("name", "")
            args = data.get("arguments")
            if not isinstance(args, dict):
                return _ALLOW

            # Path 1: Omnigent runner tools (load_skill / read_skill_file).
            if tool in _SKILL_TOOLS:
                # load_skill uses "name"; read_skill_file uses "skill_name"
                skill_name = args.get("name") if tool == "load_skill" else args.get("skill_name")
                if skill_name and skill_name.lower() in blocked_lower:
                    return {
                        "result": "DENY",
                        "reason": f"Skill '{skill_name}' is blocked by policy",
                    }
                return _ALLOW

            # Path 2: Claude Code / Codex native Skill tool.
            # Fired via PreToolUse hook → Omnigent /policies/evaluate.
            if tool == _NATIVE_SKILL_TOOL:
                skill_name = args.get("skill")
                if skill_name and skill_name.lower() in blocked_lower:
                    return {
                        "result": "DENY",
                        "reason": f"Skill '{skill_name}' is blocked by policy",
                    }
                return _ALLOW

            return _ALLOW

        # ── Path 3: request phase for slash-command skill loads ───────
        # The Omnigent server converts ``/skill-name args`` into a synthetic
        # user message ``"/<name> <args>"`` and evaluates it at the
        # REQUEST phase.  Match ``/<blocked-name>`` at the start.
        if event_type == "request":
            text = request_user_text(event.get("data"))
            if text.startswith("/"):
                # Extract the command name: first token after "/".
                # ``split(None, ...)`` drops empty tokens, so a bare "/"
                # or a slash followed only by whitespace ("/   ") yields
                # an empty list — guard against IndexError.
                tokens = text[1:].split(None, 1)
                command = tokens[0] if tokens else ""
                if command.lower() in blocked_lower:
                    return {
                        "result": "DENY",
                        "reason": f"Skill '{command}' is blocked by policy",
                    }
            return _ALLOW

        return _ALLOW

    return evaluate


# ── Sandbox enforcement ────────────────────────────────────────────────────

_AGENT_START_TOOL = "sys_agent_start"

# Keys from ``OSEnvSandboxSpec`` that can be overridden by the policy.
# If the admin supplies a key not in this set, the policy silently
# ignores it — prevents injection of unsupported fields.
_SANDBOX_OVERRIDE_KEYS = frozenset(
    {
        "type",
        "read_paths",
        "write_paths",
        "write_files",
        "allow_network",
        "cwd_allow_hidden",
        "env_passthrough",
        "egress_rules",
        "egress_allow_private_destinations",
        "cwd_hidden_scan_max_entries",
        "cwd_hidden_scan_overflow",
    }
)


def enforce_sandbox(
    sandbox_type: str = "linux_bwrap",
    allow_network: bool = True,
    write_paths: list[str] | None = None,
    read_paths: list[str] | None = None,
    env_passthrough: list[str] | None = None,
) -> PolicyCallable:
    """Factory: force a sandbox configuration on every agent start.

    Intercepts the synthetic ``__agent_start`` tool call emitted by the
    runner before spawning an agent subprocess.  On match, returns ALLOW
    with a ``data`` payload whose ``sandbox`` field overrides the agent's
    declared sandbox config.  Fields not specified in the policy are
    inherited from the agent's existing config (merge, not replace).

    If the agent has no sandbox config at all, one is created from scratch
    using the policy's parameters.

    :param sandbox_type: Sandbox backend to force, e.g.
        ``"linux_bwrap"``, ``"darwin_seatbelt"``, ``"none"``.
        Defaults to ``"linux_bwrap"``.
    :param allow_network: Whether to allow network access.
        Defaults to ``True``.
    :param write_paths: Writable paths to enforce, e.g. ``["."]``.
        ``None`` means inherit the agent's existing ``write_paths``.
    :param read_paths: Read-only paths to enforce.
        ``None`` means inherit the agent's existing ``read_paths``.
    :param env_passthrough: Env vars to allow through the sandbox.
        ``None`` means inherit the agent's existing ``env_passthrough``.
    :returns: A policy callable that forces sandbox config on
        ``__agent_start`` tool calls.
    """
    # Build the override dict — only include keys the admin explicitly set.
    override: dict[str, object] = {
        "type": sandbox_type,
        "allow_network": allow_network,
    }
    if write_paths is not None:
        override["write_paths"] = write_paths
    if read_paths is not None:
        override["read_paths"] = read_paths
    if env_passthrough is not None:
        override["env_passthrough"] = env_passthrough

    def evaluate(event: PolicyEvent) -> PolicyResponse:
        """Evaluate whether the agent start should have sandbox forced.

        :param event: Policy event dict.
        :returns: ALLOW with forced sandbox ``data`` on
            ``__agent_start``, plain ALLOW otherwise.
        """
        if event.get("type") != "tool_call":
            return _ALLOW
        data = event.get("data")
        if not isinstance(data, dict):
            return _ALLOW
        tool = data.get("name", "")
        if tool != _AGENT_START_TOOL:
            return _ALLOW

        args = data.get("arguments")
        if not isinstance(args, dict):
            args = {}

        # Merge: existing sandbox config as base, policy overrides on top.
        raw_sandbox = args.get("sandbox")
        current_sandbox: dict[str, object] = (
            {key: value for key, value in raw_sandbox.items() if isinstance(key, str)}
            if isinstance(raw_sandbox, dict)
            else {}
        )
        forced_sandbox = {
            k: v for k, v in {**current_sandbox, **override}.items() if k in _SANDBOX_OVERRIDE_KEYS
        }

        return {
            "result": "ALLOW",
            "data": {
                "name": _AGENT_START_TOOL,
                "arguments": {**args, "sandbox": forced_sandbox},
            },
        }

    return evaluate


# ── PII detection on LLM requests ────────────────────────────────────────────

# Built-in PII categories with their regex patterns. The UI shows
# these as a multi-select checklist; authors never write raw regex.
_PII_CATEGORY_PATTERNS: dict[str, _re.Pattern[str]] = {
    "ssn": _re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": _re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "email": _re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    # Phone: international (+cc ...) and common local formats
    # (US, UK, JP, DE, etc.) in a single pattern.
    "phone": _re.compile(
        r"\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b"
        r"|\b\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"
        r"|\b0\d{1,4}[-.\s]?\d{2,4}[-.\s]?\d{3,4}\b"
    ),
}

# Labels shown in the UI checklist for each category.
_PII_CATEGORY_LABELS: dict[str, str] = {
    "ssn": "Social Security Number (US)",
    "credit_card": "Credit Card Number",
    "email": "Email Address",
    "phone": "Phone Number",
}


def deny_pii_in_llm_request(
    pii_types: list[str] | None = None,
    action: str = "DENY",
) -> PolicyCallable:
    """Factory: scan the system prompt preview in ``llm_request`` for PII.

    Selects PII categories from the built-in set and scans the
    ``system_prompt_preview`` field of the LLM request data. When
    any pattern matches, the policy returns the configured *action*
    (``DENY`` by default) with a reason naming the matched category.

    Only fires on ``llm_request`` events — all other phases pass
    through with ALLOW.

    :param pii_types: List of PII category keys to scan for, e.g.
        ``["ssn", "email"]``. Defaults to all built-in categories
        when ``None`` or empty. Unknown keys are silently ignored.
    :param action: The verdict to emit on match — ``"DENY"`` or
        ``"ASK"``. Defaults to ``"DENY"``.
    :returns: A policy callable that scans LLM request prompts
        for PII patterns.
    """
    if pii_types:
        selected = {k: v for k, v in _PII_CATEGORY_PATTERNS.items() if k in pii_types}
    else:
        # None or empty list → all categories enabled.
        selected = dict(_PII_CATEGORY_PATTERNS)

    effective_action: Literal["DENY", "ASK"] = "ASK" if action == "ASK" else "DENY"

    def evaluate(event: PolicyEvent) -> PolicyResponse:
        """Evaluate user input or LLM request for PII matches.

        Fires on two phases so PII is caught regardless of harness:

        - ``request``: user message text, enforced universally on
          the Omnigent server for every harness (including supervisor,
          native).
        - ``llm_request``: full LLM call metadata (system prompt +
          last user message), enforced via the harness callback
          for executors that support it.

        :param event: Policy event dict.
        :returns: DENY/ASK if PII found, ALLOW otherwise.
        """
        event_type = event.get("type")

        if event_type == "request":
            # REQUEST phase: scan the typed message plus each text attachment.
            # ``data`` is {"user_content", "attachments"} from the input gate.
            data = event.get("data")
            result = _scan_text(request_user_text(data))
            if result is not _ALLOW:
                return result
            for attachment in request_attachments(data):
                att_text = attachment.get("text", "")
                if isinstance(att_text, str) and att_text:
                    result = _scan_text(att_text)
                    if result is not _ALLOW:
                        return result
            return _ALLOW

        if event_type == "llm_request":
            # LLM_REQUEST phase: scan system prompt + user message.
            data = event.get("data")
            if not isinstance(data, dict):
                return _ALLOW
            for field in ("system_prompt_preview", "last_user_message"):
                text = data.get(field, "")
                if isinstance(text, str) and text:
                    result = _scan_text(text)
                    if result is not _ALLOW:
                        return result
            return _ALLOW

        return _ALLOW

    def _scan_text(text: str) -> PolicyResponse:
        """Scan a text string against selected PII patterns.

        :param text: The string to scan.
        :returns: DENY/ASK if PII found, ALLOW otherwise.
        """
        for category, regex in selected.items():
            match = regex.search(text)
            if match:
                label = _PII_CATEGORY_LABELS.get(category, category)
                return {
                    "result": effective_action,
                    "reason": (f"PII detected ({label}): '{match.group()[:20]}...'"),
                }
        return _ALLOW

    return evaluate


# ── Registry ─────────────────────────────────────────────────────────────────

POLICY_REGISTRY: list[dict[str, object]] = [
    {
        "handler": "omnigent.policies.builtins.safety.max_tool_calls_per_session",
        "kind": "factory",
        "name": "Limit Tool Calls Per Session",
        "description": "Limits the total number of tool calls across the entire session "
        "using session_state to persist the counter",
        "params_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum tool calls allowed across the session",
                    "default": 100,
                },
            },
            "required": ["limit"],
        },
    },
    {
        "handler": "omnigent.policies.builtins.safety.tool_budget_per_turn",
        "kind": "factory",
        "name": "Limit Tool Calls Per Turn",
        "description": (
            "Resets on each user request, limits tool calls within the turn, "
            "and stops further calls after a bounded run of consecutive failures."
        ),
        "params_schema": {
            "type": "object",
            "properties": {
                "max_calls": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum tool calls allowed in one user turn.",
                    "default": 12,
                },
                "max_failures": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Failed tool results allowed before later calls are denied.",
                    "default": 2,
                },
            },
        },
    },
    {
        "handler": "omnigent.policies.builtins.safety.detect_loop",
        "kind": "factory",
        "name": "Detect Tool Call Retry Loops",
        "description": "Detects when the agent is stuck retrying the same tool call. "
        "Returns the configured action when the same (tool, args) repeats N times "
        "within a sliding window of recent calls. Set ignore_arg_keys and "
        "normalize_uri_args to key on the call's target instead, which catches an "
        "agent guessing at a ref grammar rather than repeating one ref",
        "params_schema": {
            "type": "object",
            "properties": {
                "window": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Number of recent tool calls to consider",
                    "default": 10,
                },
                "threshold": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Number of identical repeats within the window to trigger",
                    "default": 3,
                },
                "action": {
                    "type": "string",
                    "enum": ["ASK", "DENY"],
                    "description": "Response when a repeated-call loop is detected",
                    "default": "ASK",
                },
                "exempt_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tool names allowed to repeat without loop detection",
                    "default": [],
                },
                "reset_on_request": {
                    "type": "boolean",
                    "description": "Clear recent-call history for each user turn",
                    "default": True,
                },
                "ignore_arg_keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Argument names dropped before hashing, for fields "
                    "that pick a backend rather than a target",
                    "default": [],
                },
                "normalize_uri_args": {
                    "type": "boolean",
                    "description": "Key string arguments on their target: percent-decode, "
                    "strip scheme:// prefixes, drop a trailing slash, lowercase",
                    "default": False,
                },
                "latch": {
                    "type": "boolean",
                    "description": "Once this guard or the thrashing guard trips, deny "
                    "every remaining tool call in the turn with the same instruction",
                    "default": False,
                },
            },
        },
    },
    {
        "handler": "omnigent.policies.builtins.safety.require_evidence_before_write",
        "kind": "factory",
        "name": "Require Evidence Before Authoring A Contract",
        "description": "Denies writing a file that states a contract owned outside the "
        "repository (third-party env vars, image tags, endpoints) until the session has "
        "consulted a primary source. Which files and what counts as evidence are the "
        "agent author's parameters, not policy baked in here",
        "params_schema": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Globs naming files that state an external contract",
                    "default": [],
                },
                "evidence_skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Skill names whose loading counts as evidence",
                    "default": [],
                },
                "evidence_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tool names whose use counts as evidence",
                    "default": [],
                },
                "write_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tools treated as authoring",
                    "default": ["sys_os_write", "sys_os_edit"],
                },
                "action": {
                    "type": "string",
                    "enum": ["ASK", "DENY"],
                    "description": "Response when evidence is missing",
                    "default": "DENY",
                },
            },
        },
    },
    {
        "handler": "omnigent.policies.builtins.safety.require_progress",
        "kind": "factory",
        "name": "Require Progress Before More Evidence",
        "description": "Denies further tool calls once the agent has spent its budget "
        "gathering evidence without changing anything. Distinct from the retry-loop "
        "guard: a hundred different reads repeat no call and still make no progress",
        "params_schema": {
            "type": "object",
            "properties": {
                "budget": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Tool calls allowed since anything last changed",
                    "default": 25,
                },
                "progress_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tool names that count as progress and reset the budget",
                    "default": [],
                },
                "action": {
                    "type": "string",
                    "enum": ["ASK", "DENY"],
                    "description": "Response once the budget is spent",
                    "default": "DENY",
                },
                "exempt_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tool names that neither spend nor reset the budget",
                    "default": [],
                },
                "reset_on_request": {
                    "type": "boolean",
                    "description": "Restore a full budget for each user turn",
                    "default": True,
                },
            },
        },
    },
    {
        "handler": "omnigent.policies.builtins.safety.ask_on_os_tools",
        "kind": "callable",
        "name": "Require Approval for File & Shell Operations",
        "description": "Asks for user approval before any file or shell tool call — "
        "covers Omnigent sys_os_* tools, Claude Code native tools "
        "(Bash, Read, Write, Edit, Glob, Grep), Codex native tools, "
        "opencode native tools (bash, edit, read, grep, glob), "
        "and Hermes Agent tools (terminal, execute_code, read_file, write_file, search_files)",
        "params_schema": None,
    },
    {
        "handler": "omnigent.policies.builtins.safety.block_skills",
        "kind": "factory",
        "name": "Block Specific Skills",
        "description": "Prevents the agent from loading specific skills. "
        "Intercepts load_skill/read_skill_file (non-native harnesses) and the "
        "native Skill tool (claude-native/codex-native via PreToolUse hook)",
        "params_schema": {
            "type": "object",
            "properties": {
                "blocked": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Skill names to block (case-insensitive)",
                },
            },
            "required": ["blocked"],
        },
    },
    {
        "handler": "omnigent.policies.builtins.safety.enforce_sandbox",
        "kind": "factory",
        "name": "Enforce Sandbox on Agent Start",
        "description": "Forces a specific sandbox configuration (e.g. linux_bwrap) "
        "on every agent start. Intercepts the synthetic __agent_start tool call "
        "and overrides the agent's sandbox config.",
        "params_schema": {
            "type": "object",
            "properties": {
                "sandbox_type": {
                    "type": "string",
                    "description": "Sandbox backend to force (linux_bwrap, darwin_seatbelt, none)",
                    "default": "linux_bwrap",
                },
                "allow_network": {
                    "type": "boolean",
                    "description": "Whether to allow network access",
                    "default": True,
                },
                "write_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Writable paths to enforce (null inherits agent's config)",
                },
                "read_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Read-only paths to enforce (null inherits agent's config)",
                },
                "env_passthrough": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Env vars to allow through the sandbox "
                    "(null inherits agent's config)",
                },
            },
        },
    },
    {
        "handler": "omnigent.policies.builtins.safety.deny_pii_in_llm_request",
        "kind": "factory",
        "name": "Deny PII in LLM Requests",
        "description": "Scans user messages and LLM request prompts for PII "
        "(SSN, credit card, email, phone). Works with all harnesses.",
        "params_schema": {
            "type": "object",
            "properties": {
                "pii_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["ssn", "credit_card", "email", "phone"],
                    },
                    "uniqueItems": True,
                    "description": "PII categories to scan for. Leave empty to enable all.",
                    "default": ["ssn", "credit_card", "email", "phone"],
                },
                "action": {
                    "type": "string",
                    "enum": ["DENY", "ASK"],
                    "description": "Action when PII is detected",
                    "default": "DENY",
                },
            },
        },
    },
]
